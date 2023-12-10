import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1dWithPooling(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.pool(x)


class Upsample1dWithUpsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.upsample(x)


class ConditionalResidualBlock(nn.Module):
    """Modified from DiffusionPolicy's 1D ConditionalResidualBlock"""

    def __init__(self, in_channels, out_channels, cond_dim) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Linear(out_channels, out_channels),
                    nn.GELU(),
                )
            ]
        )

        # FiLM modulation
        # predict per-feature scale and bias
        cond_channels = 2 * out_channels
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.GELU(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimension matches
        self.residual_linear = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        # FiLM modulation
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels)
        scales = embed[:, 0, ...]
        biases = embed[:, 1, ...]

        out = out * scales + biases
        out = self.blocks[1](out)
        out = out + self.residual_linear(x)
        return out


class ConditionalUnetMLP(nn.Module):
    """Multi-layer perceptron model."""

    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim: int = 8, down_dims=[256, 512, 1024]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.down_dims = down_dims

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, 4 * dsed),
            nn.GELU(),
            nn.Linear(4 * dsed, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock(mid_dim, mid_dim, cond_dim),
                ConditionalResidualBlock(mid_dim, mid_dim, cond_dim),
            ]
        )

        down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                        ),
                        ConditionalResidualBlock(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                        ),
                        Downsample1dWithPooling(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                        ),
                        ConditionalResidualBlock(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                        ),
                        Upsample1dWithUpsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_linear = nn.Sequential(
            nn.Linear(start_dim, input_dim),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_liner = final_linear

    def forward(self, sample, timestep, global_cond=None):

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x = sample
        h = []  # skip connections
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            # x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            # x = upsample(x)

        x = self.final_liner(x)
        return x
