import torch
import torch.nn as nn
from vil2.model.embeddings.pct import PointTransformerEncoderSmall, DropoutSampler, EncoderMLP, PointNet
from vil2.model.embeddings.sinusoidal import PositionalEmbedding

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
    def __init__(self, in_channels, out_channels, cond_dim) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(in_channels, out_channels), nn.GELU()),
                                     nn.Sequential(nn.Linear(out_channels, out_channels), nn.GELU())])
        cond_channels = 2 * out_channels
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.GELU(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1)))
        self.residual_linear = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(-1, 2, self.out_channels)
        out = out * embed[:, 0, ...] + embed[:, 1, ...]
        out = self.blocks[1](out) + self.residual_linear(x)
        return out
    
class ConditionalUnetMLP(nn.Module):
    def __init__(self, input_dim: int = 256, global_cond_dim: int = 1024, diffusion_step_embed_dim: int = 128, 
                 down_dims: list = [256, 512, 1024], use_global_geometry: bool = False, downsample_pcd_enc: bool=False, 
                 downsample_size: int = 256, use_pointnet: bool = False, use_dropout_sampler: bool = False, rotation_orthogonalization: bool = False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.down_dims = down_dims

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        self.diffusion_step_encoder = nn.Sequential(
            PositionalEmbedding(num_channels=diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
            nn.ReLU(inplace=True)
        )
        if downsample_pcd_enc:
            global_cond_dim = downsample_size
        cond_dim = diffusion_step_embed_dim + global_cond_dim*2

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

        self.final_linear = nn.Sequential(nn.Linear(start_dim, 9)) if not use_dropout_sampler else DropoutSampler(start_dim, 9, dropout_rate=0.0)

        self.pcd_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=False) if not use_pointnet else PointNet(emb_dims=256)
        self.use_pointnet = use_pointnet
        self.pcd_mlp_encoder = EncoderMLP(256, global_cond_dim, uses_pt=True)
        self.pose_encoder = nn.Sequential(
                                nn.Linear(9, input_dim),
                                nn.ReLU(True),
                                nn.Linear(input_dim, input_dim),
                                nn.ReLU(True),
                                )
        self.psenc = nn.Sequential(
                        nn.Linear(global_cond_dim, downsample_size))

        self.up_modules = up_modules
        self.down_modules = down_modules
        self.use_global_geometry = use_global_geometry
        self.downsample_pcd_enc = downsample_pcd_enc
        self.rotation_orthogonalization = rotation_orthogonalization
  
    def forward(self, sample, timestep, geometry1=None, geometry2=None):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if geometry1 is not None:
            if self.use_global_geometry:
                if self.downsample_pcd_enc:
                    geometry1 = self.psenc(geometry1.to(self.device))
                    geometry2 = self.psenc(geometry2.to(self.device))
                if len(geometry1.shape) == 3:
                    geometry1 = geometry1.squeeze(1)
                    geometry2 = geometry2.squeeze(1)
                global_feature = torch.cat([geometry1, geometry2, global_feature], dim=-1)
            else:
                center1, enc_pcd1 = self.pcd_encoder(geometry1[:, :, :3], geometry1[:, :, 3:]) if not self.use_pointnet else self.pcd_encoder(geometry1[:, :, :3])
                center2, enc_pcd2 = self.pcd_encoder(geometry2[:, :, :3], geometry2[:, :, 3:]) if not self.use_pointnet else self.pcd_encoder(geometry2[:, :, :3])
                enc_pcd1 = self.pcd_mlp_encoder(enc_pcd1, center1)
                enc_pcd2 = self.pcd_mlp_encoder(enc_pcd2, center2)
                global_feature = torch.cat([enc_pcd1, enc_pcd2, global_feature], dim=-1)

        x = self.pose_encoder(sample)
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

        x = self.final_linear(x)
        if self.rotation_orthogonalization:
            process_decoder_output = x.clone()
            v1 = process_decoder_output[:, 3:6]
            v2 = process_decoder_output[:, 6:]
            # Ensure the norm of v1 is non-zero
            if torch.all(torch.norm(v1, dim=-1) > 0):
                v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
                v2 = v2 - torch.sum(v2 * v1, dim=-1, keepdim=True) * v1
                # Ensure the norm of v2 is non-zero
                if torch.all(torch.norm(v2, dim=-1) > 0):
                    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
                    output_scene = torch.concatenate([process_decoder_output[:, :3], v1, v2], dim=-1)
                    return output_scene
        return x
