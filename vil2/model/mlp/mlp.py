from __future__ import annotations
import torch
import torch.nn as nn
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


class ResBlock(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, input_dim: int = 8, global_cond_dim: int = 32, diffusion_step_embed_dim: int = 8, nlayers: int = 3, hidden_size: int = 1024):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_size = input_dim
        self.time_emb_size = diffusion_step_embed_dim
        self.use_conditioning = global_cond_dim > 0

        input_size = input_dim + global_cond_dim + self.time_emb_size if self.use_conditioning else input_dim + self.time_emb_size
        layers = [nn.Linear(input_size, hidden_size), nn.GELU()]
        for _ in range(nlayers):
            layers.append(ResBlock(hidden_size))

        layers.append(nn.Linear(hidden_size, input_size))

        self.joint_mlp = nn.Sequential(*layers)
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, 4 * dsed),
            nn.GELU(),
            nn.Linear(4 * dsed, dsed),
        )

    def forward(self, sample: torch.Tensor,  timestep: int, global_cond: torch.Tensor | None):
        sample = sample.to(dtype=torch.float32, device=self.device)

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            embeddings = torch.concat([sample, global_cond, time_emb], dim=-1)
        else:
            embeddings = torch.concat([sample, time_emb], dim=-1)

        output_scene = self.joint_mlp(embeddings)[:, :self.emb_size]

        return output_scene
