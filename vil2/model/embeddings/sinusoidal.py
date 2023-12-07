import torch
import torch.nn as nn
import math


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale
        half_size = self.size // 2
        self.emb = torch.log(torch.Tensor([24.0])) / (half_size - 1)
        self.emb = torch.exp(torch.arange(half_size, dtype=torch.float) * -self.emb)

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        emb = x.unsqueeze(-1) * self.emb.unsqueeze(0).to(x.device)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

    def __len__(self):
        return self.size


def SinusoidalTimeEmbedding(timesteps, embedding_dim, base=10000.0):
    position = timesteps.unsqueeze(1).float()

    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(base) / embedding_dim)).to(timesteps.device)
    ste = torch.zeros(timesteps.shape[0], embedding_dim).to(timesteps.device)

    ste[:, 0::2] = torch.sin(position * div_term)
    ste[:, 1::2] = torch.cos(position * div_term)

    return ste
