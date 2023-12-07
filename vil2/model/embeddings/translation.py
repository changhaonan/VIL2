import torch
import torch.nn as nn


class TranslationEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale
        half_size = self.size // 2
        self.emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        self.emb = torch.exp(torch.arange(half_size, dtype=torch.float) * -self.emb)

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        emb = x.unsqueeze(-1) * self.emb.unsqueeze(0).to(x.device)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

    def __len__(self):
        return self.size
