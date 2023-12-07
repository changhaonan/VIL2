import torch.nn as nn
import torch


class SemanticEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor):
        return x

    def __len__(self):
        return self.size
