import torch.nn as nn
import torch
from .sinusoidal import SinusoidalEmbedding
from .semantic import SemanticEmbedding
from .learnable import LearnableEmbedding
from .positional import PositionalEncoding


class EmbeddingsHub(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()
        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "semantic":
            self.layer = SemanticEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size, **kwargs)
        elif type == "positional":
            self.layer = PositionalEncoding(**kwargs)
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)
