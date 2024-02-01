"""Pose UVIT network."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vil2.model.embeddings.pct import PointTransformerEncoderSmall, EncoderMLP, DropoutSampler
from vil2.model.embeddings.sinusoidal import PositionalEmbedding
from vil2.model.network.genpose_modules import Linear


class PoseUVIT(nn.Module):
    def __init__(
        self,
        pcd_input_dim,
        pcd_output_dim,
        points_pyramid: list,
        use_pcd_mean_center: bool,
        num_attention_heads: int = 2,
        encoder_num_layers: int = 2,
        encoder_hidden_dim: int = 256,
        encoder_dropout: float = 0.1,
        encoder_activation: str = "relu",
        fusion_projection_dim: int = 512,
        max_semantic_size: int = 10,
        use_semantic_label: bool = True,
        max_converge_step: int = 10,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Encode pcd features
        self.pcd_encoder = PointTransformerEncoderSmall(
            output_dim=pcd_output_dim,
            input_dim=pcd_input_dim,
            points_pyramid=points_pyramid,
            mean_center=use_pcd_mean_center,
        )
        # Encode position
        self.position_encoder = nn.Sequential(
            nn.Linear(3, pcd_output_dim),
            nn.LayerNorm(pcd_output_dim),
            nn.ReLU(),
            nn.Linear(pcd_output_dim, pcd_output_dim),
            nn.LayerNorm(pcd_output_dim),
            nn.ReLU(),
        )