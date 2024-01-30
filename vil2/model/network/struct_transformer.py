"""Pose transformer network."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vil2.model.embeddings.pct import StructPointTransformerEncoderSmall, EncoderMLP, DropoutSampler


class StructTransformer(nn.Module):
    def __init__(
        self,
        pcd_output_dim,
        use_pcd_mean_center: bool,
        num_attention_heads: int = 2,
        encoder_num_layers: int = 2,
        encoder_hidden_dim: int = 256,
        encoder_dropout: float = 0.1,
        obj_dropout: float = 0.1,
        encoder_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encode pcd features
        self.object_encoder = StructPointTransformerEncoderSmall(
            output_dim=pcd_output_dim,
            input_dim=6,
            mean_center=use_pcd_mean_center,
        )
        # Encode position
        self.position_encoder = nn.Sequential(nn.Linear(3 + 3 * 3, 120))
        encoder_layers = TransformerEncoderLayer(2*240, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.transformer = TransformerEncoder(
            encoder_layers, encoder_num_layers
        )
        # self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)
        self.mlp = EncoderMLP(pcd_output_dim, 240, uses_pt=True)
        self.obj_dist = DropoutSampler(2*240, 3 + 6, dropout_rate=obj_dropout)

    def encode_cond(
        self,
        coord1: torch.Tensor,
        color1: torch.Tensor,
        coord2: torch.Tensor,
        color2: torch.Tensor,
    ):
        """
        Args:
            coord1: (B, N, 3)
            color1: (B, N, 3)
            coord2: (B, M, 3)
            color2: (B, M, 3)
        Returns:
            (B, N + M, C)
        """
        # Encode geometry1 and geometry2
        center1, enc_pcd1 = self.object_encoder(coord1, color1)
        center2, enc_pcd2 = self.object_encoder(coord2, color2)

        enc_pcd1 = self.mlp(enc_pcd1, center1)
        enc_pcd2 = self.mlp(enc_pcd2, center2)
        cond_feat = torch.cat((enc_pcd1, enc_pcd2), dim=1)  # (B, N + M, C)
        return cond_feat

    def forward(
        self,
        coord1: torch.Tensor,
        color1: torch.Tensor,
        coord2: torch.Tensor,
        color2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coord1: (B, N, 3)
            normal1: (B, N, 3)
            color1: (B, N, 3)
            coord2: (B, M, 3)
            normal2: (B, M, 3)
            color2: (B, M, 3)
            label1: (B, 1)
            label2: (B, 1)
        Returns:
            (B, 3)
        """
        total_feat = self.encode_cond(coord1, color1, coord2, color2)
        encoder_output = self.transformer(total_feat)  # (B, N + M, C)
        x = self.obj_dist(encoder_output)  # (B, 9)
        return x
