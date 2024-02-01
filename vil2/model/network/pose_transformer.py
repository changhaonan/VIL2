"""Pose transformer network."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vil2.model.embeddings.pct import PointTransformerEncoderSmall, EncoderMLP, DropoutSampler
from vil2.model.embeddings.sinusoidal import PositionalEmbedding
from vil2.model.network.genpose_modules import Linear


class PoseTransformer(nn.Module):
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
        # Learnable embedding
        self.seg_embedding_1 = nn.Embedding(1, pcd_output_dim)
        self.seg_embedding_2 = nn.Embedding(1, pcd_output_dim)
        self.semantic_embedding = nn.Embedding(max_semantic_size, pcd_output_dim)
        self.use_semantic_label = use_semantic_label
        self.encoder_layer = TransformerEncoderLayer(
            d_model=pcd_output_dim,
            nhead=num_attention_heads,
            dim_feedforward=encoder_hidden_dim,
            dropout=encoder_dropout,
            activation=encoder_activation,
            batch_first=True,
            norm_first=True,
        )
        self.converge_step_embedding = nn.Embedding(max_converge_step, pcd_output_dim)
        self.joint_transformer = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=encoder_num_layers)

        """ rotation regress head """
        init_zero = dict(
            init_mode="kaiming_uniform", init_weight=0, init_bias=0
        )  # init the final output layer's weights to zeros
        self.fusion_tail_rot_x = nn.Sequential(
            nn.Linear(pcd_output_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            Linear(fusion_projection_dim, 3, **init_zero),
        )
        self.fusion_tail_rot_y = nn.Sequential(
            nn.Linear(pcd_output_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            Linear(fusion_projection_dim, 3, **init_zero),
        )

        """ translation regress head """
        self.fusion_tail_trans = nn.Sequential(
            nn.Linear(pcd_output_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            Linear(fusion_projection_dim, 3, **init_zero),
        )

    def encode_cond(
        self,
        coord1: torch.Tensor,
        normal1: torch.Tensor,
        color1: torch.Tensor | None,
        label1: torch.Tensor,
        coord2: torch.Tensor,
        normal2: torch.Tensor,
        color2: torch.Tensor | None,
        label2: torch.Tensor,
    ):
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
            (B, N + M + 1, C)
        """
        # Encode geometry1 and geometry2
        if color1 is None and color2 is None:
            pcd_feat1 = normal1  # (B, N, 3)
            pcd_feat2 = normal2  # (B, M, 3)
        elif color1 is not None and color2 is not None:
            pcd_feat1 = torch.cat((normal1, color1), dim=-1)  # (B, N, 6)
            pcd_feat2 = torch.cat((normal2, color2), dim=-1)  # (B, M, 6)
        else:
            raise ValueError("Color must be provided for both or neither")
        center1, enc_pcd1 = self.pcd_encoder(coord1, pcd_feat1)
        center2, enc_pcd2 = self.pcd_encoder(coord2, pcd_feat2)

        enc_pcd1 = enc_pcd1.transpose(1, 2)  # (B, N, C)
        enc_pcd2 = enc_pcd2.transpose(1, 2)  # (B, M, C)
        enc_position1 = self.position_encoder(center1).view(center1.size(0), 1, -1)  # (B, 1, C)
        enc_position2 = self.position_encoder(center2).view(center2.size(0), 1, -1)  # (B, 1, C)
        enc_pcd1 = torch.cat((enc_pcd1, enc_position1), dim=1)
        enc_pcd2 = torch.cat((enc_pcd2, enc_position2), dim=1)
        # Apply segment embedding
        enc_pcd1 += self.seg_embedding_1.weight[None, ...]  # (B, N, C)
        enc_pcd2 += self.seg_embedding_2.weight[None, ...]  # (B, M, C)

        # Add special tokens as a sum of semantic embedding
        batch_size = enc_pcd1.size(0)
        if self.use_semantic_label:
            special_token = self.semantic_embedding(label1.view(-1)) + self.semantic_embedding(label2.view(-1))
            special_token = special_token.view(batch_size, 1, -1)
        else:
            special_token = self.semantic_embedding(torch.zeros(batch_size, dtype=torch.long, device=self.device))
            special_token = special_token.view(batch_size, 1, -1)
        cond_feat = torch.cat((special_token, enc_pcd1, enc_pcd2), dim=1)  # (B, N + M + 1, C)
        return cond_feat

    def forward(
        self,
        cond_feat: torch.Tensor,
        converge_step: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cond_feat: (B, N + M + 1, C)
            converge_step: (B, 1)
        Returns:
            (B, 3)
        """
        enc_conv_step = self.converge_step_embedding(converge_step)
        total_feat = torch.cat((cond_feat, enc_conv_step), dim=1)
        encoder_output = self.joint_transformer(total_feat)  # (B, N + M + 1 + 1, C)
        encoder_output = encoder_output[:, 0, :]  # (B, C)  # only use the first token

        # Predict pose
        rot_x = self.fusion_tail_rot_x(encoder_output)
        rot_y = self.fusion_tail_rot_y(encoder_output)
        trans = self.fusion_tail_trans(encoder_output)
        x = torch.cat([trans, rot_x, rot_y], dim=-1)
        return x
