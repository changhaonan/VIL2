"""Pose transformer noise network."""
from __future__ import annotations
import torch
import torch.nn as nn
from vil2.model.network.pose_transformer import PoseTransformer


class PoseTransformerNoiseNet(PoseTransformer):
    """Pose transformer noise network."""

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
    ) -> None:
        super().__init__(
            pcd_input_dim,
            pcd_output_dim,
            points_pyramid,
            use_pcd_mean_center,
            num_attention_heads,
            encoder_num_layers,
            encoder_hidden_dim,
            encoder_dropout,
            encoder_activation,
            fusion_projection_dim,
            max_semantic_size,
            use_semantic_label,
        )

        # Encode noisy input
        self.noisy_encoder = nn.Sequential(
            nn.Linear(9, pcd_output_dim),
            nn.LayerNorm(pcd_output_dim),
            nn.ReLU(),
            nn.Linear(pcd_output_dim, pcd_output_dim),
            nn.LayerNorm(pcd_output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        noisy_pose: torch.Tensor,
        coord1: torch.Tensor,
        normal1: torch.Tensor,
        color1: torch.Tensor,
        label1: torch.Tensor,
        coord2: torch.Tensor,
        normal2: torch.Tensor,
        color2: torch.Tensor,
        label2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_pose: (B, 9)
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
        # Encode geometry1 and geometry2
        pcd_feat1 = torch.cat((normal1, color1), dim=-1)  # (B, N, 6)
        pcd_feat2 = torch.cat((normal2, color2), dim=-1)  # (B, M, 6)
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

        # Encode noisy input
        noisy_pose = noisy_pose.view(batch_size, 1, -1)  # (B, 1, 9)
        enc_noisy_pose = self.noisy_encoder(noisy_pose)  # (B, 1, C)

        # Concatenate noisy input
        cond_feat = torch.cat((cond_feat, enc_noisy_pose), dim=1)  # (B, N + M + 2, C)
        encoder_output = self.joint_transformer(cond_feat)  # (B, N + M + 2, C)
        encoder_output = encoder_output[:, 0, :]  # (B, C)  # only use the first token

        # Predict pose
        rot_x = self.fusion_tail_rot_x(encoder_output)
        rot_y = self.fusion_tail_rot_y(encoder_output)
        trans = self.fusion_tail_trans(encoder_output)
        x = torch.cat([trans, rot_x, rot_y], dim=-1)  # (B, 9)
        return x
