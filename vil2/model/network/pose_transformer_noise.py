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
        max_timestep: int = 100,
        translation_only: bool = True,
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
        self.translation_only = translation_only
        # Encode noisy input
        self.noisy_encoder = nn.Sequential(
            nn.Linear(9, pcd_output_dim),
            nn.LayerNorm(pcd_output_dim),
            nn.ReLU(),
            nn.Linear(pcd_output_dim, pcd_output_dim),
            nn.LayerNorm(pcd_output_dim),
            nn.ReLU(),
        )
        self.time_embedding = nn.Embedding(max_timestep, pcd_output_dim)

    def forward(
        self,
        noisy_pose: torch.Tensor,
        t: torch.Tensor,
        cond_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_pose: (B, 9)
            t: (B, 1)
            cond_feat: (B, N + M + 1, C)
        Returns:
            (B, 3)
        """
        batch_size = cond_feat.shape[0]

        # Encode noisy input
        noisy_pose = noisy_pose.view(batch_size, 1, -1)  # (B, 1, 9)
        enc_noisy_pose = self.noisy_encoder(noisy_pose)  # (B, 1, C)

        # Encode time
        time = t.view(batch_size, 1)
        enc_time = self.time_embedding(time)  # (B, 1, C)

        # Concatenate noisy input & time
        total_feat = torch.cat((cond_feat, enc_noisy_pose, enc_time), dim=1)  # (B, N + M + 1 + 1 + 1, C)
        encoder_output = self.joint_transformer(total_feat)  # (B, N + M + 1 + 1 + 1, C)
        encoder_output = encoder_output[:, 0, :]  # (B, C)  # only use the first token

        # Predict pose
        if not self.translation_only:
            rot_x = self.fusion_tail_rot_x(encoder_output)
            rot_y = self.fusion_tail_rot_y(encoder_output)
            trans = self.fusion_tail_trans(encoder_output)
            x = torch.cat([trans, rot_x, rot_y], dim=-1)  # (B, 9)
        else:
            rot_x = torch.zeros((batch_size, 3), device=noisy_pose.device)
            rot_y = torch.zeros((batch_size, 3), device=noisy_pose.device)
            trans = self.fusion_tail_trans(encoder_output)  # (B, 3)
            x = torch.cat([trans, rot_x, rot_y], dim=-1)  # (B, 9)
        return x
