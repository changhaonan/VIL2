"""Pose transformer network Version 2. Using PointTransformerV2 (Unet)"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from vil2.model.embeddings.pct import PointTransformerEncoderSmall, EncoderMLP, DropoutSampler
from vil2.model.network.geometric import PointTransformerNetwork
from vil2.model.network.genpose_modules import Linear


class PoseTransformerV2(nn.Module):
    def __init__(
        self,
        grid_sizes,  # (2)
        depths,  # (3)
        dec_depths,  # (2)
        hidden_dims,  # (3)
        n_heads,  # (3)
        ks,  # (3)
        in_dim,
        fusion_projection_dim,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encode pcd features
        self.target_pcd_transformer = PointTransformerNetwork(
            grid_sizes,  # (2)
            depths,  # (3)
            dec_depths,  # (2)
            hidden_dims,  # (3)
            n_heads,  # (3)
            ks,  # (3)
            in_dim,
            skip_dec=True,
        )
        self.fixed_pcd_transformer = PointTransformerNetwork(
            grid_sizes,  # (2)
            depths,  # (3)
            dec_depths,  # (2)
            hidden_dims,  # (3)
            n_heads,  # (3)
            ks,  # (3)
            in_dim,
            skip_dec=False,
        )

        # Learnable embedding
        self.seg_embedding = nn.ModuleList([nn.Embedding(2, feat_dim) for feat_dim in hidden_dims])
        self.pose_embedding = nn.Embedding(1, hidden_dims[-1])

        # Pose refiner
        self.refine_decoders = nn.ModuleList()
        for i in range(len(hidden_dims)):
            refine_encoder_layer = TransformerDecoderLayer(
                d_model=hidden_dims[i],
                nhead=n_heads[i],
                dim_feedforward=hidden_dims[i] * 4,
                dropout=0.1,
                activation="relu",
            )
            refine_decoder = TransformerDecoder(refine_encoder_layer, num_layers=dec_depths[i])
            self.refine_decoders.append(refine_decoder)

        # Pose decoder
        init_zero = dict(
            init_mode="kaiming_uniform", init_weight=0, init_bias=0
        )  # init the final output layer's weights to zeros
        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            Linear(fusion_projection_dim, 3, **init_zero),
        )

    def encode_cond(
        self,
        coord1: torch.Tensor,
        normal1: torch.Tensor,
        color1: torch.Tensor | None,
        coord2: torch.Tensor,
        normal2: torch.Tensor,
        color2: torch.Tensor | None,
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

        target_points = torch.cat((coord1, pcd_feat1), dim=-1)
        fixed_points = torch.cat((coord2, pcd_feat2), dim=-1)
        # Encode target pcd
        target_points = self.target_pcd_transformer(target_points)
        # Encode fixed pcd
        fixed_points, all_fixed_points, cluster_indexes = self.fixed_pcd_transformer(fixed_points, return_full=True)
        return target_points, all_fixed_points, cluster_indexes

    def forward(
        self,
        target_points: torch.Tensor,
        all_fixed_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_points: (B, N, C)
            all_fixed_points: (B, M, C)
        Returns:
            (B, 3)
        """
        pose_token = self.pose_embedding.weight[0].unsqueeze(0).expand(target_points.size(0), -1)
        target_points = torch.cat((pose_token[:, None, :].expand(-1, target_points.size(1), -1), target_points), dim=1)

        # Refine pose tokens
        for i in range(len(self.refine_decoders)):
            target_points = self.refine_decoders[i](
                target_points.transpose(0, 1), all_fixed_points[-i].transpose(0, 1)
            ).transpose(0, 1)

        # Decode pose
        pose_pred = self.pose_decoder(target_points[:, 0])
        return pose_pred
