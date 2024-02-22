"""Pose transformer network Version 3. Using PointTransformerV2 (Unet) & Cross knn attention"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from vil2.model.embeddings.pct import PointTransformerEncoderSmall, EncoderMLP, DropoutSampler
from vil2.model.network.geometric import (
    PointTransformerNetwork,
    to_dense_batch,
    to_flat_batch,
    batch2offset,
    offset2batch,
    knn,
    PointBatchNorm,
    KnnTransformer,
)
from timm.models.layers import DropPath
from vil2.model.network.genpose_modules import Linear
from vil2.utils.pcd_utils import visualize_tensor_pcd
import open3d as o3d


class KnnTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_channels,
        n_heads,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_channels = embed_channels
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.enable_checkpoint = enable_checkpoint
        # Module
        self.self_attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        for i in range(num_layers):
            self.self_attns.append(
                KnnTransformer(
                    embed_channels,
                    n_heads,
                    qkv_bias=qkv_bias,
                    pe_multiplier=pe_multiplier,
                    pe_bias=pe_bias,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    enable_checkpoint=enable_checkpoint,
                )
            )
            self.cross_attns.append(
                KnnTransformer(
                    embed_channels,
                    n_heads,
                    qkv_bias=qkv_bias,
                    pe_multiplier=pe_multiplier,
                    pe_bias=pe_bias,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    enable_checkpoint=enable_checkpoint,
                )
            )
        # MLP
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, feat, coord, knn_indexes, cross_knn_indexes, context_feat, context_coord):
        for i in range(self.num_layers):
            feat = self.act(self.norm1(self.fc1(feat)))
            # Self attention
            sattn = self.self_attns[i](feat=feat, coord=coord, knn_indexes=knn_indexes)
            feat = feat + self.drop_path(sattn)
            feat = self.norm1(feat)
            # Crop attention
            cattn = self.cross_attns[i](
                feat=None,
                query_feat=feat,
                coord=coord,
                context_feat=context_feat,
                context_coord=context_coord,
                knn_indexes=cross_knn_indexes,
            )
            feat = feat + self.drop_path(cattn)
            # MLP
            proj = self.norm3(self.fc3(self.act(self.norm2(feat))))
            feat = feat + self.drop_path(proj)
        return feat


class PoseTransformerV3(nn.Module):
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
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Parameters
        self.k_coarse = 16  # knn
        self.k_fine = 16
        self.rot_axis = "yz"
        # Encode pcd features
        self.target_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=False)
        self.anchor_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=False)
        # Pcd decoders
        self.coarse_pcd_decoder = KnnTransformerDecoder(num_layers=2, embed_channels=hidden_dims[-1], n_heads=8)
        self.fine_pcd_decoder = KnnTransformerDecoder(num_layers=2, embed_channels=hidden_dims[0], n_heads=8)
        self.coarse_linear_proj = nn.Linear(hidden_dims[-1], fusion_projection_dim)
        self.fine_linear_proj = nn.Linear(hidden_dims[0], fusion_projection_dim)
        # Pose decoder; Coarse & Fine
        self.coarse_pose_decoder = nn.Sequential(
            nn.Linear(fusion_projection_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_projection_dim, 9),
        )
        self.coarse_status_decoder = nn.Sequential(
            nn.Linear(fusion_projection_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_projection_dim, 2),
        )
        self.fine_pose_decoder = nn.Sequential(
            nn.Linear(fusion_projection_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_projection_dim, 9),
        )
        self.fine_status_decoder = nn.Sequential(
            nn.Linear(fusion_projection_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_projection_dim, 2),
        )

    def encode_cond(
        self,
        target_points,
        anchor_points,
    ):
        """
        Encode target and fixed pcd to features
        """
        target_points, all_target_points, target_cluster_indexes = self.target_pcd_transformer(target_points, return_full=True)
        # Encode fixed pcd
        anchor_points, all_anchor_points, anchor_cluster_indexes = self.anchor_pcd_transformer(anchor_points, return_full=True)
        # Check the existence of nan
        if torch.isnan(target_points[1]).any() or torch.isnan(anchor_points[1]).any():
            print("Nan exists in the feature")
        return all_target_points, all_anchor_points

    def predict_coarse(self, all_target_points: list[list[torch.Tensor]], all_anchor_points: list[list[torch.Tensor]]):
        anchor_coord, anchor_feat, anchor_offset = all_anchor_points[0]  # Use the last layer
        target_coord, target_feat, target_offset = all_target_points[0]  # Use the last layer

        # Do pcd decoder
        self_knn_indexes, self_knn_dists = knn(
            target_coord,
            target_coord,
            self.k_coarse,
            query_offset=target_offset,
            base_offset=target_offset,
        )
        cross_knn_indexes, cross_knn_dists = knn(
            target_coord,
            anchor_coord,
            self.k_coarse,
            query_offset=target_offset,
            base_offset=anchor_offset,
        )
        target_feat = self.coarse_pcd_decoder(
            feat=target_feat,
            coord=target_coord,
            knn_indexes=self_knn_indexes,
            cross_knn_indexes=cross_knn_indexes,
            context_feat=anchor_feat,
            context_coord=anchor_coord,
        )
        target_feat = self.coarse_linear_proj(target_feat)
        # Convert to batch & mask
        target_batch_index = offset2batch(target_offset)
        target_feat, target_feat_mask = to_dense_batch(target_feat, target_batch_index)
        # Compute mean over mask
        target_feat = target_feat * target_feat_mask[:, :, None]
        target_feat = target_feat.sum(dim=1) / target_feat_mask.sum(dim=1)[:, None]
        # Decode pose & status
        pose_pred = self.coarse_pose_decoder(target_feat)
        status_pred = self.coarse_status_decoder(target_feat)
        return pose_pred, status_pred

    def predict_fine(
        self,
        all_target_points: list[torch.Tensor],
        all_anchor_points: list[list[torch.Tensor]],
        pose_pred: torch.Tensor | None,
    ):
        anchor_coord, anchor_feat, anchor_offset = all_anchor_points[-1]  # Use the first layer
        target_coord, target_feat, target_offset = all_target_points[-1]  # Use the first layer
        target_batch_index = offset2batch(target_offset)
        if pose_pred is not None:
            # reposition
            target_batch_coord, target_batch_mask = to_dense_batch(target_coord, target_batch_index)
            target_batch_coord = self.reposition(target_batch_coord, pose_pred.detach(), rot_axis=self.rot_axis)  # Detach pose_pred
            target_coord, _ = to_flat_batch(target_batch_coord, target_batch_mask)

        # # [DEBUG]: Visualize local structure
        # anchor_batch_index = offset2batch(anchor_offset)
        # anchor_batch_coord, anchor_batch_mask = to_dense_batch(anchor_coord, anchor_batch_index)

        # check_idx = 1
        # anchor_pcd_np = anchor_batch_coord[check_idx].detach().cpu().numpy()
        # target_pcd_np = target_batch_coord[check_idx].detach().cpu().numpy()
        # anchor_pcd_o3d = o3d.geometry.PointCloud()
        # anchor_pcd_o3d.points = o3d.utility.Vector3dVector(anchor_pcd_np)
        # anchor_pcd_o3d.paint_uniform_color([0, 1, 0])
        # target_pcd_o3d = o3d.geometry.PointCloud()
        # target_pcd_o3d.points = o3d.utility.Vector3dVector(target_pcd_np)
        # target_pcd_o3d.paint_uniform_color([0, 0, 1])
        # o3d.visualization.draw_geometries([anchor_pcd_o3d, target_pcd_o3d])

        # Do pcd decoder
        self_knn_indexes, self_knn_dists = knn(
            target_coord,
            target_coord,
            self.k_fine,
            query_offset=target_offset,
            base_offset=target_offset,
        )
        cross_knn_indexes, cross_knn_dists = knn(
            target_coord,
            anchor_coord,
            self.k_fine,
            query_offset=target_offset,
            base_offset=anchor_offset,
        )
        target_feat = self.fine_pcd_decoder(
            feat=target_feat,
            coord=target_coord,
            knn_indexes=self_knn_indexes,
            cross_knn_indexes=cross_knn_indexes,
            context_feat=anchor_feat,
            context_coord=anchor_coord,
        )
        target_feat = self.fine_linear_proj(target_feat)
        # Convert to batch & mask
        target_feat, target_feat_mask = to_dense_batch(target_feat, target_batch_index)
        # Compute mean over mask
        target_feat = target_feat * target_feat_mask[:, :, None]
        target_feat = target_feat.sum(dim=1) / target_feat_mask.sum(dim=1)[:, None]
        # Decode pose & status
        pose_pred = self.fine_pose_decoder(target_feat)
        status_pred = self.fine_status_decoder(target_feat)
        return pose_pred, status_pred

    def reposition(self, coord: torch.Tensor, pose_pred: torch.Tensor, rot_axis: str = "xy"):
        """Reposition the points using the pose9d"""
        t = pose_pred[:, 0:3]
        r1 = pose_pred[:, 3:6]
        r2 = pose_pred[:, 6:9]
        # Normalize & Gram-Schmidt
        r1 = r1 / (torch.norm(r1, dim=1, keepdim=True) + 1e-8)
        r1_r2_dot = torch.sum(r1 * r2, dim=1, keepdim=True)
        r2_orth = r2 - r1_r2_dot * r1
        r2 = r2_orth / (torch.norm(r2_orth, dim=1, keepdim=True) + 1e-8)
        r3 = torch.cross(r1, r2)
        # Rotate
        if rot_axis == "xy":
            R = torch.stack([r1, r2, r3], dim=2)
        elif rot_axis == "yz":
            R = torch.stack([r3, r1, r2], dim=2)
        elif rot_axis == "zx":
            R = torch.stack([r2, r3, r1], dim=2)
        R = R.permute(0, 2, 1)
        # Reposition
        coord = torch.bmm(coord, R)
        coord = coord + t[:, None, :]
        return coord
