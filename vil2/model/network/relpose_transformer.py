"""Relative pose transfomer"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from vil2.model.embeddings.pct import PointTransformerEncoderSmall, EncoderMLP, DropoutSampler
from vil2.model.network.geometric import PointTransformerNetwork, to_dense_batch, offset2batch
from vil2.model.network.genpose_modules import Linear
from vil2.utils.pcd_utils import visualize_tensor_pcd


class RelPoseTransformer(nn.Module):
    """Relative pose transformer network"""

    def __init__(self, grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, fusion_projection_dim) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Encode pcd features
        self.target_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=True)
        self.anchor_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=False)
        # Learnable embedding
        self.pose_embedding = nn.Embedding(1, hidden_dims[-1])
        self.status_embedding = nn.Embedding(1, hidden_dims[-1])
        # Pose refiner
        self.refine_decoders = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        self.linear_projs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            refine_encoder_layer = TransformerDecoderLayer(d_model=hidden_dims[i + 1], nhead=n_heads[i], dim_feedforward=hidden_dims[i + 1] * 4, dropout=0.1, activation="relu", batch_first=True)
            refine_decoder = TransformerDecoder(refine_encoder_layer, num_layers=dec_depths[i])
            self.refine_decoders.insert(0, refine_decoder)
            self.conv1x1.insert(0, nn.Conv1d(hidden_dims[i + 1], hidden_dims[i], 1))
            self.linear_projs.insert(0, nn.Linear(hidden_dims[i + 1], hidden_dims[i]))
        # Pose decoder
        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_dims[0], fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_projection_dim, 9),
        )
        self.status_decoder = nn.Sequential(
            nn.Linear(hidden_dims[0], fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_projection_dim, 2),
        )

    def encode_cond(self, target_points, anchor_points):
        """
        Encode target and anchor pcd to features
        """
        target_points = self.target_pcd_transformer(target_points)
        # Encode fixed pcd
        anchor_points, all_anchor_points, cluster_indexes = self.anchor_pcd_transformer(anchor_points, return_full=True)
        # Check the existence of nan
        if torch.isnan(target_points[1]).any() or torch.isnan(anchor_points[1]).any():
            print("Nan exists in the feature")
        return target_points, all_anchor_points, cluster_indexes

    def forward(self, target_points: list[torch.Tensor], all_anchor_points: list[list[torch.Tensor]]) -> torch.Tensor:
        target_coord, target_feat, target_offset = target_points
        # Convert to batch & mask
        target_batch_index = offset2batch(target_offset)
        target_feat, target_feat_mask = to_dense_batch(target_feat, target_batch_index)

        pose_token = self.pose_embedding.weight[0].unsqueeze(0).expand(target_feat.size(0), -1)
        status_token = self.status_embedding.weight[0].unsqueeze(0).expand(target_feat.size(0), -1)
        target_feat = torch.cat((pose_token[:, None, :], status_token[:, None, :], target_feat), dim=1)
        target_feat_mask = torch.cat(
            (
                torch.ones(
                    [target_feat_mask.shape[0], 2],
                    dtype=target_feat_mask.dtype,
                    device=target_feat_mask.device,
                ),
                target_feat_mask,
            ),
            dim=1,
        )
        target_feat_padding_mask = target_feat_mask == 0

        # Check the existence of nan
        if torch.isnan(target_feat).any():
            print("Nan exists in the feature")

        # Refine pose tokens
        for i in range(len(self.refine_decoders)):
            anchor_coord, anchor_feat, anchor_offset = all_anchor_points[i]
            anchor_batch_index = offset2batch(anchor_offset)
            anchor_feat, anchor_feat_mask = to_dense_batch(anchor_feat, anchor_batch_index)
            anchor_feat_padding_mask = anchor_feat_mask == 0
            if target_feat.shape[0] != anchor_feat.shape[0]:
                print("anchor_offset:")
                print(anchor_offset)
                print("target_offset:")
                print(target_offset)
                print(f"target_feat: {target_feat.shape}, anchor_feat: {anchor_feat.shape}, target_mask: {target_feat_padding_mask.shape}, anchor_mask: {anchor_feat_padding_mask.shape}")
                print("Batch size mismatch")
            target_feat = self.refine_decoders[i](
                target_feat,
                anchor_feat,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=target_feat_padding_mask,
                memory_key_padding_mask=anchor_feat_padding_mask,
            )
            if torch.isnan(target_feat).any():
                print("Nan exists in the feature")
                # Check which batch has nan
                for j in range(target_feat.shape[0]):
                    if torch.isnan(target_feat[j]).any():
                        print(f"Batch {j} has nan")
                        # Check fixed pcd of that batch
                        anchor_coord_raw, anchor_feat_raw, anchor_offset_raw = all_anchor_points[2]
                        anchor_batch_raw = offset2batch(anchor_offset_raw)
                        anchor_coord_batch, mask = to_dense_batch(anchor_coord_raw, anchor_batch_raw)
                        visualize_tensor_pcd(anchor_coord_batch[j])
            target_feat = self.conv1x1[i](target_feat.permute(0, 2, 1)).permute(0, 2, 1)
            # Sanity check
            if torch.isnan(target_feat).any():
                print("Nan exists in the feature")

        # Decode pose & status
        pose_pred = self.pose_decoder(target_feat[:, 0, :])
        status_pred = self.status_decoder(target_feat[:, 1, :])

        # Sanity check
        if torch.isnan(pose_pred).any():
            print("Nan exists in the pose")
        return pose_pred, status_pred
