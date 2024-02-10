"""Pose transformer network Version 2. Using PointTransformerV2 (Unet)"""

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
        self.pose_embedding = nn.Embedding(1, hidden_dims[-1])

        # Pose refiner
        self.refine_decoders = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        self.linear_projs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            refine_encoder_layer = TransformerDecoderLayer(
                d_model=hidden_dims[i + 1],
                nhead=n_heads[i],
                dim_feedforward=hidden_dims[i + 1] * 4,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            )
            refine_decoder = TransformerDecoder(refine_encoder_layer, num_layers=dec_depths[i])
            self.refine_decoders.insert(0, refine_decoder)
            self.conv1x1.insert(0, nn.Conv1d(hidden_dims[i + 1], hidden_dims[i], 1))
            self.linear_projs.insert(0, nn.Linear(hidden_dims[i + 1], hidden_dims[i]))

        # Pose decoder
        # init_zero = dict(
        #     init_mode="kaiming_uniform", init_weight=0, init_bias=0
        # )  # init the final output layer's weights to zeros
        # self.pose_decoder = nn.Sequential(
        #     nn.Linear(hidden_dims[0], fusion_projection_dim),
        #     nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     Linear(fusion_projection_dim, 9, **init_zero),
        # )
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

    def encode_cond(
        self,
        target_points,
        fixed_points,
    ):
        """
        Encode target and fixed pcd to features
        """
        target_points = self.target_pcd_transformer(target_points)
        # Encode fixed pcd
        fixed_points, all_fixed_points, cluster_indexes = self.fixed_pcd_transformer(fixed_points, return_full=True)
        # DEBUG:
        # Check the existence of nan
        if torch.isnan(target_points[1]).any() or torch.isnan(fixed_points[1]).any():
            print("Nan exists in the feature")
        return target_points, all_fixed_points, cluster_indexes

    def forward(
        self,
        target_points: list[torch.Tensor],
        all_fixed_points: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        target_coord, target_feat, target_offset = target_points
        # Convert to batch & mask
        target_batch_index = offset2batch(target_offset)
        target_feat, target_feat_mask = to_dense_batch(target_feat, target_batch_index)

        pose_token = self.pose_embedding.weight[0].unsqueeze(0).expand(target_feat.size(0), -1)
        target_feat = torch.cat((pose_token[:, None, :], target_feat), dim=1)
        target_feat_mask = torch.cat((torch.ones_like(target_feat_mask[:, :1]), target_feat_mask), dim=1)
        target_feat_padding_mask = target_feat_mask == 0

        # Check the existence of nan
        if torch.isnan(target_feat).any():
            print("Nan exists in the feature")

        # DEBUG:
        # target_coord_batch, mask = to_dense_batch(target_coord, target_batch_index)
        # visualize_tensor_pcd(target_coord_batch[0])
        # Refine pose tokens
        for i in range(len(self.refine_decoders)):
            fixed_coord, fixed_feat, fixed_offset = all_fixed_points[i]
            fixed_batch_index = offset2batch(fixed_offset)
            fixed_feat, fixed_feat_mask = to_dense_batch(fixed_feat, fixed_batch_index)
            fixed_feat_padding_mask = fixed_feat_mask == 0
            if target_feat.shape[0] != fixed_feat.shape[0]:
                print("fixed_offset:")
                print(fixed_offset)
                print("target_offset:")
                print(target_offset)
                print(
                    f"target_feat: {target_feat.shape}, fixed_feat: {fixed_feat.shape}, target_mask: {target_feat_padding_mask.shape}, fixed_mask: {fixed_feat_padding_mask.shape}"
                )
                print("Batch size mismatch")
            target_feat = self.refine_decoders[i](
                target_feat,
                fixed_feat,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=target_feat_padding_mask,
                memory_key_padding_mask=fixed_feat_padding_mask,
            )
            if torch.isnan(target_feat).any():
                print("Nan exists in the feature")
                # Check which batch has nan
                for j in range(target_feat.shape[0]):
                    if torch.isnan(target_feat[j]).any():
                        print(f"Batch {j} has nan")
                        # Check fixed pcd of that batch
                        fixed_coord_raw, fixed_feat_raw, fixed_offset_raw = all_fixed_points[2]
                        fixed_batch_raw = offset2batch(fixed_offset_raw)
                        fixed_coord_batch, mask = to_dense_batch(fixed_coord_raw, fixed_batch_raw)
                        visualize_tensor_pcd(fixed_coord_batch[j])
            target_feat = self.conv1x1[i](target_feat.permute(0, 2, 1)).permute(0, 2, 1)
            # DEBUG:
            # Check the existence of nan
            if torch.isnan(target_feat).any():
                print("Nan exists in the feature")
            # DEBUG:
            # fixed_coord_batch, mask = to_dense_batch(fixed_coord, fixed_batch_index)
            # visualize_tensor_pcd(fixed_coord_batch[0])

        # Decode pose & status
        pose_pred = self.pose_decoder(target_feat[:, 0, :])
        status_pred = self.status_decoder(target_feat[:, 0, :])

        # DEBUG:
        # Check the existence of nan
        if torch.isnan(pose_pred).any():
            print("Nan exists in the pose")
        return pose_pred, status_pred

    # def check_exist_batch(self, batch):
    #     """Debug method"""
    #     pass
