"""PointCloud Diffusion Network"""

from __future__ import annotations
import torch
import torch.nn as nn
from vil2.model.network.geometric import PointTransformerNetwork, to_dense_batch, to_flat_batch, batch2offset, offset2batch, knn, PointBatchNorm, KnnTransformer, KnnTransformerDecoder
from timm.models.layers import DropPath
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class PcdNoiseNet(nn.Module):
    """Generate noise for point cloud diffusion process."""

    def __init__(self, grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, fusion_projection_dim, **kwargs) -> None:
        super().__init__()
        # Module
        half_hidden_dims = [int(hd / 2) for hd in hidden_dims]
        self.target_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, half_hidden_dims, n_heads, ks, in_dim, skip_dec=True)
        self.anchor_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=False)
        self.denoise_decoders = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        self.linear_proj = nn.Linear(hidden_dims[0], 3)
        for i in range(len(hidden_dims) - 1):
            refine_encoder_layer = TransformerDecoderLayer(
                d_model=hidden_dims[i + 1],
                nhead=n_heads[i],
                dim_feedforward=hidden_dims[i + 1] * 4,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            )
            denoise_decoder = TransformerDecoder(refine_encoder_layer, num_layers=dec_depths[i])
            self.denoise_decoders.insert(0, denoise_decoder)
            self.conv1x1.insert(0, nn.Conv1d(hidden_dims[i + 1], hidden_dims[i], 1))

        # Diffusion related
        max_timestep = kwargs.get("max_timestep", 200)
        self.time_embedding = nn.Embedding(max_timestep, hidden_dims[-1])

    def encode_anchor(self, anchor_points: list[torch.Tensor]) -> torch.Tensor:
        """
        Encode anchor point cloud into feat pyramid.
        Args:
            anchor_points (list[torch.Tensor]): coord, point, offset
        """
        anchor_points, all_anchor_points, anchor_cluster_indexes = self.anchor_pcd_transformer(anchor_points, return_full=True)
        return anchor_points, all_anchor_points, anchor_cluster_indexes

    def encode_target(self, target_points: list[torch.Tensor]) -> torch.Tensor:
        """
        Encode target point cloud into feat pyramid.
        Args:
            target_points (list[torch.Tensor]): coord, point, offset
        """
        target_points = self.target_pcd_transformer(target_points)
        return target_points

    def forward(self, target_points: list[torch.Tensor], target_coord_t: torch.tensor, all_anchor_points: list[list[torch.Tensor]], t: int) -> torch.Tensor:
        """
        Args:
            target_points (torch.Tensor): encoded target point cloud
            target_coord_t (torch.Tensor): target coordinate at t
            all_anchor_points (torch.Tensor): anchor point cloud in pyramid
            t (int): diffusion time step
        """
        target_coord, target_feat_c, target_offset = target_points  # Encode canonical target
        # Convert to batch & mask
        target_batch_index = offset2batch(target_offset)
        target_feat_c, target_feat_mask = to_dense_batch(target_feat_c, target_batch_index)

        time_token = self.time_embedding(t)
        target_feat_mask = torch.cat(
            (
                torch.ones(
                    [target_feat_mask.shape[0], 1],
                    dtype=target_feat_mask.dtype,
                    device=target_feat_mask.device,
                ),
                target_feat_mask,
            ),
            dim=1,
        )
        target_feat_padding_mask = target_feat_mask == 0

        # Encode feat at t
        target_points_t = [target_coord, target_coord_t, target_offset]
        target_feat_t = self.encode_target(target_points_t)
        target_feat_t, _ = to_dense_batch(target_feat_t, target_batch_index)
        target_feat_t = torch.cat((target_feat_c, target_feat_t), dim=-1)  # (B, N, C + C)
        target_feat_t = torch.cat((time_token[None, None, :].expand(target_feat_t.size(0), target_feat_t.size(1), -1), target_feat_t), dim=1)

        # Sanity check
        if torch.isnan(target_feat_c).any():
            print("Nan exists in the feature")

        for i in range(len(self.denoise_decoders)):
            anchor_coord, anchor_feat, anchor_offset = all_anchor_points[i]
            anchor_batch_index = offset2batch(anchor_offset)
            anchor_feat, anchor_feat_mask = to_dense_batch(anchor_feat, anchor_batch_index)
            anchor_feat_padding_mask = anchor_feat_mask == 0
            assert target_feat_t.shape[0] == anchor_feat.shape[0], f"Batch size mismatch: {target_feat_c.shape[0]} != {anchor_feat.shape[0]}"
            target_feat_t = self.denoise_decoders[i](
                target_feat_t,
                anchor_feat,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=target_feat_padding_mask,
                memory_key_padding_mask=anchor_feat_padding_mask,
            )
            target_feat_t = self.conv1x1[i](target_feat_t.permute(0, 2, 1)).permute(0, 2, 1)
            # Sanity check
            if torch.isnan(target_feat_t).any():
                print("Nan exists in the feature")

        # Decode pos
        target_feat_t = target_feat_t[:, 1:]  # Remove time token
        target_pos_noise_t = self.linear_proj(target_feat_t)
        return target_pos_noise_t
