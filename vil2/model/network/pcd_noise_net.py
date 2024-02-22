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
        num_denoise_layers = kwargs.get("denoise_layers", 3)
        # Module
        self.target_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=True)
        self.anchor_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=True)
        self.denoise_decoders = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        self.linear_proj_up = nn.Linear(6, hidden_dims[-1])
        self.linear_proj_down = nn.Linear(hidden_dims[-1], 3)
        for i in range(num_denoise_layers):
            decoder_layer = TransformerDecoderLayer(
                d_model=hidden_dims[-1],
                nhead=n_heads[-1],
                dim_feedforward=hidden_dims[-1] * 4,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            )
            denoise_decoder = TransformerDecoder(decoder_layer, num_layers=dec_depths[-1])
            self.denoise_decoders.insert(0, denoise_decoder)
            self.conv1x1.insert(0, nn.Conv1d(hidden_dims[-1], hidden_dims[-1], 1))

        # Diffusion related
        max_timestep = kwargs.get("max_timestep", 200)
        self.time_embedding = nn.Embedding(max_timestep, hidden_dims[-1])

    def encode_anchor(self, anchor_points: list[torch.Tensor]) -> torch.Tensor:
        """
        Encode anchor point cloud into feat pyramid.
        Args:
            anchor_points (list[torch.Tensor]): coord, point, offset
        """
        anchor_points = self.anchor_pcd_transformer(anchor_points)
        return anchor_points

    def encode_target(self, target_points: list[torch.Tensor]) -> torch.Tensor:
        """
        Encode target point cloud into feat pyramid.
        Args:
            target_points (list[torch.Tensor]): coord, point, offset
        """
        target_points = self.target_pcd_transformer(target_points)
        return target_points

    def forward(self, target_coord_t: torch.tensor, target_points: list[torch.Tensor], anchor_points: list[list[torch.Tensor]], t: int) -> torch.Tensor:
        """
        Args:
            target_points (torch.Tensor): encoded target point cloud
            target_coord_t (torch.Tensor): target coordinate at t
            anchor_points (torch.Tensor): anchor point cloud in pyramid
            t (int): diffusion time step
        """
        # Convert to batch & mask
        anchor_coord, anchor_feat, anchor_offset = anchor_points  # Encode anchor
        time_token = self.time_embedding(t)
        anchor_batch_index = offset2batch(anchor_offset)
        anchor_feat, anchor_feat_mask = to_dense_batch(anchor_feat, anchor_batch_index)
        anchor_feat = torch.cat((time_token, anchor_feat), dim=1)
        anchor_feat_mask = torch.cat(
            (
                torch.ones(
                    [anchor_feat_mask.shape[0], 1],
                    dtype=anchor_feat_mask.dtype,
                    device=anchor_feat_mask.device,
                ),
                anchor_feat_mask,
            ),
            dim=1,
        )
        anchor_feat_padding_mask = anchor_feat_mask == 0

        # Use coord as feature: We need involve conanical feature or coord
        target_coord, target_feat_c, target_offset = target_points  # Encode canonical target
        target_coord_t = torch.cat((target_coord, target_coord_t), dim=1)
        target_batch_index = offset2batch(target_offset)
        target_feat_t, target_feat_mask = to_dense_batch(target_coord_t, target_batch_index)
        target_feat_t = self.linear_proj_up(target_feat_t)
        target_feat_padding_mask = target_feat_mask == 0

        # Sanity check
        if torch.isnan(anchor_feat).any():
            print("Nan exists in the feature")

        for i in range(len(self.denoise_decoders)):
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
        target_pos_noise_t = self.linear_proj_down(target_feat_t)
        return target_pos_noise_t
