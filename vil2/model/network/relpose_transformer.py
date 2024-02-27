"""Relative pose transfomer"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from vil2.model.embeddings.pct import PointTransformerEncoderSmall, EncoderMLP, DropoutSampler
from vil2.model.network.geometric import PointTransformerNetwork, to_dense_batch, offset2batch
from vil2.model.network.genpose_modules import Linear
from vil2.utils.pcd_utils import visualize_tensor_pcd


class PositionEmbeddingCoordsSine(nn.Module):
    """From Mask3D"""

    def __init__(self, temperature=10000, normalize=False, scale=None, pos_type="fourier", d_pos=None, d_in=3, gauss_scale=1.0):
        super().__init__()
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        num_channels = self.d_pos
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert ndim % 2 == 0, f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(bsize, npoints, d_out)
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                out = self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                out = self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

        return out


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RelPoseTransformer(nn.Module):
    """Relative pose transformer network"""

    def __init__(self, grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, fusion_projection_dim, **kwargs) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Encode pcd features
        self.target_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=True)
        self.anchor_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=False)
        # Learnable embedding
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier", d_pos=hidden_dims[-1], gauss_scale=1.0, normalize=False)
        self.pose_embedding = nn.Embedding(1, hidden_dims[-1])
        self.status_embedding = nn.Embedding(1, hidden_dims[-1])
        # Pose refiner
        self.refine_decoders = nn.ModuleList()
        self.linear_projs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            refine_encoder_layer = TransformerDecoderLayer(d_model=hidden_dims[i + 1], nhead=n_heads[i], dim_feedforward=hidden_dims[i + 1] * 4, dropout=0.1, activation="relu", batch_first=True)
            refine_decoder = TransformerDecoder(refine_encoder_layer, num_layers=dec_depths[i])
            self.refine_decoders.insert(0, refine_decoder)
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
        target_feat_mask = torch.cat((torch.ones([target_feat_mask.shape[0], 2], dtype=target_feat_mask.dtype, device=target_feat_mask.device), target_feat_mask), dim=1)
        target_feat_padding_mask = target_feat_mask == 0

        # Check the existence of nan
        if torch.isnan(target_feat).any():
            print("Nan exists in the feature")

        # Refine pose tokens
        target_coord = to_dense_batch(target_coord, target_batch_index)[0]
        target_pos_embedding = self.pos_enc(target_coord).permute(0, 2, 1)
        for i in range(len(self.refine_decoders)):
            anchor_coord, anchor_feat, anchor_offset = all_anchor_points[i]
            anchor_batch_index = offset2batch(anchor_offset)
            anchor_coord = to_dense_batch(anchor_coord, anchor_batch_index)[0]
            anchor_pos_embedding = self.pos_enc(anchor_coord).permute(0, 2, 1)
            anchor_feat, anchor_feat_mask = to_dense_batch(anchor_feat, anchor_batch_index)
            anchor_feat_padding_mask = anchor_feat_mask == 0
            assert target_feat.shape[0] == anchor_feat.shape[0], "Batch size mismatch"
            # Apply position embedding
            anchor_feat = anchor_feat + anchor_pos_embedding
            target_feat[:, 2:, :] = target_feat[:, 2:, :] + target_pos_embedding
            target_feat = self.refine_decoders[i](
                target_feat, anchor_feat, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=target_feat_padding_mask, memory_key_padding_mask=anchor_feat_padding_mask
            )
            if torch.isnan(target_feat).any():
                assert False, "Nan exists in the feature"
            # Sanity check
            if torch.isnan(target_feat).any():
                print("Nan exists in the feature")
            # Projection
            target_feat = self.linear_projs[i](target_feat)
        # Decode pose & status
        pose_pred = self.pose_decoder(target_feat[:, 0, :])
        status_pred = self.status_decoder(target_feat[:, 1, :])

        # Sanity check
        if torch.isnan(pose_pred).any():
            print("Nan exists in the pose")
        return pose_pred, status_pred
