"""PointCloud Diffusion Network"""

from __future__ import annotations
import torch
import torch.nn as nn
from vil2.model.network.geometric import PointTransformerNetwork, to_dense_batch, to_flat_batch, batch2offset, offset2batch, knn, PointBatchNorm, KnnTransformer, KnnTransformerDecoder
from timm.models.layers import DropPath
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Apply mask to attention scores
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attn = attn.masked_fill(mask, float("-inf"))  # Apply mask where mask is True

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, output_dim, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PcdNoiseNet(nn.Module):
    """Generate noise for point cloud diffusion process."""

    def __init__(self, grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, **kwargs) -> None:
        super().__init__()
        num_denoise_layers = kwargs.get("num_denoise_layers", 3)
        num_denoise_depths = kwargs.get("num_denoise_depths", 3)
        hidden_dim_denoise = kwargs.get("hidden_dim_denoise", 512)
        n_heads_denoise = kwargs.get("n_heads_denoise", 16)
        condition_pooling = kwargs.get("condition_pooling", "max")  # max, mean
        condition_strategy = kwargs.get("condition_strategy", "cross_attn")  # cross_attn, FiLM
        self.condition_strategy = condition_strategy
        self.condition_pooling = condition_pooling
        # Module
        self.target_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=True)
        self.anchor_pcd_transformer = PointTransformerNetwork(grid_sizes, depths, dec_depths, hidden_dims, n_heads, ks, in_dim, skip_dec=True)
        self.conv1x1 = nn.ModuleList()
        self.linear_proj_up = nn.Linear(6, hidden_dims[-1])
        if condition_strategy == "cross_attn":
            self.pos_decoder = nn.Sequential(
                nn.LayerNorm(hidden_dims[-1], elementwise_affine=False, eps=1e-6),
                nn.Linear(hidden_dims[-1], 3),
            )
        elif condition_strategy == "FiLM":
            self.pos_decoder = DiTFinalLayer(hidden_dims[-1], 3, 3)
        else:
            raise NotImplementedError
        self.decoders = nn.ModuleList()
        if self.condition_strategy == "cross_attn":
            for i in range(num_denoise_layers):
                decoder_layer = TransformerDecoderLayer(
                    d_model=hidden_dim_denoise,
                    nhead=n_heads_denoise,
                    dim_feedforward=2048,
                    dropout=0.1,
                    activation="relu",
                    batch_first=True,
                )
                decoder = TransformerDecoder(decoder_layer, num_layers=num_denoise_depths)
                self.decoders.insert(0, decoder)
                self.conv1x1.insert(0, nn.Conv1d(hidden_dims[-1], hidden_dims[-1], 1))
        elif self.condition_strategy == "FiLM":
            for i in range(num_denoise_layers):
                self.decoders.insert(0, DiTBlock(hidden_size=hidden_dim_denoise, num_heads=n_heads_denoise))

        # Diffusion related
        max_timestep = kwargs.get("max_timestep", 200)
        self.time_embedding = nn.Embedding(max_timestep, hidden_dims[-1])

    def initialize_weights(self):
        if self.condition_strategy == "FiLM":
            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.decoders:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

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
            anchor_points (torch.Tensor): anchor point cloud in pyramid
            t (int): diffusion time step
        """
        if self.condition_strategy == "cross_attn":
            return self.forward_cross_attn(target_coord_t, target_points, anchor_points, t)
        elif self.condition_strategy == "FiLM":
            return self.forward_FiLM(target_coord_t, target_points, anchor_points, t)
        else:
            raise NotImplementedError

    def forward_cross_attn(self, target_coord_t: torch.tensor, target_points: list[torch.Tensor], anchor_points: list[list[torch.Tensor]], t: int) -> torch.Tensor:
        """
        Args:
            target_points (torch.Tensor): encoded target point cloud
            target_coord_t (torch.Tensor): target coordinate at t
            anchor_points (torch.Tensor): anchor point cloud in pyramid
            t (int): diffusion time step
        """
        if len(t.shape) == 1:
            t = t[:, None]
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
        # target_coord_t = torch.cat((target_feat_c, target_coord_t), dim=1)
        target_coord_t = torch.cat((target_coord, target_coord_t), dim=1)
        target_batch_index = offset2batch(target_offset)
        target_feat_t, target_feat_mask = to_dense_batch(target_coord_t, target_batch_index)
        target_feat_t = self.linear_proj_up(target_feat_t)
        target_feat_padding_mask = target_feat_mask == 0

        # Sanity check
        if torch.isnan(anchor_feat).any():
            print("Nan exists in the feature")

        for i in range(len(self.decoders)):
            target_feat_t = self.decoders[i](
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
        target_pos_noise_t = self.pos_decoder(target_feat_t)
        return target_pos_noise_t

    def forward_FiLM(self, target_coord_t: torch.tensor, target_points: list[torch.Tensor], anchor_points: list[list[torch.Tensor]], t: int) -> torch.Tensor:
        """
        FiLM condition.
        Args:
            target_points (torch.Tensor): encoded target point cloud
            target_coord_t (torch.Tensor): target coordinate at t
            anchor_points (torch.Tensor): anchor point cloud in pyramid
            t (int): diffusion time step
        """
        if len(t.shape) == 1:
            t = t[:, None]
        # Convert to batch & mask
        anchor_coord, anchor_feat, anchor_offset = anchor_points  # Encode anchor
        time_token = self.time_embedding(t)
        anchor_batch_index = offset2batch(anchor_offset)
        anchor_feat, anchor_feat_mask = to_dense_batch(anchor_feat, anchor_batch_index)
        anchor_feat = torch.cat((time_token, anchor_feat), dim=1)

        # Sanity check
        if torch.isnan(anchor_feat).any():
            print("Nan exists in the feature")

        # pooling anchor_feat to 1-d
        if self.condition_pooling == "max":
            anchor_feat = torch.max(anchor_feat, dim=1).values
        elif self.condition_pooling == "mean":
            anchor_feat = torch.mean(anchor_feat, dim=1)
        else:
            raise NotImplementedError

        # Use coord as feature: We need involve conanical feature or coord
        target_coord, target_feat_c, target_offset = target_points  # Encode canonical target
        # target_coord_t = torch.cat((target_feat_c, target_coord_t), dim=1)
        target_coord_t = torch.cat((target_coord, target_coord_t), dim=1)
        target_batch_index = offset2batch(target_offset)
        target_feat_t, target_feat_mask = to_dense_batch(target_coord_t, target_batch_index)
        target_feat_t = self.linear_proj_up(target_feat_t)
        target_feat_padding_mask = target_feat_mask == 0

        # Do DiT
        for i in range(len(self.decoders)):
            # Mask out the padding
            target_mask = target_feat_padding_mask.unsqueeze(1)
            target_feat_t = self.decoders[i](target_feat_t, anchor_feat, mask=target_mask)
            # Sanity check
            if torch.isnan(target_feat_t).any():
                print("Nan exists in the feature")

        # Decode pos
        if self.condition_strategy == "FiLM":
            target_pos_noise_t = self.pos_decoder(target_feat_t, anchor_feat)
        elif self.condition_strategy == "cross_attn":
            target_pos_noise_t = self.pos_decoder(target_feat_t)
        else:
            raise NotImplementedError
        return target_pos_noise_t
