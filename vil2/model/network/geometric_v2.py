import numpy as np
import einops
from copy import deepcopy
import pytorch3d.ops as p3dops
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
import torch.nn as nn
import torch
from timm.models.layers import DropPath


def knn_gather(idx, feat, coord=None, with_coord=False, gather_coord=None):
    """
    indexes: (n, k)
    feat: (m, c)
    coord: (m, 3)
    return: (n, k, c), (n, k, 3)
    """
    assert feat.is_contiguous()
    m, nsample, c = idx.shape[0], idx.shape[1], feat.shape[1]
    # Append a zero feature for the last index
    feat = torch.cat([feat, torch.zeros([1, c]).to(feat.device)], dim=0)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)
    if with_coord:
        coord = coord.contiguous()
        coord = torch.cat([coord, torch.zeros([1, 3]).to(coord.device)], dim=0)
        mask = torch.sign(idx + 1)
        if gather_coord is None:
            gather_coord = coord[:-1]
        grouped_coord = coord[idx.view(-1).long(), :].view(m, nsample, 3) - gather_coord.unsqueeze(1)
        grouped_coord = torch.einsum("n s c, n s -> n s c", grouped_coord, mask)
        return grouped_feat, grouped_coord
    else:
        return grouped_feat


def fallback(*args):
    for a in args:
        if a is not None:
            return a


class VoxelPooling(nn.Module):

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(VoxelPooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, skip_fc=False, **point_attributes):
        coord, feat, offset = points
        batch = offset2batch(offset)
        if not skip_fc:
            feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]), reduce="min")
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster, stable=True)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        # Performal voxel for each batch
        offset = batch2offset(batch)
        out_point_attributes = {}
        if len(point_attributes) > 0:
            for k, v in point_attributes.items():
                if k in ["cluster", "indices", "idx_ptr"]:
                    continue
                reduce = "max"
                if isinstance(v, (list, tuple)):
                    reduce, v = v
                prev_bool = False
                if v.dtype == torch.bool:
                    prev_bool = True
                    v = v.long()
                elif v.dtype in [torch.float32, torch.float64]:
                    reduce = "mean"
                out_point_attributes[k] = segment_csr(v[sorted_cluster_indices], idx_ptr, reduce=reduce)
                if prev_bool:
                    v = v.bool()
        return [coord, feat, offset], {
            "cluster": cluster,
            "indices": sorted_cluster_indices,
            "idx_ptr": idx_ptr,
            **out_point_attributes,
        }


class UnpoolWithSkip(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, bias=True, skip=True, backend="map"):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = in_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]  # Up pooling method

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = self.proj(feat)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


class PointBatchNorm(nn.Module):

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor):
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedVectorAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat=None, coord=None, knn_indexes=None, query_feat=None, context_feat=None, context_coord=None):
        is_cross = context_feat is not None
        query_feat = fallback(query_feat, feat)
        context_feat = fallback(context_feat, feat)
        context_coord = fallback(context_coord, coord)
        query, key, value = (
            self.linear_q(query_feat),
            self.linear_k(context_feat),
            self.linear_v(context_feat),
        )
        key, pos = knn_gather(
            knn_indexes, key, context_coord, with_coord=True, gather_coord=coord if is_cross else None
        )
        value = knn_gather(knn_indexes, key, context_coord, with_coord=True, gather_coord=coord if is_cross else None)
        relation_qk = key - query.unsqueeze(1)  # relative position
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(knn_indexes + 1)  # all one
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


class KnnTransformer(nn.Module):
    def __init__(
        self,
        embed_channels,
        n_heads,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(KnnTransformer, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=n_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, feat=None, coord=None, knn_indexes=None, query_feat=None, context_feat=None, context_coord=None):
        assert knn_indexes is not None
        if feat is not None:
            identity = feat
            feat = self.act(self.norm1(self.fc1(feat)))
            feat = (
                self.attn(feat, coord, knn_indexes)
                if not self.enable_checkpoint
                else checkpoint(self.attn, feat, coord, knn_indexes, use_reentrant=True)
            )
        else:
            identity = query_feat
            feat = self.act(self.norm1(self.fc1(query_feat)))
            feat = (
                self.attn(
                    query_feat=feat,
                    coord=coord,
                    context_coord=context_coord,
                    context_feat=context_feat,
                    knn_indexes=knn_indexes,
                )
                if not self.enable_checkpoint
                else checkpoint(
                    self.attn,
                    query_feat=feat,
                    coord=coord,
                    context_coord=context_coord,
                    context_feat=context_feat,
                    knn_indexes=knn_indexes,
                    use_reentrant=True,
                )
            )

        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return feat


PointTransformer = KnnTransformer


class PointTransformerSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbors=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(PointTransformerSequence, self).__init__()
        if isinstance(drop_path_rate, list):
            drop_path_rate = drop_path_rate
            assert len(drop_path_rate) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbors = neighbors
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = PointTransformer(
                embed_channels=embed_channels,
                n_heads=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points, return_knn_indexes=False):
        coord, feat, offset = points
        knn_index, _ = knn(coord, coord, self.neighbors, query_offsets=offset)
        for block in self.blocks:
            feat = block(feat, coord, knn_index)
        if return_knn_indexes:
            return [coord, feat, offset], knn_index
        else:
            return [coord, feat, offset]


class PointPatchEmbed(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(PointPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(),
        )
        self.blocks = PointTransformerSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bais=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

        def forward(self, points):
            coord, feat, offset = points
            feat = self.proj(feat)
            return self.blocks(coord, feat, offset)


class PointEncoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        grid_size=None,
        neighbors=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
    ):
        super(PointEncoder, self).__init__()

        self.down = VoxelPooling(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = PointTransformerSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbors=neighbors,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, return_knn_indexes=False, **point_attributes):
        points, cluster = self.down(points, **point_attributes)  # Pooling
        return self.blocks(points, return_knn_indexes=return_knn_indexes), cluster


class PointTransformerNetwork(nn.Module):
    def __init__(
        self,
        grid_sizes,
        depths,
        dec_depths,
        hidden_dims,
        n_heads,
        ks,
        in_dim=None,
        skip_dec=False,
    ):
        super().__init__()
        self.skip_dec = skip_dec
        self.enc_stages = nn.ModuleList()
        if not skip_dec:
            self.dec_stages = nn.ModuleList()

        self.patch_embed = PointPatchEmbed(
            in_channels=in_dim or hidden_dims[0],
            embed_channels=hidden_dims[0],
            groups=n_heads[0],
            depth=depths[0],
            neighbours=ks[0],
        )

        for i in range(len(depths) - 1):
            self.enc_stages.append(
                PointEncoder(
                    depth=depths[i + 1],
                    in_channels=hidden_dims[i],
                    groups=n_heads[i + 1],
                    grid_size=grid_sizes[i],
                    neighbors=ks[i + 1],
                )
            )

            if not skip_dec:
                self.dec_stages.insert(
                    0,
                    PointDecoder(
                        depth=dec_depths[i],
                        in_channels=hidden_dims[i + 1],
                        skip_channels=hidden_dims[i],
                        embed_channels=hidden_dims[i],
                        groups=n_heads[i],
                        neighbors=ks[i],
                    ),
                )

    def forward(self, points, return_full=False):
        points = self.patch_embed(points)
        cluster_indexes = []
        all_points = []
        for i, stage in enumerate(self.enc_stages):
            points, attrs = stage(points)
            cluster_indexes.insert(0, attrs["cluster"])
            all_points.insert(0, points)

        if not self.skip_dec:
            for i, dec_stage in enumerate(self.dec_stages):
                points, skip_points = all_points[i], all_points[i + 1]
                cluster = cluster_indexes[i]
                points = dec_stage(points, skip_points, cluster)
                all_points[i + 1] = points

        if return_full:
            return points, all_points, cluster_indexes
        else:
            return points
