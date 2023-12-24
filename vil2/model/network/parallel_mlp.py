import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import functools
from vil2.model.embeddings.pct import PointTransformerEncoderSmall, EncoderMLP, PointNet
from vil2.model.embeddings.sinusoidal import PositionalEmbedding
from vil2.model.network.genpose_modules import Linear


    
class ParallelMLP(nn.Module):
    def __init__(self, input_dim: int = 256, global_cond_dim: int = 1024, diffusion_step_embed_dim: int = 128, 
                 down_dims: list = [256, 512, 1024], use_global_geometry: bool = False, downsample_pcd_enc: bool = False, 
                 downsample_size: bool = False, fusion_projection_dim: int = 512, use_pointnet: bool = False, 
                 use_dropout_sampler: bool = False, rotation_orthogonalization: bool = False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion_step_encoder = nn.Sequential(
            PositionalEmbedding(num_channels=diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
            nn.ReLU(inplace=True)
        )
        if downsample_pcd_enc:
            global_cond_dim = downsample_size

        self.pcd_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=False) if not use_pointnet else PointNet(emb_dims=256)
        self.use_pointnet = use_pointnet
        self.pcd_mlp_encoder = EncoderMLP(256, global_cond_dim, uses_pt=True)
        self.pose_encoder = nn.Sequential(
                                nn.Linear(9, input_dim),
                                nn.ReLU(True),
                                nn.Linear(input_dim, input_dim),
                                nn.ReLU(True),
                                )
        self.psenc = nn.Sequential(
                        nn.Linear(1024, global_cond_dim))

        self.use_global_geometry = use_global_geometry
        self.downsample_pcd_enc = downsample_pcd_enc
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0) # init the final output layer's weights to zeros
        self.sigma_encoder = nn.Sequential(
        PositionalEmbedding(num_channels=diffusion_step_embed_dim),
        nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
        nn.ReLU(inplace=True),
        )
        fusion_dim = diffusion_step_embed_dim + input_dim + global_cond_dim*2
        self.fusion_tail_rot_x = nn.Sequential(
                nn.Linear(fusion_dim, fusion_projection_dim),
                nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                nn.ReLU(inplace=True),
                nn.Linear(fusion_projection_dim, fusion_projection_dim),
                nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                nn.ReLU(inplace=True),
                Linear(fusion_projection_dim, 3, **init_zero),
            )
        self.fusion_tail_rot_y = nn.Sequential(
            nn.Linear(fusion_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True),
            nn.Linear(fusion_projection_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True),
            Linear(fusion_projection_dim, 3, **init_zero),
        )
        
        ''' translation regress head '''
        self.fusion_tail_trans = nn.Sequential(
            nn.Linear(fusion_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True),
            Linear(fusion_projection_dim, 3, **init_zero),
        )    

        self.rotation_orthogonalization = rotation_orthogonalization

    def forward(self, sample, timestep, geometry1=None, geometry2=None):

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if geometry1 is not None:
            if self.use_global_geometry:
                if self.downsample_pcd_enc:
                    geometry1 = self.psenc(geometry1.to(self.device))
                    geometry2 = self.psenc(geometry2.to(self.device))
                if len(geometry1.shape) == 3:
                    geometry1 = geometry1.squeeze(1)
                    geometry2 = geometry2.squeeze(1)
                global_feature = torch.cat([geometry1, geometry2, global_feature], dim=-1)
            else:
                center1, enc_pcd1 = self.pcd_encoder(geometry1[:, :, :3], geometry1[:, :, 3:]) if not self.use_pointnet else self.pcd_encoder(geometry1[:, :, :3])
                center2, enc_pcd2 = self.pcd_encoder(geometry2[:, :, :3], geometry2[:, :, 3:]) if not self.use_pointnet else self.pcd_encoder(geometry2[:, :, :3])
                enc_pcd1 = self.pcd_mlp_encoder(enc_pcd1, center1)
                enc_pcd2 = self.pcd_mlp_encoder(enc_pcd2, center2)
                global_feature = torch.cat([enc_pcd1, enc_pcd2, global_feature], dim=-1)

        x = self.pose_encoder(sample)
        total_feat = torch.cat([global_feature, x], dim=-1)
        rot_x = self.fusion_tail_rot_x(total_feat)
        rot_y = self.fusion_tail_rot_y(total_feat)
        trans = self.fusion_tail_trans(total_feat)

        x = torch.cat([trans, rot_x, rot_y], dim=-1)
        if self.rotation_orthogonalization:
            process_decoder_output = x.clone()
            v1 = process_decoder_output[:, 3:6]
            v2 = process_decoder_output[:, 6:]
            # Ensure the norm of v1 is non-zero
            if torch.all(torch.norm(v1, dim=-1) > 0):
                v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
                v2 = v2 - torch.sum(v2 * v1, dim=-1, keepdim=True) * v1
                if torch.all(torch.norm(v2, dim=-1) > 0):
                    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
                    output_scene = torch.concatenate([process_decoder_output[:, :3], v1, v2], dim=-1)
                    return output_scene
        return x
