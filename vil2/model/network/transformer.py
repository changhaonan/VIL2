from __future__ import annotations
import torch
import numpy as np
import torch.nn as nn
from vil2.model.embeddings.pct import PointTransformerEncoderSmall, EncoderMLP, PointNet, DropoutSampler 
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vil2.model.embeddings.sinusoidal import PositionalEmbedding
from vil2.model.network.genpose_modules import Linear

class Transformer(nn.Module):
    def __init__(self, 
                  input_dim: int = 80, 
                  global_cond_dim: int = 80,
                  diffusion_step_embed_dim: int = 80, 
                  num_attention_heads: int = 2, 
                  encoder_hidden_dim: int = 256,
                  encoder_dropout: float = 0.5,
                  encoder_activation: str = "relu",
                  encoder_num_layers: int = 3,
                  use_global_geometry: bool = True,
                  fusion_projection_dim: int = 512,
                  use_pointnet: bool = False,
                  rotation_orthogonalization: bool = False,
                  downsample_pcd_enc: bool = False,
                  downsample_size: int = 256,
                  use_dropout_sampler: bool = False,
                  ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_size = input_dim

        encoder_input_dim = input_dim + diffusion_step_embed_dim + 2*global_cond_dim

        self.encoder_layer = TransformerEncoderLayer(d_model=encoder_input_dim, 
                                                 nhead=num_attention_heads,
                                                 dim_feedforward=encoder_hidden_dim, 
                                                 dropout=encoder_dropout, 
                                                 activation=encoder_activation,
                                                #  batch_first=True,
                                                #  norm_first=True
                                                 )
        self.transformer_encoder = TransformerEncoder(encoder_layer=self.encoder_layer, 
                                                      num_layers=encoder_num_layers)
        
        self.use_global_geometry = use_global_geometry  

        self.transformer_decoder = nn.Linear(encoder_input_dim, 9)
        self.diffusion_step_encoder = nn.Sequential(
            PositionalEmbedding(num_channels=diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
            nn.ReLU(inplace=True)
        )
        self.pcd_encoder = PointTransformerEncoderSmall(output_dim=global_cond_dim, input_dim=6, mean_center=False) if not use_pointnet else PointNet(emb_dims=256)
        self.pcd_mlp_encoder = EncoderMLP(256, global_cond_dim, uses_pt=True)
        self.pose_encoder = nn.Sequential(
                                        nn.Linear(9, input_dim),
                                        nn.ReLU(),
                                        nn.Linear(input_dim, input_dim),
                                        nn.ReLU(),
                                        )
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0) # init the final output layer's weights to zeros
        fusion_dim = diffusion_step_embed_dim + input_dim + global_cond_dim*2
        self.fusion_tail_rot_x = nn.Sequential(
                nn.Linear(fusion_dim, fusion_projection_dim),
                nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                nn.ReLU(inplace=True),
                # nn.Linear(fusion_projection_dim, fusion_projection_dim),
                # nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                # nn.ReLU(inplace=True),
                Linear(fusion_projection_dim, 9, **init_zero),
            )
        self.fusion_tail_rot_y = nn.Sequential(
            nn.Linear(fusion_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True),
            # nn.Linear(fusion_projection_dim, fusion_projection_dim),
            # nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            # nn.ReLU(inplace=True),
            Linear(fusion_projection_dim, 3, **init_zero),
        )
        
        ''' translation regress head '''
        self.fusion_tail_trans = nn.Sequential(
            nn.Linear(fusion_dim, fusion_projection_dim),
            nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True),
            # nn.Linear(fusion_projection_dim, fusion_projection_dim),
            # nn.BatchNorm1d(fusion_projection_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            # nn.ReLU(inplace=True),
            Linear(fusion_projection_dim, 3, **init_zero),
        )    
        self.final_linear = nn.Sequential(nn.Linear(fusion_dim, 9)) if not use_dropout_sampler else DropoutSampler(fusion_dim, 9, dropout_rate=0.0)
        self.use_dropout_sampler = use_dropout_sampler
        self.downsample_pcd_enc = downsample_pcd_enc
        self.psenc = nn.Sequential(
                nn.Linear(global_cond_dim, downsample_size))
        self.rotation_orthogonalization = rotation_orthogonalization

    def forward(self, x: torch.Tensor,  timestep: int, geometry1: torch.Tensor | None, geometry2: torch.Tensor | None):
        x = x.to(dtype=torch.float32, device=self.device)
        sample_enc = self.pose_encoder(x)

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample_enc.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample_enc.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)

        if geometry1 is not None:
            if self.use_global_geometry:
                if self.downsample_pcd_enc:
                    geometry1 = self.psenc(geometry1.to(self.device))
                    geometry2 = self.psenc(geometry2.to(self.device))
                if len(geometry1.shape) == 3:
                    geometry1 = geometry1.squeeze(1)
                    geometry2 = geometry2.squeeze(1)
                global_feature = torch.concat([geometry1, geometry2, global_feature], dim=-1)
            else:
                center1, enc_pcd1 = self.pcd_encoder(geometry1[:, :, :3], geometry1[:, :, 3:])
                center2, enc_pcd2 = self.pcd_encoder(geometry2[:, :, :3], geometry2[:, :, 3:])
                enc_pcd1 = self.pcd_mlp_encoder(enc_pcd1, center1)
                enc_pcd2 = self.pcd_mlp_encoder(enc_pcd2, center2)
                global_feature = torch.concat([sample_enc, enc_pcd1, enc_pcd2, global_feature], dim=-1)
        else:
            global_feature = torch.concat([sample_enc, global_feature], dim=-1)

        # Observation: concatenating input_x (sample_enc) and global_feature (global_cond) causes cuda error and concatenating them later below
        total_feat = torch.cat([sample_enc, global_feature], dim=-1)

        encoder_output = self.transformer_encoder(total_feat)

        if self.use_dropout_sampler:
            output_scene = self.final_linear(encoder_output)
            return output_scene
        
        rot_x = self.fusion_tail_rot_x(encoder_output.detach().cpu().cuda(0).clone())
        if self.rotation_orthogonalization:
            process_decoder_output = rot_x.clone()
            v1 = process_decoder_output[:, 3:6]
            v2 = process_decoder_output[:, 6:]
            # Ensure the norm of v1 is non-zero
            if torch.all(torch.norm(v1, dim=-1) > 0):
                v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
                v2 = v2 - torch.sum(v2 * v1, dim=-1, keepdim=True) * v1
                # Ensure the norm of v2 is non-zero
                if torch.all(torch.norm(v2, dim=-1) > 0):
                    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
                    output_scene = torch.concatenate([process_decoder_output[:, :3], v1, v2], dim=-1)
                    return output_scene
        return rot_x.clone()