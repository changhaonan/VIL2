from __future__ import annotations
import torch
import torch.nn as nn
import math
from vil2.model.embeddings.pct import PointTransformerEncoderSmall, EncoderMLP, PointNet
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Transformer(nn.Module):
    def __init__(self, 
                  input_dim: int = 80, 
                  global_cond_dim: int = 80,
                  diffusion_step_embed_dim: int = 80, 
                  num_attention_heads: int = 8, 
                  encoder_hidden_dim: int = 16,
                  encoder_dropout: float = 0.1,
                  encoder_activation: str = "relu",
                  encoder_num_layers: int = 8,
                  ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_size = input_dim
        self.time_emb_size = diffusion_step_embed_dim
        self.use_conditioning = global_cond_dim > 0

        encoder_input_dim = input_dim + diffusion_step_embed_dim + 2*global_cond_dim

        encoder_layers = TransformerEncoderLayer(encoder_input_dim, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, encoder_num_layers)
        # transformer_decoder_layer = TransformerDecoderLayer(encoder_input_dim, num_attention_heads, encoder_hidden_dim, encoder_dropout)
        # self.transformer_decoder = TransformerDecoder(transformer_decoder_layer, encoder_num_layers)
        self.transformer_decoder = nn.Linear(encoder_input_dim, input_dim)
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, 4 * dsed),
            nn.GELU(),
            nn.Linear(4 * dsed, dsed),
        )
        # self.pcd_encoder = PointTransformerEncoderSmall(output_dim=global_cond_dim, input_dim=6, mean_center=False)
        self.pcd_encoder = PointNet(emb_dims=global_cond_dim)
        self.pcd_mlp_encoder = EncoderMLP(256, global_cond_dim, uses_pt=True)
        self.pose_encoder = nn.Sequential(nn.Linear(9, input_dim))

    def forward(self, sample: torch.Tensor,  timestep: int, global_cond1: torch.Tensor | None, global_cond2: torch.Tensor | None):
        sample = sample.to(dtype=torch.float32, device=self.device)
        sample_enc = self.pose_encoder(sample)
        sample = sample_enc
        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample_enc.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample_enc.shape[0])
        time_emb = self.diffusion_step_encoder(timesteps)

        if global_cond1 is not None:
            center1, enc_pcd1 = self.pcd_encoder(global_cond1[:, :, :3])
            center2, enc_pcd2 = self.pcd_encoder(global_cond2[:, :, :3])
            # center1, enc_pcd1 = self.pcd_encoder(global_cond1[:, :, :3], global_cond1[:, :, 3:])
            # center2, enc_pcd2 = self.pcd_encoder(global_cond2[:, :, :3], global_cond2[:, :, 3:])
            # enc_pcd1 = self.pcd_mlp_encoder(enc_pcd1, center1)
            # enc_pcd2 = self.pcd_mlp_encoder(enc_pcd2, center2)
            embeddings = torch.concat([sample, enc_pcd1, enc_pcd2, time_emb], dim=-1)
        else:
            embeddings = torch.concat([sample, time_emb], dim=-1)

        encoder_output = self.transformer_encoder(embeddings)
        
        # decoder_output = self.transformer_decoder(encoder_output, encoder_output)
        decoder_output = self.transformer_decoder(encoder_output)

        output_scene = decoder_output[:, :9]

        return output_scene
