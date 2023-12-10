"""Modified from official Diffusion Policy"""
from __future__ import annotations
import diffusers
import collections
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from vil2.data.obj_dp_dataset import normalize_data, unnormalize_data
import vil2.utils.torch_utils as torch_utils
import clip


class DmorpModel:
    """Diffusion Model for multi-object relative Pose Generation"""

    def __init__(self, cfg, vision_encoder, noise_pred_net, device):
        self.cfg = cfg
        # check cuda and mps
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
        # self.device = "cpu"
        # parameters
        self.max_scene_size = cfg.MODEL.MAX_SCENE_SIZE
        self.action_dim = cfg.MODEL.NOISE_NET.INIT_ARGS["input_dim"]
        self.pose_dim = cfg.MODEL.POSE_DIM
        self.time_emb_dim = cfg.MODEL.TIME_EMB_DIM if cfg.MODEL.USE_POSITIONAL_EMBEDDING else 1
        self.geometry_feat_dim = cfg.MODEL.GEOMETRY_FEAT_DIM
        self.semantic_feat_dim = cfg.MODEL.SEMANTIC_FEAT_DIM
        self.use_positional_embedding = cfg.MODEL.USE_POSITIONAL_EMBEDDING

        # ablation module
        self.recon_semantic_feature = cfg.MODEL.RECON_SEMANTIC_FEATURE
        self.recon_data_stamp = cfg.MODEL.RECON_DATA_STAMP
        self.cond_geometry_feature = cfg.MODEL.COND_GEOMETRY_FEATURE
        self.cond_semantic_feature = cfg.MODEL.COND_SEMANTIC_FEATURE
        self.guide_data_consistency = cfg.MODEL.GUIDE_DATA_CONSISTENCY
        self.guide_semantic_consistency = cfg.MODEL.GUIDE_SEMANTIC_CONSISTENCY
        # vision encoder
        # self.vision_encoder = vision_encoder
        # scheduler
        self.num_diffusion_iters = cfg.MODEL.NUM_DIFFUSION_ITERS
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )
        # noise net
        self.noise_pred_net = noise_pred_net
        self.nets = nn.ModuleDict(
            {
                # "vision_encoder": self.vision_encoder,
                "noise_pred_net": self.noise_pred_net,
                "positional_embedding": torch_utils.SinusoidalEmbedding(size=self.time_emb_dim),
            }
        ).to(self.device)
        # build embedding key
        self.build_embedding_key()
        # semantic feature related
        self.semantic_feat_type = cfg.MODEL.SEMANTIC_FEAT_TYPE
        if self.semantic_feat_type == "clip":
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
            self.clip_model.eval()
            self.clip_model.requires_grad_(False)

    def train(self, num_epochs: int, data_loader):
        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        ema = EMAModel(parameters=self.nets.parameters(), power=0.75)

        # Standard ADAM optimizer
        # Note that EMA parametesr are not optimized
        optimizer = torch.optim.AdamW(params=self.nets.parameters(), lr=1e-4, weight_decay=1e-6)

        # Cosine LR schedule with linear warmup
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(data_loader) * num_epochs,
        )

        with tqdm(range(num_epochs), desc="Epoch") as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(data_loader, desc="Batch", leave=False) as tepoch:
                    for nbatch in tepoch:
                        B = nbatch["sem_feat"].shape[0]  # (B, M, D)
                        M = nbatch["sem_feat"].shape[1]
                        ncond = self.assemble_cond(nbatch, B, M)  # (B*M, cond_dim)
                        naction = self.assemble_action(nbatch, B, M).to(self.device)  # (B*V, pred_horizon, action_dim)
                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device)
                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (B*M,),
                            device=self.device,
                        ).long()

                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

                        # predict the noise residual
                        noise_pred = self.noise_pred_net(
                            noisy_actions, timesteps, global_cond=ncond
                        )

                        # L2 loss
                        loss = nn.functional.mse_loss(noise_pred, noise)

                        # optimize
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        ema.step(self.nets.parameters())

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))
        # copy ema back to model
        ema.copy_to(self.nets.parameters())

    def inference(self, obs, stats: dict, batch_size: int):
        """Inference with the model"""
        B = batch_size  # inference batch size is 1
        M = obs["sem_feat"].shape[0]  # (M, D)

        sem_feat = obs["sem_feat"].reshape((M, self.semantic_feat_dim))
        geo_feat = obs["geo_feat"].reshape((M, self.geometry_feat_dim))

        # normalize observation
        nsem_feat = normalize_data(sem_feat, stats=stats["sem_feat"])
        # ngeo_feat = normalize_data(geo_feat, stats=stats["geo_feat"])
        nsem_feat = sem_feat
        ngeo_feat = geo_feat

        # device transfer
        nsem_feat = torch.from_numpy(nsem_feat).to(self.device).to(torch.float32).tile((B, 1, 1))
        ngeo_feat = torch.from_numpy(ngeo_feat).to(self.device).to(torch.float32).tile((B, 1, 1))
        nbatch = {
            "sem_feat": nsem_feat,
            "geo_feat": ngeo_feat,
        }

        # infer action
        with torch.no_grad():
            ncond = self.assemble_cond(nbatch, B, M)  # (B*V, pred_horizon, cond_dim)

            # initialize action from Guassian noise
            noisy_action = torch.randn((B * M, self.action_dim), device=self.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.nets["noise_pred_net"](
                    sample=naction, timestep=k, global_cond=ncond
                )
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample
                # gradient guided time consistency
                naction_grad = torch.zeros_like(naction)
                offset = 0
                gamma = 0.1
                if self.guide_data_consistency and self.recon_data_stamp:
                    ndata_stamp_mean = naction[..., offset:offset+self.time_emb_dim].reshape((B, M, self.time_emb_dim)).mean(
                        dim=1, keepdim=True).repeat(1, M, 1).flatten(start_dim=0, end_dim=1)
                    ndata_stamp_grad = naction[..., offset:offset+self.time_emb_dim] - ndata_stamp_mean
                    naction_grad[..., offset:offset+self.time_emb_dim] = gamma * ndata_stamp_grad
                    offset += self.time_emb_dim
                # apply gradient
                naction = naction - naction_grad

        # unnormalize action
        naction = naction.detach().to("cpu").numpy().reshape((B, M, self.action_dim))
        return self.parse_action(naction, stats)

    def save(self, export_path):
        """Save model weights"""
        torch.save(self.nets.state_dict(), export_path)

    def load(self, export_path):
        """Load model weights"""
        state_dict = torch.load(export_path, map_location=self.device)
        self.nets.load_state_dict(state_dict)
        print("Pretrained weights loaded.")

    def assemble_action(self, nbatch, B, M):
        """Dynamically assemble action"""
        action_list = []
        if self.recon_data_stamp:
            nobj_data_stamp = nbatch["data_stamp"].to(self.device)  # (B, M, D)
            nobj_data_stamp = nobj_data_stamp.flatten(start_dim=0, end_dim=1)  # (B * M, D)
            if self.use_positional_embedding:
                nobj_data_stamp = self.nets["positional_embedding"](nobj_data_stamp).reshape((-1, self.time_emb_dim))
            action_list.append(nobj_data_stamp)
        if self.recon_semantic_feature:
            nsem_feat = nbatch["sem_feat"].to(self.device)  # (B, M, D)
            nsem_feat = nsem_feat.flatten(start_dim=0, end_dim=1)  # (B * M, D)
            action_list.append(nsem_feat)
        if len(action_list) == 0:
            naction = None
        else:
            naction = torch.cat(action_list, dim=-1)  # (B * M, action_dim)
        return naction

    def assemble_cond(self, nbatch, B, M):
        """Dynamically assemble condition"""
        cond_list = []
        if self.cond_semantic_feature:
            nsem_feat = nbatch["sem_feat"].to(self.device)  # (B, M, D)
            nsem_feat = nsem_feat.flatten(start_dim=0, end_dim=1)  # (B * M, D)
            cond_list.append(nsem_feat)
        if self.cond_geometry_feature:
            ngeo_feat = nbatch["geo_feat"].to(self.device)  # (B, M, D)
            ngeo_feat = ngeo_feat.flatten(start_dim=0, end_dim=1)  # (B * M, D)
            cond_list.append(ngeo_feat)
        if len(cond_list) == 0:
            ncond = None
        else:
            ncond = torch.cat(cond_list, dim=-1)  # (B * M, pred_horizon, cond_dim)
        return ncond

    def parse_action(self, naction, stats):
        """Parse action into timestamp and pose"""
        offset = 0
        actions = {}
        if self.recon_data_stamp:
            ndata_stamp = naction[..., offset:offset + self.time_emb_dim]
            if not self.use_positional_embedding:
                # unnormalize
                data_stamp = unnormalize_data(ndata_stamp, stats=stats["data_stamp"])
            else:
                data_stamp = self.parse_embedding(ndata_stamp)
            actions["data_stamp"] = data_stamp
            offset += self.time_emb_dim
        if self.recon_semantic_feature:
            nsem_feat = naction[..., offset:offset + self.semantic_feat_dim]
            actions["sem_feat"] = unnormalize_data(nsem_feat, stats=stats["sem_feat"])
            offset += self.semantic_feat_dim
        return actions

    def build_embedding_key(self, max_len=1000):
        """Build embedding key"""
        self.embedding_key = torch.arange(max_len, device=self.device).reshape((-1))
        self.embedding_key = self.nets["positional_embedding"](self.embedding_key).detach().to("cpu").numpy()
        self.embedding_key = self.embedding_key / np.linalg.norm(self.embedding_key, axis=-1, keepdims=True)

    def parse_embedding(self, embedding_query):
        """Parse embedding query into timestamp and pose"""
        B, M, D = embedding_query.shape
        embedding_query = embedding_query.reshape((-1, self.time_emb_dim))
        # normalize
        embedding_query = embedding_query / np.linalg.norm(embedding_query, axis=-1, keepdims=True)
        embedding_attention = np.dot(embedding_query, self.embedding_key.T)
        # argmax along the last dimension
        embedding_attention = np.argmax(embedding_attention, axis=-1)
        return embedding_attention.reshape((B, M, 1))

    def encode_text(self, text: str):
        """Encode text into semantic feature"""
        if self.semantic_feat_type == "clip":
            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(text)]).to(self.device)
                semantic_feature = self.clip_model.encode_text(text_inputs).detach().cpu().numpy()[0]
            return semantic_feature

    def parse_vocab(self, sem_feats: np.ndarray, vocab: list[str]):
        """Parse semantic feature to vocabulary"""
        # Build vocabulary dictionary
        B, M, D = sem_feats.shape
        sem_feats = sem_feats.reshape((-1, D))
        vocab_dict = {}
        for i, obj_name in enumerate(vocab):
            vocab_dict[obj_name] = self.encode_text(obj_name)
            # vocab_dict[obj_name] = vocab_dict[obj_name] / np.linalg.norm(vocab_dict[obj_name])  # normalize
        vocab_feats = np.stack(list(vocab_dict.values()))
        # Match semantic feature to vocabulary; by nearest neighbor
        vocab_names = [[]]
        vocab_ids = [[]]
        idx_batch = 0
        for i in range(sem_feats.shape[0]):
            if i % M == 0 and i > 0:
                idx_batch += 1
                vocab_names.append([])
                vocab_ids.append([])
            # nsem_feat = sem_feats[i] / np.linalg.norm(sem_feats[i])
            dist = np.linalg.norm(sem_feats[i] - vocab_feats, axis=-1)
            vocab_names[idx_batch].append(vocab[np.argmin(dist)])
            vocab_ids[idx_batch].append(np.argmin(dist) + 1)  # vocab id starts from 1
        return vocab_names, vocab_ids
