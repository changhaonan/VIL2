"""Modified from official Diffusion Policy"""
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


class ObjDPModel:
    """Object-wise Diffusion policy Model"""

    def __init__(self, cfg, vision_encoder, noise_pred_net):
        self.cfg = cfg
        # check cuda and mps
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.device = "cpu"
        # parameters
        self.obs_horizon = cfg.MODEL.OBS_HORIZON
        self.action_horizon = cfg.MODEL.ACTION_HORIZON
        self.pred_horizon = cfg.MODEL.PRED_HORIZON
        self.action_dim = cfg.MODEL.NOISE_NET.INIT_ARGS["input_dim"]
        self.pose_dim = cfg.MODEL.POSE_DIM
        self.time_emb_dim = cfg.MODEL.TIME_EMB_DIM if cfg.MODEL.USE_POSITIONAL_EMBEDDING else 1
        self.geometry_feat_dim = cfg.MODEL.GEOMETRY_FEAT_DIM
        self.use_positional_embedding = cfg.MODEL.USE_POSITIONAL_EMBEDDING

        self.recon_voxel_center = cfg.MODEL.RECON_VOXEL_CENTER
        self.recon_time_stamp = cfg.MODEL.RECON_TIME_STAMP
        self.recon_data_stamp = cfg.MODEL.RECON_DATA_STAMP
        self.cond_geometry_feature = cfg.MODEL.COND_GEOMETRY_FEATURE
        self.cond_voxel_center = cfg.MODEL.COND_VOXEL_CENTER
        self.guide_time_consistency = cfg.MODEL.GUIDE_TIME_CONSISTENCY
        self.guide_data_consistency = cfg.MODEL.GUIDE_DATA_CONSISTENCY
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
        # ablation module
        self.recon_voxel_center = cfg.MODEL.RECON_VOXEL_CENTER
        # build embedding key
        self.build_embedding_key()

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
                        B = nbatch["obj_voxel_feat"].shape[0]
                        V = nbatch["obj_voxel_feat"].shape[2]
                        ncond = self.assemble_cond(nbatch, B, V)  # (B*V, pred_horizon, cond_dim)
                        naction = self.assemble_action(nbatch, B, V).to(self.device)  # (B*V, pred_horizon, action_dim)
                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device)
                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (B * V,),
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

    def inference(self, obs_deque: collections.deque, stats: dict, batch_size: int):
        """Inference with the model"""
        B = batch_size  # inference batch size is 1
        V = obs_deque[0]["obj_voxel_feat"].shape[0]

        # stack the last obs_horizon number of observations
        obj_voxel_feat = np.stack([x["obj_voxel_feat"] for x in obs_deque])[None, ...]
        obj_voxel_center = np.stack([x["obj_voxel_center"] for x in obs_deque])[None, ...]

        # normalize observation
        nobj_voxel_feat = normalize_data(obj_voxel_feat, stats=stats["obj_voxel_feat"])
        nobj_voxel_center = normalize_data(obj_voxel_center, stats=stats["obj_voxel_center"])

        # device transfer
        nobj_voxel_feat = torch.from_numpy(nobj_voxel_feat).to(self.device).to(torch.float32).tile((B, 1, 1, 1))
        nobj_voxel_center = torch.from_numpy(nobj_voxel_center).to(self.device).to(torch.float32).tile((B, 1, 1, 1))
        nbatch = {
            "obj_voxel_feat": nobj_voxel_feat,
            "obj_voxel_center": nobj_voxel_center,
        }

        # infer action
        with torch.no_grad():
            ncond = self.assemble_cond(nbatch, B, V)  # (B*V, pred_horizon, cond_dim)

            # initialize action from Guassian noise
            noisy_action = torch.randn((B * V, self.pred_horizon, self.action_dim), device=self.device)
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
                gamma = 0.5
                if self.guide_time_consistency and self.recon_time_stamp:
                    # guide the time to be the same
                    # gamma = 1.0 * torch.sqrt(1 - self.noise_scheduler.alphas[k])
                    # gamma = 0.5
                    # FIXME: old-version: but seems working better???
                    # ntime_stamp_mean = naction[..., 0:1].mean(dim=0, keepdim=True)
                    # naction = (1 - gamma) * naction + gamma * ntime_stamp_mean
                    ntime_stamp_mean = naction[..., offset:offset+self.time_emb_dim].reshape((B, V, self.pred_horizon, self.time_emb_dim)).mean(
                        dim=1, keepdim=True).repeat(1, V, 1, 1).flatten(start_dim=0, end_dim=1)  # (B*V, pred_horizon, time_emb_dim)
                    ntime_stamp_grad = naction[..., offset:offset+self.time_emb_dim] - ntime_stamp_mean
                    naction_grad[..., offset:offset+self.time_emb_dim] = gamma * ntime_stamp_grad
                    offset += self.time_emb_dim
                if self.guide_data_consistency and self.recon_data_stamp:
                    ndata_stamp_mean = naction[..., offset:offset+self.time_emb_dim].reshape((B, V, self.pred_horizon, self.time_emb_dim)).mean(
                        dim=1, keepdim=True).repeat(1, V, 1, 1).flatten(start_dim=0, end_dim=1)
                    ndata_stamp_grad = naction[..., offset:offset+self.time_emb_dim] - ndata_stamp_mean
                    naction_grad[..., offset:offset+self.time_emb_dim] = gamma * ndata_stamp_grad
                    offset += self.time_emb_dim
                # apply gradient
                naction = naction - naction_grad

        # unnormalize action
        naction = naction.detach().to("cpu").numpy().reshape((B, V, self.pred_horizon, self.action_dim))
        return self.parse_action(naction, stats)

    def save(self, export_path):
        """Save model weights"""
        torch.save(self.nets.state_dict(), export_path)

    def load(self, export_path):
        """Load model weights"""
        state_dict = torch.load(export_path, map_location=self.device)
        self.nets.load_state_dict(state_dict)
        print("Pretrained weights loaded.")

    def assemble_action(self, nbatch, B, V):
        """Dynamically assemble action"""
        action_list = []
        if self.recon_time_stamp:
            nobj_timestamp = nbatch["t"].to(self.device)
            nobj_timestamp = nobj_timestamp[:, None, ...].repeat(1, V, 1, 1)  # (B, V, pred_horizon, 1)
            nobj_timestamp = nobj_timestamp.flatten(start_dim=0, end_dim=2)  # (B * V, pred_horizon, 1)
            if self.use_positional_embedding:
                nobj_timestamp = self.nets["positional_embedding"](nobj_timestamp).reshape((B * V, self.pred_horizon, self.time_emb_dim))
            action_list.append(nobj_timestamp)
        if self.recon_data_stamp:
            nobj_data_stamp = nbatch["data_stamp"].to(self.device)
            nobj_data_stamp = nobj_data_stamp[:, None, ...].repeat(1, V, 1, 1)  # (B, V, pred_horizon, 1)
            nobj_data_stamp = nobj_data_stamp.flatten(start_dim=0, end_dim=2)  # (B * V, pred_horizon, 1)
            if self.use_positional_embedding:
                nobj_data_stamp = self.nets["positional_embedding"](nobj_data_stamp).reshape((B * V, self.pred_horizon, self.time_emb_dim))
            action_list.append(nobj_data_stamp)
        if self.recon_voxel_center:
            nobj_voxel_center = nbatch["obj_voxel_center"].to(self.device)  # (B, pred_horizon, V, 3)
            nobj_voxel_center = nobj_voxel_center[:, : self.pred_horizon, :, :3]  # (B, pred_horizon, V, 3)
            nobj_voxel_center = nobj_voxel_center.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)  # (B * V, pred_horizon, 3)
            # append voxel center to action
            action_list.append(nobj_voxel_center)
        if len(action_list) == 0:
            naction = None
        else:
            naction = torch.cat(action_list, dim=-1)  # (B * V, pred_horizon, action_dim)
        return naction

    def assemble_cond(self, nbatch, B, V):
        """Dynamically assemble condition"""
        cond_list = []
        if self.cond_geometry_feature:
            nobj_voxel_feat = nbatch["obj_voxel_feat"].to(self.device)  # (B, pred_horizon, V, D)
            nobj_voxel_feat = nobj_voxel_feat[:, : self.pred_horizon, :, :]
            nobj_voxel_feat = nobj_voxel_feat.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)  # (B * V, pred_horizon, D)
            cond_list.append(nobj_voxel_feat)
        if self.cond_voxel_center:
            nobj_voxel_center = nbatch["obj_voxel_center"].to(self.device)
            nobj_voxel_center = nobj_voxel_center[:, : self.pred_horizon, :, :3]
            nobj_voxel_center = nobj_voxel_center.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)
            cond_list.append(nobj_voxel_center)
        if len(cond_list) == 0:
            ncond = None
        else:
            ncond = torch.cat(cond_list, dim=-1)  # (B * V, pred_horizon, cond_dim)
        return ncond

    def parse_action(self, naction, stats):
        """Parse action into timestamp and pose"""
        offset = 0
        actions = {}
        if self.recon_time_stamp:
            ntime_stamp = naction[..., offset:self.time_emb_dim]
            if not self.use_positional_embedding:
                # unnormalize
                time_stamp = unnormalize_data(ntime_stamp, stats=stats["t"])
            else:
                time_stamp = self.parse_embedding(ntime_stamp)
            actions["t"] = time_stamp
            offset += self.time_emb_dim
        if self.recon_data_stamp:
            ndata_stamp = naction[..., offset:offset + self.time_emb_dim]
            if not self.use_positional_embedding:
                # unnormalize
                data_stamp = unnormalize_data(ndata_stamp, stats=stats["data_stamp"])
            else:
                data_stamp = self.parse_embedding(ndata_stamp)
            actions["data_stamp"] = data_stamp
            offset += self.time_emb_dim
        if self.recon_voxel_center:
            npose = naction[..., offset:offset + 3]
            # unnormalize
            pose = unnormalize_data(npose, stats=stats["obj_voxel_center"])
            actions["obj_voxel_center"] = pose
            offset += 3
        return actions

    def build_embedding_key(self, max_len=100):
        """Build embedding key"""
        self.embedding_key = torch.arange(max_len, device=self.device).reshape((-1))
        self.embedding_key = self.nets["positional_embedding"](self.embedding_key).detach().to("cpu").numpy()
        self.embedding_key = self.embedding_key / np.linalg.norm(self.embedding_key, axis=-1, keepdims=True)

    def parse_embedding(self, embedding_query):
        """Parse embedding query into timestamp and pose"""
        B, V, L, D = embedding_query.shape
        embedding_query = embedding_query.reshape((-1, self.time_emb_dim))
        # normalize
        embedding_query = embedding_query / np.linalg.norm(embedding_query, axis=-1, keepdims=True)
        embedding_attention = np.dot(embedding_query, self.embedding_key.T)
        # argmax along the last dimension
        embedding_attention = np.argmax(embedding_attention, axis=-1)
        return embedding_attention.reshape((B, V, L, 1))
