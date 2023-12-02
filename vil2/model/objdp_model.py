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
        self.action_dim = cfg.MODEL.ACTION_DIM
        # vision encoder
        # self.vision_encoder = vision_encoder
        # scheduler
        self.num_diffusion_iters = 100
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
            }
        ).to(self.device)

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
                        # data normalized in dataset
                        # device transfer
                        # voxel_feature
                        nobj_voxel_feat = nbatch["obj_voxel_feat"].to(self.device)  # (B, obs_horizon, num_voxel, dim_feat)
                        nobj_voxel_center = nbatch["obj_voxel_center"].to(self.device)  # (B, obs_horizon, num_voxel, 3)
                        nobj_voxel_obs = torch.cat([nobj_voxel_feat, nobj_voxel_center], dim=-1)  # (B, obs_horizon, num_voxel, D)
                        # voxel_pose (to predict)
                        nobj_voxel_pose = nbatch["obj_voxel_pose"].to(self.device)  # (B, pred_horizon, num_voxel, dim_pose)

                        # ------------------- obs -------------------
                        # (B, obs_horizon, num_voxel, D) -> (B * num_voxel, obs_horizon * D)
                        obs_features = nobj_voxel_obs.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)
                        # (B * num_voxel, obs_horizon, D)
                        obs_cond = obs_features.flatten(start_dim=1)
                        # (B * num_voxel, obs_horizon * D)

                        # ------------------- action -------------------
                        # (B, pred_horizon, num_voxel, D) -> (B * num_voxel, pred_horizon * D)
                        naction = nobj_voxel_pose.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)
                        # (B * num_voxel, pred_horizon, D)

                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device)
                        B = naction.shape[0]
                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (B,),
                            device=self.device,
                        ).long()

                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

                        # predict the noise residual
                        noise_pred = self.noise_pred_net(
                            noisy_actions, timesteps, global_cond=obs_cond
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

    def inference(self, obs_deque: collections.deque, stats: dict):
        """Inference with the model"""
        B = 1  # inference batch size is 1
        num_voxel = obs_deque[0]["obj_voxel_feat"].shape[0]

        # stack the last obs_horizon number of observations
        obj_voxel_feat = np.stack([x["obj_voxel_feat"] for x in obs_deque])
        obj_voxel_center = np.stack([x["obj_voxel_center"] for x in obs_deque])

        # normalize observation
        nobj_voxel_feat = normalize_data(obj_voxel_feat, stats=stats["obj_voxel_feat"])
        nobj_voxel_center = normalize_data(obj_voxel_center, stats=stats["obj_voxel_center"])

        # device transfer
        nobj_voxel_feat = torch.from_numpy(nobj_voxel_feat).to(self.device).to(torch.float32)
        nobj_voxel_center = torch.from_numpy(nobj_voxel_center).to(self.device).to(torch.float32)

        # infer action
        with torch.no_grad():
            # concat with low-dim observations
            obs_features = torch.cat([nobj_voxel_feat, nobj_voxel_center], dim=-1).unsqueeze(0)  # (B, obs_horizon, num_voxel, D)

            # reshape observation to (B * num_voxel, obs_horizon * D)
            obs_features = obs_features.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)
            obs_cond = obs_features.flatten(start_dim=1)
            # (B * num_voxel, obs_horizon * D)

            # initialize action from Guassian noise
            noisy_action = torch.randn((B * num_voxel, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.nets["noise_pred_net"](
                    sample=naction, timestep=k, global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to("cpu").numpy()
        # (B * num_voxel, pred_horizon, action_dim)
        naction = naction.reshape(B, num_voxel, self.pred_horizon, self.action_dim)
        action_pred = unnormalize_data(naction, stats=stats["obj_voxel_pose"])

        return action_pred

    def save(self, export_path):
        """Save model weights"""
        torch.save(self.nets.state_dict(), export_path)

    def load(self, export_path):
        """Load model weights"""
        state_dict = torch.load(export_path, map_location=self.device)
        self.nets.load_state_dict(state_dict)
        print("Pretrained weights loaded.")
