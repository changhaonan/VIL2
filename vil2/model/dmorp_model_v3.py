"""Diffusion model for generating multi-object relative pose using PCD diffusion net"""

from __future__ import annotations
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import math
import os
import time
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vil2.model.network.pcd_noise_net import PcdNoiseNet
from box import Box
import yaml
import vil2.utils.misc_utils as utils


class LitPcdDiffusion(L.LightningModule):
    """Lignthing module for PCD diffusion model"""

    def __init__(self, pcd_noise_net: PcdNoiseNet, cfg: Box, **kwargs: Any) -> None:
        super().__init__()
        self.pcd_noise_net = pcd_noise_net
        self.lr = cfg.TRAIN.LR
        self.start_time = time.time()
        self.diffusion_process = cfg.MODEL.DIFFUSION_PROCESS
        self.num_diffusion_iters = cfg.MODEL.NUM_DIFFUSION_ITERS
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=False,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

    def criterion(self, batch):
        target_coord = batch["target_coord"].to(torch.float32)
        target_normal = batch["target_normal"].to(torch.float32)
        target_color = batch["target_color"].to(torch.float32)
        target_label = batch["target_label"].to(torch.long)
        fixed_coord = batch["fixed_coord"].to(torch.float32)
        fixed_normal = batch["fixed_normal"].to(torch.float32)
        fixed_color = batch["fixed_color"].to(torch.float32)
        fixed_label = batch["fixed_label"].to(torch.long)
        pose9d = batch["target_pose"].to(torch.float32)

        # Compute conditional features
        cond_feat = self.pose_transformer.encode_cond(target_coord, target_normal, target_color, target_label, fixed_coord, fixed_normal, fixed_color, fixed_label)
        if self.diffusion_process == "ddpm":
            # sample noise to add to actions
            noise = torch.randn((pose9d.shape[0], pose9d.shape[1]), device=self.device)
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                low=0,
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(pose9d.shape[0],),
                device=self.device,
            ).long()
            # add noisy actions
            noisy_pose9d = self.noise_scheduler.add_noise(pose9d, noise, timesteps)
            # predict noise residual
            noise_pred = self.pose_transformer(noisy_pose9d, timesteps, cond_feat)
            # compute loss
            trans_loss = F.mse_loss(noise_pred[:, :3], noise[:, :3])
            loss = trans_loss
            if not self.translation_only:
                rx_loss = F.mse_loss(noise_pred[:, 3:6], noise[:, 3:6])
                ry_loss = F.mse_loss(noise_pred[:, 6:9], noise[:, 6:9])
                loss += rx_loss + ry_loss
            else:
                rx_loss = 0
                ry_loss = 0
            return trans_loss, rx_loss, ry_loss, loss
        else:
            raise ValueError(f"Diffusion process {self.diffusion_process} not supported.")
