"""Diffusion model for generating multi-object relative pose using Pose Transformer"""
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
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vil2.model.network.pose_transformer_noise import PoseTransformerNoiseNet


class LitPoseDiffusion(L.LightningModule):
    """Lightning module for Pose Diffusion"""

    def __init__(self, pose_transformer: PoseTransformerNoiseNet, cfg=None) -> None:
        super().__init__()
        self.pose_transformer = pose_transformer
        self.lr = cfg.TRAIN.LR
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

    def training_step(self, batch, batch_idx):
        """Training step; DDPM loss"""
        target_coord = batch["target_coord"].to(torch.float32)
        target_normal = batch["target_normal"].to(torch.float32)
        target_color = batch["target_color"].to(torch.float32)
        target_label = batch["target_label"].to(torch.long)
        fixed_coord = batch["fixed_coord"].to(torch.float32)
        fixed_normal = batch["fixed_normal"].to(torch.float32)
        fixed_color = batch["fixed_color"].to(torch.float32)
        fixed_label = batch["fixed_label"].to(torch.long)
        pose9d = batch["target_pose"].to(torch.float32)

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
            noise_pred = self.pose_transformer(
                noisy_pose9d,
                target_coord,
                target_normal,
                target_color,
                target_label,
                fixed_coord,
                fixed_normal,
                fixed_color,
                fixed_label,
            )
            # compute loss
            trans_loss = F.mse_loss(noise_pred[:, :3], noise[:, :3])
            rx_loss = F.mse_loss(noise_pred[:, 3:6], noise[:, 3:6])
            ry_loss = F.mse_loss(noise_pred[:, 6:9], noise[:, 6:9])
            loss = trans_loss + rx_loss + ry_loss
            # log
            self.log("tr_trans_loss", trans_loss, sync_dist=True)
            self.log("tr_rx_loss", rx_loss, sync_dist=True)
            self.log("tr_ry_loss", ry_loss, sync_dist=True)
            # log
            self.log("train_loss", loss, sync_dist=True)
            return loss
        else:
            raise ValueError(f"Diffusion process {self.diffusion_process} not supported.")

    def validation_step(self, batch, batch_idx):
        target_coord = batch["target_coord"].to(torch.float32)
        target_normal = batch["target_normal"].to(torch.float32)
        target_color = batch["target_color"].to(torch.float32)
        target_label = batch["target_label"].to(torch.long)
        fixed_coord = batch["fixed_coord"].to(torch.float32)
        fixed_normal = batch["fixed_normal"].to(torch.float32)
        fixed_color = batch["fixed_color"].to(torch.float32)
        fixed_label = batch["fixed_label"].to(torch.long)
        pose9d = batch["target_pose"].to(torch.float32)

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
            noise_pred = self.pose_transformer(
                noisy_pose9d,
                target_coord,
                target_normal,
                target_color,
                target_label,
                fixed_coord,
                fixed_normal,
                fixed_color,
                fixed_label,
            )
            # compute loss
            trans_loss = F.mse_loss(noise_pred[:, :3], noise[:, :3])
            rx_loss = F.mse_loss(noise_pred[:, 3:6], noise[:, 3:6])
            ry_loss = F.mse_loss(noise_pred[:, 6:9], noise[:, 6:9])
            loss = trans_loss + rx_loss + ry_loss
            # log
            self.log("v_trans_loss", trans_loss, sync_dist=True)
            self.log("v_rx_loss", rx_loss, sync_dist=True)
            self.log("v_ry_loss", ry_loss, sync_dist=True)
            self.log("val_loss", loss, sync_dist=True)
            return loss
        else:
            raise ValueError(f"Diffusion process {self.diffusion_process} not supported.")

    def test_step(self, batch, batch_idx):
        target_coord = batch["target_coord"].to(torch.float32)
        target_normal = batch["target_normal"].to(torch.float32)
        target_color = batch["target_color"].to(torch.float32)
        target_label = batch["target_label"].to(torch.long)
        fixed_coord = batch["fixed_coord"].to(torch.float32)
        fixed_normal = batch["fixed_normal"].to(torch.float32)
        fixed_color = batch["fixed_color"].to(torch.float32)
        fixed_label = batch["fixed_label"].to(torch.long)
        pose9d = batch["target_pose"].to(torch.float32)

        if self.diffusion_process == "ddpm":
            # initialize action from Guassian noise
            noisy_pose9d = torch.randn((pose9d.shape[0], pose9d.shape[1]), device=self.device)
            pose9d_pred = noisy_pose9d
            for k in self.noise_scheduler.timesteps:
                # predict noise residual
                noise_pred = self.pose_transformer(
                    noisy_pose9d,
                    target_coord,
                    target_normal,
                    target_color,
                    target_label,
                    fixed_coord,
                    fixed_normal,
                    fixed_color,
                    fixed_label,
                )
                # inverse diffusion step (remove noise)
                pose9d_pred = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=pose9d_pred
                ).prev_sample
            # compute loss
            trans_loss = F.mse_loss(pose9d_pred[:, :3], pose9d[:, :3])
            rx_loss = F.mse_loss(pose9d_pred[:, 3:6], pose9d[:, 3:6])
            ry_loss = F.mse_loss(pose9d_pred[:, 6:9], pose9d[:, 6:9])
            loss = trans_loss + rx_loss + ry_loss
            # log
            self.log("te_trans_loss", trans_loss, sync_dist=True)
            self.log("te_rx_loss", rx_loss, sync_dist=True)
            self.log("te_ry_loss", ry_loss, sync_dist=True)
            self.log("test_loss", loss, sync_dist=True)
            return loss
        else:
            raise ValueError(f"Diffusion process {self.diffusion_process} not supported.")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class DmorpModel:
    """Transformer Model for multi-object relative Pose Generation"""

    def __init__(self, cfg, pose_transformer: PoseTransformerNoiseNet) -> None:
        self.cfg = cfg
        # parameters
        self.logger_project = cfg.LOGGER.PROJECT
        # build model
        self.pose_transformer = LitPoseDiffusion(pose_transformer, cfg).to(torch.float32)

    def train(self, num_epochs: int, train_data_loader, val_data_loader, save_path: str):
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(save_path, "checkpoints"),
            filename="Dmorp_model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )
        # Trainer
        # If not mac, using ddp_find_unused_parameters_true
        strategy = "ddp_find_unused_parameters_true" if os.uname().sysname != "Darwin" else "auto"
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = L.Trainer(
            max_epochs=num_epochs,
            logger=WandbLogger(
                name=self.experiment_name(), project=self.logger_project, save_dir=os.path.join(save_path, "logs")
            ),
            callbacks=[checkpoint_callback],
            strategy=strategy,
            log_every_n_steps=5,
            accelerator=accelerator,
        )
        trainer.fit(self.pose_transformer, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

    def test(self, test_data_loader, save_path: str):
        # Trainer
        strategy = "ddp_find_unused_parameters_true" if os.uname().sysname != "Darwin" else "auto"
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = L.Trainer(
            logger=WandbLogger(
                name=self.experiment_name(), project=self.logger_project, save_dir=os.path.join(save_path, "logs")
            ),
            strategy=strategy,
        )
        trainer.test(self.pose_transformer, test_dataloaders=test_data_loader, accelerator=accelerator)

    def experiment_name(self):
        noise_net_name = self.cfg.MODEL.NOISE_NET.NAME
        init_args = self.cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name]
        pod = init_args["pcd_output_dim"]
        na = init_args["num_attention_heads"]
        ehd = init_args["encoder_hidden_dim"]
        fpd = init_args["fusion_projection_dim"]
        pp_str = ""
        for points in init_args.points_pyramid:
            pp_str += str(points) + "-"
        usl = f"usl{init_args.use_semantic_label}"
        return f"Dmorp_model_pod{pod}_na{na}_ehd{ehd}_fpd{fpd}_pp{pp_str}_usl{usl}"
