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
import time
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vil2.model.network.pose_transformer_noise import PoseTransformerNoiseNet
from box import Box
import yaml
import vil2.utils.misc_utils as utils


class LitPoseDiffusion(L.LightningModule):
    """Lightning module for Pose Diffusion"""

    def __init__(self, pose_transformer: PoseTransformerNoiseNet, cfg) -> None:
        super().__init__()
        self.pose_transformer = pose_transformer
        self.lr = cfg.TRAIN.LR
        self.start_time = time.time()
        self.diffusion_process = cfg.MODEL.DIFFUSION_PROCESS
        self.num_diffusion_iters = cfg.MODEL.NUM_DIFFUSION_ITERS
        self.translation_only = self.pose_transformer.translation_only
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
        anchor_coord = batch["anchor_coord"].to(torch.float32)
        anchor_normal = batch["anchor_normal"].to(torch.float32)
        anchor_color = batch["anchor_color"].to(torch.float32)
        anchor_label = batch["anchor_label"].to(torch.long)
        pose9d = batch["target_pose"].to(torch.float32)

        # Compute conditional features
        cond_feat = self.pose_transformer.encode_cond(
            target_coord, target_normal, target_color, target_label, anchor_coord, anchor_normal, anchor_color, anchor_label
        )
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

    def training_step(self, batch, batch_idx):
        """Training step; DDPM loss"""
        trans_loss, rx_loss, ry_loss, loss = self.criterion(batch)
        # log
        self.log("tr_trans_loss", trans_loss, sync_dist=True)
        self.log("tr_rx_loss", rx_loss, sync_dist=True)
        self.log("tr_ry_loss", ry_loss, sync_dist=True)
        # log
        self.log("train_loss", loss, sync_dist=True)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        trans_loss, rx_loss, ry_loss, loss = self.criterion(batch)
        self.log("v_trans_loss", trans_loss, sync_dist=True)
        self.log("v_rx_loss", rx_loss, sync_dist=True)
        self.log("v_ry_loss", ry_loss, sync_dist=True)
        self.log("val_loss", loss, sync_dist=True)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("val_runtime(hrs)", elapsed_time, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        target_coord = batch["target_coord"].to(torch.float32)
        target_normal = batch["target_normal"].to(torch.float32)
        target_color = batch["target_color"].to(torch.float32)
        target_label = batch["target_label"].to(torch.long)
        anchor_coord = batch["anchor_coord"].to(torch.float32)
        anchor_normal = batch["anchor_normal"].to(torch.float32)
        anchor_color = batch["anchor_color"].to(torch.float32)
        anchor_label = batch["anchor_label"].to(torch.long)
        pose9d = batch["target_pose"].to(torch.float32)

        # Compute conditional features
        cond_feat = self.pose_transformer.encode_cond(
            target_coord, target_normal, target_color, target_label, anchor_coord, anchor_normal, anchor_color, anchor_label
        )
        if self.diffusion_process == "ddpm":
            # initialize action from Guassian noise
            pose9d_pred = torch.randn((pose9d.shape[0], pose9d.shape[1]), device=self.device)
            for k in self.noise_scheduler.timesteps:
                timesteps = torch.tensor([k], device=self.device).to(torch.long).repeat(pose9d.shape[0])
                # predict noise residual

                noise_pred = self.pose_transformer(pose9d_pred, timesteps, cond_feat)
                # inverse diffusion step (remove noise)
                pose9d_pred = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=pose9d_pred
                ).prev_sample
            # compute loss
            trans_loss = F.mse_loss(pose9d_pred[:, :3], pose9d[:, :3])
            rx_loss = F.mse_loss(pose9d_pred[:, 3:6], pose9d[:, 3:6])
            ry_loss = F.mse_loss(pose9d_pred[:, 6:9], pose9d[:, 6:9])
            loss = F.mse_loss(pose9d_pred, pose9d)
            # log
            self.log("te_trans_loss", trans_loss, sync_dist=True)
            self.log("te_rx_loss", rx_loss, sync_dist=True)
            self.log("te_ry_loss", ry_loss, sync_dist=True)
            self.log("test_loss", loss, sync_dist=True)
            elapsed_time = (time.time() - self.start_time) / 3600
            self.log("test_runtime(hrs)", elapsed_time, sync_dist=True)
            return loss
        else:
            raise ValueError(f"Diffusion process {self.diffusion_process} not supported.")

    def forward(self, batch):
        target_coord = batch["target_coord"].to(torch.float32)
        target_normal = batch["target_normal"].to(torch.float32)
        target_color = batch["target_color"].to(torch.float32)
        target_label = batch["target_label"].to(torch.long)
        anchor_coord = batch["anchor_coord"].to(torch.float32)
        anchor_normal = batch["anchor_normal"].to(torch.float32)
        anchor_color = batch["anchor_color"].to(torch.float32)
        anchor_label = batch["anchor_label"].to(torch.long)

        B = target_coord.shape[0]
        # Compute conditional features
        cond_feat = self.pose_transformer.encode_cond(
            target_coord, target_normal, target_color, target_label, anchor_coord, anchor_normal, anchor_color, anchor_label
        )
        if self.diffusion_process == "ddpm":
            # initialize action from Guassian noise
            pose9d_pred = torch.randn((B, 9), device=self.device)
            for k in self.noise_scheduler.timesteps:
                timesteps = torch.tensor([k], device=self.device).to(torch.long).repeat(B)
                # predict noise residual

                noise_pred = self.pose_transformer(pose9d_pred, timesteps, cond_feat)
                # inverse diffusion step (remove noise)
                pose9d_pred = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=pose9d_pred
                ).prev_sample
            return pose9d_pred
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
        self.pose_transformer = pose_transformer
        self.lightning_pose_transformer = LitPoseDiffusion(pose_transformer, cfg).to(torch.float32)

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
        os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
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
        trainer.fit(
            self.lightning_pose_transformer, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader
        )

    def test(self, test_data_loader, save_path: str):
        strategy = "ddp_find_unused_parameters_true" if os.uname().sysname != "Darwin" else "auto"
        os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
        trainer = L.Trainer(
            logger=WandbLogger(
                name=self.experiment_name(), project=self.logger_project, save_dir=os.path.join(save_path, "logs")
            ),
            strategy=strategy,
        )
        checkpoint_path = f"{save_path}/checkpoints/Dmorp_model-epoch=5115-val_loss=0.08.ckpt"
        trainer.test(
            self.lightning_pose_transformer.__class__.load_from_checkpoint(
                checkpoint_path, pose_transformer=self.pose_transformer, cfg=self.cfg
            ),
            dataloaders=test_data_loader,
        )

    def predict(
        self,
        target_pcd_arr: np.ndarray,
        anchor_pcd_arr: np.ndarray,
        target_label: np.ndarray,
        anchor_label: np.ndarray,
        target_pose=None,
    ) -> Any:
        self.lightning_pose_transformer.eval()
        # Assemble batch
        assert target_pcd_arr.shape[0] == anchor_pcd_arr.shape[0]
        assert target_pcd_arr.shape[1] == anchor_pcd_arr.shape[1]
        batch = {
            "target_coord": target_pcd_arr[None, :, :3],
            "target_normal": target_pcd_arr[None, :, 3:6],
            "target_color": target_pcd_arr[None, :, 6:],
            "target_label": target_label[None, :],
            "anchor_coord": anchor_pcd_arr[None, :, :3],
            "anchor_normal": anchor_pcd_arr[None, :, 3:6],
            "anchor_color": anchor_pcd_arr[None, :, 6:],
            "anchor_label": anchor_label[None, :],
        }
        # Put to torch
        for key in batch.keys():
            batch[key] = torch.from_numpy(batch[key]).to(torch.float32)
        pred_pose9d = self.lightning_pose_transformer(batch)
        pred_pose9d = pred_pose9d.detach().cpu().numpy()[0]

        print(f"Pred pose: {pred_pose9d}")
        if target_pose is not None:
            print(f"Gt pose: {target_pose}")
            trans_loss = np.mean(np.square(pred_pose9d[:3] - target_pose[:3]))
            rx_loss = np.mean(np.square(pred_pose9d[3:6] - target_pose[3:6]))
            ry_loss = np.mean(np.square(pred_pose9d[6:9] - target_pose[6:9]))
            print(f"trans_loss: {trans_loss}, rx_loss: {rx_loss}, ry_loss: {ry_loss}")
        # Convert pose9d to matrix
        pred_pose9d = utils.pose9d_to_mat(pred_pose9d)
        pred_pose_mat = np.eye(4, dtype=np.float32)
        pred_pose_mat[:3, 0] = pred_pose9d[3:6]
        pred_pose_mat[:3, 1] = pred_pose9d[6:9]
        pred_pose_mat[:3, 2] = np.cross(pred_pose9d[3:6], pred_pose9d[6:9])
        pred_pose_mat[:3, 3] = pred_pose9d[:3]
        return pred_pose_mat

    def load(self, checkpoint_path: str) -> None:
        self.lightning_pose_transformer.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    def save(self, save_path: str) -> None:
        torch.save(self.lightning_pose_transformer.state_dict(), save_path)

    def experiment_name(self):
        root_path = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
        dataset_name = self.cfg.MODEL.DATASET_CONFIG
        noise_net_name = self.cfg.MODEL.NOISE_NET.NAME
        init_args = self.cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name]
        rsdr = self.cfg.DATALOADER.AUGMENTATION.RANDOM_SEGMENT_DROP_RATE
        volume_augmentation_file = self.cfg.DATALOADER.AUGMENTATION.VOLUME_AUGMENTATION_FILE
        volume_augmentations_path = (
            os.path.join(root_path, "config", volume_augmentation_file)
            if volume_augmentation_file is not None
            else None
        )
        volume_augmentations = Box(yaml.load(open(volume_augmentations_path, "r"), Loader=yaml.FullLoader))
        sdp = volume_augmentations.segment_drop.prob
        pod = init_args["pcd_output_dim"]
        na = init_args["num_attention_heads"]
        ehd = init_args["encoder_hidden_dim"]
        fpd = init_args["fusion_projection_dim"]
        di = f"{self.cfg.MODEL.NUM_DIFFUSION_ITERS}"
        goal_type = f"{self.cfg.ENV.GOAL_TYPE}"
        lr = f"{self.cfg.TRAIN.LR}"
        pp_str = ""
        for points in init_args.points_pyramid:
            pp_str += str(points) + "-"
        usl = f"{init_args.use_semantic_label}"
        return f"Dmorp_{goal_type}_sdp{sdp}_rsdr{rsdr}_lr{lr}_pod{pod}_na{na}_ehd{ehd}_fpd{fpd}_pp{pp_str}_di{di}_usl{usl}_dc{dataset_name}"
