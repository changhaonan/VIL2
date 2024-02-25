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
from vil2.model.network.geometric import batch2offset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vil2.model.network.pcd_noise_net import PcdNoiseNet
from vil2.model.network.geometric import batch2offset, offset2batch, to_dense_batch, to_flat_batch
from box import Box
import yaml
import vil2.utils.misc_utils as utils
from torch.optim.lr_scheduler import LambdaLR


class LPcdDiffusion(L.LightningModule):
    """Lignthing module for PCD diffusion model"""

    def __init__(self, pcd_noise_net: PcdNoiseNet, cfg: Box, **kwargs: Any) -> None:
        super().__init__()
        self.pcd_noise_net = pcd_noise_net
        self.lr = cfg.TRAIN.LR
        self.start_time = time.time()
        self.diffusion_process = cfg.MODEL.DIFFUSION_PROCESS
        self.num_diffusion_iters = cfg.MODEL.NUM_DIFFUSION_ITERS
        self.warm_up_step = cfg.TRAIN.WARM_UP_STEP
        self.rot_axis = cfg.DATALOADER.AUGMENTATION.ROT_AXIS
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
        # Logging
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        # Init
        self.pcd_noise_net.initialize_weights()

    def criterion(self, batch):
        # Prepare data
        target_coord = batch["target_coord"]
        target_feat = batch["target_feat"]
        target_batch_idx = batch["target_batch_index"]
        target_offset = batch2offset(target_batch_idx)
        target_points = [target_coord, target_feat, target_offset]
        anchor_coord = batch["anchor_coord"]
        anchor_feat = batch["anchor_feat"]
        anchor_batch_idx = batch["anchor_batch_index"]
        anchor_offset = batch2offset(anchor_batch_idx)
        anchor_points = [anchor_coord, anchor_feat, anchor_offset]
        target_pose = batch["target_pose"]
        # Do one-time encoding
        enc_anchor_points = self.pcd_noise_net.encode_anchor(anchor_points)
        enc_target_points = self.pcd_noise_net.encode_target(target_points)
        enc_target_coord, enc_target_feat, enc_target_offset = enc_target_points
        # Reposition to get goal coordinate
        target_batch_index = offset2batch(enc_target_offset)
        target_batch_coord, target_coord_mask = to_dense_batch(enc_target_coord, target_batch_index)
        target_batch_coord = self.reposition(target_batch_coord, target_pose, self.rot_axis)
        flat_target_coord, _ = to_flat_batch(target_batch_coord, target_coord_mask)

        if self.diffusion_process == "ddpm":
            # sample noise to add to actions
            noise = torch.randn(target_batch_coord.shape, device=self.device)
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                low=0,
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(target_batch_coord.shape[0], 1),
                device=self.device,
            ).long()
            # add noisy actions
            noisy_target_coord = self.noise_scheduler.add_noise(target_batch_coord, noise, timesteps)
            noisy_target_coord, _ = to_flat_batch(noisy_target_coord, target_coord_mask)
            # predict noise residual
            noise_pred = self.pcd_noise_net(noisy_target_coord, enc_target_points, enc_anchor_points, timesteps)

            # Compute loss with mask
            loss = F.mse_loss(noise_pred[target_coord_mask], noise[target_coord_mask])
        else:
            raise ValueError(f"Diffusion process {self.diffusion_process} not supported.")
        return loss

    def training_step(self, batch, batch_idx):
        """Training step; DDPM loss"""
        loss = self.criterion(batch)
        # log
        self.log("train_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.criterion(batch)
        # log
        self.log("val_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def forward(self, batch):
        # Prepare data
        target_coord = batch["target_coord"]
        target_feat = batch["target_feat"]
        target_batch_idx = batch["target_batch_index"]
        target_offset = batch2offset(target_batch_idx)
        target_points = [target_coord, target_feat, target_offset]
        anchor_coord = batch["anchor_coord"]
        anchor_feat = batch["anchor_feat"]
        anchor_batch_idx = batch["anchor_batch_index"]
        anchor_offset = batch2offset(anchor_batch_idx)
        anchor_points = [anchor_coord, anchor_feat, anchor_offset]

        # Do one-time encoding
        enc_anchor_points = self.pcd_noise_net.encode_anchor(anchor_points)
        enc_target_points = self.pcd_noise_net.encode_target(target_points)

        # Batchify
        enc_target_coord, enc_target_feat, enc_target_offset = enc_target_points
        target_batch_index = offset2batch(enc_target_offset)
        target_batch_coord, target_coord_mask = to_dense_batch(enc_target_coord, target_batch_index)
        B = target_batch_coord.shape[0]
        if self.diffusion_process == "ddpm":
            # initialize action from Guassian noise
            pred_target_coord = torch.randn(target_batch_coord.shape, device=self.device)
            pred_target_coord, _ = to_flat_batch(pred_target_coord, target_coord_mask)
            for k in self.noise_scheduler.timesteps:
                timesteps = torch.tensor([k], device=self.device).to(torch.long).repeat(B)
                # predict noise residual
                noise_pred = self.pcd_noise_net(pred_target_coord, enc_target_points, enc_anchor_points, timesteps)

                # Batchify
                # inverse diffusion step (remove noise)
                pred_target_coord = self.noise_scheduler.step(model_output=noise_pred[target_coord_mask], timestep=k, sample=pred_target_coord).prev_sample
        else:
            raise ValueError(f"Diffusion process {self.diffusion_process} not supported.")

        # Batchify & Numpy
        pred_target_coord, _ = to_dense_batch(pred_target_coord, target_batch_index)
        pred_target_coord = pred_target_coord.detach().cpu().numpy()
        prev_target_coord = target_batch_coord.detach().cpu().numpy()
        return pred_target_coord, prev_target_coord

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        def lr_foo(epoch):
            if epoch < self.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (self.warm_up_step - epoch)
            else:
                lr_scale = 0.95**epoch

            return lr_scale

        scheduler = LambdaLR(optimizer, lr_lambda=lr_foo)
        return [optimizer], [scheduler]

    def reposition(self, coord: torch.Tensor, pose9d: torch.Tensor, rot_axis: str = "xy"):
        """Reposition the points using the pose9d"""
        t = pose9d[:, 0:3]
        r1 = pose9d[:, 3:6]
        r2 = pose9d[:, 6:9]
        # Normalize & Gram-Schmidt
        r1 = r1 / (torch.norm(r1, dim=1, keepdim=True) + 1e-8)
        r1_r2_dot = torch.sum(r1 * r2, dim=1, keepdim=True)
        r2_orth = r2 - r1_r2_dot * r1
        r2 = r2_orth / (torch.norm(r2_orth, dim=1, keepdim=True) + 1e-8)
        r3 = torch.cross(r1, r2)
        # Rotate
        if rot_axis == "xy":
            R = torch.stack([r1, r2, r3], dim=2)
        elif rot_axis == "yz":
            R = torch.stack([r3, r1, r2], dim=2)
        elif rot_axis == "zx":
            R = torch.stack([r2, r3, r1], dim=2)
        R = R.permute(0, 2, 1)
        # Reposition
        coord = torch.bmm(coord, R)
        coord = coord + t[:, None, :]
        return coord


class PCDDModel:
    """PCD diffusion model"""

    def __init__(self, cfg, pcd_noise_net: PcdNoiseNet) -> None:
        self.cfg = cfg
        # parameters
        self.logger_project = cfg.LOGGER.PROJECT
        # build model
        self.pcd_noise_net = pcd_noise_net
        self.lpcd_noise_net = LPcdDiffusion(pcd_noise_net, cfg).to(torch.float32)

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
            logger=WandbLogger(name=self.experiment_name(), project=self.logger_project, save_dir=os.path.join(save_path, "logs")),
            callbacks=[checkpoint_callback],
            strategy=strategy,
            log_every_n_steps=5,
            accelerator=accelerator,
        )
        trainer.fit(self.lpcd_noise_net, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

    def test(self, test_data_loader, save_path: str):
        strategy = "ddp_find_unused_parameters_true" if os.uname().sysname != "Darwin" else "auto"
        os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
        trainer = L.Trainer(
            logger=WandbLogger(name=self.experiment_name(), project=self.logger_project, save_dir=os.path.join(save_path, "logs")),
            strategy=strategy,
        )
        checkpoint_path = f"{save_path}/checkpoints"
        # Select the best checkpoint
        checkpoints = os.listdir(checkpoint_path)
        sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
        checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
        checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
        trainer.test(
            self.lpcd_noise_net.__class__.load_from_checkpoint(checkpoint_file, pcd_noise_net=self.pcd_noise_net, cfg=self.cfg),
            dataloaders=test_data_loader,
        )

    def predict(
        self,
        target_coord: np.ndarray | None = None,
        target_feat: np.ndarray | None = None,
        anchor_coord: np.ndarray | None = None,
        anchor_feat: np.ndarray | None = None,
        batch=None,
        target_pose=None,
        **kwargs,
    ) -> Any:
        self.lpcd_noise_net.eval()
        # Assemble batch
        if batch is None:
            batch_size = kwargs.get("batch_size", 16)
            target_coord_list = []
            target_feat_list = []
            anchor_coord_list = []
            anchor_feat_list = []
            target_batch_idx_list = []
            anchor_batch_idx_list = []
            for i in range(batch_size):
                target_coord_list.append(target_coord)
                target_feat_list.append(target_feat)
                anchor_coord_list.append(anchor_coord)
                anchor_feat_list.append(anchor_feat)
                target_batch_idx_list.append(np.array([i] * target_coord.shape[0], dtype=np.int64))
                anchor_batch_idx_list.append(np.array([i] * anchor_coord.shape[0], dtype=np.int64))
            batch = {
                "target_coord": np.concatenate(target_coord_list, axis=0),
                "target_feat": np.concatenate(target_feat_list, axis=0),
                "target_batch_index": np.concatenate(target_batch_idx_list, axis=0),
                "anchor_coord": np.concatenate(anchor_coord_list, axis=0),
                "anchor_feat": np.concatenate(anchor_feat_list, axis=0),
                "anchor_batch_index": np.concatenate(anchor_batch_idx_list, axis=0),
            }
            # Put to torch
            for key in batch.keys():
                batch[key] = torch.from_numpy(batch[key])
        else:
            check_batch_idx = kwargs.get("check_batch_idx", 0)
            batch_size = kwargs.get("batch_size", 8)
            target_coord = batch["target_coord"]
            target_feat = batch["target_feat"]
            target_batch_index = batch["target_batch_index"]

            target_coord_i = target_coord[target_batch_index == check_batch_idx]
            target_feat_i = target_feat[target_batch_index == check_batch_idx]
            anchor_coord = batch["anchor_coord"]
            anchor_feat = batch["anchor_feat"]
            anchor_batch_index = batch["anchor_batch_index"]
            anchor_coord_i = anchor_coord[anchor_batch_index == check_batch_idx]
            anchor_feat_i = anchor_feat[anchor_batch_index == check_batch_idx]
            target_coord_list = []
            target_feat_list = []
            anchor_coord_list = []
            anchor_feat_list = []
            target_batch_idx_list = []
            anchor_batch_idx_list = []
            for i in range(batch_size):
                target_coord_list.append(target_coord_i)
                target_feat_list.append(target_feat_i)
                anchor_coord_list.append(anchor_coord_i)
                anchor_feat_list.append(anchor_feat_i)
                target_batch_idx_list.append(torch.tensor([i] * target_coord_i.shape[0], dtype=torch.int64))
                anchor_batch_idx_list.append(torch.tensor([i] * anchor_coord_i.shape[0], dtype=torch.int64))
            batch = {
                "target_coord": torch.cat(target_coord_list, dim=0),
                "target_feat": torch.cat(target_feat_list, dim=0),
                "target_batch_index": torch.cat(target_batch_idx_list, dim=0),
                "anchor_coord": torch.cat(anchor_coord_list, dim=0),
                "anchor_feat": torch.cat(anchor_feat_list, dim=0),
                "anchor_batch_index": torch.cat(anchor_batch_idx_list, dim=0),
            }
        # Put to device
        for key in batch.keys():
            batch[key] = batch[key].to(self.lpcd_noise_net.device)
        # [Debug]
        # self.lpcd_noise_net.criterion(batch)
        pred_target_coord, prev_target_coord = self.lpcd_noise_net(batch)
        # Full points
        anchor_coord = batch["anchor_coord"]
        anchor_batch_index = batch["anchor_batch_index"]
        anchor_batch_coord, anchor_coord_mask = to_dense_batch(anchor_coord, anchor_batch_index)
        target_coord = batch["target_coord"]
        target_batch_index = batch["target_batch_index"]
        target_batch_coord, target_coord_mask = to_dense_batch(target_coord, target_batch_index)
        return pred_target_coord, prev_target_coord, anchor_batch_coord, target_batch_coord

    def load(self, checkpoint_path: str) -> None:
        print(f"Loading checkpoint from {checkpoint_path}")
        self.lpcd_noise_net.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    def save(self, save_path: str) -> None:
        torch.save(self.lpcd_noise_net.state_dict(), save_path)

    def experiment_name(self):
        noise_net_name = self.cfg.MODEL.NOISE_NET.NAME
        init_args = self.cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name]
        crop_strategy = self.cfg.DATALOADER.AUGMENTATION.CROP_STRATEGY
        return f"PCDD_model_{crop_strategy}"

    @staticmethod
    def kabsch_transform(point_1, point_2, filter_zero=True):
        """Compute the 6D transformation between two point clouds"""
        if filter_zero:
            # Filter zero points
            mask_1 = np.linalg.norm(point_1, axis=1) > 0
            mask_2 = np.linalg.norm(point_2, axis=1) > 0
            mask = mask_1 * mask_2
            point_1 = point_1[mask]
            point_2 = point_2[mask]
        # Center the points
        centroid_1 = np.mean(point_1, axis=0)
        centroid_2 = np.mean(point_2, axis=0)
        centered_1 = point_1 - centroid_1
        centered_2 = point_2 - centroid_2

        # Compute the covariance matrix
        H = np.dot(centered_1.T, centered_2)

        # Compute the SVD
        U, S, Vt = np.linalg.svd(H)

        # Compute the rotation
        R = np.dot(Vt.T, U.T)

        # Ensure a right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Compute the translation
        t = centroid_2 - np.dot(R, centroid_1)

        # Combine R and t to form the transformation matrix
        transform = np.identity(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        # Compute transform residual
        residual = np.mean(np.linalg.norm(np.dot(point_1, R.T) + t - point_2, axis=1))
        return transform, residual
