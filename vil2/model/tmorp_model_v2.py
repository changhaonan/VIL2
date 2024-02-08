"""Transformer Model for multi-object relative Pose Generation"""

from __future__ import annotations
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import time
from vil2.model.network.geometric import batch2offset
from vil2.model.network.pose_transformer_v2 import PoseTransformerV2
import vil2.utils.misc_utils as utils


class LitPoseTransformerV2(L.LightningModule):
    """Lightning module for Pose Transformer"""

    def __init__(self, pose_transformer: PoseTransformerV2, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.pose_transformer = pose_transformer
        self.lr = cfg.TRAIN.LR
        self.warm_up_step = cfg.TRAIN.WARM_UP_STEP
        self.start_time = time.time()
        # Logging
        self.batch_size = cfg.DATALOADER.BATCH_SIZE

    def training_step(self, batch, batch_idx):
        pose9d = batch["target_pose"].to(torch.float32)
        pose9d_pred = self.forward(batch)
        # compute loss
        trans_loss = F.mse_loss(pose9d_pred[:, :3], pose9d[:, :3])
        rx_loss = F.mse_loss(pose9d_pred[:, 3:6], pose9d[:, 3:6])
        ry_loss = F.mse_loss(pose9d_pred[:, 6:9], pose9d[:, 6:9])
        loss = trans_loss + rx_loss + ry_loss
        # log
        self.log("tr_trans_loss", trans_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("tr_rx_loss", rx_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("tr_ry_loss", ry_loss, sync_dist=True, batch_size=self.batch_size)
        # log
        self.log("train_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        pose9d = batch["target_pose"].to(torch.float32)
        pose9d_pred = self.forward(batch)
        # compute loss
        trans_loss = F.mse_loss(pose9d_pred[:, :3], pose9d[:, :3])
        rx_loss = F.mse_loss(pose9d_pred[:, 3:6], pose9d[:, 3:6])
        ry_loss = F.mse_loss(pose9d_pred[:, 6:9], pose9d[:, 6:9])
        loss = trans_loss + rx_loss + ry_loss
        # log
        self.log("te_trans_loss", trans_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("te_rx_loss", rx_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("te_ry_loss", ry_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("test_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        pose9d = batch["target_pose"].to(torch.float32)
        pose9d_pred = self.forward(batch)
        # compute loss
        trans_loss = F.mse_loss(pose9d_pred[:, :3], pose9d[:, :3])
        rx_loss = F.mse_loss(pose9d_pred[:, 3:6], pose9d[:, 3:6])
        ry_loss = F.mse_loss(pose9d_pred[:, 6:9], pose9d[:, 6:9])
        loss = trans_loss + rx_loss + ry_loss
        # log
        self.log("v_trans_loss", trans_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("v_rx_loss", rx_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("v_ry_loss", ry_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("val_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def forward(self, batch) -> Any:
        # Assemble input
        target_coord = batch["target_coord"]
        target_feat = batch["target_feat"]
        target_batch_idx = batch["target_batch_index"]
        target_offset = batch2offset(target_batch_idx)
        target_points = [target_coord, target_feat, target_offset]
        fixed_coord = batch["fixed_coord"]
        fixed_feat = batch["fixed_feat"]
        fixed_batch_idx = batch["fixed_batch_index"]
        fixed_offset = batch2offset(fixed_batch_idx)
        fixed_points = [fixed_coord, fixed_feat, fixed_offset]

        # Compute conditional features
        enc_target_points, all_enc_fixed_points, cluster_indexes = self.pose_transformer.encode_cond(
            target_points, fixed_points
        )

        # forward
        pose9d_pred = self.pose_transformer(enc_target_points, all_enc_fixed_points)
        return pose9d_pred

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


class TmorpModelV2:
    """Transformer Model for multi-object relative Pose Generation"""

    def __init__(self, cfg, pose_transformer: PoseTransformerV2) -> None:
        self.cfg = cfg
        # parameters
        # build model
        self.pose_transformer = pose_transformer
        self.lightning_pose_transformer = LitPoseTransformerV2(pose_transformer, cfg).to(torch.float32)
        # parameters
        self.rot_axis = cfg.DATALOADER.AUGMENTATION.ROT_AXIS
        self.gradient_clip_val = cfg.TRAIN.GRADIENT_CLIP_VAL
        self.logger_project = cfg.LOGGER.PROJECT

    def train(self, num_epochs: int, train_data_loader, val_data_loader, save_path: str):
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(save_path, "checkpoints"),
            filename="Tmorp_modelV2-{epoch:02d}-{val_loss:.2f}",
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
            logger=WandbLogger(name=self.experiment_name(), save_dir=os.path.join(save_path, "logs")),
            callbacks=[checkpoint_callback],
            strategy=strategy,
            log_every_n_steps=5,
            accelerator=accelerator,
            gradient_clip_val=self.gradient_clip_val,
        )
        trainer.fit(
            self.lightning_pose_transformer, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader
        )

    def test(self, test_data_loader, save_path: str):
        # Trainer
        strategy = "ddp_find_unused_parameters_true" if os.uname().sysname != "Darwin" else "auto"
        os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
        trainer = L.Trainer(
            logger=WandbLogger(
                name=self.experiment_name(), project=self.logger_project, save_dir=os.path.join(save_path, "logs")
            ),
            strategy=strategy,
        )
        checkpoint_path = f"{save_path}/checkpoints"
        # Select the best checkpoint
        checkpoints = os.listdir(checkpoint_path)
        sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
        checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
        trainer.test(
            self.lightning_pose_transformer.__class__.load_from_checkpoint(
                checkpoint_file, pose_transformer=self.pose_transformer, cfg=self.cfg
            ),
            dataloaders=test_data_loader,
        )

    def predict(
        self,
        target_coord: np.ndarray,
        target_feat: np.ndarray,
        fixed_coord: np.ndarray,
        fixed_feat: np.ndarray,
        target_pose=None,
    ) -> Any:
        self.lightning_pose_transformer.eval()
        # Assemble batch
        target_batch_idx = np.array([0] * target_coord.shape[0], dtype=np.int64)
        fixed_batch_idx = np.array([0] * fixed_coord.shape[0], dtype=np.int64)
        batch = {
            "target_coord": target_coord,
            "target_feat": target_feat,
            "target_batch_index": target_batch_idx,
            "fixed_coord": fixed_coord,
            "fixed_feat": fixed_feat,
            "fixed_batch_index": fixed_batch_idx,
        }
        # Put to torch
        for key in batch.keys():
            batch[key] = torch.from_numpy(batch[key])
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
        pred_pose_mat = utils.pose9d_to_mat(pred_pose9d, rot_axis=self.rot_axis)
        return pred_pose_mat

    def load(self, checkpoint_path: str) -> None:
        print(f"Loading checkpoint from {checkpoint_path}")
        self.lightning_pose_transformer.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    def save(self, save_path: str) -> None:
        print(f"Saving checkpoint to {save_path}")
        torch.save(self.lightning_pose_transformer.state_dict(), save_path)

    def experiment_name(self):
        noise_net_name = self.cfg.MODEL.NOISE_NET.NAME
        init_args = self.cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name]
        return f"Tmorp_model"

    def sample_bbox(self, coord, feat, crop_size: float, fake_crop=False):
        """Sample a bbox in the point cloud"""
        if fake_crop:
            return coord, feat
        max_try = 10
        for i in range(max_try):
            # Compute the occupied bbox
            min_coord = np.min(coord, axis=0)
            max_coord = np.max(coord, axis=0)
            bbox_size = max_coord - min_coord
            # Sample a bbox
            bbox_center = min_coord + bbox_size / 2
            bbox_center = bbox_center * (1 + np.random.uniform(-0.2, 0.2, 3))
            bbox_min = bbox_center - crop_size
            bbox_max = bbox_center + crop_size

            # Crop the point cloud
            inds = self.crop(coord, *bbox_min, *bbox_max)
            if inds.sum() > 0:
                break
            if i == max_try - 1:
                inds = np.arange(len(coord))

        coord = coord[inds]
        feat = feat[inds]

        coord_center = bbox_center
        coord -= coord_center
        feat[:, :3] -= coord_center

        return coord, feat

    def crop(self, points, x_min, y_min, z_min, x_max, y_max, z_max):
        if x_max <= x_min or y_max <= y_min or z_max <= z_min:
            raise ValueError(
                "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
                " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
                " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z_min,
                    z_max=z_max,
                )
            )
        inds = np.all(
            [
                (points[:, 0] >= x_min),
                (points[:, 0] < x_max),
                (points[:, 1] >= y_min),
                (points[:, 1] < y_max),
                (points[:, 2] >= z_min),
                (points[:, 2] < z_max),
            ],
            axis=0,
        )
        return inds
