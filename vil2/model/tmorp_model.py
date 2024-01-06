"""Transformer Model for multi-object relative Pose Generation"""
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

from vil2.model.network.pose_transformer import PoseTransformer


class LitPoseTransformer(L.LightningModule):
    """Lightning module for Pose Transformer"""

    def __init__(self, pose_transformer: PoseTransformer, cfg=None) -> None:
        super().__init__()
        self.pose_transformer = pose_transformer
        self.lr = cfg.TRAIN.LR

    def training_step(self, batch, batch_idx):
        target_coord = batch["target_coord"].to(torch.float32)
        target_normal = batch["target_normal"].to(torch.float32)
        target_color = batch["target_color"].to(torch.float32)
        target_pose = batch["target_pose"].to(torch.float32)
        fixed_coord = batch["fixed_coord"].to(torch.float32)
        fixed_normal = batch["fixed_normal"].to(torch.float32)
        fixed_color = batch["fixed_color"].to(torch.float32)
        pose9d = batch["target_pose"].to(torch.float32)
        # forward
        pose9d_pred = self.pose_transformer(
            fixed_coord, fixed_normal, fixed_color, target_coord, target_normal, target_color
        )
        # compute loss
        loss = F.mse_loss(pose9d_pred, pose9d)
        # log
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        target_coord = batch["target_coord"].to(torch.float32)
        target_normal = batch["target_normal"].to(torch.float32)
        target_color = batch["target_color"].to(torch.float32)
        target_pose = batch["target_pose"].to(torch.float32)
        fixed_coord = batch["fixed_coord"].to(torch.float32)
        fixed_normal = batch["fixed_normal"].to(torch.float32)
        fixed_color = batch["fixed_color"].to(torch.float32)
        pose9d = batch["target_pose"].to(torch.float32)
        # forward
        pose9d_pred = self.pose_transformer(
            fixed_coord, fixed_normal, fixed_color, target_coord, target_normal, target_color
        )
        # compute loss
        loss = F.mse_loss(pose9d_pred, pose9d)
        # log
        self.log("test_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target_coord = batch["target_coord"].to(torch.float32)
        target_normal = batch["target_normal"].to(torch.float32)
        target_color = batch["target_color"].to(torch.float32)
        target_pose = batch["target_pose"].to(torch.float32)
        fixed_coord = batch["fixed_coord"].to(torch.float32)
        fixed_normal = batch["fixed_normal"].to(torch.float32)
        fixed_color = batch["fixed_color"].to(torch.float32)
        pose9d = batch["target_pose"].to(torch.float32)
        # forward
        pose9d_pred = self.pose_transformer(
            fixed_coord, fixed_normal, fixed_color, target_coord, target_normal, target_color
        )
        # compute loss
        loss = F.mse_loss(pose9d_pred, pose9d)
        # log
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class TmorpModel:
    """Transformer Model for multi-object relative Pose Generation"""

    def __init__(self, cfg, pose_transformer: PoseTransformer) -> None:
        self.cfg = cfg
        # parameters
        # build model
        self.pose_transformer = LitPoseTransformer(pose_transformer, cfg)

    def train(self, num_epochs: int, train_data_loader, val_data_loader, save_path: str, num_gpus: int = 1):
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(save_path, "checkpoints"),
            filename="Tmorp_model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )
        # Trainer
        trainer = L.Trainer(
            max_epochs=num_epochs,
            logger=WandbLogger(name="Tmorp_model", save_dir=os.path.join(save_path, "logs")),
            callbacks=[checkpoint_callback],
            strategy="ddp_find_unused_parameters_true",
            log_every_n_steps=5,
        )
        trainer.fit(self.pose_transformer, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

    def test(self, test_data_loader, save_path: str, num_gpus: int = 1):
        # Trainer
        trainer = L.Trainer(
            logger=WandbLogger(name="Tmorp_model", save_dir=os.path.join(save_path, "logs")),
        )
        trainer.test(self.pose_transformer, test_dataloaders=test_data_loader)
