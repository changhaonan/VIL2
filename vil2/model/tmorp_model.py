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
        target_label = batch["target_label"].to(torch.long)
        fixed_coord = batch["fixed_coord"].to(torch.float32)
        fixed_normal = batch["fixed_normal"].to(torch.float32)
        fixed_color = batch["fixed_color"].to(torch.float32)
        fixed_label = batch["fixed_label"].to(torch.long)
        pose9d = batch["target_pose"].to(torch.float32)
        # forward
        pose9d_pred = self.pose_transformer(
            target_coord, target_normal, target_color, target_label, fixed_coord, fixed_normal, fixed_color, fixed_label
        )
        # compute loss
        trans_loss = F.mse_loss(pose9d_pred[:, :3], pose9d[:, :3])
        rx_loss = F.mse_loss(pose9d_pred[:, 3:6], pose9d[:, 3:6])
        ry_loss = F.mse_loss(pose9d_pred[:, 6:9], pose9d[:, 6:9])
        loss = trans_loss + rx_loss + ry_loss
        # log
        self.log("tr_trans_loss", trans_loss, sync_dist=True)
        self.log("tr_rx_loss", rx_loss, sync_dist=True)
        self.log("tr_ry_loss", ry_loss, sync_dist=True)
        # log
        self.log("train_loss", loss, sync_dist=True)
        return loss

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
        # forward
        pose9d_pred = self.pose_transformer(
            target_coord, target_normal, target_color, target_label, fixed_coord, fixed_normal, fixed_color, fixed_label
        )
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
        # forward
        pose9d_pred = self.pose_transformer(
            target_coord, target_normal, target_color, target_label, fixed_coord, fixed_normal, fixed_color, fixed_label
        )
        # compute loss
        trans_loss = F.mse_loss(pose9d_pred[:, :3], pose9d[:, :3])
        rx_loss = F.mse_loss(pose9d_pred[:, 3:6], pose9d[:, 3:6])
        ry_loss = F.mse_loss(pose9d_pred[:, 6:9], pose9d[:, 6:9])
        loss = trans_loss + rx_loss + ry_loss
        # log
        self.log("v_trans_loss", trans_loss, sync_dist=True)
        self.log("v_rx_loss", rx_loss, sync_dist=True)
        self.log("v_ry_loss", ry_loss, sync_dist=True)
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
        self.pose_transformer = LitPoseTransformer(pose_transformer, cfg).to(torch.float32)
        net_name = cfg.MODEL.NOISE_NET.NAME
        net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
        self.model_name = "Tmorp_model" 
        self.model_name += f"_pod{net_init_args.pcd_output_dim}" 
        self.model_name += f"_na{net_init_args.num_attention_heads}"
        self.model_name += f"_ehd{net_init_args.encoder_hidden_dim}"
        self.model_name += f"_fpd{net_init_args.fusion_projection_dim}"
        pp_str = ""
        for points in net_init_args.points_pyramid:
            pp_str += str(points) + "_"
        self.model_name += f"_pp{pp_str}"
        self.model_name += f"usl{net_init_args.use_semantic_label}"

    def train(self, num_epochs: int, train_data_loader, val_data_loader, save_path: str):
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(save_path, "checkpoints"),
            filename="Tmorp_model-{epoch:02d}-{val_loss:.2f}",
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
            logger=WandbLogger(name=self.model_name, save_dir=os.path.join(save_path, "logs")),
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
        os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
        trainer = L.Trainer(
            logger=WandbLogger(name=self.model_name, save_dir=os.path.join(save_path, "logs")),
            strategy=strategy,
        )
        trainer.test(self.pose_transformer, test_dataloaders=test_data_loader, accelerator=accelerator)
