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
import time
from vil2.model.network.pose_transformer import PoseTransformer


class LitPoseTransformer(L.LightningModule):
    """Lightning module for Pose Transformer"""

    def __init__(self, pose_transformer: PoseTransformer, cfg=None) -> None:
        super().__init__()
        self.pose_transformer = pose_transformer
        self.lr = cfg.TRAIN.LR
        self.start_time = time.time()

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
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True)
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
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True)
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
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True)
        return loss

    def forward(self, batch) -> Any:
        target_coord = batch["target_coord"].to(torch.float32)
        target_normal = batch["target_normal"].to(torch.float32)
        target_color = batch["target_color"].to(torch.float32)
        target_label = batch["target_label"].to(torch.long)
        fixed_coord = batch["fixed_coord"].to(torch.float32)
        fixed_normal = batch["fixed_normal"].to(torch.float32)
        fixed_color = batch["fixed_color"].to(torch.float32)
        fixed_label = batch["fixed_label"].to(torch.long)
        pose9d_pred = self.pose_transformer(
            target_coord, target_normal, target_color, target_label, fixed_coord, fixed_normal, fixed_color, fixed_label
        )
        return pose9d_pred

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class TmorpModel:
    """Transformer Model for multi-object relative Pose Generation"""

    def __init__(self, cfg, pose_transformer: PoseTransformer) -> None:
        self.cfg = cfg
        # parameters
        # build model
        self.pose_transformer = pose_transformer
        self.lightning_pose_transformer = LitPoseTransformer(pose_transformer, cfg).to(torch.float32)
        # parameters
        self.logger_project = cfg.LOGGER.PROJECT

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
            logger=WandbLogger(name=self.experiment_name(), save_dir=os.path.join(save_path, "logs")),
            callbacks=[checkpoint_callback],
            strategy=strategy,
            log_every_n_steps=5,
            accelerator=accelerator,
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

    def predict(self, batch) -> Any:
        return self.lightning_pose_transformer(batch)

    def load(self, checkpoint_path: str) -> None:
        self.lightning_pose_transformer.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    def save(self, save_path: str) -> None:
        torch.save(self.lightning_pose_transformer.state_dict(), save_path)

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
        usl = f"{init_args.use_semantic_label}"
        return f"Tmorp_model_pod{pod}_na{na}_ehd{ehd}_fpd{fpd}_pp{pp_str}_usl{usl}"
