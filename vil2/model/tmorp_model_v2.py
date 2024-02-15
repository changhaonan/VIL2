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
import open3d as o3d
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
from vil2.utils.pcd_utils import normalize_pcd, check_collision

# DataUtils
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.data.pcd_datalodaer import PcdPairCollator


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
        is_valid_crop = batch["is_valid_crop"].to(torch.long)
        pred_pose9d, pred_status = self.forward(batch)
        # compute loss
        status_loss = F.cross_entropy(pred_status, is_valid_crop)
        # mask out invalid crops
        pose9d_valid = pose9d[is_valid_crop == 1]
        pose9d_pred_valid = pred_pose9d[is_valid_crop == 1]
        trans_loss = F.mse_loss(pose9d_pred_valid[:, :3], pose9d_valid[:, :3])
        rx_loss = F.mse_loss(pose9d_pred_valid[:, 3:6], pose9d_valid[:, 3:6])
        ry_loss = F.mse_loss(pose9d_pred_valid[:, 6:9], pose9d_valid[:, 6:9])
        # sum
        loss = trans_loss + rx_loss + ry_loss + 0.1 * status_loss
        # log
        self.log("tr_status_loss", status_loss, sync_dist=True, batch_size=self.batch_size)
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
        is_valid_crop = batch["is_valid_crop"].to(torch.long)
        pred_pose9d, pred_status = self.forward(batch)
        # compute loss
        status_loss = F.cross_entropy(pred_status, is_valid_crop)
        # mask out invalid crops
        pose9d_valid = pose9d[is_valid_crop == 1]
        pose9d_pred_valid = pred_pose9d[is_valid_crop == 1]
        trans_loss = F.mse_loss(pose9d_pred_valid[:, :3], pose9d_valid[:, :3])
        rx_loss = F.mse_loss(pose9d_pred_valid[:, 3:6], pose9d_valid[:, 3:6])
        ry_loss = F.mse_loss(pose9d_pred_valid[:, 6:9], pose9d_valid[:, 6:9])
        # sum
        loss = trans_loss + rx_loss + ry_loss + 0.1 * status_loss
        # log
        self.log("te_status_loss", status_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("te_trans_loss", trans_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("te_rx_loss", rx_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("te_ry_loss", ry_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("test_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        pose9d = batch["target_pose"].to(torch.float32)
        is_valid_crop = batch["is_valid_crop"].to(torch.long)
        pred_pose9d, pred_status = self.forward(batch)
        # compute loss
        status_loss = F.cross_entropy(pred_status, is_valid_crop)
        # mask out invalid crops
        pose9d_valid = pose9d[is_valid_crop == 1]
        pose9d_pred_valid = pred_pose9d[is_valid_crop == 1]
        trans_loss = F.mse_loss(pose9d_pred_valid[:, :3], pose9d_valid[:, :3])
        rx_loss = F.mse_loss(pose9d_pred_valid[:, 3:6], pose9d_valid[:, 3:6])
        ry_loss = F.mse_loss(pose9d_pred_valid[:, 6:9], pose9d_valid[:, 6:9])
        # sum
        loss = trans_loss + rx_loss + ry_loss + 0.1 * status_loss
        # log
        self.log("v_status_loss", status_loss, sync_dist=True, batch_size=self.batch_size)
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
        pred_pose9d, pred_status = self.pose_transformer(enc_target_points, all_enc_fixed_points)
        return pred_pose9d.to(torch.float32), pred_status.to(torch.float32)

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
        target_coord: np.ndarray | None = None,
        target_feat: np.ndarray | None = None,
        fixed_coord: np.ndarray | None = None,
        fixed_feat: np.ndarray | None = None,
        batch=None,
        target_pose=None,
    ) -> Any:
        self.lightning_pose_transformer.eval()
        # Assemble batch
        if batch is None:
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
        # Put to device
        for key in batch.keys():
            batch[key] = batch[key].to(self.lightning_pose_transformer.device)
        pred_pose9d, pred_status = self.lightning_pose_transformer(batch)
        pred_status = torch.softmax(pred_status, dim=1)
        pred_pose9d = pred_pose9d.detach().cpu().numpy()
        pred_status = pred_status.detach().cpu().numpy()
        if target_pose is not None:
            trans_loss = np.mean(np.square(pred_pose9d[:, :3] - target_pose[:, :3]))
            rx_loss = np.mean(np.square(pred_pose9d[:, 3:6] - target_pose[:, 3:6]))
            ry_loss = np.mean(np.square(pred_pose9d[:, 6:9] - target_pose[:, 6:9]))
            print(f"trans_loss: {trans_loss}, rx_loss: {rx_loss}, ry_loss: {ry_loss}")
        # HACK: Add a small offset on z-axis
        pred_pose9d[:, 2] += 0.1
        return pred_pose9d, pred_status

    def load(self, checkpoint_path: str) -> None:
        print(f"Loading checkpoint from {checkpoint_path}")
        self.lightning_pose_transformer.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    def save(self, save_path: str) -> None:
        print(f"Saving checkpoint to {save_path}")
        torch.save(self.lightning_pose_transformer.state_dict(), save_path)

    def experiment_name(self):
        noise_net_name = self.cfg.MODEL.NOISE_NET.NAME
        init_args = self.cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name]
        crop_strategy = self.cfg.DATALOADER.AUGMENTATION.CROP_STRATEGY
        return f"Tmorp_model_{crop_strategy}"

    def batch_random_sample(
        self,
        sample_size: int,
        sample_strategy: str,
        target_coord: np.ndarray,
        target_feat: np.ndarray,
        fixed_coord: np.ndarray,
        fixed_feat: np.ndarray,
        crop_strategy: str,
        **kwargs,
    ) -> Any:
        crop_size = kwargs.get("crop_size", 0.2)
        knn_k = kwargs.get("knn_k", 20)
        num_grid = kwargs.get("num_grid", 5)
        samples = []
        # Randomly sample batch
        for i in range(sample_size):
            # Do random crop sampling
            x_min, y_min, z_min = fixed_coord.min(axis=0)
            x_max, y_max, z_max = fixed_coord.max(axis=0)
            if sample_strategy == "random":
                x_center = np.random.uniform(0.1, 0.9) * (x_max - x_min) + x_min
                y_center = np.random.uniform(0.1, 0.9) * (y_max - y_min) + y_min
                z_center = np.random.uniform(0.1, 0.9) * (z_max - z_min) + z_min
                crop_center = np.array([x_center, y_center, z_center])
            elif sample_strategy == "grid":
                x_grid = np.linspace(0.1, 0.9, num_grid) * (x_max - x_min) + x_min
                y_grid = np.linspace(0.1, 0.9, num_grid) * (y_max - y_min) + y_min
                z_grid = np.linspace(0.1, 0.9, num_grid) * (z_max - z_min) + z_min
                crop_center = np.array(
                    [
                        np.random.choice(x_grid),
                        np.random.choice(y_grid),
                        np.random.choice(z_grid),
                    ]
                )
            crop_indices = PcdPairDataset.crop(
                pcd=fixed_coord,
                crop_center=crop_center,
                crop_strategy=crop_strategy,
                ref_points=target_coord,
                crop_size=crop_size,
                knn_k=knn_k,
            )
            crop_fixed_coord = fixed_coord[crop_indices]
            crop_fixed_feat = fixed_feat[crop_indices]
            crop_fixed_coord[:, :3] -= crop_center
            crop_fixed_feat[:, :3] -= crop_center
            # Convert to batch
            sample = {
                "target_coord": target_coord,
                "target_feat": target_feat,
                "fixed_coord": crop_fixed_coord,
                "fixed_feat": crop_fixed_feat,
                "target_pose": np.zeros(9),
                "is_valid_crop": np.ones(1),
                "crop_center": crop_center,  # Aux info
            }
            samples.append(sample)
        return PcdPairCollator()(samples), samples

    # Utility functions
    def preprocess_input_rpdiff(self, target_coord: np.ndarray, fixed_coord: np.ndarray):
        """Preprocess data for eval on RPDiff"""
        # Build o3d object
        target_pcd_o3d = o3d.geometry.PointCloud()
        target_pcd_o3d.points = o3d.utility.Vector3dVector(target_coord)
        target_pcd_o3d.paint_uniform_color([0, 0, 1])
        fixed_pcd_o3d = o3d.geometry.PointCloud()
        fixed_pcd_o3d.points = o3d.utility.Vector3dVector(fixed_coord)
        fixed_pcd_o3d.paint_uniform_color([1, 0, 0])

        # Estimate the pose of fixed coord using a rotating bbox
        fixed_pcd_bbox = fixed_pcd_o3d.get_minimal_oriented_bounding_box()
        fixed_pcd_bbox.color = [0, 1, 0]
        fixed_R = fixed_pcd_bbox.R
        fixed_t = (np.max(fixed_coord, axis=0) + np.min(fixed_coord, axis=0)) / 2
        fixed_extent = fixed_pcd_bbox.extent
        print(fixed_extent)
        # Play around axis
        fixed_R_z = np.array([0, 0, 1])
        # Remove the axis that is parallel to the z-axis
        fixed_R_z_dot = fixed_R_z @ fixed_R
        fixed_R_z_idx = np.argmax(np.abs(fixed_R_z_dot))
        fixed_R_axis = np.delete(fixed_R, fixed_R_z_idx, axis=1)
        fixed_R_extent = np.delete(fixed_extent, fixed_R_z_idx)
        # The one with shorter extent is the x-axis
        fixed_R_x_idx = np.argmin(fixed_R_extent)
        fixed_R_x = fixed_R_axis[:, fixed_R_x_idx]
        # The other one is the y-axis
        fixed_R_y = np.cross(fixed_R_z, fixed_R_x)
        fixed_R = np.column_stack([fixed_R_x, fixed_R_y, fixed_R_z])
        fixed_pose = np.eye(4)
        fixed_pose[:3, 3] = fixed_t
        fixed_pose[:3, :3] = fixed_R
        target_t = (np.max(target_coord, axis=0) + np.min(target_coord, axis=0)) / 2
        target_pose = np.eye(4)
        target_pose[:3, :3] = fixed_R
        target_pose[:3, 3] = target_t

        # Shift the target coord to the origin
        fixed_pcd_o3d.transform(np.linalg.inv(fixed_pose))
        target_pcd_o3d.transform(np.linalg.inv(target_pose))

        # Normalize pcd
        # fixed_pcd_o3d, target_pcd_o3d, _, scale_xyz = normalize_pcd(fixed_pcd_o3d, target_pcd_o3d)
        target_pcd_o3d, fixed_pcd_o3d, _, scale_xyz = normalize_pcd(target_pcd_o3d, fixed_pcd_o3d)

        # Downsample the point cloud
        downsample_grid_size = self.cfg.PREPROCESS.GRID_SIZE
        fixed_pcd_o3d = fixed_pcd_o3d.voxel_down_sample(voxel_size=downsample_grid_size)
        target_pcd_o3d = target_pcd_o3d.voxel_down_sample(voxel_size=downsample_grid_size)

        # Compute normal
        fixed_pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # DEBUG:
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        # o3d.visualization.draw_geometries([target_pcd_o3d, fixed_pcd_o3d, origin])

        # Build the input
        target_coord = np.array(target_pcd_o3d.points).astype(np.float32)
        target_normal = np.array(target_pcd_o3d.normals).astype(np.float32)
        fixed_coord = np.array(fixed_pcd_o3d.points).astype(np.float32)
        fixed_normal = np.array(fixed_pcd_o3d.normals).astype(np.float32)
        target_feat = np.concatenate([target_coord, target_normal], axis=1)
        fixed_feat = np.concatenate([fixed_coord, fixed_normal], axis=1)
        data = {
            "target_coord": target_coord,
            "target_feat": target_feat,
            "fixed_coord": fixed_coord,
            "fixed_feat": fixed_feat,
            "fixed_pose": np.linalg.inv(fixed_pose),
            "target_pose": np.linalg.inv(target_pose),
            "scale_xyz": scale_xyz,
        }
        return data

    def pose_recover_rpdiff(self, pred_pose9d: np.ndarray, crop_center: np.ndarray, data: dict):
        """Recover the pose back to the original coordinate system for RPDiff"""
        pred_pose_mat = utils.pose9d_to_mat(pred_pose9d, rot_axis=self.rot_axis)
        T_shift = np.eye(4)
        T_shift[:3, 3] = crop_center
        fixed_pose = data["fixed_pose"]
        target_pose = data["target_pose"]
        scale_xyz = data["scale_xyz"]
        T_scale = np.eye(4)
        T_scale[:3, :3] = np.diag([1.0 / scale_xyz, 1.0 / scale_xyz, 1.0 / scale_xyz])
        recover_pose = np.linalg.inv(T_scale @ fixed_pose) @ T_shift @ pred_pose_mat @ T_scale @ target_pose
        return recover_pose

    def post_filter_rpdiff(self, pred_pose9d: np.ndarray, samples: list, collision_threshold: float = 0.01):
        """Post-process for rpdiff; filter out collision results"""
        collison_scores = []
        for i in range(pred_pose9d.shape[0]):
            fixed_coord = samples[i]["fixed_coord"]
            target_coord = samples[i]["target_coord"]
            pred_pose_mat = utils.pose9d_to_mat(pred_pose9d[i], rot_axis=self.rot_axis)

            # Transform
            target_coord = (pred_pose_mat[:3, :3] @ target_coord.T).T + pred_pose_mat[:3, 3]

            # Check collision
            collision = check_collision(fixed_coord, target_coord, threshold=collision_threshold)
            collison_scores.append(collision)
        collison_scores = np.array(collison_scores)
        return collison_scores
