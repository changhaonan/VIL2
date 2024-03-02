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
import cv2
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import time
from vil2.model.network.geometric import batch2offset, to_dense_batch, offset2batch
from vil2.model.network.rigpose_transformer import RigPoseTransformer
import vil2.utils.misc_utils as utils
from vil2.utils.pcd_utils import normalize_pcd, check_collision
import torch_scatter
import wandb

# DataUtils
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.data.pcd_datalodaer import PcdPairCollator


class LRigPoseTransformer(L.LightningModule):
    """Lightning module for Rigstration Pose Transformer"""

    def __init__(self, pose_transformer: RigPoseTransformer, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.pose_transformer: RigPoseTransformer = pose_transformer
        self.lr = cfg.TRAIN.LR
        self.warm_up_step = cfg.TRAIN.WARM_UP_STEP
        self.start_time = time.time()
        # Logging
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        self.valid_crop_threshold = cfg.MODEL.VALID_CROP_THRESHOLD
        self.rot_axis = cfg.DATALOADER.AUGMENTATION.ROT_AXIS
        self.sample_batch = None

    def training_step(self, batch, batch_idx):
        loss = self.criterion(batch)
        # log
        self.log("train_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.criterion(batch)
        # log
        self.log("test_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.sample_batch is None:
            self.sample_batch = batch
        loss = self.criterion(batch)
        # log
        self.log("val_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Validate results by visualization"""
        if (self.current_epoch + 1) % 1 == 0:
            batch = self.sample_batch
            anchor_batch_idx = batch["anchor_batch_index"]
            anchor_coord = batch["anchor_coord"]
            batch_anchor_coord = to_dense_batch(anchor_coord, anchor_batch_idx)[0]
            target_batch_idx = batch["target_batch_index"]
            target_coord = batch["target_coord"]
            batch_target_coord = to_dense_batch(target_coord, target_batch_idx)[0]
            conf_matrix, gt_corr, (pred_R, pred_t) = self.forward(batch)
            # Visualize the result
            for i in range(min(4, batch_anchor_coord.shape[0])):
                image = self.view_result(batch_anchor_coord[i, :], batch_target_coord[i, :], pred_R[i, :], pred_t[i, :], conf_matrix[i, :], show_3d=False)
                if image is not None:
                    self.logger.experiment.log({"val_result": [wandb.Image(image, caption=f"val_result_{i}")]})

    def criterion(self, batch):
        conf_matrix, gt_corr, (gt_R, gt_t) = self.forward(batch)
        # compute loss
        loss = self.pose_transformer.dual_softmax_reposition.compute_matching_loss(conf_matrix, gt_matrix=gt_corr)
        return loss

    def forward(self, batch) -> Any:
        # Assemble input
        target_coord = batch["target_coord"]
        target_normal = batch["target_normal"]
        target_feat = batch["target_feat"]
        target_batch_idx = batch["target_batch_index"]
        target_offset = batch2offset(target_batch_idx)
        target_points = [target_coord, target_feat, target_offset]
        anchor_coord = batch["anchor_coord"]
        anchor_normal = batch["anchor_normal"]
        anchor_feat = batch["anchor_feat"]
        anchor_batch_idx = batch["anchor_batch_index"]
        anchor_offset = batch2offset(anchor_batch_idx)
        anchor_points = [anchor_coord, anchor_feat, anchor_offset]
        prev_pose9d = batch.get("prev_pose9d", None)

        if prev_pose9d is not None:
            target_points, target_normal = self.pose_transformer.reposition(target_points, target_normal, prev_pose9d, self.rot_axis)

        # Compute conditional features
        enc_target_points, enc_anchor_points, enc_target_attr, enc_anchor_attr = self.pose_transformer.encode_cond(
            target_points, anchor_points, target_attrs={"normal": target_normal}, anchor_attrs={"normal": anchor_normal}
        )

        # Compute correspondence
        gt_corr = batch["corr"]
        corr_batch_idx = batch["corr_batch_index"]
        corr_offset = batch2offset(corr_batch_idx)
        gt_corr = to_dense_batch(gt_corr, corr_batch_idx)[0]

        conf_matrix = self.pose_transformer.forward(enc_anchor_points, enc_target_points)
        gt_corr_matrix = self.pose_transformer.to_gt_correspondence_matrix(conf_matrix, gt_corr)

        # [DEBUG] Output estimation for evaluation
        R, t, condition = self.pose_transformer.dual_softmax_reposition.arun(
            conf_matrix=conf_matrix, coord_a=target_coord, coord_b=anchor_coord, batch_index_a=offset2batch(target_offset), batch_index_b=offset2batch(anchor_offset)
        )

        return conf_matrix, gt_corr_matrix, (R, t)

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

    def view_result(self, anchor_coord: torch.Tensor, target_coord: torch.Tensor, R: torch.Tensor, t: torch.Tensor, corr: torch.Tensor, show_3d: bool = False):
        """Visualize the prediction"""
        target_coord = target_coord.detach().cpu().numpy()
        anchor_coord = anchor_coord.detach().cpu().numpy()
        R = R.detach().cpu().numpy()
        t = t.detach().cpu().numpy()
        corr = corr.detach().cpu().numpy()
        # Transform
        target_coord = (R @ target_coord.T).T + t
        # Visualize
        pcd = o3d.geometry.PointCloud()
        combined_coord = np.concatenate([target_coord, anchor_coord], axis=0)
        combined_color = np.zeros((combined_coord.shape[0], 3))
        combined_color[: target_coord.shape[0], 0] = 1
        combined_color[target_coord.shape[0] :, 2] = 1
        pcd.points = o3d.utility.Vector3dVector(combined_coord)
        pcd.colors = o3d.utility.Vector3dVector(combined_color)
        # Draw correspondence
        corr = (corr - corr.min()) / (corr.max() - corr.min() + 1e-8)  # Normalize
        lines = []
        colors = []
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if corr[i, j] > 0.5:
                    lines.append([i, j + target_coord.shape[0]])
                    intensity = corr[i, j]
                    colors.append([intensity, 0, 1 - intensity])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(combined_coord)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        if not show_3d:
            # Offscreen rendering
            viewer = o3d.visualization.Visualizer()
            viewer.create_window(width=540, height=540, visible=False)
            viewer.add_geometry(pcd)
            if len(lines) > 0:
                viewer.add_geometry(line_set)
            # Render image
            image = viewer.capture_screen_float_buffer(do_render=True)
            viewer.destroy_window()
            return np.asarray(image)
        else:
            o3d.visualization.draw_geometries([pcd, line_set])
            return None


class RGTModel:
    """RigPose Transformer Model for multi-object relative Pose Generation"""

    def __init__(self, cfg, pose_transformer: RigPoseTransformer) -> None:
        self.cfg = cfg
        # parameters
        # build model
        self.pose_transformer = pose_transformer
        self.lightning_pose_transformer = LRigPoseTransformer(pose_transformer, cfg).to(torch.float32)
        # parameters
        self.rot_axis = cfg.DATALOADER.AUGMENTATION.ROT_AXIS
        self.gradient_clip_val = cfg.TRAIN.GRADIENT_CLIP_VAL
        self.logger_project = cfg.LOGGER.PROJECT

    def train(self, num_epochs: int, train_data_loader, val_data_loader, save_path: str):
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=os.path.join(save_path, "checkpoints"), filename="RTmorp_model-{epoch:02d}-{val_loss:.2f}", save_top_k=3, mode="min")
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
        trainer.fit(self.lightning_pose_transformer, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

    def test(self, test_data_loader, save_path: str):
        # Trainer
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
        trainer.test(
            self.lightning_pose_transformer.__class__.load_from_checkpoint(checkpoint_file, pose_transformer=self.pose_transformer, cfg=self.cfg),
            dataloaders=test_data_loader,
        )

    def predict(self, target_coord=None, target_feat=None, anchor_coord=None, anchor_feat=None, batch=None, target_pose=None, vis: bool = False) -> Any:
        self.lightning_pose_transformer.eval()
        # Assemble batch
        if batch is None:
            target_batch_idx = np.array([0] * target_coord.shape[0], dtype=np.int64)
            anchor_batch_idx = np.array([0] * anchor_coord.shape[0], dtype=np.int64)
            batch = {
                "target_coord": target_coord,
                "target_feat": target_feat,
                "target_batch_index": target_batch_idx,
                "anchor_coord": anchor_coord,
                "anchor_feat": anchor_feat,
                "anchor_batch_index": anchor_batch_idx,
            }
            # Put to torch
            for key in batch.keys():
                batch[key] = torch.from_numpy(batch[key])
        # Put to device
        for key in batch.keys():
            batch[key] = batch[key].to(self.lightning_pose_transformer.device)
        conf_matrix, gt_corr, (pred_R, pred_t) = self.lightning_pose_transformer(batch)
        if vis:
            anchor_batch_idx = batch["anchor_batch_index"]
            anchor_coord = batch["anchor_coord"]
            batch_anchor_coord = to_dense_batch(anchor_coord, anchor_batch_idx)[0]
            target_batch_idx = batch["target_batch_index"]
            target_coord = batch["target_coord"]
            batch_target_coord = to_dense_batch(target_coord, target_batch_idx)[0]
            # Visualize the result
            for i in range(min(8, batch_anchor_coord.shape[0])):
                self.lightning_pose_transformer.view_result(batch_anchor_coord[i, :], batch_target_coord[i, :], pred_R[i, :], pred_t[i, :], conf_matrix[i, :], show_3d=True)

        conf_matrix = conf_matrix.detach().cpu().numpy()
        gt_corr = gt_corr.detach().cpu().numpy()
        pred_R = pred_R.detach().cpu().numpy()
        pred_t = pred_t.detach().cpu().numpy()
        return conf_matrix, gt_corr, (pred_R, pred_t)

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
        return f"RPT_model_{crop_strategy}"

    def batch_random_sample(
        self, sample_size: int, sample_strategy: str, target_coord: np.ndarray, target_feat: np.ndarray, anchor_coord: np.ndarray, anchor_feat: np.ndarray, crop_strategy: str, **kwargs
    ) -> Any:
        crop_size = kwargs.get("crop_size", 0.2)
        knn_k = kwargs.get("knn_k", 20)
        num_grid = kwargs.get("num_grid", 5)
        samples = []
        # Randomly sample batch
        for i in range(sample_size):
            # Do random crop sampling
            x_min, y_min, z_min = anchor_coord.min(axis=0)
            x_max, y_max, z_max = anchor_coord.max(axis=0)
            if sample_strategy == "random":
                x_center = np.random.uniform(0.1, 0.9) * (x_max - x_min) + x_min
                y_center = np.random.uniform(0.1, 0.9) * (y_max - y_min) + y_min
                z_center = np.random.uniform(0.1, 0.9) * (z_max - z_min) + z_min
                crop_center = np.array([x_center, y_center, z_center])
            elif sample_strategy == "grid":
                x_grid = np.linspace(0.1, 0.9, num_grid) * (x_max - x_min) + x_min
                y_grid = np.linspace(0.1, 0.9, num_grid) * (y_max - y_min) + y_min
                z_grid = np.linspace(0.1, 0.9, num_grid) * (z_max - z_min) + z_min
                crop_center = np.array([np.random.choice(x_grid), np.random.choice(y_grid), np.random.choice(z_grid)])
            crop_indices = PcdPairDataset.crop(pcd=anchor_coord, crop_center=crop_center, crop_strategy=crop_strategy, ref_points=target_coord, crop_size=crop_size, knn_k=knn_k)
            crop_anchor_coord = anchor_coord[crop_indices]
            crop_anchor_feat = anchor_feat[crop_indices]
            crop_anchor_coord[:, :3] -= crop_center
            crop_anchor_feat[:, :3] -= crop_center
            # Convert to batch
            sample = {
                "target_coord": target_coord,
                "target_feat": target_feat,
                "anchor_coord": crop_anchor_coord,
                "anchor_feat": crop_anchor_feat,
                "target_pose": np.zeros(9),
                "is_valid_crop": np.ones(1),
                "crop_center": crop_center,
            }
            samples.append(sample)
        return PcdPairCollator()(samples), samples

    # Utility functions
    def preprocess_input_rpdiff(self, target_coord: np.ndarray, anchor_coord: np.ndarray):
        """Preprocess data for eval on RPDiff"""
        # Build o3d object
        target_pcd_o3d = o3d.geometry.PointCloud()
        target_pcd_o3d.points = o3d.utility.Vector3dVector(target_coord)
        target_pcd_o3d.paint_uniform_color([0, 0, 1])
        anchor_pcd_o3d = o3d.geometry.PointCloud()
        anchor_pcd_o3d.points = o3d.utility.Vector3dVector(anchor_coord)
        anchor_pcd_o3d.paint_uniform_color([1, 0, 0])

        # Estimate the pose of fixed coord using a rotating bbox
        anchor_pcd_bbox = anchor_pcd_o3d.get_minimal_oriented_bounding_box()
        anchor_pcd_bbox.color = [0, 1, 0]
        anchor_R = anchor_pcd_bbox.R
        anchor_t = (np.max(anchor_coord, axis=0) + np.min(anchor_coord, axis=0)) / 2
        anchor_extent = anchor_pcd_bbox.extent
        print(anchor_extent)
        # Play around axis
        anchor_R_z = np.array([0, 0, 1])
        # Remove the axis that is parallel to the z-axis
        anchor_R_z_dot = anchor_R_z @ anchor_R
        anchor_R_z_idx = np.argmax(np.abs(anchor_R_z_dot))
        anchor_R_axis = np.delete(anchor_R, anchor_R_z_idx, axis=1)
        anchor_R_extent = np.delete(anchor_extent, anchor_R_z_idx)
        # The one with shorter extent is the x-axis
        anchor_R_x_idx = np.argmin(anchor_R_extent)
        anchor_R_x = anchor_R_axis[:, anchor_R_x_idx]
        # The other one is the y-axis
        anchor_R_y = np.cross(anchor_R_z, anchor_R_x)
        anchor_R = np.column_stack([anchor_R_x, anchor_R_y, anchor_R_z])
        anchor_pose = np.eye(4)
        anchor_pose[:3, 3] = anchor_t
        anchor_pose[:3, :3] = anchor_R
        target_t = (np.max(target_coord, axis=0) + np.min(target_coord, axis=0)) / 2
        target_pose = np.eye(4)
        target_pose[:3, :3] = anchor_R
        target_pose[:3, 3] = target_t

        # Shift the target coord to the origin
        anchor_pcd_o3d.transform(np.linalg.inv(anchor_pose))
        target_pcd_o3d.transform(np.linalg.inv(target_pose))

        # Normalize pcd
        # anchor_pcd_o3d, target_pcd_o3d, _, scale_xyz = normalize_pcd(anchor_pcd_o3d, target_pcd_o3d)
        target_pcd_o3d, anchor_pcd_o3d, _, scale_xyz = normalize_pcd(target_pcd_o3d, anchor_pcd_o3d)

        # Downsample the point cloud
        downsample_grid_size = self.cfg.PREPROCESS.GRID_SIZE
        anchor_pcd_o3d = anchor_pcd_o3d.voxel_down_sample(voxel_size=downsample_grid_size)
        target_pcd_o3d = target_pcd_o3d.voxel_down_sample(voxel_size=downsample_grid_size)

        # Compute normal
        anchor_pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Build the input
        target_coord = np.array(target_pcd_o3d.points).astype(np.float32)
        target_normal = np.array(target_pcd_o3d.normals).astype(np.float32)
        anchor_coord = np.array(anchor_pcd_o3d.points).astype(np.float32)
        anchor_normal = np.array(anchor_pcd_o3d.normals).astype(np.float32)
        target_feat = np.concatenate([target_coord, target_normal], axis=1)
        anchor_feat = np.concatenate([anchor_coord, anchor_normal], axis=1)
        data = {
            "target_coord": target_coord,
            "target_feat": target_feat,
            "anchor_coord": anchor_coord,
            "anchor_feat": anchor_feat,
            "anchor_pose": np.linalg.inv(anchor_pose),
            "target_pose": np.linalg.inv(target_pose),
            "scale_xyz": scale_xyz,
        }
        return data

    def pose_recover_rpdiff(self, pred_pose9d: np.ndarray, crop_center: np.ndarray, data: dict):
        """Recover the pose back to the original coordinate system for RPDiff"""
        pred_pose_mat = utils.pose9d_to_mat(pred_pose9d, rot_axis=self.rot_axis)
        T_shift = np.eye(4)
        T_shift[:3, 3] = crop_center
        anchor_pose = data["anchor_pose"]
        target_pose = data["target_pose"]
        scale_xyz = data["scale_xyz"]
        T_scale = np.eye(4)
        T_scale[:3, :3] = np.diag([1.0 / scale_xyz, 1.0 / scale_xyz, 1.0 / scale_xyz])
        recover_pose = np.linalg.inv(T_scale @ anchor_pose) @ T_shift @ pred_pose_mat @ T_scale @ target_pose
        return recover_pose

    def post_filter_rpdiff(self, pred_pose9d: np.ndarray, samples: list, collision_threshold: float = 0.01):
        """Post-process for rpdiff; filter out collision results"""
        collison_scores = []
        for i in range(pred_pose9d.shape[0]):
            anchor_coord = samples[i]["anchor_coord"]
            target_coord = samples[i]["target_coord"]
            pred_pose_mat = utils.pose9d_to_mat(pred_pose9d[i], rot_axis=self.rot_axis)

            # Transform
            target_coord = (pred_pose_mat[:3, :3] @ target_coord.T).T + pred_pose_mat[:3, 3]

            # Check collision
            collision = check_collision(anchor_coord, target_coord, threshold=collision_threshold)
            collison_scores.append(collision)
        collison_scores = np.array(collison_scores)
        return collison_scores

    def reposition(self, coord: torch.Tensor, pose_pred: torch.Tensor, rot_axis: str = "xy"):
        """Reposition the points using the pose9d"""
        t = pose_pred[:, 0:3]
        r1 = pose_pred[:, 3:6]
        r2 = pose_pred[:, 6:9]
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
