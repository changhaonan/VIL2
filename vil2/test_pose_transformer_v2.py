"""Test pose transformer."""

import os
import open3d as o3d
import torch
import numpy as np
import pickle
import argparse
import vil2.utils.misc_utils as utils
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.data.pcd_datalodaer import PcdPairCollator
from vil2.model.network.pose_transformer_v2 import PoseTransformerV2
from vil2.model.tmorp_model_v2 import TmorpModelV2
from detectron2.config import LazyConfig
from torch.utils.data.dataset import random_split
from vil2.vil2_utils import build_dmorp_dataset
import random


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--random_index", type=int, default=0)
    args = argparser.parse_args()
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # Load config
    task_name = "Dmorp"
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", "pose_transformer_rdiff.py")
    cfg = LazyConfig.load(cfg_file)

    # Load dataset & data loader
    train_dataset, val_dataset, test_dataset = build_dmorp_dataset(root_path, cfg)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=PcdPairCollator(),
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=PcdPairCollator(),
    )

    # Build model
    net_name = cfg.MODEL.NOISE_NET.NAME
    net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
    pose_transformer = PoseTransformerV2(**net_init_args)
    tmorp_model = TmorpModelV2(cfg, pose_transformer)

    model_name = tmorp_model.experiment_name()
    noise_net_name = cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", noise_net_name)
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = f"{save_path}/checkpoints"
    # Select the best checkpoint
    checkpoints = os.listdir(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
    checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])

    tmorp_model.load(checkpoint_file)
    for i in range(20):
        # test_idx = np.random.randint(len(test_dataset))
        test_idx = i
        # Load the best checkpoint
        test_data = test_dataset[test_idx]
        # test_data = train_dataset[test_idx]
        target_coord = test_data["target_coord"]
        target_feat = test_data["target_feat"]
        fixed_coord = test_data["fixed_coord"]
        fixed_feat = test_data["fixed_feat"]
        target_pose = test_data["target_pose"]

        # Init crop center
        crop_center = target_pose[:3]
        crop_size = cfg.DATALOADER.AUGMENTATION.CROP_SIZE
        knn_k = cfg.DATALOADER.AUGMENTATION.KNN_K
        crop_strategy = cfg.DATALOADER.AUGMENTATION.CROP_STRATEGY

        # Batch sampling
        batch_size = 32
        sample_batch, samples = tmorp_model.batch_random_sample(
            batch_size,
            target_coord=target_coord,
            target_feat=target_feat,
            fixed_coord=fixed_coord,
            fixed_feat=fixed_feat,
            crop_strategy=crop_strategy,
            crop_size=crop_size,
            knn_k=knn_k,
        )

        pred_pose9d, pred_status = tmorp_model.predict(batch=sample_batch)

        # Rank the prediction by status
        sorted_indices = np.argsort(pred_status[:, 1])
        sorted_indices = sorted_indices[::-1]
        pred_pose9d = pred_pose9d[sorted_indices]
        pred_status = pred_status[sorted_indices]

        # Check the result
        fixed_pcd = o3d.geometry.PointCloud()
        fixed_pcd.points = o3d.utility.Vector3dVector(fixed_coord)
        fixed_pcd.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([fixed_pcd])

        for j in range(3):
            print(f"Status: {pred_status[j]} for {j}-th sample")
            vis_list = [fixed_pcd]
            # Crop fixed
            crop_fixed_coord = samples[sorted_indices[j]]["fixed_coord"]
            crop_fixed_pcd = o3d.geometry.PointCloud()
            crop_fixed_pcd.points = o3d.utility.Vector3dVector(crop_fixed_coord)
            crop_fixed_pcd.paint_uniform_color([1, 0, 0])
            crop_center = samples[sorted_indices[j]]["crop_center"]
            crop_fixed_pcd.translate(crop_center)
            vis_list.append(crop_fixed_pcd)

            # Target
            pred_pose_mat = utils.pose9d_to_mat(pred_pose9d[j], rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_coord)
            target_pcd.paint_uniform_color([0, 0, 1])
            target_pcd.transform(pred_pose_mat)
            target_pcd.translate(crop_center)
            vis_list.append(target_pcd)

            # Origin
            origin_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis_list.append(origin_pcd)
            o3d.visualization.draw_geometries(vis_list)

        # for j in range(10):
        #     # Iterate multiple times
        #     # DEBUG: iterative seems to be not working
        #     if not has_gt_crop:
        #         # Do random crop sampling
        #         x_min, y_min, z_min = fixed_coord.min(axis=0)
        #         x_max, y_max, z_max = fixed_coord.max(axis=0)
        #         crop_center = np.random.rand(3) * (
        #             np.array([x_max, y_max, z_max]) - np.array([x_min, y_min, z_min])
        #         ) + np.array([x_min, y_min, z_min])
        #         crop_indices = PcdPairDataset.crop(
        #             pcd=fixed_coord,
        #             crop_center=crop_center,
        #             crop_strategy=crop_strategy,
        #             ref_points=target_coord,
        #             crop_size=crop_size,
        #             knn_k=knn_k,
        #         )
        #         crop_fixed_coord = fixed_coord[crop_indices]
        #         crop_fixed_feat = fixed_feat[crop_indices]
        #         crop_fixed_coord[:, :3] -= crop_center
        #         crop_fixed_feat[:, :3] -= crop_center
        #     else:
        #         crop_indices = PcdPairDataset.crop(
        #             pcd=fixed_coord,
        #             crop_center=crop_center,
        #             crop_strategy=crop_strategy,
        #             ref_points=target_coord,
        #             crop_size=crop_size,
        #             knn_k=knn_k,
        #         )
        #         crop_fixed_coord = fixed_coord[crop_indices]
        #         crop_fixed_feat = fixed_feat[crop_indices]
        #         crop_fixed_coord[:, :3] -= crop_center
        #         crop_fixed_feat[:, :3] -= crop_center

        #     # Do prediction
        #     pred_pose_mat, pred_status = tmorp_model.predict(
        #         target_coord=target_coord,
        #         target_feat=target_feat,
        #         fixed_coord=crop_fixed_coord,
        #         fixed_feat=crop_fixed_feat,
        #         target_pose=target_pose,
        #     )

        #     # DEBUG: Check the cropping result
        #     print(pred_status)
        #     fixed_pcd = o3d.geometry.PointCloud()
        #     fixed_pcd.points = o3d.utility.Vector3dVector(fixed_coord)
        #     fixed_color = np.zeros_like(fixed_coord)
        #     fixed_color[:, 1] = 1
        #     fixed_color[crop_indices, 0] = 1
        #     fixed_pcd.colors = o3d.utility.Vector3dVector(fixed_color)
        #     o3d.visualization.draw_geometries([fixed_pcd])

        #     if pred_status[1] >= 0.7:
        #         break

        # DEBUG & VISUALIZATION
        # target_color = np.zeros_like(target_coord)
        # target_color[:, 0] = 1
        # fixed_color = np.zeros_like(crop_fixed_coord)
        # fixed_color[:, 1] = 1

        # fixed_normal = crop_fixed_feat[:, 3:6]
        # target_normal = target_feat[:, 3:6]
        # target_pose_mat = utils.pose9d_to_mat(target_pose, rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)
        # # utils.visualize_pcd_list(
        # #     coordinate_list=[target_coord, crop_fixed_coord],
        # #     normal_list=[target_normal, fixed_normal],
        # #     color_list=[target_color, fixed_color],
        # #     pose_list=[np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)],
        # # )

        # # utils.visualize_pcd_list(
        # #     coordinate_list=[target_coord, fixed_coord],
        # #     normal_list=[target_normal, fixed_normal],
        # #     color_list=[target_color, fixed_color],
        # #     pose_list=[target_pose_mat, np.eye(4, dtype=np.float32)],
        # # )
        # # # Check the prediction
        # utils.visualize_pcd_list(
        #     coordinate_list=[target_coord, crop_fixed_coord],
        #     normal_list=[target_normal, fixed_normal],
        #     color_list=[target_color, fixed_color],
        #     pose_list=[pred_pose_mat, np.eye(4, dtype=np.float32)],
        # )

        # # Update crop_center
        # crop_center = pred_pose_mat[:3, 3] + crop_center
        # target_coord = (pred_pose_mat[:3, :3] @ target_coord.T).T + pred_pose_mat[:3, 3]
        # target_feat[:, :3] = (pred_pose_mat[:3, :3] @ target_feat[:, :3].T).T + pred_pose_mat[:3, 3]
        # target_feat[:, 3:6] = (pred_pose_mat[:3, :3] @ target_feat[:, 3:6].T).T
