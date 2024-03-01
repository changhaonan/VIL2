"""Test rel pose transformer only on the training data."""

import os
import open3d as o3d
import torch
import numpy as np
import pickle
import argparse
import vil2.utils.misc_utils as utils
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.data.pcd_datalodaer import PcdPairCollator
from vil2.model.network.relpose_transformer import RelPoseTransformer
from vil2.model.rpt_model import RPTModel
from detectron2.config import LazyConfig
from torch.utils.data.dataset import random_split
from vil2.vil2_utils import build_dmorp_dataset
import random


# Read data
def parse_child_parent(arr):
    pcd_dict = arr[()]
    parent_val = pcd_dict["parent"]
    child_val = pcd_dict["child"]
    return parent_val, child_val


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--random_index", type=int, default=0)
    argparser.add_argument(
        "--task_name",
        type=str,
        default="book_in_bookshelf",
        help="stack_can_in_cabinet, book_in_bookshelf, mug_on_rack_multi",
    )
    args = argparser.parse_args()
    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    task_name = args.task_name
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", f"pose_transformer_rpdiff_{task_name}.py")
    cfg = LazyConfig.load(cfg_file)
    # Overriding config
    cfg.MODEL.NOISE_NET.NAME = "RPTModel"
    cfg.DATALOADER.AUGMENTATION.CROP_PCD = True
    cfg.DATALOADER.BATCH_SIZE = 32

    # Load dataset & data loader
    train_dataset, val_dataset, test_dataset = build_dmorp_dataset(root_path, cfg)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=PcdPairCollator(),
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=PcdPairCollator(),
    )

    # Build model
    net_name = cfg.MODEL.NOISE_NET.NAME
    net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
    pose_transformer = RelPoseTransformer(**net_init_args)
    rpt_model = RPTModel(cfg, pose_transformer)

    model_name = rpt_model.experiment_name()
    noise_net_name = cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", noise_net_name)
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = f"{save_path}/checkpoints"
    # Select the best checkpoint
    checkpoints = os.listdir(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
    checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])

    rpt_model.load(checkpoint_file)
    for i in range(20):
        batch = next(iter(val_data_loader))
        for k in range(3):
            if k != 0:
                batch["prev_pose9d"] = torch.from_numpy(prev_pose9d)
            else:
                prev_pose9d = None
            pred_pose9d, pred_status = rpt_model.predict(batch=batch, target_pose=batch["target_pose"].cpu().numpy())
            # Check the results
            target_batch_idx = batch["target_batch_index"]
            anchor_batch_idx = batch["anchor_batch_index"]
            prev_pose9d_list = []
            for j in range(pred_pose9d.shape[0]):
                print(f"Prediction status: {pred_status[j]}")
                target_idx = target_batch_idx == j
                anchor_idx = anchor_batch_idx == j
                target_coord = batch["target_coord"][target_idx].cpu().numpy()
                anchor_coord = batch["anchor_coord"][anchor_idx].cpu().numpy()
                target_pose = batch["target_pose"][j].cpu().numpy()
                target_pose_mat = utils.pose9d_to_mat(target_pose, rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)
                # Visualize the results
                target_pcd_o3d = o3d.geometry.PointCloud()
                target_pcd_o3d.points = o3d.utility.Vector3dVector(target_coord)
                target_pcd_o3d.paint_uniform_color([0.0, 0.0, 1.0])
                pred_pose_mat = utils.pose9d_to_mat(pred_pose9d[j], rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)
                if prev_pose9d is not None:
                    prev_pose_mat = utils.pose9d_to_mat(prev_pose9d[j], rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)
                    target_pcd_o3d.transform(prev_pose_mat)
                    prev_pose_mat = pred_pose_mat @ prev_pose_mat
                    prev_pose9d_list.append(utils.mat_to_pose9d(prev_pose_mat, rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS))
                else:
                    prev_pose9d_list.append(pred_pose9d[j])
                target_pcd_o3d.transform(pred_pose_mat)

                gt_target_pcd_o3d = o3d.geometry.PointCloud()
                gt_target_pcd_o3d.points = o3d.utility.Vector3dVector(target_coord)
                gt_target_pcd_o3d.paint_uniform_color([0.0, 1.0, 0.0])
                gt_target_pcd_o3d.transform(target_pose_mat)

                s_target_pcd_o3d = o3d.geometry.PointCloud()
                s_target_pcd_o3d.points = o3d.utility.Vector3dVector(target_coord)
                s_target_pcd_o3d.paint_uniform_color([1.0, 1.0, 0.0])

                # Check numerical difference
                trans_loss = np.linalg.norm(pred_pose9d[j, :3] - target_pose[:3])
                rx_loss = np.linalg.norm(pred_pose9d[j, 3:6] - target_pose[3:6])
                ry_loss = np.linalg.norm(pred_pose9d[j, 6:9] - target_pose[6:9])
                print(f"Translation loss: {trans_loss}, Rotation loss: {rx_loss}, {ry_loss}")

                anchor_pcd_o3d = o3d.geometry.PointCloud()
                anchor_pcd_o3d.points = o3d.utility.Vector3dVector(anchor_coord)
                anchor_pcd_o3d.paint_uniform_color([1.0, 0.0, 0.0])
                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                if j <= 4:
                    o3d.visualization.draw_geometries([target_pcd_o3d, gt_target_pcd_o3d, s_target_pcd_o3d, anchor_pcd_o3d, origin])

            prev_pose9d = np.array(prev_pose9d_list)
