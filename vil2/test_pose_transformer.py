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
from vil2.model.network.pose_transformer_v3 import PoseTransformerV3
from vil2.model.tmorp_model_v3 import TmorpModelV3
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

    # Use raw data
    # Prepare path
    data_path_dict = {
        "stack_can_in_cabinet": "/home/harvey/Project/VIL2/vil2/external/rpdiff/data/task_demos/can_in_cabinet_stack/task_name_stack_can_in_cabinet",
        "book_in_bookshelf": "/home/harvey/Project/VIL2/vil2/external/rpdiff/data/task_demos/book_on_bookshelf_double_view_rnd_ori/task_name_book_in_bookshelf",
        "mug_on_rack_multi": "/home/harvey/Project/VIL2/vil2/external/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos/task_name_mug_on_rack_multi",
    }
    data_format = "test"  # "rpdiff_fail" or "raw", "test"
    rpdiff_path = data_path_dict[task_name]
    rpdiff_file_list = os.listdir(rpdiff_path)
    rpdiff_file_list = [f for f in rpdiff_file_list if f.endswith(".npz")]

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

    # Test dataset
    if data_format == "rpdiff_fail":
        failed_data_path = (
            "/home/harvey/Project/VIL2/vil2/external/rpdiff/eval_data/eval_data/can_on_cabinet_nosc/seed_0/failed"
        )
        failed_data_list = os.listdir(failed_data_path)
        failed_data_list = [os.path.join(failed_data_path, f) for f in failed_data_list if f.endswith(".npz")]

    # Build model
    net_name = cfg.MODEL.NOISE_NET.NAME
    net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
    pose_transformer = PoseTransformerV3(**net_init_args)
    tmorp_model = TmorpModelV3(cfg, pose_transformer)

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
        batch = next(iter(test_data_loader))
        pred_pose9d_coarse, pred_status_coarse, pred_pose9d_fine, pred_status_fine = tmorp_model.predict(batch=batch, target_pose=batch["target_pose"].cpu().numpy())

        # # Rank the prediction by status
        # sorted_indices = np.argsort(pred_status[:, 1])
        # sorted_indices = sorted_indices[::-1]
        # pred_pose9d = pred_pose9d[sorted_indices]
        # pred_status = pred_status[sorted_indices]

        # Check the results
        target_batch_idx = batch["target_batch_index"]
        fixed_batch_idx = batch["fixed_batch_index"]
        for j in range(pred_pose9d_coarse.shape[0]):
            print(f"Prediction status: {pred_status_fine[j]}")
            target_idx = target_batch_idx == j
            fixed_idx = fixed_batch_idx == j
            target_coord = batch["target_coord"][target_idx].cpu().numpy()
            fixed_coord = batch["fixed_coord"][fixed_idx].cpu().numpy()
            target_pose = batch["target_pose"][j].cpu().numpy()
            target_pose_mat = utils.pose9d_to_mat(target_pose, rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)
            # Visualize the results
            target_pcd_o3d = o3d.geometry.PointCloud()
            target_pcd_o3d.points = o3d.utility.Vector3dVector(target_coord)
            target_pcd_o3d.paint_uniform_color([0.0, 0.0, 1.0])
            pred_pose_mat = utils.pose9d_to_mat(pred_pose9d_coarse[j], rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)
            target_pcd_o3d.transform(pred_pose_mat)

            gt_target_pcd_o3d = o3d.geometry.PointCloud()
            gt_target_pcd_o3d.points = o3d.utility.Vector3dVector(target_coord)
            gt_target_pcd_o3d.paint_uniform_color([0.0, 1.0, 0.0])
            gt_target_pcd_o3d.transform(target_pose_mat)

            s_target_pcd_o3d = o3d.geometry.PointCloud()
            s_target_pcd_o3d.points = o3d.utility.Vector3dVector(target_coord)
            s_target_pcd_o3d.paint_uniform_color([1.0, 1.0, 0.0])

            # target_pcd_o3d.transform(target_pose_mat)
            # Check numerical difference
            trans_loss = np.linalg.norm(pred_pose9d_coarse[j, :3] - target_pose[:3])
            rx_loss = np.linalg.norm(pred_pose9d_coarse[j, 3:6] - target_pose[3:6])
            ry_loss = np.linalg.norm(pred_pose9d_coarse[j, 6:9] - target_pose[6:9])
            print(f"Translation loss: {trans_loss}, Rotation loss: {rx_loss}, {ry_loss}")

            fixed_pcd_o3d = o3d.geometry.PointCloud()
            fixed_pcd_o3d.points = o3d.utility.Vector3dVector(fixed_coord)
            fixed_pcd_o3d.paint_uniform_color([1.0, 0.0, 0.0])
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([target_pcd_o3d, gt_target_pcd_o3d, s_target_pcd_o3d, fixed_pcd_o3d, origin])
