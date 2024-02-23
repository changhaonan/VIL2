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
from vil2.model.network.pcd_noise_net import PcdNoiseNet
from vil2.model.pcdd_model import PCDDModel
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
    pcd_noise_net = PcdNoiseNet(**net_init_args)
    pcdd_model = PCDDModel(cfg, pcd_noise_net)

    model_name = pcdd_model.experiment_name()
    noise_net_name = cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", noise_net_name)
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = f"{save_path}/checkpoints"
    # Select the best checkpoint
    checkpoints = os.listdir(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
    checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])

    pcdd_model.load(checkpoint_file)
    for _i in range(20):
        batch = next(iter(test_data_loader))
        if _i != 4:
            continue
        pred_target_coord, prev_target_coord, anchor_coord_full, target_coord_full = pcdd_model.predict(batch=batch)

        # Visualize
        for check_idx in range(pred_target_coord.shape[0]):
            # Compute transform
            transform, residual = PCDDModel.kabsch_transform(prev_target_coord[check_idx], pred_target_coord[check_idx])
            print(f"Residual: {residual:.4f}")
            anchor_full_pcd = o3d.geometry.PointCloud()
            anchor_full_pcd.points = o3d.utility.Vector3dVector(anchor_coord_full[check_idx])
            anchor_full_pcd.paint_uniform_color([1, 0, 0])

            target_full_pcd = o3d.geometry.PointCloud()
            target_full_pcd.points = o3d.utility.Vector3dVector(target_coord_full[check_idx])
            target_full_pcd.paint_uniform_color([0, 0, 1])
            target_full_pcd.transform(transform)
            prev_target_full_pcd = o3d.geometry.PointCloud()
            prev_target_full_pcd.points = o3d.utility.Vector3dVector(target_coord_full[check_idx])
            prev_target_full_pcd.paint_uniform_color([1, 1, 0])

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(pred_target_coord[check_idx])
            target_pcd.paint_uniform_color([0, 1, 1])
            prev_target_pcd = o3d.geometry.PointCloud()
            prev_target_pcd.points = o3d.utility.Vector3dVector(prev_target_coord[check_idx])
            prev_target_pcd.paint_uniform_color([0, 1, 0])
            # Connecting points within target_pcd and prev_target_pcd
            lines = []
            for i in range(len(pred_target_coord[check_idx])):
                lines.append([i, i])
            for i in range(len(pred_target_coord[check_idx])):
                lines.append([i, i + len(pred_target_coord[check_idx])])
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.concatenate([pred_target_coord[check_idx], prev_target_coord[check_idx]], axis=0)),
                lines=o3d.utility.Vector2iVector(lines),
            )
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([origin, anchor_full_pcd, target_full_pcd, target_pcd, prev_target_full_pcd, prev_target_pcd, line_set])
