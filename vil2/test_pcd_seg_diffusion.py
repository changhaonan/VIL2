"""Test pcd segmentation diffusion."""

import os
import open3d as o3d
import torch
import numpy as np
import pickle
import argparse
import vil2.utils.misc_utils as utils
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.data.pcd_datalodaer import PcdPairCollator
from vil2.model.network.pcd_seg_noise_net import PcdSegNoiseNet
from vil2.model.pcdd_seg_model import PCDDModel
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
    argparser.add_argument("--task_name", type=str, default="book_in_bookshelf", help="stack_can_in_cabinet, book_in_bookshelf, mug_on_rack_multi")
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
        "book_in_bookshelf": "/home/harvey/Data/rpdiff_V3/book_in_bookshelf",
        "mug_on_rack_multi": "/home/harvey/Project/VIL2/vil2/external/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos/task_name_mug_on_rack_multi",
    }
    data_format = "test"  # "rpdiff_fail" or "raw", "test"
    rpdiff_path = data_path_dict[task_name]
    rpdiff_file_list = os.listdir(rpdiff_path)
    rpdiff_file_list = [f for f in rpdiff_file_list if f.endswith(".npz")]

    # Load dataset & data loader
    train_dataset, val_dataset, test_dataset = build_dmorp_dataset(root_path, cfg)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=PcdPairCollator())
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=PcdPairCollator())

    # Build model
    net_name = cfg.MODEL.NOISE_NET.NAME
    net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
    pcd_noise_net = PcdSegNoiseNet(**net_init_args)
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
        # batch = next(iter(test_data_loader))
        batch = next(iter(train_data_loader))
        if _i < 11:
            continue
        # Extract one data from batch
        check_batch_idx = 1
        pred_anchor_label, anchor_coord, anchor_super_index = pcdd_model.predict(batch=batch, check_batch_idx=check_batch_idx)

        # Visualize
        for check_idx in range(pred_anchor_label.shape[0]):
            anchor_full_pcd = o3d.geometry.PointCloud()
            anchor_full_pcd.points = o3d.utility.Vector3dVector(anchor_coord[check_idx])
            anchor_full_pcd.paint_uniform_color([1, 0, 0])

            super_point_list = []
            for super_idx in np.unique(anchor_super_index[check_idx, :, 0]):
                pcd_superpoint_idx = anchor_super_index[check_idx, :, 0] == super_idx
                superpoint_coord = anchor_coord[check_idx][pcd_superpoint_idx]
                superpoint_pcd = o3d.geometry.PointCloud()
                superpoint_pcd.points = o3d.utility.Vector3dVector(superpoint_coord)
                # Color by label3
                super_idx = super_idx - np.min(anchor_super_index[check_idx, :, 0])
                superpoint_label = pred_anchor_label[check_idx][super_idx]  # (-1, 1)
                color = superpoint_label * np.array([0.0, 1.0, 0.0]) + (1 - superpoint_label) * np.array([1.0, 0.0, 0.0])
                superpoint_pcd.paint_uniform_color(color)
                super_point_list.append(superpoint_pcd)
            o3d.visualization.draw_geometries(super_point_list)
