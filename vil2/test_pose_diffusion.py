"""Run pose transformer."""

import os
import numpy as np
import torch
import pickle
import argparse
from vil2.data.pcd_dataset import PcdPairDataset
import vil2.utils.misc_utils as utils
from vil2.model.network.pose_transformer_noise import PoseTransformerNoiseNet
from vil2.model.dmorp_model_v2 import DmorpModel
from detectron2.config import LazyConfig
from torch.utils.data.dataset import random_split
from vil2.vil2_utils import build_dmorp_dataset, build_dmorp_model
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

    # Build model
    dmorp_model = build_dmorp_model(cfg)

    # Prepare save path
    model_name = dmorp_model.experiment_name()
    noise_net_name = cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", noise_net_name)
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load the best checkpoint
    checkpoint_path = f"{save_path}/checkpoints"
    # Select the best checkpoint
    checkpoints = os.listdir(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
    checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
    dmorp_model.load(checkpoint_file)

    # Do prediction
    for i in range(10):
        test_idx = np.random.randint(len(test_dataset))
        test_data = test_dataset[test_idx]
        target_coord = test_data["target_coord"]
        target_normal = test_data["target_normal"]
        target_color = test_data["target_color"]
        target_label = test_data["target_label"]
        fixed_coord = test_data["fixed_coord"]
        fixed_normal = test_data["fixed_normal"]
        fixed_color = test_data["fixed_color"]
        fixed_label = test_data["fixed_label"]
        target_pcd_arr = np.concatenate([target_coord, target_normal, target_color], axis=-1)
        fixed_pcd_arr = np.concatenate([fixed_coord, fixed_normal, fixed_color], axis=-1)
        target_label = test_data["target_label"]
        fixed_label = test_data["fixed_label"]
        target_pose = test_data["target_pose"]
        converge_step = test_data["converge_step"]

        # Do prediction
        pred_pose_mat = np.eye(4, dtype=np.float32)

        # Move the pcd accordingly
        target_pcd_arr[:, :3] = (pred_pose_mat[:3, :3] @ target_pcd_arr[:, :3].T + pred_pose_mat[:3, 3:4]).T  # coord
        target_pcd_arr[:, 3:6] = (pred_pose_mat[:3, :3] @ target_pcd_arr[:, 3:6].T).T  # normal
        # Move both back to center
        target_pcd_arr[:, :3] -= np.mean(target_pcd_arr[:, :3], axis=0)
        fixed_pcd_arr[:, :3] -= np.mean(fixed_pcd_arr[:, :3], axis=0)
        pred_pose_mat = dmorp_model.predict(
            target_pcd_arr=target_pcd_arr,
            fixed_pcd_arr=fixed_pcd_arr,
            target_label=target_label,
            fixed_label=fixed_label,
            target_pose=target_pose,
        )

        # Check the prediction
        utils.visualize_pcd_list(
            [target_coord, fixed_coord],
            [target_normal, fixed_normal],
            [target_color, fixed_color],
            [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)],
        )

        utils.visualize_pcd_list(
            [target_coord, fixed_coord],
            [target_normal, fixed_normal],
            [target_color, fixed_color],
            [pred_pose_mat, np.eye(4, dtype=np.float32)],
        )
