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

    # Do test
    # tmorp_model.test(
    #     test_data_loader=test_data_loader,
    #     save_path=save_path,
    # )

    checkpoint_path = f"{save_path}/checkpoints"
    # Select the best checkpoint
    checkpoints = os.listdir(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
    checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])

    tmorp_model.load(checkpoint_file)
    for i in range(10):
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

        # Do prediction
        pred_pose_mat = tmorp_model.predict(target_coord, target_feat, fixed_coord, fixed_feat, target_pose=target_pose)

        # DEBUG & VISUALIZATION
        target_color = np.zeros_like(target_coord)
        target_color[:, 0] = 1
        fixed_color = np.zeros_like(fixed_coord)
        fixed_color[:, 1] = 1

        fixed_normal = fixed_feat[:, 3:6]
        target_normal = target_feat[:, 3:6]
        target_pose_mat = utils.pose9d_to_mat(target_pose, rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)
        utils.visualize_pcd_list(
            coordinate_list=[target_coord, fixed_coord],
            normal_list=[target_normal, fixed_normal],
            color_list=[target_color, fixed_color],
            pose_list=[np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)],
        )

        utils.visualize_pcd_list(
            coordinate_list=[target_coord, fixed_coord],
            normal_list=[target_normal, fixed_normal],
            color_list=[target_color, fixed_color],
            pose_list=[target_pose_mat, np.eye(4, dtype=np.float32)],
        )
        # # Check the prediction
        utils.visualize_pcd_list(
            coordinate_list=[target_coord, fixed_coord],
            normal_list=[target_normal, fixed_normal],
            color_list=[target_color, fixed_color],
            pose_list=[pred_pose_mat, np.eye(4, dtype=np.float32)],
        )
