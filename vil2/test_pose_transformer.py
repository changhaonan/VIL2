"""Test pose transformer."""
import os
import open3d as o3d
import torch
import numpy as np
import pickle
import argparse
import vil2.utils.misc_utils as utils
from vil2.data.pcd_dataset import PointCloudDataset
from vil2.model.network.pose_transformer import PoseTransformer
from vil2.model.tmorp_model import TmorpModel
from detectron2.config import LazyConfig
from torch.utils.data.dataset import random_split
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
    cfg_file = os.path.join(root_path, "config", "pose_transformer.py")
    cfg = LazyConfig.load(cfg_file)
    retrain = cfg.MODEL.RETRAIN
    pcd_size = cfg.MODEL.PCD_SIZE
    is_elastic_distortion = cfg.DATALOADER.AUGMENTATION.IS_ELASTIC_DISTORTION
    is_random_distortion = cfg.DATALOADER.AUGMENTATION.IS_RANDOM_DISTORTION
    random_distortion_rate = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_RATE
    random_distortion_mag = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_MAG
    volume_augmentation_file = cfg.DATALOADER.AUGMENTATION.VOLUME_AUGMENTATION_FILE
    random_segment_drop_rate = cfg.DATALOADER.AUGMENTATION.RANDOM_SEGMENT_DROP_RATE
    max_converge_step = cfg.DATALOADER.AUGMENTATION.MAX_CONVERGE_STEP
    # Load dataset & data loader
    data_id_list = [0]
    filter_key = "random"
    if cfg.ENV.GOAL_TYPE == "multimodal":
        dataset_folder = "dmorp_multimodal"
    if "real" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "dmorp_real"
    else:
        dataset_folder = "dmorp_faster"
    if filter_key is not None:
        data_file_list = [
            os.path.join(
                root_path,
                "test_data",
                dataset_folder,
                f"diffusion_dataset_{data_id}_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_{filter_key}.pkl",
            )
            for data_id in data_id_list
        ]
    else:
        data_file_list = [
            os.path.join(
                root_path,
                "test_data",
                dataset_folder,
                f"diffusion_dataset_{data_id}_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}.pkl",
            )
            for data_id in data_id_list
        ]

    volume_augmentations_path = (
        os.path.join(root_path, "config", volume_augmentation_file) if volume_augmentation_file is not None else None
    )
    dataset = PointCloudDataset(
        data_file_list=data_file_list,
        dataset_name="dmorp",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=is_elastic_distortion,
        is_random_distortion=is_random_distortion,
        random_distortion_rate=random_distortion_rate,
        random_distortion_mag=random_distortion_mag,
        volume_augmentations_path=volume_augmentations_path,
        random_segment_drop_rate=random_segment_drop_rate,
        max_converge_step=max_converge_step,
    )
    # dataset.set_mode("test")
    dataset.set_mode("test")
    # Load test data
    data_id_list = [0]
    if cfg.ENV.GOAL_TYPE == "multimodal":
        dataset_folder = "dmorp_multimodal"
    if "real" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "dmorp_real"
    else:
        dataset_folder = "dmorp_faster"

    # Split dataset
    train_size = int(cfg.MODEL.TRAIN_SPLIT * len(dataset))
    val_size = int(cfg.MODEL.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
    )

    # Build model
    net_name = cfg.MODEL.NOISE_NET.NAME
    net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
    net_init_args["max_converge_step"] = cfg.DATALOADER.AUGMENTATION.MAX_CONVERGE_STEP
    pose_transformer = PoseTransformer(**net_init_args)
    tmorp_model = TmorpModel(cfg, pose_transformer)

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

    # Do prediction
    test_idx = np.random.randint(len(test_dataset))

    # Load the best checkpoint
    checkpoint_path = f"{save_path}/checkpoints"
    # Select the best checkpoint
    checkpoints = os.listdir(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
    checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
    tmorp_model.load(checkpoint_file)

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
    for i in range(1, 4):
        print(f"---------- Converge step {i} ----------")
        # Move the pcd accordingly
        target_pcd_arr[:, :3] = (pred_pose_mat[:3, :3] @ target_pcd_arr[:, :3].T + pred_pose_mat[:3, 3:4]).T  # coord
        target_pcd_arr[:, 3:6] = (pred_pose_mat[:3, :3] @ target_pcd_arr[:, 3:6].T).T  # normal
        # Move both back to center
        target_pcd_arr[:, :3] -= np.mean(target_pcd_arr[:, :3], axis=0)
        fixed_pcd_arr[:, :3] -= np.mean(fixed_pcd_arr[:, :3], axis=0)
        pred_pose_mat = tmorp_model.predict(
            target_pcd_arr=target_pcd_arr,
            fixed_pcd_arr=fixed_pcd_arr,
            target_label=target_label,
            fixed_label=fixed_label,
            converge_step=np.array([i]),
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
