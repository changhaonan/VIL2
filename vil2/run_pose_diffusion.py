"""Run pose transformer."""
import os
import torch
import pickle
import argparse
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.model.network.pose_transformer_noise import PoseTransformerNoiseNet
from vil2.model.dmorp_model_v2 import DmorpModel
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
    cfg_file = os.path.join(root_path, "config", "pose_transformer_rdiff.py")
    cfg = LazyConfig.load(cfg_file)
    retrain = cfg.MODEL.RETRAIN
    pcd_size = cfg.MODEL.PCD_SIZE
    is_elastic_distortion = cfg.DATALOADER.AUGMENTATION.IS_ELASTIC_DISTORTION
    is_random_distortion = cfg.DATALOADER.AUGMENTATION.IS_RANDOM_DISTORTION
    random_distortion_rate = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_RATE
    random_distortion_mag = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_MAG
    volume_augmentation_file = cfg.DATALOADER.AUGMENTATION.VOLUME_AUGMENTATION_FILE
    random_segment_drop_rate = cfg.DATALOADER.AUGMENTATION.RANDOM_SEGMENT_DROP_RATE
    # Load dataset & data loader
    if cfg.ENV.GOAL_TYPE == "multimodal":
        dataset_folder = "dmorp_multimodal"
    elif "real" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "dmorp_real"
    elif "struct" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "dmorp_struct"
    elif "rdiff" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "dmorp_rdiff"
    else:
        dataset_folder = "dmorp_faster"

    # Get different split
    splits = ["train", "val", "test"]
    data_file_dict = {}
    for split in splits:
        data_file_dict[split] = os.path.join(
            root_path,
            "test_data",
            dataset_folder,
            f"diffusion_dataset_0_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_{split}.pkl",
        )

    volume_augmentations_path = (
        os.path.join(root_path, "config", volume_augmentation_file) if volume_augmentation_file is not None else None
    )
    train_dataset = PcdPairDataset(
        data_file_list=[data_file_dict["train"]],
        dataset_name="dmorp",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=is_elastic_distortion,
        is_random_distortion=is_random_distortion,
        random_distortion_rate=random_distortion_rate,
        random_distortion_mag=random_distortion_mag,
        volume_augmentations_path=volume_augmentations_path,
        random_segment_drop_rate=random_segment_drop_rate,
    )
    val_dataset = PcdPairDataset(
        data_file_list=[data_file_dict["val"]],
        dataset_name="dmorp",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=False,
        is_random_distortion=False,
        random_distortion_rate=0,
        random_distortion_mag=0,
        volume_augmentations_path=None,
        random_segment_drop_rate=0,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

    # Build model
    net_name = cfg.MODEL.NOISE_NET.NAME
    net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
    net_init_args["max_timestep"] = cfg.MODEL.NUM_DIFFUSION_ITERS

    pose_transformer = PoseTransformerNoiseNet(**net_init_args)
    dmorp_model = DmorpModel(cfg, pose_transformer)

    model_name = dmorp_model.experiment_name()
    noise_net_name = cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", noise_net_name)
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    dmorp_model.train(
        num_epochs=cfg.TRAIN.NUM_EPOCHS,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        save_path=save_path,
    )
