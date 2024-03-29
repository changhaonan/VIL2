"""Run pose transformer."""
import os
import torch
import pickle
import argparse
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.model.network.struct_transformer import StructTransformer
from vil2.model.structmorp_model import StructmorpModel
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
    cfg_file = os.path.join(root_path, "config", "struct_transformer.py")
    cfg = LazyConfig.load(cfg_file)
    retrain = cfg.MODEL.RETRAIN
    pcd_size = cfg.MODEL.PCD_SIZE
    is_elastic_distortion = cfg.DATALOADER.AUGMENTATION.IS_ELASTIC_DISTORTION
    is_random_distortion = cfg.DATALOADER.AUGMENTATION.IS_RANDOM_DISTORTION
    random_distortion_rate = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_RATE
    random_distortion_mag = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_MAG
    volume_augmentation_file = cfg.DATALOADER.AUGMENTATION.VOLUME_AUGMENTATION_FILE
    # Load dataset & data loader
    data_id_list = [0]
    if cfg.ENV.GOAL_TYPE == "multimodal":
        dataset_folder = "dmorp_multimodal"
    elif "real" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "dmorp_real"
    elif "struct" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "dmorp_struct"
    else:
        dataset_folder = "dmorp_faster"
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
    dataset = PcdPairDataset(
        data_file_list=data_file_list,
        dataset_name="dmorp",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=is_elastic_distortion,
        is_random_distortion=is_random_distortion,
        random_distortion_rate=random_distortion_rate,
        random_distortion_mag=random_distortion_mag,
        volume_augmentations_path=volume_augmentations_path,
    )
    # Split dataset
    train_size = int(cfg.MODEL.TRAIN_SPLIT * len(dataset))
    val_size = int(cfg.MODEL.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # Save test dataset to pickle
    test_dataset_file = os.path.join(
        root_path,
        "test_data",
        "dmorp_struct",
        f"diffusion_dataset_test_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}.pkl",
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

    pose_transformer = StructTransformer(**net_init_args)
    structmorp_model = StructmorpModel(cfg, pose_transformer)

    model_name = "Structmorp_model" 
    model_name += f"_pod{net_init_args.pcd_output_dim}" 
    model_name += f"_na{net_init_args.num_attention_heads}"
    model_name += f"_ehd{net_init_args.encoder_hidden_dim}"
    noise_net_name = cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", noise_net_name)
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    structmorp_model.train(
        num_epochs=cfg.TRAIN.NUM_EPOCHS,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        save_path=save_path,
    )
