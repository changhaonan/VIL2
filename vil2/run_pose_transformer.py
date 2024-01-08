"""Run pose transformer."""
import os
import torch
import pickle
import argparse
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
    argparser.add_argument("--random_index", type=int, default=0)
    args = argparser.parse_args()
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
    # Load dataset & data loader
    dataset_file = os.path.join(
        root_path, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}.pkl"
    )
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)
        data_size = len(dataset)
    # Select 0.2 for validation
    train_size = int(data_size * 0.8)
    val_size = data_size - train_size
    val_indices = []
    train_indices = []
    val_scenes = random.sample(range(1000), 200)
    for s in val_scenes:
        val_indices += range(s * 20, (s+1) * 20)
    train_scenes = list(set(range(1000)) - set(val_scenes))
    for s in train_scenes:
        train_indices += range(s * 20, (s+1) * 20)

    for el in range(data_size, 1000 * 20):
        if el in val_indices:
            val_indices.remove(el)
        if el in train_indices:    
            train_indices.remove(el)
    
    # val_indices = list(range(args.random_index * val_size, (args.random_index + 1) * val_size))
    # train_indices = list(set(range(data_size)) - set(val_indices))
    volume_augmentations_path=os.path.join(root_path, "config", volume_augmentation_file) if volume_augmentation_file is not None else None
    train_dataset = PointCloudDataset(
        data_file=dataset_file,
        dataset_name="dmorp",
        indices=train_indices,
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=is_elastic_distortion,
        is_random_distortion=is_random_distortion,
        random_distortion_rate=random_distortion_rate,
        random_distortion_mag=random_distortion_mag,
        volume_augmentations_path=volume_augmentations_path
    )
    train_dataset.set_mode("train")
    
    val_dataset = PointCloudDataset(
        data_file=dataset_file,
        dataset_name="dmorp",
        indices=val_indices,
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=is_elastic_distortion,
        is_random_distortion=is_random_distortion,
        random_distortion_rate=random_distortion_rate,
        random_distortion_mag=random_distortion_mag,
        volume_augmentations_path=volume_augmentations_path)
    val_dataset.set_mode("val")

    # # Use cached dataset if available
    # if os.path.exists(
    #     os.path.join(
    #         root_path,
    #         "test_data",
    #         "dmorp_augmented",
    #         f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_train.pkl",
    #     )
    # ):
    #     print("Loading cached dataset....")
    #     with open(
    #         os.path.join(
    #             root_path,
    #             "test_data",
    #             "dmorp_augmented",
    #             f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_train.pkl",
    #         ),
    #         "rb",
    #     ) as f:
    #         train_dataset = pickle.load(f)
    #     with open(
    #         os.path.join(
    #             root_path,
    #             "test_data",
    #             "dmorp_augmented",
    #             f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_val.pkl",
    #         ),
    #         "rb",
    #     ) as f:
    #         val_dataset = pickle.load(f)
    # else:
    #     print("Caching dataset....")
    #     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    #     with open(
    #         os.path.join(
    #             root_path,
    #             "test_data",
    #             "dmorp_augmented",
    #             f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_train.pkl",
    #         ),
    #         "wb",
    #     ) as f:
    #         pickle.dump(train_dataset, f)
    #     with open(
    #         os.path.join(
    #             root_path,
    #             "test_data",
    #             "dmorp_augmented",
    #             f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_val.pkl",
    #         ),
    #         "wb",
    #     ) as f:
    #         pickle.dump(val_dataset, f)

    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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

    pose_transformer = PoseTransformer(**net_init_args)
    tmorp_model = TmorpModel(cfg, pose_transformer)

    model_name = "tmorp_model"
    noise_net_name = cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", noise_net_name)
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    tmorp_model.train(
        num_epochs=cfg.TRAIN.NUM_EPOCHS,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        save_path=save_path,
    )
