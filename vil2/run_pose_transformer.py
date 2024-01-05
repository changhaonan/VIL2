"""Run pose transformer."""
import os
import torch
import pickle
import argparse
import copy
from vil2.data_gen.preprocess_data import DiffDataset
from vil2.model.network.pose_transformer import PoseTransformer
from vil2.model.tmorp_model import TmorpModel
from detectron2.config import LazyConfig
from torch.utils.data.dataset import random_split


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
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
    # Load dataset & data loader
    with open(
        os.path.join(
            root_path, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}.pkl"
        ),
        "rb",
    ) as f:
        dtset = pickle.load(f)
    dataset = DiffDataset(dtset=dtset)
    train_size = int(cfg.MODEL.TRAIN_TEST_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    if os.path.exists(
        os.path.join(
            root_path,
            "test_data",
            "dmorp_augmented",
            f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_train.pkl",
        )
    ):
        print("Loading cached dataset....")
        with open(
            os.path.join(
                root_path,
                "test_data",
                "dmorp_augmented",
                f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_train.pkl",
            ),
            "rb",
        ) as f:
            train_dataset = pickle.load(f)
        with open(
            os.path.join(
                root_path,
                "test_data",
                "dmorp_augmented",
                f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_val.pkl",
            ),
            "rb",
        ) as f:
            val_dataset = pickle.load(f)
    else:
        print("Caching dataset....")
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        with open(
            os.path.join(
                root_path,
                "test_data",
                "dmorp_augmented",
                f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_train.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(train_dataset, f)
        with open(
            os.path.join(
                root_path,
                "test_data",
                "dmorp_augmented",
                f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_val.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(val_dataset, f)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=True,
        # num_workers=23,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=False,
        # num_workers=23,
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
