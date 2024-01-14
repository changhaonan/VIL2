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
    # Load dataset & data loader
    data_id_list = [0, 1]
    if cfg.ENV.GOAL_TYPE == "multimodal":
        dataset_folder = "dmorp_multimodal"
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
    )
    # Split dataset
    train_size = int(cfg.MODEL.TRAIN_TEST_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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

    model_name = "Tmorp_model" 
    model_name += f"_pod{net_init_args.pcd_output_dim}" 
    model_name += f"_na{net_init_args.num_attention_heads}"
    model_name += f"_ehd{net_init_args.encoder_hidden_dim}"
    model_name += f"_fpd{net_init_args.fusion_projection_dim}"
    pp_str = ""
    for points in net_init_args.points_pyramid:
        pp_str += str(points) + "_"
    model_name += f"_pp{pp_str}"
    model_name += f"usl{net_init_args.use_semantic_label}"

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
