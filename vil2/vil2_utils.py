import os
from detectron2.config import LazyConfig
from vil2.data.pcd_dataset import PcdPairDataset
from vil2.model.network.pose_transformer_noise import PoseTransformerNoiseNet
from vil2.model.dmorp_model_v2 import DmorpModel


def build_dmorp_dataset(root_path, cfg):
    pcd_size = cfg.MODEL.PCD_SIZE
    is_elastic_distortion = cfg.DATALOADER.AUGMENTATION.IS_ELASTIC_DISTORTION
    is_random_distortion = cfg.DATALOADER.AUGMENTATION.IS_RANDOM_DISTORTION
    random_distortion_rate = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_RATE
    random_distortion_mag = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_MAG
    volume_augmentation_file = cfg.DATALOADER.AUGMENTATION.VOLUME_AUGMENTATION_FILE
    crop_pcd = cfg.DATALOADER.AUGMENTATION.CROP_PCD
    crop_size = cfg.DATALOADER.AUGMENTATION.CROP_SIZE
    crop_noise = cfg.DATALOADER.AUGMENTATION.CROP_NOISE
    noise_level = cfg.DATALOADER.AUGMENTATION.NOISE_LEVEL
    rot_axis = cfg.DATALOADER.AUGMENTATION.ROT_AXIS
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
    print("Data loaded from: ", data_file_dict)
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
        crop_pcd=crop_pcd,
        crop_size=crop_size,
        crop_noise=crop_noise,
        noise_level=noise_level,
        rot_axis=rot_axis,
    )
    val_dataset = PcdPairDataset(
        data_file_list=[data_file_dict["val"]],
        dataset_name="dmorp",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=is_elastic_distortion,
        is_random_distortion=is_random_distortion,
        random_distortion_rate=random_distortion_rate,
        random_distortion_mag=random_distortion_mag,
        volume_augmentations_path=volume_augmentations_path,
        crop_pcd=crop_pcd,
        crop_size=crop_size,
        crop_noise=crop_noise,
        noise_level=noise_level,
        rot_axis=rot_axis,
    )
    test_dataset = PcdPairDataset(
        data_file_list=[data_file_dict["test"]],
        dataset_name="dmorp",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=True,
        is_random_distortion=True,
        random_distortion_rate=random_distortion_rate,
        random_distortion_mag=random_distortion_mag,
        volume_augmentations_path=volume_augmentations_path,
        crop_pcd=True,
        crop_size=crop_size,
        crop_noise=crop_noise,
        noise_level=noise_level,
        rot_axis=rot_axis,
    )
    return train_dataset, val_dataset, test_dataset


def build_dmorp_model(cfg):
    net_name = cfg.MODEL.NOISE_NET.NAME
    net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
    net_init_args["max_timestep"] = cfg.MODEL.NUM_DIFFUSION_ITERS

    pose_transformer = PoseTransformerNoiseNet(**net_init_args)
    dmorp_model = DmorpModel(cfg, pose_transformer)
    return dmorp_model
