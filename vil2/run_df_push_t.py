"""Run Diffusion policy on PushT scene"""
import os
import torch
import gdown
from vil2.env import env_builder
from vil2.model.df_model import DFModel
from vil2.model.net_factory import build_vision_encoder, build_noise_net
from vil2.data.dataset import DFImageDataset
from detectron2.config import LazyConfig, instantiate


if __name__ == "__main__":
    # load env
    task_name = "push_t"
    env = env_builder(task_name, render_mode="rgb_array")

    # build model
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", "df.py")

    cfg = LazyConfig.load(cfg_file)

    vision_encoder = build_vision_encoder(cfg.MODEL.VISION_ENCODER.NAME)
    noise_net = build_noise_net(cfg.MODEL.NOISE_NET.NAME, **cfg.MODEL.NOISE_NET.INIT_ARGS)

    df_model = DFModel(cfg, vision_encoder=vision_encoder, noise_net=noise_net)

    # load dataset
    # download demonstration data from Google Drive
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    dataset = DFImageDataset(
        dataset_path=dataset_path,
        pred_horizon=cfg.MODEL.PRED_HORIZON,
        obs_horizon=cfg.MODEL.OBS_HORIZON,
        action_horizon=cfg.MODEL.ACTION_HORIZON,
    )

    stats = dataset.stats
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    df_model.train(num_epochs=cfg.TRAIN.NUM_EPOCHS, data_loader=data_loader)
