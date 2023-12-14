"""Run Diffusion Model for Object Relative Pose Generation"""
import os
import torch
import pickle
import argparse
import json
import random
import numpy as np
import collections
from tqdm.auto import tqdm
from vil2.env import env_builder
from torch import nn
from vil2.data_gen.data_loader import DiffDataset, visualize_pcd_with_open3d
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vil2.model.dmorp_model import DmorpModel
from vil2.model.net_factory import build_vision_encoder, build_noise_pred_net
import vil2.utils.misc_utils as utils
from detectron2.config import LazyConfig
from vil2.model.mlp.cond_unet_mlp import ConditionalUnetMLP
from vil2.model.mlp.mlp import MLP
import vil2.utils.eval_utils as eval_utils

if __name__ == "__main__":
    # Load config
    task_name = "Dmorp"
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", "dmorp_simplify.py")
    cfg = LazyConfig.load(cfg_file)
    retrain = False
    # Load dataset & data loader
    with open(os.path.join(root_path, "test_data", "dmorp_augmented", "diffusion_dataset.pkl"), "rb") as f:
        dtset = pickle.load(f)
    dataset = DiffDataset(dtset=dtset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # Compute network input/output dimension
    noise_net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS
    input_dim = 0
    global_cond_dim = 0
    # condition related
    cond_geometry_feature = cfg.MODEL.COND_GEOMETRY_FEATURE
    cond_semantic_feature = cfg.MODEL.COND_SEMANTIC_FEATURE
    # i/o related
    recon_data_stamp = cfg.MODEL.RECON_DATA_STAMP
    recon_semantic_feature = cfg.MODEL.RECON_SEMANTIC_FEATURE
    recon_pose = cfg.MODEL.RECON_POSE
    aggregate_list = cfg.MODEL.AGGREGATE_LIST
    if recon_data_stamp:
        for agg_type in aggregate_list:
            if agg_type == "SEMANTIC":
                input_dim += cfg.MODEL.SEMANTIC_FEAT_DIM
            elif agg_type == "POSE":
                input_dim += cfg.MODEL.POSE_DIM
            else:
                raise NotImplementedError
    if recon_semantic_feature:
        input_dim += cfg.MODEL.SEMANTIC_FEAT_DIM
    if recon_pose:
        input_dim += cfg.MODEL.POSE_DIM
    if cond_geometry_feature:
        global_cond_dim += cfg.MODEL.GEOMETRY_FEAT_DIM
    if cond_semantic_feature:
        global_cond_dim += cfg.MODEL.SEMANTIC_FEAT_DIM
    noise_net_init_args["input_dim"] = input_dim
    noise_net_init_args["global_cond_dim"] = global_cond_dim
    noise_net_init_args["diffusion_step_embed_dim"] = cfg.MODEL.TIME_EMB_DIM
    dmorp_model = DmorpModel(
        cfg,
        vision_encoder=None,
        noise_pred_net=build_noise_pred_net(
            cfg.MODEL.NOISE_NET.NAME, **noise_net_init_args
        ),
        device="cuda"
    )

    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    if retrain:
        # dmorp_model.module.train(num_epochs=cfg.TRAIN.NUM_EPOCHS, data_loader=data_loader)
        best_model, best_epoch = dmorp_model.train(num_epochs=cfg.TRAIN.NUM_EPOCHS, data_loader=data_loader)
        # save the data
        save_path = os.path.join(save_dir, f"dmorp_model_rel.pt")
        torch.save(best_model, save_path)
        print(f"Saving the best epoch:{best_epoch}. Model saved to {save_path}")
    else:
        # load the model
        save_path = os.path.join(save_dir, f"dmorp_model_rel.pt")
        dmorp_model.nets.load_state_dict(torch.load(save_path))
        print(f"Model loaded from {save_path}")

    # Test inference
    # Load vocabulary
    if not retrain:
        dmorp_model.debug_inference(dataset, sample_size=750, consider_only_one_pair=True)
    pass
    