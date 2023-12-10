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
from box import Box
from vil2.env import env_builder
from vil2.model.dmorp_model import DmorpModel
from vil2.model.net_factory import build_vision_encoder, build_noise_pred_net
from vil2.data.obj_dp_dataset import normalize_data, unnormalize_data
from vil2.data.dmorp_dataset import DmorpDataset
import vil2.utils.misc_utils as utils
from vil2.env.obj_sim.obj_movement import ObjSim
from detectron2.config import LazyConfig, instantiate
import vil2.utils.eval_utils as eval_utils
from vil2.run_dmorp import run_dmorp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_param_list", type=str, default="NUM_DIFFUSION_ITERS,SEMANTIC_FEAT_DIM,GEOMETRY_FEAT_DIM,NUM_EPOCHS")
    parser.add_argument("--index_list", type=str, default="1,6,4,1")
    parser.add_argument("--cuda_device", type=str, default="cuda:0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load config
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", "dmorp_simplify.py")
    cfg = LazyConfig.load(cfg_file)
    retrain = True

    param_list = args.grid_param_list.split(",")
    main_param = param_list[0]
    index_list = args.index_list.split(",")
    index_grid = dict()

    for ky in cfg.PARAM_GRID.keys():
        index_grid[ky] = 0
    for search_param, param_index in zip(param_list, index_list):
        for ky in cfg.PARAM_GRID.keys():
            if ky == search_param:
                index_grid[ky] = int(param_index)
                break

    export_cfg = dict()
    export_cfg["ENV"] = dict(
        NUM_OBJ=8,
        NUM_STRUCTURE=4,
        SEMANTIC_FEAT_DIM=cfg.PARAM_GRID['SEMANTIC_FEAT_DIM'][index_grid['SEMANTIC_FEAT_DIM']],
        SEMANTIC_FEAT_TYPE=cfg.PARAM_GRID["SEMANTIC_FEAT_TYPE"][index_grid["SEMANTIC_FEAT_TYPE"]]
    )
    export_cfg["DATALOADER"] = dict(
        BATCH_SIZE=cfg.PARAM_GRID['BATCH_SIZE'][index_grid['BATCH_SIZE']],
        NUM_WORKERS=cfg.PARAM_GRID['NUM_WORKERS'][index_grid['NUM_WORKERS']],
    )
    export_cfg["MODEL"] = dict(
        MAX_SCENE_SIZE=cfg.PARAM_GRID['MAX_SCENE_SIZE'][index_grid['MAX_SCENE_SIZE']],
        ACTION_DIM=cfg.PARAM_GRID['ACTION_DIM'][index_grid['ACTION_DIM']],
        POSE_DIM=cfg.PARAM_GRID['POSE_DIM'][index_grid['POSE_DIM']],
        GEOMETRY_FEAT_DIM=cfg.PARAM_GRID['GEOMETRY_FEAT_DIM'][index_grid['GEOMETRY_FEAT_DIM']],
        SEMANTIC_FEAT_DIM=cfg.PARAM_GRID['SEMANTIC_FEAT_DIM'][index_grid['SEMANTIC_FEAT_DIM']],
        NUM_DIFFUSION_ITERS=cfg.PARAM_GRID['NUM_DIFFUSION_ITERS'][index_grid['NUM_DIFFUSION_ITERS']],
        VISION_ENCODER=dict(
            NAME="resnet18",
            PRETRAINED=True,
        ),
        NOISE_NET=dict(
            NAME="MLP",
            INIT_ARGS=dict(
                input_dim=3+1,
                global_cond_dim=262,  # (128 (dim_feat) + 3 (pos)) * 2 (obs_horizon)
                diffusion_step_embed_dim=256,
                down_dims=[1024, 2048, 2048],
            ),
        ),
        RECON_DATA_STAMP=cfg.PARAM_GRID['RECON_DATA_STAMP'][index_grid['RECON_DATA_STAMP']],
        RECON_SEMANTIC_FEATURE=cfg.PARAM_GRID['RECON_SEMANTIC_FEATURE'][index_grid['RECON_SEMANTIC_FEATURE']],
        RECON_POSE=cfg.PARAM_GRID['RECON_POSE'][index_grid['RECON_POSE']],
        COND_GEOMETRY_FEATURE=cfg.PARAM_GRID['COND_GEOMETRY_FEATURE'][index_grid['COND_GEOMETRY_FEATURE']],
        COND_SEMANTIC_FEATURE=cfg.PARAM_GRID['COND_SEMANTIC_FEATURE'][index_grid['COND_SEMANTIC_FEATURE']],
        GUIDE_DATA_CONSISTENCY=cfg.PARAM_GRID['GUIDE_DATA_CONSISTENCY'][index_grid['GUIDE_DATA_CONSISTENCY']],
        GUIDE_SEMANTIC_CONSISTENCY=cfg.PARAM_GRID['GUIDE_SEMANTIC_CONSISTENCY'][index_grid['GUIDE_SEMANTIC_CONSISTENCY']],
        USE_POSITIONAL_EMBEDDING=cfg.PARAM_GRID['USE_POSITIONAL_EMBEDDING'][index_grid['USE_POSITIONAL_EMBEDDING']],
        TIME_EMB_DIM=cfg.PARAM_GRID['TIME_EMB_DIM'][index_grid['TIME_EMB_DIM']],
        SEMANTIC_FEAT_TYPE=cfg.PARAM_GRID["SEMANTIC_FEAT_TYPE"][index_grid["SEMANTIC_FEAT_TYPE"]],
    )
    export_cfg["TRAIN"] = dict(
        NUM_EPOCHS=cfg.PARAM_GRID['NUM_EPOCHS'][index_grid['NUM_EPOCHS']],
    )
    export_cfg["CUDA_DEVICE"] = args.cuda_device
    param_shortcuts = dict(
        NUM_EPOCHS = "e",
        NUM_DIFFUSION_ITERS = "iter",
        BATCH_SIZE = "bs",
        SEMANTIC_FEAT_DIM = "sem",
        GEOMETRY_FEAT_DIM = "geo",
        USE_POSITIONAL_EMBEDDING = "pos",
        RECON_DATA_STAMP = "data",
        RECON_SEMANTIC_FEATURE = "rec_sem",
        RECON_POSE = "rec_pose",
        COND_GEOMETRY_FEATURE = "cond_geo",
        COND_SEMANTIC_FEATURE = "cond_sem",
        GUIDE_DATA_CONSISTENCY = "guide_data",
        GUIDE_SEMANTIC_CONSISTENCY = "guide_sem",
        ACTION_DIM = "act",
        POSE_DIM = "pose",
        MAX_SCENE_SIZE = "m_sc_size",
        SEMANTIC_FEAT_TYPE = "sem_type",
        NUM_WORKERS = "num_workers",
        TIME_EMB_DIM = "time",
    )

    subdir_child = ""
    for ky in cfg.PARAM_GRID.keys():
        if ky != main_param:
            subdir_child += f"{param_shortcuts[ky]}({cfg.PARAM_GRID[ky][index_grid[ky]]})_"
    save_str = f"{param_shortcuts[main_param]}{cfg.PARAM_GRID[main_param][index_grid[main_param]]}"
    
    if args.debug:
        print("save_subsubdir:", subdir_child)
        print("save_str:", save_str)

    export_cfg = Box(export_cfg)
    run_dmorp(root_path=root_path, cfg=export_cfg, retrain=True, task_name="Dmorp", save_subdir=main_param, subdir_child=subdir_child, save_str=save_str)
