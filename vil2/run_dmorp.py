"""Run Diffusion Model for Object Relative Pose Generation"""
import os
import torch
import pickle
import json
import numpy as np
import collections
from tqdm.auto import tqdm
from vil2.env import env_builder
from vil2.model.dmorp_model import DmorpModel
from vil2.model.net_factory import build_vision_encoder, build_noise_pred_net
from vil2.data.obj_dp_dataset import normalize_data, unnormalize_data
from vil2.data.dmorp_dataset import DmorpDataset
import vil2.utils.misc_utils as utils
from vil2.env.obj_sim.obj_movement import ObjSim
from detectron2.config import LazyConfig, instantiate
import vil2.utils.eval_utils as eval_utils


if __name__ == "__main__":
    # Load config
    task_name = "Dmorp"
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", "dmorp_simplify.py")
    cfg = LazyConfig.load(cfg_file)
    retrain = False
    # Load dataset & data loader
    dataset = DmorpDataset(
        dataset_path=f"{root_path}/test_data/Dmorp/raw_data/dmorp_data.zarr",
        max_scene_size=cfg.MODEL.MAX_SCENE_SIZE,
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
    dmorp_model = DmorpModel(
        cfg,
        vision_encoder=None,
        noise_pred_net=build_noise_pred_net(
            cfg.MODEL.NOISE_NET.NAME, **noise_net_init_args
        ),
    )

    if retrain:
        dmorp_model.train(num_epochs=cfg.TRAIN.NUM_EPOCHS, data_loader=data_loader)
        # save the data
        save_dir = os.path.join(root_path, "test_data", "Dmorp", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "dmorp_model.pt")
        torch.save(dmorp_model.nets.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    else:
        # load the model
        save_dir = os.path.join(root_path, "test_data", "Dmorp", "checkpoints")
        save_path = os.path.join(save_dir, "dmorp_model.pt")
        dmorp_model.nets.load_state_dict(torch.load(save_path))
        print(f"Model loaded from {save_path}")

    # Test inference
    # Load vocabulary
    vocab_path = os.path.join(root_path, "test_data", "Dmorp", "raw_data", "vocab.json")
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_names = list(vocab.keys())
    for i in range(1):
        # Gt sem_feat distribution
        gt_sem_feat = dataset.normalized_train_data["sem_feat"]

        # Sample a scene query
        batch_size = 1000
        num_query_obj = 4
        objs_query = np.random.choice(
            vocab_names, size=num_query_obj, replace=False
        )
        obs = dict()
        obs["sem_feat"] = np.zeros((num_query_obj, cfg.MODEL.SEMANTIC_FEAT_DIM))
        obs["geo_feat"] = np.zeros((num_query_obj, cfg.MODEL.GEOMETRY_FEAT_DIM))
        pred = dmorp_model.inference(obs, stats=stats, batch_size=batch_size, init_noise=None, draw_sample=False)
        pred_vocab_names, pred_vocab_ids = dmorp_model.parse_vocab(pred["sem_feat"],  vocab_names)

        # Checking the distribution of poses
        if cfg.MODEL.RECON_POSE:
            gt_pose = dataset.normalized_train_data["pose"]
            gt_pose = gt_pose.reshape(-1, cfg.MODEL.POSE_DIM)
            gt_pose = unnormalize_data(gt_pose, stats=stats["pose"])
            eval_utils.compare_distribution(gt_pose, pred["pose"].reshape(-1, cfg.MODEL.POSE_DIM), dim_start=0, dim_end=3, title="Pose")

        # Checking the distribution of data stamp
        # Build batch data first
        if cfg.MODEL.RECON_DATA_STAMP:
            nbatch = {}
            nbatch["sem_feat"] = torch.from_numpy(dataset.normalized_train_data["sem_feat"]).reshape(-1,
                                                                                                     cfg.MODEL.MAX_SCENE_SIZE, cfg.MODEL.SEMANTIC_FEAT_DIM)
            nbatch["pose"] = torch.from_numpy(dataset.normalized_train_data["pose"]).reshape(-1, cfg.MODEL.MAX_SCENE_SIZE, cfg.MODEL.POSE_DIM)
            naction = dmorp_model.assemble_action(nbatch, B=1000, M=cfg.MODEL.MAX_SCENE_SIZE)
            gt_data_stamp = naction[:, :dmorp_model.data_stamp_dim].detach().cpu().numpy().reshape(-1, dmorp_model.data_stamp_dim)
            pred_data_stamp = pred["data_stamp"].reshape(-1, dmorp_model.data_stamp_dim)
            eval_utils.compare_distribution(gt_data_stamp, pred_data_stamp, dim_start=16, dim_end=23, title="Data Stamp")

        # Check if the relative pose is correct
        pass
    # DEBUG TRAIN
    # dmorp_model.debug_train(data_loader=data_loader)
