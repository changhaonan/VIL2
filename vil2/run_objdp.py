"""Run Object-wise Diffusion policy"""
import os
import torch
import pickle
import numpy as np
import collections
from tqdm.auto import tqdm
from vil2.env import env_builder
from vil2.model.objdp_model import ObjDPModel
from vil2.model.net_factory import build_vision_encoder, build_noise_pred_net
from vil2.data.obj_dp_dataset import ObjDPDataset, normalize_data, unnormalize_data, parser_obs
import vil2.utils.misc_utils as utils
from vil2.env.obj_sim.obj_movement import ObjSim
from detectron2.config import LazyConfig, instantiate
import vil2.utils.eval_utils as eval_utils


if __name__ == "__main__":
    # load env
    task_name = "objdp"
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", "obj_sim_simplify.py")
    cfg = LazyConfig.load(cfg_file)

    retrain = False
    # load dataset & data loader
    dataset = ObjDPDataset(
        dataset_path=f"{root_path}/test_data/ObjSim/obj_dp_dataset.zarr",
        obs_horizon=cfg.MODEL.OBS_HORIZON,
        action_horizon=cfg.MODEL.ACTION_HORIZON,
        pred_horizon=cfg.MODEL.PRED_HORIZON,
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

    # start training
    # compute network input/output dimension
    noise_net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS
    input_dim = 0
    global_cond_dim = 0
    # condition related
    cond_geometry_feature = cfg.MODEL.COND_GEOMETRY_FEATURE
    cond_voxel_center = cfg.MODEL.COND_VOXEL_CENTER
    # horizon related
    obs_horizon = cfg.MODEL.OBS_HORIZON
    # i/o related
    recon_time_stamp = cfg.MODEL.RECON_TIME_STAMP
    recon_data_stamp = cfg.MODEL.RECON_DATA_STAMP
    recon_voxel_center = cfg.MODEL.RECON_VOXEL_CENTER
    embedding_dim = cfg.MODEL.TIME_EMB_DIM if cfg.MODEL.USE_POSITIONAL_EMBEDDING else 1
    if recon_time_stamp:
        input_dim += embedding_dim
    if recon_data_stamp:
        input_dim += embedding_dim
    if recon_voxel_center:
        input_dim += 3
    if cond_geometry_feature:
        global_cond_dim += cfg.MODEL.GEOMETRY_FEAT_DIM
    if cond_voxel_center:
        global_cond_dim += 3
    global_cond_dim = global_cond_dim * obs_horizon
    noise_net_init_args["input_dim"] = input_dim
    noise_net_init_args["global_cond_dim"] = global_cond_dim
    objdp_model = ObjDPModel(
        cfg,
        vision_encoder=None,
        noise_pred_net=build_noise_pred_net(
            cfg.MODEL.NOISE_NET.NAME, **noise_net_init_args
        ),
    )

    if retrain:
        objdp_model.train(num_epochs=cfg.TRAIN.NUM_EPOCHS, data_loader=data_loader)

        # save the data
        save_dir = os.path.join(root_path, "test_data", "ObjSim", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "objdp_model.pt")
        torch.save(objdp_model.nets.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    else:
        # load the model
        save_dir = os.path.join(root_path, "test_data", "ObjSim", "checkpoints")
        save_path = os.path.join(save_dir, "objdp_model.pt")
        objdp_model.nets.load_state_dict(torch.load(save_path))
        print(f"Model loaded from {save_path}")
    # do inference
    env = ObjSim(cfg=cfg)
    obs_deque = collections.deque(maxlen=cfg.MODEL.OBS_HORIZON)

    # load the action used for training
    action_trajectory_file = os.path.join(root_path, "test_data", "ObjSim", "raw_data", "action_trajectory.pkl")
    with open(action_trajectory_file, "rb") as f:
        action_trajectory = pickle.load(f)

    for i in range(100):
        env.reset()
        obs_deque.clear()
        while True:
            action = action_trajectory[env._t % len(action_trajectory)]
            obs, reward, done, info = env.step(action)
            if done:
                break
            # parse obs
            t, img, depth, active_obj_super_voxel_pose, active_obj_geometry_feat, active_obj_voxel_center, data_stamp = parser_obs(
                obs, dataset.carrier_type, dataset.geometry_encoder, dataset.aggretator
            )
            obs = {
                "t": t,
                "img": img,
                "depth": depth,
                "obj_super_voxel_pose": active_obj_super_voxel_pose,
                "obj_voxel_feat": active_obj_geometry_feat,
                "obj_voxel_center": active_obj_voxel_center,
                "data_stamp": data_stamp,
            }
            obs_deque.append(obs)
            batch_size = 1
            if len(obs_deque) == cfg.MODEL.OBS_HORIZON:
                pred = objdp_model.inference(obs_deque, stats=stats, batch_size=batch_size)
                # visualize prediction
                check_horizon = 1
                # pred_voxel_poses = pred[0, :, :check_horizon, 1:4]
                pred_voxel_time_stamps = pred["t"]
                pred_voxel_poses = pred["obj_voxel_center"]
                pred_voxel_data_stamps = pred["data_stamp"]
                # env.render(return_image=False, pred_voxel_poses=pred_voxel_poses[0, :, :check_horizon, :])
                eval_utils.draw_pose_distribution(
                    poses=pred_voxel_poses.reshape(-1, 3),
                    color=pred_voxel_time_stamps[..., 0].reshape(-1),
                    scale=1.0,
                    xyz_range=np.array([0.0, 10.0, 0.0, 10.0, -5, 5]),
                )
                eval_utils.draw_pose_distribution(
                    poses=pred_voxel_poses.reshape(-1, 3),
                    color=pred_voxel_data_stamps[..., 0].reshape(-1),
                    scale=1.0,
                    xyz_range=np.array([0.0, 10.0, 0.0, 10.0, -5, 5]),
                )
                pass
