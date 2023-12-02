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


if __name__ == "__main__":
    # load env
    task_name = "objdp"
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", "obj_sim.py")
    cfg = LazyConfig.load(cfg_file)

    retrain = True
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
    objdp_model = ObjDPModel(
        cfg,
        vision_encoder=None,
        noise_pred_net=build_noise_pred_net(
            cfg.MODEL.NOISE_NET.NAME, **cfg.MODEL.NOISE_NET.INIT_ARGS
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
            img, depth, active_obj_super_voxel_pose, active_obj_geometry_feat, active_obj_voxel_center = parser_obs(
                obs, dataset.carrier_type, dataset.geometry_encoder, dataset.aggretator
            )
            obs = {
                "img": img,
                "depth": depth,
                "obj_super_voxel_pose": active_obj_super_voxel_pose,
                "obj_voxel_feat": active_obj_geometry_feat,
                "obj_voxel_center": active_obj_voxel_center,
            }
            obs_deque.append(obs)
            if len(obs_deque) == cfg.MODEL.OBS_HORIZON:
                pred = objdp_model.inference(obs_deque, stats=stats)
                # visualize prediction
                check_horizon = 1
                env.render(return_image=False, pred_voxel_poses=pred[0, :, :check_horizon, :])
