from __future__ import annotations
import os
import cv2
import numpy as np
from vil2.env.obj_sim.obj_movement import ObjSim
from detectron2.config import LazyConfig
import pickle
from tqdm.auto import tqdm
import time


def collect_data_sim_obj(
    env_name: str,
    env: ObjSim,
    num_eposides: int | list[int],
    max_steps: int,
    obs_noise_level: float,
    action_trajecotry: dict[int, np.ndarray],
    output_path=None,
):
    """Collect offline data for SimObj"""
    for i in tqdm(range(num_eposides), "Collecting Sim Obj data"):
        env.reset()
        count = 0
        epoch_path = os.path.join(output_path, f"{i}")
        os.makedirs(epoch_path, exist_ok=True)
        while True:
            action = action_trajecotry[env._t % len(action_trajecotry)]
            obs, reward, done, info = env.step(action)
            if done or count > max_steps:
                break
            if epoch_path is not None:
                # image, depth_image = env.render(return_image=False)
                # cv2.imwrite(os.path.join(epoch_path, f"{env._t}.png"), image)
                with open(os.path.join(epoch_path, f"{env._t}.pkl"), "wb") as f:
                    pickle.dump(obs, f)
            count += 1


if __name__ == "__main__":
    # prepare path
    root_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_name = "ObjSim"
    export_path = os.path.join(root_path, "test_data", env_name, "raw_data")
    # remove old data
    if os.path.exists(export_path):
        os.system(f"rm -r {export_path}")
    os.makedirs(export_path, exist_ok=True)
    cfg = LazyConfig.load(os.path.join(root_path, "config", "obj_sim.py"))
    env = ObjSim(cfg=cfg)

    # generate a random trajectory
    action_trajecotry = dict()
    for i in range(12):
        random_action = np.zeros((7,), dtype=np.float32)
        random_action[:3] = np.array([0.1, 0.1, 0.0], dtype=np.float32)
        random_action[3:] = np.array([0.0, 0.0, 0.0, 1.0])  # no rotation
        action = {
            0: random_action,
        }
        action_trajecotry[i] = action

    # collect data
    collect_data_sim_obj(
        env_name=env_name,
        env=env,
        num_eposides=10,
        max_steps=100,
        obs_noise_level=0.0,
        action_trajecotry=action_trajecotry,
        output_path=export_path,
    )

    # save action trajectory
    with open(os.path.join(export_path, "action_trajectory.pkl"), "wb") as f:
        pickle.dump(action_trajecotry, f)
