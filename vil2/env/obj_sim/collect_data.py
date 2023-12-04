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
    action_trajecotry_list: dict[int, np.ndarray],
    output_path=None,
):
    """Collect offline data for SimObj"""
    for i in tqdm(range(num_eposides), "Collecting Sim Obj data"):
        action_trajecotry = action_trajecotry_list[i // (num_eposides // 2)]  # first half of the data is trajectory 1, second half is trajectory 2
        env.reset()
        count = 0
        epoch_path = os.path.join(output_path, f"{i}")
        os.makedirs(epoch_path, exist_ok=True)
        while True:
            action = action_trajecotry[env._t % len(action_trajecotry)]
            obs, reward, done, info = env.step(action, noise=obs_noise_level)
            if done or count > max_steps:
                break
            if epoch_path is not None:
                # image, depth_image = env.render(return_image=False)
                # cv2.imwrite(os.path.join(epoch_path, f"{env._t}.png"), image)
                with open(os.path.join(epoch_path, f"{env._t}.pkl"), "wb") as f:
                    obs["data_stamp"] = i
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
    cfg = LazyConfig.load(os.path.join(root_path, "config", "obj_sim_simplify.py"))
    env = ObjSim(cfg=cfg)

    # generate a random trajectory
    action_trajecotry_list = [{}, {}]
    # trajectory 1
    for i in range(100):
        random_action = np.zeros((6,), dtype=np.float32)
        random_action[:3] = np.array([1*np.log(i + 10), 2*np.log(i + 1), 0.0], dtype=np.float32)
        random_action[3:] = np.array([0.0, 0.0, 0.0])  # no rotation
        action = {
            0: random_action,
        }
        action_trajecotry_list[0][i] = action
    # trajectory 2
    for i in range(100):
        random_action = np.zeros((6,), dtype=np.float32)
        random_action[:3] = np.array([2*np.log(i + 10), 1*np.log(i + 1), 0.0], dtype=np.float32)
        random_action[3:] = np.array([0.0, 0.0, 0.0])  # no rotation
        action = {
            0: random_action,
        }
        action_trajecotry_list[1][i] = action

    # collect data
    collect_data_sim_obj(
        env_name=env_name,
        env=env,
        num_eposides=10,
        max_steps=100,
        obs_noise_level=0.03,
        action_trajecotry_list=action_trajecotry_list,
        output_path=export_path,
    )

    # save action trajectory
    with open(os.path.join(export_path, "action_trajectory.pkl"), "wb") as f:
        pickle.dump(action_trajecotry_list[0], f)
