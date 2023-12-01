from __future__ import annotations
import os
import cv2
import numpy as np
from vil2.env.obj_sim.obj_movement import ObjSim
from detectron2.config import LazyConfig
import pickle
from tqdm.auto import tqdm


def collect_data_sim_obj(
    env_name: str,
    env,
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
                image, depth_image = env.render(return_image=True)
                cv2.imwrite(os.path.join(epoch_path, f"{env._t}.png"), image)
                with open(os.path.join(epoch_path, f"{env._t}.pkl"), "wb") as f:
                    pickle.dump(obs, f)
            count += 1


if __name__ == "__main__":
    # prepare path
    root_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_name = "ObjSim"
    export_path = os.path.join(root_path, "test_data", env_name)
    check_point_path = os.path.join(export_path, 'oqdp', 'checkpoint')
    log_path = os.path.join(export_path, 'oqdp', 'log')
    # remove old data
    if os.path.exists(export_path):
        os.system(f"rm -r {export_path}")
    os.makedirs(export_path, exist_ok=True)
    os.makedirs(check_point_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    cfg = LazyConfig.load(os.path.join(root_path, "config", "obj_sim.py"))
    env = ObjSim(cfg=cfg)

    # generate a random trajectory
    action_trajecotry = dict()
    for i in range(10):
        random_action = np.random.uniform(-1.0, 1.0, size=7) * 0.1
        random_action[3:] = random_action[3:] / np.linalg.norm(random_action[3:])  # normalize quaternion
        action = {
            0: random_action,
        }
        action_trajecotry[i] = action

    # collect data
    collect_data_sim_obj(
        env_name=env_name,
        env=env,
        num_eposides=10,
        max_steps=10,
        obs_noise_level=0.0,
        action_trajecotry=action_trajecotry,
        output_path=export_path,
    )
