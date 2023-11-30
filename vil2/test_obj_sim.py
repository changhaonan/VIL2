
import os
import cv2
import numpy as np
from vil2.env.obj_sim.obj_movement import ObjSim
from detectron2.config import LazyConfig


if __name__ == "__main__":
    # prepare path
    root_path = os.path.dirname((os.path.abspath(__file__)))
    env_name = "GYM-PointMaze_UMaze-v3"
    export_path = os.path.join(root_path, "test_data", env_name)
    check_point_path = os.path.join(export_path, 'oqdp', 'checkpoint')
    log_path = os.path.join(export_path, 'oqdp', 'log')
    os.makedirs(export_path, exist_ok=True)
    os.makedirs(check_point_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    cfg = LazyConfig.load(os.path.join(root_path, "config", "obj_sim.py"))

    env = ObjSim(cfg=cfg)

    for i in range(100):
        action = {
            0: np.array([0.0, 0.0, 0.5 * np.sin(i * 0.1), 0.0, 0.0, 0.0, 1.0]),
        }
        obs, reward, done, info = env.step(action)
        env.render(show_super_patch=True)
        print(obs)
