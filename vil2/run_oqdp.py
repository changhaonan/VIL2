"""Run Online Q-value Diffusion Policy"""
import os
import cv2
import numpy as np
from vil2.env import env_builder
from vil2.algo.oqdp import OQDP
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
    cfg = LazyConfig.load(os.path.join(root_path, "config", "mini_grid_lm.py"))

    # prepare data
    env = env_builder(env_name, render_mode="rgb_array", cfg=cfg.ENV)

    config = {
        'batch_size': 512,
        'value_epochs': 1000,
        'eval_period': 1000,
        'q_hidden_dim': 256,
        'v_hidden_dim': 256,
        'policy_hidden_dim': 256,
        'tau': 0.7,
        'beta': 10.0,
        'enable_save': True,
        'enable_load': False,
        'log_path': log_path,
        "action_deterministic": False,
        'render_eval': True,  # render evaluation
    }

    oqdp = OQDP(env=env, config=config)

    # do train
    oqdp.train(batch_size=config['batch_size'],
               num_epochs=config['value_epochs'],)

    oqdp.save(check_point_path)
