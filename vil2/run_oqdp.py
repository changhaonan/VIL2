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
        'num_epochs': 600,
        'eval_period': 1000,
        'q_hidden_dim': 256,
        'v_hidden_dim': 256,
        'tau': 0.7,  # iql related
        'gamma': 0.99,  # reward discount
        'alpha': 0.1,
        'enable_save': True,
        'enable_load': False,
        'log_path': log_path,
        'render_eval': True,  # render evaluation
        'lamda': 10.0,  # regularization parameter
        'policy_std': 0.1,
        'sample_size': 32,
        'num_diffusion_iters': 5,
        'a_lazy_init': True,  # use last a_0 to init current a_T
    }

    oqdp = OQDP(env=env, config=config)

    # do train
    oqdp.train(batch_size=config['batch_size'],
               num_epochs=config['num_epochs'],)

    oqdp.save(os.path.join(check_point_path, 'model.pt'))
