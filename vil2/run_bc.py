"""Run IQL on the offline data"""
import os
import cv2
import numpy as np
from vil2.env import env_builder
from vil2.algo.bc import BC

if __name__ == "__main__":
    # prepare path
    root_path = os.path.dirname((os.path.abspath(__file__)))
    env_name = "MiniGrid-MM"
    export_path = os.path.join(root_path, "test_data", env_name)
    check_point_path = os.path.join(export_path, 'bc', 'checkpoint')
    log_path = os.path.join(export_path, 'bc','log')
    os.makedirs(export_path, exist_ok=True)
    os.makedirs(check_point_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # prepare data
    env = env_builder(env_name, render_mode="rgb_array")
    dataset = np.load(os.path.join(export_path, "offline-data.npz"), allow_pickle=True)
    
    config = {
        'batch_size': 512,
        'value_epochs': 50000,
        'eval_period': 1000,
        'q_hidden_dim': 256,
        'v_hidden_dim': 256,
        'policy_hidden_dim': 256,
        'tau': 0.7, 
        'beta': 3.0,
        'enable_save': True,
        'enable_load': False,
        'log_path': log_path,
        'render_eval': False, # render evaluation
        "action_deterministic": False,
    }

    bc = BC(env=env, dataset=dataset, config=config)
    # test
    if config['enable_load']:
        bc.load(os.path.join(check_point_path, 'bc.pth'))
    else:
        bc.train(env=env, batch_size=config['batch_size'], num_epochs=config['value_epochs'], eval_period=config['eval_period'])
        # save model
        if config['enable_save']:
            bc.save(os.path.join(check_point_path, 'bc.pth'))

