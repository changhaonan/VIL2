"""Collect expert/offline data"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from vil2.env import env_builder, env_data_collect


if __name__ == "__main__":
    root_path = os.path.dirname((os.path.abspath(__file__)))
    env_name = "MiniGrid-Empty-5x5-v0"
    export_path = os.path.join(root_path, "test_data", env_name)
    os.makedirs(export_path, exist_ok=True)
    env = env_builder(env_name, render_mode="rgb_array")
    data = env_data_collect(env, env_name, num_eposides=[1000, 1000], max_steps=100, strategies=["navigate", "suboptimal"])
    np.savez(os.path.join(export_path, "offline-data.npz"), **data)
    print(f"Data saved to {export_path}")