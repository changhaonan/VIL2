import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vil2.env import ENV_MAP
from stable_baselines3 import PPO


def collect_action_d(env, model, enable_vis=False):
    """Collect the action distribution"""
    obs = env.reset()
    action_d = {}
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        obs_id = env.envs[0].decode_obs(obs[0])
        if obs_id not in action_d:
            action_d[obs_id] = []
        action_d[obs_id].append(action[0])
        if enable_vis:
            images = env.get_images()
            cv2.imshow("image", images[0])
            cv2.waitKey(1)
        if terminated:
            obs = env.reset()
    return action_d


if __name__ == "__main__":
    root_path = os.path.dirname((os.path.abspath(__file__)))
    env_name = "maze"
    export_path = os.path.join(root_path, "test_data", env_name)
    
    config = {
        "seed": 0,
        "num_level": 5, 
        "num_branch": 2,
        "num_goal": 3,
        "end_probs": [0.0, 0.0, 0.0, 0.3, 0.3, 0.0],
        "noise_level": 0.0,
    }
    env = ENV_MAP[env_name](config=config)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=35000)

    vec_env = model.get_env()
    action_d = collect_action_d(vec_env, model, enable_vis=False)
    # draw distribution
    for obs_id in action_d:
        actions = action_d[obs_id]
        # draw histogram
        fig = plt.figure(figsize=(8, 8))
        plt.hist(actions, bins=100)
        plt.title(f"Obs ID: {obs_id}")
        plt.xlabel("Action")
        plt.ylabel("Count")
        # save figure
        action_d_path = os.path.join(export_path, "action_d")
        os.makedirs(action_d_path, exist_ok=True)
        image_path = os.path.join(action_d_path, f"{obs_id}.png")
        plt.savefig(image_path)

