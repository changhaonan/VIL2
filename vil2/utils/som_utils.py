"""Utility functions for State Occupancy Measures (SOM)"""
import gymnasium as gym
import numpy as np
from tqdm import tqdm

def sample_som(env: gym.Env, policy_fn, num_steps: int, num_samples: int, seed: int = 0, **kwargs):
    """Sample a set of states from the environment.

    Args:
        env: Gym environment.
        policy_fn: A function that takes in a state and returns an action.
        num_samples: Number of samples to draw.
        seed: Random seed.

    Returns:
        A list of states.
    """
    env.reset(seed=seed)
    observation_list = []
    achieved_goal_list = []
    desired_goal_list = []
    for _ in tqdm(range(num_samples)):
        obs, info = env.reset()
        for i in range(num_steps):
            action = policy_fn(obs).squeeze()
            obs, _, done, truncated, info = env.step(action)
            observation_list.append(obs['observation'])
            achieved_goal_list.append(obs['achieved_goal'])
            desired_goal_list.append(obs['desired_goal'])
            if done or truncated:
                break
            # env.render()
    # Merged list of observations
    observation_list = np.vstack(observation_list)
    achieved_goal_list = np.vstack(achieved_goal_list)
    desired_goal_list = np.vstack(desired_goal_list)
    obses = {
        'observation': observation_list,
        'achieved_goal': achieved_goal_list,
        'desired_goal': desired_goal_list,
    }
    return obses
