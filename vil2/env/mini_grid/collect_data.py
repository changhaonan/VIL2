""" Generate expert data for MiniGrid with some heuristics
"""
from __future__ import annotations
import numpy as np
import tqdm
from minigrid.core.actions import Actions
from vil2.env.mini_grid.base import BaseMiniGridEnv
from vil2.env.mini_grid.multi_modality import MultiModalityMiniGridEnv


def collect_data_mini_grid(env_name: str, env: BaseMiniGridEnv, num_eposides: int|list[int], max_steps: int, strategies: list[str], random_action_prob: float = 0.0, output_path = None):
    """Collect offline data for MiniGrid"""
    data_list = []
    if isinstance(num_eposides, int):
        num_eposides = [num_eposides] * len(strategies)
    for i, strategy in enumerate(strategies):
        if strategy == "navigate":
            data_list.append(collect_navigate_data(env_name, env, num_eposides[i], max_steps))
        elif strategy == "suboptimal":
            data_list.append(collect_suboptimal_data(env_name, env, num_eposides[i], max_steps, random_action_prob=random_action_prob))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    # merge data
    data_merge = {}
    for data in data_list:
        for key, value in data.items():
            if key not in data_merge:
                data_merge[key] = []
            data_merge[key].append(value)
    # stack
    for key, value in data_merge.items():
        data_merge[key] = np.vstack(value)
    if output_path is not None:
        np.savez(output_path, **data_merge)
    return data_merge


def collect_navigate_data(env_name: str, env: BaseMiniGridEnv, num_eposides: int, max_steps: int):
    """Collect random navigation data"""
    obs_list = []
    next_obs_list = []
    reward_list = []
    action_list = []
    terminated_list = []
    truncated_list = []
    print("Collecting navigation data")
    for i in tqdm.tqdm(range(num_eposides)):
        obs, _ = env.reset(seed=i)
        for j in range(max_steps):
            # random explore
            action = env.action_space.sample()
            # collect data
            obs_list.append(obs[None, :])
            obs, reward, terminated, truncated, info = env.step(action)
            # collect data
            next_obs_list.append(obs[None, :])
            reward_list.append(reward)
            action_list.append(action)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
    data = {
        "observations": np.vstack(obs_list).astype(np.float32),
        "next_observations": np.vstack(next_obs_list).astype(np.float32),
        "rewards": np.vstack(reward_list).astype(np.float32),
        "actions": np.vstack(action_list).astype(np.float32),
        "terminals": np.vstack(terminated_list).astype(np.float32),
        "truncateds": np.vstack(truncated_list).astype(np.float32),
    }
    return data


def collect_suboptimal_data(env_name: str, env: BaseMiniGridEnv, num_eposides: int, max_steps: int, random_action_prob: float=0.0):
    """Path planning for mini-grid"""
    obs_list = []
    next_obs_list = []
    reward_list = []
    action_list = []
    terminated_list = []
    truncated_list = []
    print("Collecting optimal data")
    mini_grid_type = env_name.split("-")[1].lower()
    if mini_grid_type == "base":
        for i in tqdm.tqdm(range(num_eposides)):
            obs, _ = env.reset(seed=i)
            goal_pose = env.goal_poses[np.random.choice(len(env.goal_poses))]  # select a random goal
            path = env.optimal_path(env.agent_pos, goal_pose)
            for next_pos in path:
                while True:
                    action = env.get_action(next_pos)
                    if action is None:
                        break
                    if np.random.rand() < random_action_prob:
                        action = env.action_space.sample()
                    # collect data
                    obs_list.append(obs[None, :])
                    # step
                    obs, reward, terminated, truncated, info = env.step(action)
                    # collect data
                    next_obs_list.append(obs[None, :])
                    reward_list.append(reward)
                    action_list.append(action)
                    terminated_list.append(terminated)
                    truncated_list.append(truncated)
    else:
        raise ValueError(f"Unknown mini_grid_type: {mini_grid_type}")
    data = {
        "observations": np.vstack(obs_list).astype(np.float32),
        "next_observations": np.vstack(next_obs_list).astype(np.float32),
        "rewards": np.vstack(reward_list).astype(np.float32),
        "actions": np.vstack(action_list).astype(np.float32),
        "terminals": np.vstack(terminated_list).astype(np.float32),
        "truncateds": np.vstack(truncated_list).astype(np.float32),
    }
    return data


if __name__ == "__main__":
    env = BaseMiniGridEnv(agent_start_pos=None, render_mode="human")
    random_action_prob = 0.3
    collect_data_mini_grid(env_name="MiniGrid-Base", env=env, num_eposides=1000, max_steps=100, strategies=["suboptimal"], random_action_prob=random_action_prob)