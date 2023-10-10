""" Generate expert data for MiniGrid with some heuristics
"""
from __future__ import annotations
import numpy as np
import tqdm
from minigrid.core.actions import Actions
from vil2.env.mini_grid.base import BaseMiniGridEnv


def collect_data_mini_grid(env_name: str, env: BaseMiniGridEnv, num_eposides: int|list[int], max_steps: int, strategies: list[str], output_path=None):
    """Collect offline data for MiniGrid"""
    data_list = []
    if isinstance(num_eposides, int):
        num_eposides = [num_eposides] * len(strategies)
    for i, strategy in enumerate(strategies):
        if strategy == "navigate":
            data_list.append(collect_navigate_data(env_name, env, num_eposides[i], max_steps))
        elif strategy == "suboptimal":
            data_list.append(collect_suboptimal_data(env_name, env, num_eposides[i], max_steps))
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
        obs, _ = env.reset()
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


def collect_suboptimal_data(env_name: str, env: BaseMiniGridEnv, num_eposides: int, max_steps: int):
    """Collect suboptimal data to directly solve the task
    Note: agent_dir is 0, 1, 2, 3 for right, down, left, up
    """
    obs_list = []
    next_obs_list = []
    reward_list = []
    action_list = []
    terminated_list = []
    truncated_list = []
    print("Collecting suboptimal data")
    random_action_prob = 0.7
    for i in tqdm.tqdm(range(num_eposides)):
        obs, _ = env.reset()
        goal_pos = env.goal_poses[0]
        for j in range(max_steps):
            if np.random.rand() < random_action_prob:
                action = env.action_space.sample()
            else:
                agent_pos = env.agent_pos
                agent_dir = env.agent_dir
                # diff
                diff = np.array(goal_pos) - np.array(agent_pos)
                # go up/down first
                if diff[1] < 0:
                    if agent_dir == 3:
                        action = Actions.forward
                    else:
                        action = Actions.left 
                elif diff[1] > 0:
                    if agent_dir == 1:
                        action = Actions.forward
                    else:
                        action = Actions.right
                else:
                    # go left/right
                    if diff[0] > 0:
                        if agent_dir == 0:
                            action = Actions.forward
                        else:
                            action = Actions.left
                    elif diff[0] < 0:
                        if agent_dir == 2:
                            action = Actions.forward
                        else:
                            action = Actions.right
                    else:
                        # done
                        break
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
    collect_data_mini_grid(env_name="MiniGrid-Empty-5x5-v0", env=env, num_eposides=1000, max_steps=100, strategies=["suboptimal"], output_path="test_data.npz")