"""Generate expert data for LGMCTSBot with some heuristics
"""
from __future__ import annotations
import lgmcts
import numpy as np
from lgmcts.env import seed
import cv2
import zarr
import os
from tqdm.auto import tqdm


def collect_data_lgmcts_bot(
    env_name: str,
    env,
    num_eposides: int | list[int],
    max_steps: int,
    strategies: list[str],
    random_action_prob: float = 0.0,
    output_path=None,
):
    """Collect offline data for LGMCTSBot"""
    data_list = []
    meta_data_list = []
    if isinstance(num_eposides, int):
        num_eposides = [num_eposides] * len(strategies)
    for i, strategy in enumerate(strategies):
        if strategy == "navigate":
            data_list.append(collect_random_data(env_name, env, num_eposides[i], max_steps))
        elif strategy == "suboptimal":   
            data, meta_data = collect_suboptimal_data(
                    env_name=env_name,
                    env=env,
                    num_eposides=num_eposides[i],
                    max_steps=max_steps,
                    random_action_prob=random_action_prob,
                )
            data_list.append(data)
            meta_data_list.append(meta_data)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    # merge data & meta_data
    data_merge = {}
    for data in data_list:
        for key, value in data.items():
            if key not in data_merge:
                data_merge[key] = []
            data_merge[key].append(value)
    # stack
    for key, value in data_merge.items():
        data_merge[key] = np.vstack(value)
    # merge meta_data
    meta_data_merge = {}
    for meta_data in meta_data_list:
        for key, value in meta_data.items():
            if key not in meta_data_merge:
                meta_data_merge[key] = []
            meta_data_merge[key].append(value)
    if "episode_len" in meta_data_merge:
        meta_data_merge["episode_len"] = np.hstack(meta_data_merge["episode_len"])
        meta_data_merge["episode_ends"] = np.cumsum(meta_data_merge["episode_len"])

    if output_path is not None:
        root = zarr.open(f"{output_path}/{env_name}.zarr", mode="w")
        data = root.create_group("data")
        data.create_dataset("img", data=data_merge["img"])
        data.create_dataset("state", data=data_merge["state"])
        data.create_dataset("action", data=data_merge["action"])
        data.create_dataset("reward", data=data_merge["reward"])
        data = root.create_group("meta")
        data.create_dataset("episode_ends", data=meta_data_merge["episode_ends"])
    return data_merge


def collect_random_data(env_name: str, env, num_eposides: int, max_steps: int):
    return None


def collect_suboptimal_data(
    env_name: str,
    env,
    num_eposides: int,
    max_steps: int,
    random_action_prob: float = 0.0,
):
    """Collect suboptimal data"""
    obs_img_list = []
    obs_state_list = []
    action_list = []
    reward_list = []
    episode_len_list = []
    task = env.task
    for i in tqdm(range(num_eposides), desc="Collecting suboptimal data"):
        env.reset()
        terminated = False
        step_count = 0
        while True:
            opt_action_list = task.oracle_action(env=env)
            if len(opt_action_list) == 0:  # start with finished state
                break
            for action in opt_action_list:
                # execute action
                obs, reward, terminated, truncated, info = env.step(action)
                front_rgb = obs["rgb"]["front"].copy().transpose(1, 2, 0)
                front_rgb = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
                # cv2.imshow("front_rgb", front_rgb)
                # cv2.waitKey(1)
                obs_img_list.append(front_rgb[None, ...])
                obs_state_list.append(obs["robot_joints"][None, ...])
                reward_list.append(reward)
                action_list.append(np.hstack([action["pose0_position"], action["pose1_position"]]))
                step_count += 1
                if terminated:
                    break
            if terminated or step_count >= max_steps:
                break
        episode_len_list.append(step_count)
    # merge
    data = {
        "img": np.vstack(obs_img_list),
        "state": np.vstack(obs_state_list),
        "action": np.vstack(action_list),
        "reward": np.vstack(reward_list),
    }
    # get meta data
    episode_len = np.array(episode_len_list).astype(np.int32)
    meta_data = {
        "episode_len": episode_len,
    }
    return data, meta_data  
    


if __name__ == "__main__":
    # prepare env
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    export_dir = os.path.join(root_dir, "test_data", "lgmcts_bot")
    os.makedirs(export_dir, exist_ok=True)
    
    task_name = f"push_object_{seed}"
    debug = False
    env = lgmcts.make(
        task_name=task_name,
        task_kwargs=lgmcts.PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        seed=0,
        debug=debug,
        display_debug_window=debug,
        hide_arm_rgb=(not debug),
    )
    collect_data_lgmcts_bot(
        env_name=task_name,
        env=env,
        num_eposides=100,
        max_steps=100,
        strategies=["suboptimal"],
        random_action_prob=0.0,
        output_path=export_dir
    )
    env.close()