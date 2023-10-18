"""Generate expert data for LGMCTSBot with some heuristics
"""
from __future__ import annotations
import lgmcts
import numpy as np
from lgmcts.env import seed


def collect_data_lgmcts_bot(env_name: str, env, num_eposides: int|list[int], max_steps: int, min_steps: int, strategies: list[str], random_action_prob: float = 0.0, output_path = None):
    """Collect offline data for LGMCTSBot"""
    data_list = []
    if isinstance(num_eposides, int):
        num_eposides = [num_eposides] * len(strategies)
    for i, strategy in enumerate(strategies):
        if strategy == "navigate":
            data_list.append(collect_random_data(env_name, env, num_eposides[i], max_steps, min_steps))
        elif strategy == "suboptimal":
            data_list.append(collect_suboptimal_data(env_name, env, num_eposides[i], max_steps, min_steps, random_action_prob=random_action_prob))
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


def collect_random_data(env_name: str, env, num_eposides: int, max_steps: int, min_steps: int):
    return None


def collect_suboptimal_data(env_name: str, env, num_eposides: int, max_steps: int, min_steps: int, random_action_prob: float = 0.0):
    return None


if __name__ == "__main__":
    task_name = f"push_object_{seed}"
    debug = True
    env = lgmcts.make(
        task_name=task_name,
        task_kwargs=lgmcts.PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        seed=0,
        debug=debug,
        display_debug_window=debug,
        hide_arm_rgb=(not debug),
    )
    collect_data_lgmcts_bot(task_name, env, 100, 100, 100, ["navigate", "suboptimal"], 0.0)