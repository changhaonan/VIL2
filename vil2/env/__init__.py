from .maze_2d import MazeTree
import gymnasium as gym
from .mini_grid.base import BaseMiniGridEnv
from .mini_grid.multi_modality import MultiModalityMiniGridEnv
from .mini_grid.collect_data import collect_data_mini_grid


def env_builder(env_name, **kwargs):
    if env_name == "maze":
        return MazeTree(**kwargs)
    elif env_name.split("-")[0].lower() == "minigrid":
        render_mode = kwargs.pop("render_mode", "rgb_array")
        mini_grid_type = env_name.split("-")[1].lower()
        if mini_grid_type == "mm":
            return MultiModalityMiniGridEnv(agent_start_pos=None, render_mode=render_mode)
        else:
            return BaseMiniGridEnv(agent_start_pos=None, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")


def env_data_collect(env, env_name, **kwargs):
    if env_name.split("-")[0].lower() == "minigrid":
        num_eposides = kwargs.pop("num_eposides", 1000)
        max_steps = kwargs.pop("max_steps", 100)
        strategies = kwargs.pop("strategies", ["navigate", "suboptimal"])
        return collect_data_mini_grid(env_name=env_name, env=env, num_eposides=num_eposides, max_steps=max_steps, strategies=strategies)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")