from .maze_2d import MazeTree
import gymnasium as gym


def env_builder(env_name, **kwargs):
    if env_name == "maze":
        return MazeTree(**kwargs)
    elif env_name.split("-")[0].lower() == "minigrid":
        render_mode = kwargs.pop("render_mode", "human")
        return gym.make(env_name, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")