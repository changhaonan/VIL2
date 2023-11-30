# from .maze_2d import MazeTree
import gymnasium as gym
from .mini_grid.base import BaseMiniGridEnv
from .mini_grid.multi_modality import MultiModalityMiniGridEnv
from .mini_grid.local_min import LocalMinGridEnv
from .mini_grid.collect_data import collect_data_mini_grid
# from .push_t.push_t_base import PushTEnv, PushTImageEnv


def env_builder(env_name, render_mode="rgb_array", cfg=None):
    if env_name == "maze":
        # return MazeTree(**kwargs)
        raise ImportError("Maze env not defined")
    elif env_name.split("-")[0].lower() == "minigrid":
        mini_grid_type = env_name.split("-")[1].lower()
        agent_start_pos = cfg.pop("START_POS", None)
        goal_poses = cfg.pop("GOAL_POSES", None)
        if mini_grid_type == "mm":
            return MultiModalityMiniGridEnv(agent_start_pos=agent_start_pos, render_mode=render_mode)
        elif mini_grid_type == "lm":
            return LocalMinGridEnv(agent_start_pos=agent_start_pos, goal_poses=goal_poses, render_mode=render_mode)
        else:
            return BaseMiniGridEnv(agent_start_pos=agent_start_pos, render_mode=render_mode)
    elif env_name.split("-")[0].lower() == "push_t":
        return PushTImageEnv()  # default PushT
    elif env_name.split("-")[0].lower() == "lgmcts":
        task_name = env_name.split("-")[1].lower()
        debug = cfg.pop("debug", False)
        import lgmcts
        from .lgmcts_bot import LgmctsEnvWrapper
        env = LgmctsEnvWrapper(
            task_name=task_name,
            task_kwargs=lgmcts.PARTITION_TO_SPECS["train"][task_name],
            modalities=["rgb", "segm", "depth"],
            seed=0,
            debug=debug,
            display_debug_window=debug,
            hide_arm_rgb=(not debug),
        )
        return env
    elif env_name.split("-")[0].lower() == "gym":
        gym_env_name = env_name[4:]
        return gym.make(gym_env_name, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")


def env_data_collect(env, env_name, **kwargs):
    if env_name.split("-")[0].lower() == "minigrid":
        num_eposides = kwargs.pop("num_eposides", 1000)
        max_steps = kwargs.pop("max_steps", 100)
        min_steps = kwargs.pop("min_steps", 5)
        strategies = kwargs.pop("strategies", ["navigate", "suboptimal"])
        random_action_prob = kwargs.pop("random_action_prob", 0.0)
        way_points_list = kwargs.pop("way_points_list", None)
        return collect_data_mini_grid(
            env_name=env_name,
            env=env,
            num_eposides=num_eposides,
            max_steps=max_steps,
            min_steps=min_steps,
            strategies=strategies,
            random_action_prob=random_action_prob,
            way_points_list=way_points_list,
        )
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
