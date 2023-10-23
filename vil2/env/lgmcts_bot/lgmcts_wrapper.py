"""Wrapping the lgmcts env with df env interface"""
import numpy as np
from lgmcts.env.base import BaseEnv
import vil2.utils.misc_utils as utils


class LgmctsEnvWrapper(BaseEnv):
    def __init__(self, **kwargs):
        task_name = kwargs.pop("task_name")
        if "image_size" in kwargs:
            self.image_size = kwargs.pop("image_size")
        else:
            self.image_size = 96  # default
        kwargs["task"] = task_name
        super().__init__(**kwargs)

    def step(self, action=None):
        # process action
        if isinstance(action, np.ndarray):
            action= {
                "pose0_position": action[:3],
                "pose0_rotation": np.array([0, 0, 0, 1]),
                "pose1_position": action[3:6],
                "pose1_rotation": np.array([0, 0, 0, 1]),
            }
        obs, reward, terminated, truncated, info = super().step(action)
        parsed_obs = self.parse_obs(obs)
        return parsed_obs, reward, terminated, truncated, info
    
    def reset(self):
        obs = super().reset() 
        parsed_obs = self.parse_obs(obs)
        return parsed_obs, {}

    def parse_obs(self, obs):
        """Parse observations from lgmcts env to df env"""
        image = obs["rgb"]["front"]
        image_resized = utils.resize_image(image, self.image_size)
        parsed_obs = {
            "image": image_resized,  # (w, h, 3)
            "state": obs["robot_joints"],
        }
        return parsed_obs
