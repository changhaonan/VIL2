"""Run Bellman Diffusion Network Policy"""
import os
import cv2
import pickle
import numpy as np
from vil2.env import env_builder
from vil2.algo.bdnp import BDNPolicy
from detectron2.config import LazyConfig
from vil2.utils.som_utils import sample_som
from vil2.utils.eval_utils import compare_distribution
import gymnasium as gym
import gymnasium_robotics as gym_robotics


class CustomFetchWrapper(gym.Wrapper):
    def __init__(self, env, new_goal=None, initial_joint_values=None):
        super(CustomFetchWrapper, self).__init__(env)
        self.new_goal = new_goal
        self.initial_joint_values = initial_joint_values

    def reset(self, **kwargs):
        # Reset the environment
        obs, info = self.env.reset(**kwargs)

        # Set a new desired goal if specified
        if self.new_goal is not None:
            self.env.unwrapped.goal = self.new_goal
        observation = obs['observation']
        achieved_goal = obs['achieved_goal']
        desired_goal = obs['desired_goal']
        override_obs = {
            'observation': np.concatenate([observation, achieved_goal, desired_goal]),
            'achieved_goal': achieved_goal,
            'desired_goal': self.new_goal,
        }

        # # Set initial joint values if specified
        # if self.initial_joint_values is not None:
        #     # Ensure the length matches the robot's joints
        #     assert len(self.initial_joint_values) == len(self.env.sim.data.qpos)
        #     self.env.sim.data.qpos[:] = self.initial_joint_values
        #     self.env.sim.forward()

        return override_obs, info


if __name__ == "__main__":
    # prepare path
    root_path = os.path.dirname((os.path.abspath(__file__)))
    env_name = "GYM-FetchReach-v2"
    export_path = os.path.join(root_path, "test_data", env_name)
    check_point_path = os.path.join(export_path, 'bdnp', 'checkpoint')
    log_path = os.path.join(export_path, 'bdnp', 'log')
    os.makedirs(export_path, exist_ok=True)
    os.makedirs(check_point_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # prepare data
    env = env_builder(env_name, render_mode="rgb_array")
    config = {
        # ----- som related -----
        'som_hidden_dim': 32,
        'som_hidden_layer': 3,
        'som_time_emb': 'learnable',
        'som_input_emb': 'learnable',
        'beta_schedule': 'squaredcos_cap_v2',
        'num_diffusion_iters': 20,
        'som_gamma': 0.3,  # balancing loss; to distinguish from the gamma for discount factor
        'som_lr': 1e-4,
        'som_weight_decay': 1e-6,
        # ----- policy related -----
        'pi_hidden_dim': 256,
        'pi_hidden_layer': 3,
        'policy_std': 0.1,
        'batch_size': 512,
        'finite_horizon': 50,
        'policy_lr': 1e-4,
        'policy_weight_decay': 1e-6,
        # ----- training related -----
        'num_epochs': 500,
        'eval_period': 1000,
        'max_epoch_per_episode': 8,
        'log_period': 100,
        'her_tolerance': 0.2,  # HER tolerance
        'alpha': 0.1,
        'target_update_period': 1,
        'enable_save': True,
        'enable_load': False,
        'log_path': log_path,
        'render_eval': True,  # render evaluation
        'lamda': 10.0,  # regularization parameter
        'sample_size': 32,
        'a_lazy_init': True,  # use last a_0 to init current a_T
    }
    # env = DictConcatWrapper(env)
    # bdnp = BDNPolicy(env=env, config=config)
    # # do train
    # goal_pi = np.zeros((31,), dtype=np.float32)
    # bdnp.train(batch_size=config['batch_size'],
    #            num_episode=config['num_epochs'])

    # Build a heuristic policy
    if env_name == "GYM-FetchReach-v2":
        def heuristic_policy(obs, step_remain=0, action_noise=0.0):
            gripper_pos = obs['observation'][:3]
            goal_pos = obs['desired_goal']
            gripper_movement = goal_pos - gripper_pos
            cur_diff = np.linalg.norm(gripper_movement)
            # print(f"cur_diff: {cur_diff}")
            gripper_action = np.zeros(4)
            gripper_action[:3] = gripper_movement * 3.0
            # apply noise
            gripper_action[:3] += np.random.normal(size=3) * action_noise
            # clip by (-1, 1)
            gripper_action = np.clip(gripper_action, -1, 1)
            return gripper_action
    else:
        raise NotImplementedError

    # do train
    # goal_pi = np.zeros((31,), dtype=np.float32)
    # bdnp.train(batch_size=config['batch_size'],
    #            num_episode=config['num_epochs'],
    #            train_policy=False,
    #            heuristic_policy=heuristic,)

    # bdnp.save(os.path.join(check_point_path, 'model.pt'))

    # save the stats for replay buffer
    # stats = bdnp.replay_buffer.compute_stats()
    # with open(os.path.join(export_path, 'bdnp', 'stats.pkl'), 'wb') as f:
    #     pickle.dump(stats, f)

    #################### Compute SOM ########################
    # apply a wrapper
    desired_goal = np.array([1.3, 0.8, 0.8])
    env = CustomFetchWrapper(env, new_goal=desired_goal)
    num_steps = 30
    num_samples = 1000
    sampled_som = sample_som(env=env, policy_fn=lambda obs: heuristic_policy(obs, 0, action_noise=0.01), num_steps=num_steps, num_samples=num_samples)
    compare_distribution(sampled_som['desired_goal'], sampled_som['observation'], dim_end=3, plot_type="scatter")
    pass
