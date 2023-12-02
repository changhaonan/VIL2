"""Bellman Diffusion Network Policy"""
from __future__ import annotations
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import vil2.utils.misc_utils as misc_utils
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from vil2.algo.model import PolicyNetwork, NoiseNetwork
import gymnasium as gym
from collections import namedtuple, deque
import random
from tqdm.auto import tqdm
from collections import deque


Transition = namedtuple(
    'Transition', ('goal', 'state', 'action', 'next_state', 'reward', 'terminated', 'steps_remain'))


class DictConcatWrapper(gym.ObservationWrapper):
    """Concatenate all observations into one vector"""

    def observation(self, obs):
        """override"""
        if isinstance(obs, dict):
            return np.concatenate([obs[obs_name] for obs_name in obs.keys()])
        return obs


class ReplayMemory(object):
    """Replay Memory for Finite-Horizon, Goal-Conditioned RL
    Memory stores a list of trajectories; Each trajectory is a list of transitions.
    """

    def __init__(self, capacity, finite_horizon: int):
        self.finite_horizon = finite_horizon
        self.temp_buffer = []  # temporary buffer for storing transitions of current epoch
        self.memory = deque([], maxlen=capacity)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def push(self, goal, obs, action, next_obs, reward, terminated, is_new_epoch: bool):
        """Save a transition"""
        if is_new_epoch and len(self.temp_buffer) > 0:
            # dump from temporary buffer to memory
            trajectory = []
            for _idx, transition in enumerate(self.temp_buffer):
                # update steps_remain
                transition = transition._replace(steps_remain=torch.tensor([[self.finite_horizon - _idx]]).float().to(self.device))
                trajectory.append(transition)
            self.memory.append(trajectory)  # append a copy
            self.temp_buffer.clear()

        goal = torch.from_numpy(goal[None, :]).float().to(self.device)
        obs = torch.from_numpy(obs[None, :]).float().to(self.device)
        action = torch.from_numpy(action[None, :]).float().to(self.device)
        next_obs = torch.from_numpy(next_obs[None, :]).float().to(self.device)
        reward = torch.tensor([[reward]]).float().to(self.device)
        terminated = torch.tensor([[terminated]]).float().to(self.device)
        # push to temporary buffer
        self.temp_buffer.append(Transition(goal, obs, action, next_obs, reward, terminated, 0))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BDNPolicy:
    """Bellman Diffusion Network Policy"""

    def __init__(self, env: gym.Env, config: dict, o_to_g_fn=None) -> None:
        # Parameters
        self.config: dict = config
        self.env: gym.Env = env
        self.o_to_g_fn = o_to_g_fn if o_to_g_fn is not None else self.dummy_obs_to_goal
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        obs, _ = self.env.reset()
        self.goal_dim = self.o_to_g_fn(obs).shape[0]
        self.state_dim = obs.shape[0]
        self.action_dim = env.action_space.shape[0]
        buffer_size = config.get('buffer_size', 100000)
        self.finite_horizon = config['finite_horizon']
        self.memory = ReplayMemory(capacity=buffer_size, finite_horizon=self.finite_horizon)
        # training related
        self.log_path = config.get('log_path', None)
        self.max_step_per_epoch = config.get('max_step_per_epoch', 1000)
        self.log_period = config.get('log_period', 100)
        self.alpha = config.get('alpha', 0.1)
        self.her_tolerance = config.get('her_tolerance', 0.2)
        self.target_update_period = config.get('target_update_period', 1)

        # Policy: mapping from (goal, state) to action
        policy_input_dim = self.goal_dim + self.state_dim
        policy_hidden_dim = config['pi_hidden_dim']
        policy_output_dim = self.action_dim
        self.policy_std = config['policy_std']
        self.policy = PolicyNetwork(
            input_dim=policy_input_dim,
            hidden_dim=policy_hidden_dim,
            output_dim=policy_output_dim,
            is_gaussian=True,
        ).to(self.device)
        # SOM: state occupancy measure
        # epsilon(x| s', a', t, n) predict the distribution of future states
        # here s' is the next state
        # a' = pi(s') is the action taken by the policy
        # t is the time step
        # n is the left execution time
        som_input_dim = self.state_dim
        som_cond_dim = self.state_dim + self.action_dim + 1
        som_hidden_dim = config['som_hidden_dim']
        som_hidden_layers = config['som_hidden_layer']
        som_time_emb = config['som_time_emb']
        som_input_emb = config['som_input_emb']
        self.som_noise = NoiseNetwork(
            input_size=som_input_dim,
            condition_size=som_cond_dim,
            hidden_size=som_hidden_dim,
            hidden_layers=som_hidden_layers,
            time_emb=som_time_emb,
            input_emb=som_input_emb,
        ).to(self.device)
        self.som_noise_target = NoiseNetwork(
            input_size=som_input_dim,
            condition_size=som_cond_dim,
            hidden_size=som_hidden_dim,
            hidden_layers=som_hidden_layers,
            time_emb=som_time_emb,
            input_emb=som_input_emb,
        ).to(self.device)
        self.som_gamma = config['som_gamma']
        # noise scheduler
        self.num_diffusion_iters = config['num_diffusion_iters']
        beta_schedule = config['beta_schedule']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule=beta_schedule,
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

    def update_som_target_network(self, alpha: float = 0.1):
        """Update som target network with Polyak averaging"""
        for param, target_param in zip(self.som_noise.parameters(), self.som_noise_target.parameters()):
            target_param.data.copy_(alpha * param.data +
                                    (1 - alpha) * target_param.data)

    def train(self, batch_size: int, num_epochs: int, goal_pi: np.ndarray = None):
        """Train BDN with goal_pi as the goal"""
        enable_render = True
        self.policy.train()
        self.som_noise.train()
        self.som_noise_target.train()
        t_epoch = tqdm(range(num_epochs))
        for epoch in t_epoch:
            obs, _ = self.env.reset(seed=epoch)
            step_epoch = 0
            step_task = 0
            reward_epoch = 0.0
            actions = None
            epoch_dir = os.path.join(self.log_path, f"epoch-{epoch}")

            # ----------------- collect data -----------------
            for _ in range(self.max_step_per_epoch):
                # step the environment
                action = self.predict(obs, goal_pi, is_deterministic=False)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                # compute reward, terminated
                reward = self.her_reward(obs, goal_pi, self.her_tolerance)
                if reward > 0:
                    terminated = True  # force early termination
                if step_task >= self.finite_horizon:
                    terminated = True  # force early termination

                # save transition
                obs_goal = self.o_to_g_fn(obs)
                self.memory.push(
                    obs_goal, obs, action, next_obs, reward, terminated, is_new_epoch=(step_task == 0)
                )

                # update info
                obs = next_obs
                reward_epoch += reward
                step_epoch += 1
                step_task += 1

                # reset
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    step_task = 0

                # do visualization
                # if enable_render and epoch % self.log_period == 0:
                #     image = self.env.render()
                #     if self.log_path is not None:
                #         if not os.path.exists(epoch_dir):
                #             os.makedirs(epoch_dir, exist_ok=True)
                #         cv2.imwrite(os.path.join(epoch_dir, f"step-{step_epoch}.png"), image)

            # log
            t_epoch.set_description(f"Epoch: {epoch}, Reward: {reward_epoch}, Step: {step_epoch}")
            if epoch % self.log_period == 0:
                if self.log_path is not None:
                    with open(os.path.join(self.log_path, "eval.txt"), "a") as f:
                        f.write(
                            f"Epoch: {epoch}| reward: {reward_epoch}, step: {step_epoch}\n")

            # ----------------- optimize model -----------------
            # optimize som
            self.optimize_som(epoch, batch_size=batch_size)

            # optimize policy
            self.optimize_policy(epoch, batch_size=batch_size)

            # update som target network
            if epoch % self.target_update_period == 0:
                self.update_som_target_network(alpha=self.alpha)

    def predict(self, obs, goal, is_deterministic=False):
        """Predict action given observation & goal"""
        # move to gpu
        obs = torch.from_numpy(obs[None, :]).float().to(self.device)
        goal = torch.from_numpy(goal[None, :]).float().to(self.device)
        # predict action
        mean = self.policy(torch.cat([goal, obs], dim=1))
        if not is_deterministic:
            # build distribution
            dist = torch.distributions.Normal(mean, self.policy_std)
            # sample action
            action = dist.sample()
            return action.detach().cpu().numpy().squeeze()
        else:
            return mean.detach().cpu().numpy().squeeze()

    def her_reward(self, obs, goal, tolerance: float):
        """HER style reward function"""
        obs_goal = self.o_to_g_fn(obs)
        dist = np.linalg.norm(obs_goal - goal)
        if dist < tolerance:
            return 1.0
        else:
            return 0.0

    def optimize_som(self, epoch: int, batch_size: int):
        """Optimize SOM (State Occupancy Measure) Model"""
        pass

    def optimize_policy(self, epoch: int, batch_size: int):
        """Optimize Policy"""
        pass

    def dummy_obs_to_goal(self, obs):
        """Dummy goal function; Identity function"""
        return obs.copy()

    def save(self, path):
        """Save model to path"""
        torch.save({
            'policy': self.policy.state_dict(),
            'som_noise': self.som_noise.state_dict(),
            'som_noise_target': self.som_noise_target.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model from path"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.som_noise.load_state_dict(checkpoint['som_noise'])
        self.som_noise_target.load_state_dict(checkpoint['som_noise_target'])
        self.update_som_target_network(alpha=1.0)
        print(f"Model loaded from {path}")
