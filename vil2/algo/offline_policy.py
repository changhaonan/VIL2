"""Offline policy Base"""
from __future__ import annotations
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import vil2.utils.misc_utils as misc_utils
from vil2.algo.model import QNetwork, VNetwork, PolicyNetwork
import gymnasium as gym


class OfflinePolicy:
    """Offline policy"""
    def __init__(self, env: gym.Env, dataset: dict, config: dict):
        # parameters
        self.policy_std = config.get('policy_std', 0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path = config.get('log_path', None)
        if os.path.exists(os.path.join(self.log_path, "eval.txt")):
            os.remove(os.path.join(self.log_path, "eval.txt"))
        self.action_deterministic = config.get('action_deterministic', False)
        self.render_eval = config.get('render_eval', False)
        # extract data from dataset
        self.observations = torch.from_numpy(dataset['observations']).float().to(self.device)
        self.next_observations = torch.from_numpy(dataset['next_observations']).float().to(self.device)
        self.actions = torch.from_numpy(dataset['actions']).float().to(self.device)
        self.rewards = torch.from_numpy(dataset['rewards']).float().to(self.device)
        if len(self.rewards.shape) == 1:
            self.rewards = self.rewards.unsqueeze(1)
        self.terminals = torch.from_numpy(dataset['terminals']).float().to(self.device)
        if len(self.terminals.shape) == 1:
            self.terminals = self.terminals.unsqueeze(1)
        self.truncateds = torch.from_numpy(dataset['truncateds']).float().to(self.device)
        if len(self.truncateds.shape) == 1:
            self.truncateds = self.truncateds.unsqueeze(1)
        self.epoch_ids = torch.from_numpy(dataset['epoch_ids']).long().to(self.device)
        if len(self.epoch_ids.shape) == 1:
            self.epoch_ids = self.epoch_ids.unsqueeze(1)
        self.num_data_epochs = self.epoch_ids.max().item() + 1
        self.epoch_sizes = torch.from_numpy(dataset['epoch_sizes']).long().to(self.device)
        
        # observation
        if len(self.observations.shape) == 4:
            # image
            self.observations = self.observations.reshape(self.observations.shape[0], -1)
            self.next_observations = self.next_observations.reshape(self.next_observations.shape[0], -1)
        self.state_dim = self.observations.shape[1]
        # action
        force_action_continuous = config.get('force_action_continuous', False)
        if isinstance(env.action_space, gym.spaces.Discrete):
            if not force_action_continuous:
                self.action_discrete = True
                self.action_dim = env.action_space.n
                # one-hot encoding
                self.actions = torch.zeros((self.actions.shape[0], self.action_dim), device=self.device).scatter_(1, self.actions.long(), 1).float()
                self.action_range = None
            else:
                # discrete action space, but force to use continuous action
                self.action_discrete = False
                self.action_dim = 1
                self.action_range = torch.tensor([0, env.action_space.n - 1], device=self.device).unsqueeze(0)
                # scale to [-1, 1]
                self.actions = (self.actions - self.action_range[:, 0]) / (self.action_range[:, 1] - self.action_range[:, 0]) * 2 - 1
        else:
            self.action_discrete = False
            self.action_dim = self.actions.shape[1]
            self.action_range = torch.tensor([env.action_space.low, env.action_space.high], device=self.device).transpose(0, 1)
            # scale to [-1, 1]
            self.actions = (self.actions - self.action_range[:, 0]) / (self.action_range[:, 1] - self.action_range[:, 0]) * 2 - 1
    
    def predict(self, observations):
        raise NotImplementedError

    def evaluate(self, env, num_epochs: int, epoch: int, enable_render: bool = False):
        """Evaluate the policy on env; log the result"""
        epoch_dir = os.path.join(self.log_path, f"epoch-{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        step_sum = 0.0
        reward_sum = 0.0
        for i in range(num_epochs):
            obs, _ = env.reset(seed=i)
            step_epoch = 0
            reward_epoch = 0.0
            while True:
                if enable_render:
                    image = env.render()
                    if self.log_path is not None:
                        cv2.imwrite(os.path.join(epoch_dir, f"i-{i}-step-{step_epoch}.png"), image)
                if "image" in obs:
                    obs = obs["image"]
                action = self.predict(obs)
                action = action[:self.action_dim]  # use the first action
                obs, reward, terminated, truncated, info = env.step(action)
                reward_epoch += reward
                step_epoch += 1
                if terminated or truncated:
                    if i == 0:
                        print(f"========== Epoch: {epoch} ==========")
                    print(f"Episode: {i}, Reward: {reward}, Step: {step_epoch}")
                    break
            # log
            step_sum += step_epoch
            reward_sum += reward_epoch
            if self.log_path is not None:
                with open(os.path.join(self.log_path, "eval.txt"), "a") as f:
                    if i == 0:
                        f.write(f"========== Epoch: {epoch} ==========\n")
                    f.write(f"seed: {i}, reward: {reward_epoch}, step: {step_epoch}\n")
                    if i == num_epochs - 1:
                        f.write(f"-------- Average: --------\n")
                        f.write(f"reward: {reward_sum / num_epochs}, step: {step_sum / num_epochs}\n")