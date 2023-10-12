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
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_discrete = True
            self.action_dim = env.action_space.n
            # one-hot encoding
            self.actions = torch.zeros((self.actions.shape[0], self.action_dim), device=self.device).scatter_(1, self.actions.long(), 1).float()
        else:
            self.action_discrete = False
            self.action_dim = self.actions.shape[1]