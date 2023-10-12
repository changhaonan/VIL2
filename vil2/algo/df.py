"""Diffusion policy"""
from __future__ import annotations
import os
import cv2
import torch
import torch.nn as nn

import numpy as np
import vil2.utils.misc_utils as misc_utils
from vil2.algo.model import DiffusionModel
import gymnasium as gym
from vil2.algo.offline_policy import OfflinePolicy


class DF(OfflinePolicy):
    """Diffusion policy"""
    name: str = "DF"

    def __init__(self, env: gym.Env, dataset: dict, config: dict):
        super().__init__(env=env, dataset=dataset, config=config)
        # parameters
        policy_lr = config.get('policy_lr', 1e-3)
        weight_decay = config.get('weight_decay', 0.0)
        num_timesteps = config.get('num_timesteps', 1000)
        beta_start = config.get('beta_start', 0.0001)
        beta_end = config.get('beta_end', 0.02)

        df_input_dim = self.state_dim + self.action_dim
        df_hidden_dim = config.get('df_hidden_dim', 256)
        df_output_dim = self.state_dim

        self.diffusion_model = DiffusionModel(
            input_dim=df_input_dim, hidden_dim=df_hidden_dim, output_dim=df_output_dim,
            num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end).to(self.device)
        
    def train(self, env, batch_size: int, horizon: int, num_epochs: int, eval_period: int):
        """Train DF"""
        #TODO: maybe use torch scatter to accelerate the sampling process
        for epoch in range(num_epochs):
            # sample a batch of epochs
            # randomly select a chunk of horizon
            batch_epoch_ids = torch.randint(low=0, high=self.num_data_epochs, size=(batch_size,), device=self.device)
            batch_epoch_sizes = self.epoch_sizes[batch_epoch_ids]
            batch_start_pos_high = torch.clamp(batch_epoch_sizes - horizon, min=0) + 1
            batch_start_pos = (torch.rand([batch_size, 1], device=self.device) * batch_start_pos_high).long()
            
            # placeholder
            batch_observations = torch.zeros((batch_size, horizon, self.state_dim), device=self.device)
            batch_actions = torch.zeros((batch_size, horizon, self.action_dim), device=self.device)
            batch_rewards = torch.zeros((batch_size, horizon, 1), device=self.device)
            batch_terminals = torch.zeros((batch_size, horizon, 1), device=self.device)

            for i in range(batch_size):
                batch_start_pos_val = batch_start_pos[i, 0].item()
                batch_epoch_size = batch_epoch_sizes[i, 0].item()
                batch_idx = torch.arange(batch_start_pos_val, batch_start_pos_val + horizon, device=self.device)
                batch_epoch_ids_i = torch.where(self.epoch_ids == batch_epoch_ids[i])[0]
                batch_idx = batch_epoch_ids_i[batch_idx]
                #
                batch_observations[i] = self.observations[batch_idx] 
                batch_actions[i] = self.actions[batch_idx]
                batch_rewards[i] = self.rewards[batch_idx]
                batch_terminals[i] = self.terminals[batch_idx]  
            # sample a batch of data
            self.update_policy(batch_observations, batch_actions, batch_rewards, batch_terminals)

    def update_policy(self, observations, actions, rewards, terminals):
        """Update policy; for naive BC, the advantage is the same"""
        pass
