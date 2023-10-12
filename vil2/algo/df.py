"""Diffusion policy"""
from __future__ import annotations
import os
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import vil2.utils.misc_utils as misc_utils
from vil2.algo.model import NoiseScheduler, NoiseNetwork
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
        beta_start = config.get('beta_start', 0.0001)
        beta_end = config.get('beta_end', 0.02)
        self.num_timesteps = config.get('num_timesteps', 100)
        self.horizon = config.get('horizon', 5)

        df_hidden_dim = config.get('df_hidden_dim', 256)
        df_hidden_layer = config.get('df_hidden_layer', 3)
        df_emb_size = config.get('df_emb_size', 64)

        self.noise_scheduler = NoiseScheduler(num_timesteps=self.num_timesteps, beta_start=beta_start, beta_end=beta_end).to(self.device)
        # noise model
        self.noise_model = NoiseNetwork(
            input_size=self.action_dim * self.horizon,
            condition_size=self.state_dim,
            hidden_size=df_hidden_dim, hidden_layers=df_hidden_layer, emb_size=df_emb_size).to(self.device)
        self.noise_optimizer = torch.optim.Adam(self.noise_model.parameters(), lr=policy_lr, weight_decay=weight_decay)
        
    def train(self, env, batch_size: int, num_epochs: int, eval_period: int):
        """Train DF"""
        #TODO: maybe use torch scatter to accelerate the sampling process
        self.noise_model.train()
        for epoch in range(num_epochs):
            # sample a batch of epochs
            # randomly select a chunk of horizon
            batch_epoch_ids = torch.randint(low=0, high=self.num_data_epochs, size=(batch_size,), device=self.device)
            batch_epoch_sizes = self.epoch_sizes[batch_epoch_ids]
            batch_start_pos_high = torch.clamp(batch_epoch_sizes - self.horizon, min=0) + 1
            batch_start_pos = (torch.rand([batch_size, 1], device=self.device) * batch_start_pos_high).long()
            
            # placeholder
            batch_observations = torch.zeros((batch_size, self.state_dim), device=self.device)
            batch_actions = torch.zeros((batch_size, self.horizon * self.action_dim), device=self.device)
            batch_rewards = torch.zeros((batch_size, 1), device=self.device)
            batch_terminals = torch.zeros((batch_size, 1), device=self.device)

            for i in range(batch_size):
                batch_start_pos_val = batch_start_pos[i, 0].item()
                batch_epoch_size = batch_epoch_sizes[i, 0].item()
                batch_idx = torch.arange(batch_start_pos_val, batch_start_pos_val + self.horizon, device=self.device)
                batch_epoch_ids_i = torch.where(self.epoch_ids == batch_epoch_ids[i])[0]
                batch_idx = batch_epoch_ids_i[batch_idx]
                #
                batch_observations[i] = self.observations[batch_idx][0, :] # only use the first observation 
                batch_actions[i] = self.actions[batch_idx].reshape(-1)
                batch_rewards[i] = self.rewards[batch_idx].sum()
                batch_terminals[i] = self.terminals[batch_idx].sum() > 0 
            # sample a batch of data
            policy_loss = self.update_policy(batch_observations, batch_actions, batch_rewards, batch_terminals)
            if epoch % eval_period == 0:
                print(f"Epoch {epoch}, policy loss {policy_loss}")
                self.evaluate(env, num_epochs=10, epoch=epoch, enable_render=self.render_eval)

    def update_policy(self, observations, actions, rewards, terminals):
        """Update policy; using ddpm"""
        # train using ddpm
        batch_size = observations.shape[0]
        timesteps = torch.randint(0, self.num_timesteps, (batch_size, 1)).long().to(self.device)  # sample timesteps

        noise = torch.randn(actions.shape, device=self.device)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        noise_pred = self.noise_model(noisy_actions, observations, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)

        nn.utils.clip_grad_norm_(self.noise_scheduler.parameters(), 1.0)
        self.noise_optimizer.step()
        self.noise_optimizer.zero_grad()

        return loss.item()

    def predict(self, observations):
        """Predict using ddpm sampling"""
        observations = torch.from_numpy(observations).float().to(self.device)
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)
        self.noise_model.eval()
        with torch.no_grad():
            horizon_actions = torch.randn([observations.shape[0], self.horizon * self.action_dim], device=self.device)
            for t in range(self.num_timesteps - 1, 0, -1):
                t_tensor = torch.ones([observations.shape[0], 1], device=self.device) * t
                # gradually remove noise
                noise_pred = self.noise_model(horizon_actions, observations, t_tensor)
                horizon_actions = self.noise_scheduler.step(noise_pred, t, horizon_actions)
            # use the first action
            action = horizon_actions[:, :self.action_dim]
            # clip and rescale
            if self.action_range is not None:
                action = torch.clamp(action, -1, 1)
                action = (action + 1) / 2 * (self.action_range[:, 1] - self.action_range[:, 0]) + self.action_range[:, 0]
            return action.detach().cpu().numpy()
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'noise_model': self.noise_model.state_dict()
        }, path)
        print(f"Save model to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.noise_model.load_state_dict(checkpoint['noise_model'])
        print(f"Load model from {path}")

    def evaluate(self, env, num_epochs: int, epoch: int, enable_render: bool = False):
        # eval on env
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
