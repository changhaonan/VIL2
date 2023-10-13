"""Behavior cloning algorithm."""
from __future__ import annotations
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import vil2.utils.misc_utils as misc_utils
from vil2.algo.model import QNetwork, VNetwork, PolicyNetwork
import gymnasium as gym
from vil2.algo.offline_policy import OfflinePolicy


class BC(OfflinePolicy):
    """Behavior cloning algorithm."""
    name: str = "BC"
    
    def __init__(self, env: gym.Env, dataset: dict, config: dict):
        super().__init__(env=env, dataset=dataset, config=config)
        # parameters
        policy_lr = config.get('policy_lr', 1e-3)
        weight_decay = config.get('weight_decay', 0.0)  

        # init network 
        policy_input_dim = self.state_dim
        policy_output_dim = self.action_dim
        policy_hidden_dim = config['policy_hidden_dim']
        policy_is_gaussian = True
        self.policy = PolicyNetwork(input_dim=policy_input_dim, hidden_dim=policy_hidden_dim, output_dim=policy_output_dim, is_gaussian=policy_is_gaussian)
        self.policy.to(self.device)
        
        # init optimizer
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr, weight_decay=weight_decay)

    def update_policy(self, observations, actions):
        """Update policy; for naive BC, the advantage is the same"""
        if self.action_discrete:
            action_mean = self.policy(observations)
            # use cross entropy loss
            log_probs = - nn.CrossEntropyLoss(reduction='none')(action_mean, actions.argmax(dim=-1))
        else:
            if self.policy.is_gaussian:
                action_mean = self.policy(observations)
                action_std = torch.ones_like(action_mean) * self.policy_std
                action_dist = torch.distributions.Normal(action_mean, action_std)
                log_probs = action_dist.log_prob(actions)
            else:
                assert False, "Not implemented"
        policy_loss = -(log_probs).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.detach().cpu().numpy()

    def train(self, env, batch_size: int, num_epochs: int, eval_period: int):
        """Train BC"""
        for epoch in range(num_epochs):
            # sample a batch of data
            batch_idx = torch.randint(low=0, high=self.observations.shape[0], size=(batch_size,), device=self.device)
            observations = self.observations[batch_idx]
            # set velocity to zero
            next_observations = self.next_observations[batch_idx]
            actions = self.actions[batch_idx]
            rewards = self.rewards[batch_idx]
            terminals = self.terminals[batch_idx]
            self.update_policy(observations=observations, actions=actions)
            
            if epoch % eval_period == 0:
                self.evaluate(env=env, num_epochs=10, epoch=epoch, enable_render=self.render_eval)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict action given observations"""
        observations = torch.from_numpy(observations).float().to(self.device)
        self.policy.eval()
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)
        elif len(observations.shape) == 3:
            # image
            observations = observations.reshape(1, -1)
        
        if self.action_discrete:
            action_mean = self.policy(observations)
            actoin_prob = nn.functional.softmax(action_mean, dim=-1)
            # sample action
            if self.action_deterministic:
                action = actoin_prob.argmax(dim=-1)
            else:
                action = torch.multinomial(actoin_prob, num_samples=1)
        else:
            if self.policy.is_gaussian:
                with torch.no_grad():
                    action_mean = self.policy(observations)
                    action_std = torch.ones_like(action_mean) * self.policy_std
                    action_dist = torch.distributions.Normal(action_mean, action_std)
                    action = action_dist.sample()
            else:
                assert False, "Not implemented"
        if action.shape[0] == 1:
            action = action.squeeze(0)
        return action.detach().cpu().numpy()

    def save(self, path: str):
        """Save model"""
        torch.save({
            'policy_network': self.policy.state_dict()
        }, path)
        print(f"Save model to {path}")

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_network'])
        print(f"Load model from {path}")