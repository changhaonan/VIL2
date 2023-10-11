"""Implicit Q learning: https://arxiv.org/abs/2110.06169"""
from __future__ import annotations
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import vil2.utils.misc_utils as misc_utils
from vil2.algo.model import QNetwork, VNetwork, PolicyNetwork
import gymnasium as gym

# constants
EXP_ADV_MAX = 100.0


class IQL:
    """Implicit Q learning method"""
    def __init__(self, env: gym.Env, dataset: dict, config: dict):
        # parameters
        q_lr = config.get('q_lr', 1e-3)
        v_lr = config.get('v_lr', 1e-3)
        weight_decay = config.get('weight_decay', 0.0)
        policy_lr = config.get('policy_lr', 1e-3)
        policy_std = config.get('policy_std', 0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path = config.get('log_path', None)
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
        
        # init network
        # observation
        if len(self.observations.shape) == 4:
            # image
            self.observations = self.observations.reshape(self.observations.shape[0], -1)
            self.next_observations = self.next_observations.reshape(self.next_observations.shape[0], -1)
        state_dim = self.observations.shape[1]
        # action
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_discrete = True
            action_dim = env.action_space.n
            # one-hot encoding
            self.actions = torch.zeros((self.actions.shape[0], action_dim), device=self.device).scatter_(1, self.actions.long(), 1).float()
        else:
            self.action_discrete = False
            action_dim = self.actions.shape[1]
            
        q_input_dim = state_dim + action_dim
        v_input_dim = state_dim
        policy_input_dim = state_dim
        q_output_dim = 1
        v_output_dim = 1
        policy_output_dim = action_dim
        q_hidden_dim = config['q_hidden_dim']
        v_hidden_dim = config['v_hidden_dim']
        policy_hidden_dim = config['policy_hidden_dim']
        policy_is_gaussian = True
        self.qf1 = QNetwork(input_dim=q_input_dim, hidden_dim=q_hidden_dim, output_dim=q_output_dim)
        self.qf2 = QNetwork(input_dim=q_input_dim, hidden_dim=q_hidden_dim, output_dim=q_output_dim)
        self.target_qf1 = QNetwork(input_dim=q_input_dim, hidden_dim=q_hidden_dim, output_dim=q_output_dim)
        self.target_qf2 = QNetwork(input_dim=q_input_dim, hidden_dim=q_hidden_dim, output_dim=q_output_dim)
        self.vf = VNetwork(input_dim=v_input_dim, hidden_dim=v_hidden_dim, output_dim=v_output_dim)
        self.policy = PolicyNetwork(input_dim=policy_input_dim, hidden_dim=policy_hidden_dim, output_dim=policy_output_dim, is_gaussian=policy_is_gaussian)
        self.policy_std = policy_std
        # init network 
        self.qf1.to(self.device)
        self.qf2.to(self.device)
        self.target_qf1.to(self.device)
        self.target_qf2.to(self.device)
        self.vf.to(self.device)
        self.policy.to(self.device)
        # init optimizer
        self.qf1_optimizer = torch.optim.Adam(self.qf1.parameters(), lr=q_lr, weight_decay=weight_decay)
        self.qf2_optimizer = torch.optim.Adam(self.qf2.parameters(), lr=q_lr, weight_decay=weight_decay)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=v_lr, weight_decay=weight_decay)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr, weight_decay=weight_decay)
    
    def update_q_target_network(self, alpha: float = 0.1):
        """Update Q target network with Polyak averaging"""
        for param, target_param in zip(self.qf1.parameters(), self.target_qf1.parameters()):
            target_param.data.copy_(alpha * param.data + (1.0 - alpha) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.target_qf2.parameters()):
            target_param.data.copy_(alpha * param.data + (1.0 - alpha) * target_param.data)
    
    def train(self, env, batch_size: int, num_epochs: int, eval_period: int, alpha: float = 0.01, tau: float = 0.6, gamma: float = 0.99, beta: float = 3.0, update_action: bool = False):
        """Train IQL"""
        # train
        print("Start training Q & V...")
        v_loss_sum = 0.0
        q_loss_sum = 0.0
        q_update_freq = 1
        target_update_freq = 10
        for epoch in range(num_epochs):
            # sample a batch of data
            batch_idx = torch.randint(low=0, high=self.observations.shape[0], size=(batch_size,), device=self.device)
            observations = self.observations[batch_idx]
            # set velocity to zero
            next_observations = self.next_observations[batch_idx]
            actions = self.actions[batch_idx]
            rewards = self.rewards[batch_idx]
            terminals = self.terminals[batch_idx]
            
            # QF loss
            q1_pred = self.qf1(torch.cat([observations, actions], dim=1))
            q2_pred = self.qf2(torch.cat([observations, actions], dim=1))
            next_vf_pred = self.vf(next_observations).detach()  # don't update V network with Q loss
            q_target = rewards + gamma * (1 - terminals) * next_vf_pred
            qf1_loss = nn.MSELoss()(q1_pred, q_target)
            qf2_loss = nn.MSELoss()(q2_pred, q_target)

            # VF loss
            q1_target_pred = self.target_qf1(torch.cat([observations, actions], dim=1))
            q2_target_pred = self.target_qf2(torch.cat([observations, actions], dim=1))
            q_target_pred = torch.min(q1_target_pred, q2_target_pred).detach()  # don't update Q network with V loss
            v_pred = self.vf(observations)
            vf_loss = self.expectile(q_target_pred - v_pred, tau=tau).mean()

            # update Networks
            if epoch % q_update_freq == 0:
                # update Q network
                self.qf1_optimizer.zero_grad(set_to_none=True)
                qf1_loss.backward()
                self.qf1_optimizer.step()

                self.qf2_optimizer.zero_grad(set_to_none=True)
                qf2_loss.backward()
                self.qf2_optimizer.step()

                # update V network
                self.v_optimizer.zero_grad(set_to_none=True)
                vf_loss.backward()
                self.v_optimizer.step()
            
            if epoch % target_update_freq == 0:
                # update target Q network
                self.update_q_target_network(alpha=alpha)

            # update action
            if update_action:
                self.update_policy(observations=observations, actions=actions, beta=beta)
            
            # eval
            v_loss_sum += vf_loss.detach().cpu().numpy()
            q_loss_sum += (qf1_loss.detach().cpu().numpy() + qf2_loss.detach().cpu().numpy()) / 2.0
            print(f"Epoch: {epoch}, V Loss: {v_loss_sum / eval_period}, Q Loss: {q_loss_sum / eval_period}")
            if epoch % eval_period == 0:
                if update_action:
                    self.evaluate(env=env, num_epochs=10, epoch=epoch, enable_render=True)
                # statics
                v_loss_sum = 0.0
                q_loss_sum = 0.0
            
    def expectile(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """Expectile function"""
        return torch.where(x < 0, tau * x.square(), (1 - tau) * x.square())

    def extract_policy(self, batch_size: int = 256, num_epochs: int = 1000, beta: float = 3.0):
        """Extract policy from V & Q network"""
        print("Start extracting policy...")
        for epoch in range(num_epochs):
            # sample a batch of data
            batch_idx = torch.randint(low=0, high=self.observations.shape[0], size=(batch_size,), device=self.device)
            observations = self.observations[batch_idx]
            actions = self.actions[batch_idx]
            # update policy
            pi_loss = self.update_policy(observations=observations, actions=actions, beta=beta)
            # eval
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {pi_loss}")

    def update_policy(self, observations, actions, beta: float = 1.0):
        """Update policy; beta is inverse temperature"""
        q_target_pred = torch.min(self.target_qf1(torch.cat([observations, actions], dim=1)), self.target_qf2(torch.cat([observations, actions], dim=1)))
        advantages = q_target_pred - self.vf(observations)
        advantages_exp = torch.exp(advantages * beta).detach().clamp(max=EXP_ADV_MAX)  # don't update V & Q network with policy loss
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
        policy_loss = -(advantages_exp * log_probs).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.detach().cpu().numpy()
    
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
            'qf1_network': self.qf1.state_dict(),
            'qf2_network': self.qf2.state_dict(),
            'target_qf1_network': self.target_qf1.state_dict(),
            'target_qf2_network': self.target_qf2.state_dict(),
            'v_network': self.vf.state_dict(),
            'policy_network': self.policy.state_dict()
        }, path)
        print(f"Save model to {path}")

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.qf1.load_state_dict(checkpoint['qf1_network'])
        self.qf2.load_state_dict(checkpoint['qf2_network'])
        self.target_qf1.load_state_dict(checkpoint['target_qf1_network'])
        self.target_qf2.load_state_dict(checkpoint['target_qf2_network'])
        self.vf.load_state_dict(checkpoint['v_network'])
        self.policy.load_state_dict(checkpoint['policy_network'])
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
