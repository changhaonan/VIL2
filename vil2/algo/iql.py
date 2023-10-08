"""Implicit Q learning: https://arxiv.org/abs/2110.06169"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    """Q network; mapping from state-action to value"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(QNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # build network
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

        # activation
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)

        return x


class VNetwork(nn.Module):
    """V network; mapping from state to value"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(VNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # build network
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

        # activation
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)

        return x


class PolicyNetwork(nn.Module):
    """Policy network; mapping from state to action"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, is_gaussian: bool = True):
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.is_gaussian = is_gaussian

        # build network
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

        # activation
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)

        return x


class IQL:
    """Implicit Q learning method"""
    def __init__(self, dataset: dict, config: dict):
        # parameters
        q_lr = config.get('q_lr', 3e-4)
        v_lr = config.get('v_lr', 3e-4)
        policy_lr = config.get('policy_lr', 3e-4)
        policy_std = config.get('policy_std', 0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        state_dim = self.observations.shape[1]
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
        self.qf1_optimizer = torch.optim.Adam(self.qf1.parameters(), lr=q_lr)
        self.qf2_optimizer = torch.optim.Adam(self.qf2.parameters(), lr=q_lr)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=v_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
    
    def update_q_target_network(self, alpha: float = 0.1):
        """Update Q target network with Polyak averaging"""
        for param, target_param in zip(self.qf1.parameters(), self.target_qf1.parameters()):
            target_param.data.copy_(alpha * param.data + (1.0 - alpha) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.target_qf2.parameters()):
            target_param.data.copy_(alpha * param.data + (1.0 - alpha) * target_param.data)
    
    def train(self, batch_size: int = 256, num_epochs: int = 1000, alpha: float = 0.01, tau: float = 0.6, gamma: float = 0.99):
        """Train IQL"""
        # train
        print("Start training Q & V...")
        v_loss_sum = 0.0
        q_loss_sum = 0.0
        eval_freq = 10
        q_update_freq = 1
        target_update_freq = 10
        for epoch in range(num_epochs):
            # sample a batch of data
            batch_idx = torch.randint(low=0, high=self.observations.shape[0], size=(batch_size,), device=self.device)
            observations = self.observations[batch_idx]
            next_observations = self.next_observations[batch_idx]
            actions = self.actions[batch_idx]
            rewards = self.rewards[batch_idx]
            terminals = self.terminals[batch_idx]
            
            # QF loss
            q1_pred = self.qf1(torch.cat([observations, actions], dim=1))
            q2_pred = self.qf2(torch.cat([observations, actions], dim=1))
            target_vf_pred = self.vf(next_observations).detach()  # don't update V network with Q loss
            q_target = rewards + gamma * (1 - terminals) * target_vf_pred
            q_target = q_target.detach()
            qf1_loss = nn.MSELoss()(q1_pred, q_target)
            qf2_loss = nn.MSELoss()(q2_pred, q_target)

            # VF loss
            q_pred = torch.min(q1_pred, q2_pred).detach()  # don't update Q network with V loss
            v_pred = self.vf(observations)
            vf_loss = self.expectile(q_pred - v_pred, tau=tau).mean()

            # update Networks
            if epoch % q_update_freq == 0:
                # update Q network
                self.qf1_optimizer.zero_grad()
                qf1_loss.backward()
                self.qf1_optimizer.step()

                self.qf2_optimizer.zero_grad()
                qf2_loss.backward()
                self.qf2_optimizer.step()

                # update V network
                self.v_optimizer.zero_grad()
                vf_loss.backward()
                self.v_optimizer.step()
            
            if epoch % target_update_freq == 0:
                # update target Q network
                self.update_q_target_network(alpha=alpha)
            
            # eval
            v_loss_sum += vf_loss.detach().cpu().numpy()
            q_loss_sum += (qf1_loss.detach().cpu().numpy() + qf2_loss.detach().cpu().numpy()) / 2.0
            if epoch % eval_freq == 0:
                print(f"Epoch: {epoch}, V Loss: {v_loss_sum / eval_freq}, Q Loss: {q_loss_sum / eval_freq}")
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
        q_pred = torch.min(self.qf1(torch.cat([observations, actions], dim=1)), self.qf2(torch.cat([observations, actions], dim=1)))
        advantages = q_pred - self.vf(observations)
        advantages_exp = torch.exp(advantages / beta).detach()  # don't update V & Q network with policy loss
        if self.policy.is_gaussian:
            action_mean = self.policy(observations)
            action_std = torch.ones_like(action_mean) * self.policy_std
            action_dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = action_dist.log_prob(actions)
        else:
            assert False, "Not implemented"
        policy_loss = -(advantages_exp * log_probs).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.detach().cpu().numpy()
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict action given observations"""
        observations = torch.from_numpy(observations).float().to(self.device)
        self.policy.eval()
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)
        if self.policy.is_gaussian:
            action_mean = self.policy(observations)
            action_std = torch.ones_like(action_mean) * self.policy_std
            action_dist = torch.distributions.Normal(action_mean, action_std)
            action = action_dist.sample()
            if action.shape[0] == 1:
                action = action.squeeze(0)
        else:
            assert False, "Not implemented"
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

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.qf1.load_state_dict(checkpoint['qf1_network'])
        self.qf2.load_state_dict(checkpoint['qf2_network'])
        self.target_qf1.load_state_dict(checkpoint['target_qf1_network'])
        self.target_qf2.load_state_dict(checkpoint['target_qf2_network'])
        self.vf.load_state_dict(checkpoint['v_network'])
        self.policy.load_state_dict(checkpoint['policy_network'])


if __name__ == "__main__":
    import d4rl
    import gym
    import os

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    check_point_path = os.path.join(root_path, 'checkpoint')
    os.makedirs(check_point_path, exist_ok=True)
    # Create the environment
    env = gym.make('maze2d-umaze-v1')

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    dataset = env.get_dataset()
    print(dataset['observations']) # An N x dim_observation Numpy array of observations

    # Alternatively, use d4rl.qlearning_dataset which
    # also adds next_observations.
    dataset = d4rl.qlearning_dataset(env)

    # Bulid IQL module
    config = {
        'batch_size': 256,
        'value_epochs': 100000,
        'policy_epochs': 100000,
        'q_hidden_dim': 64,
        'v_hidden_dim': 64,
        'policy_hidden_dim': 64,
        'enable_save': True,
        'enable_load': False
    }
    iql = IQL(dataset=dataset, config=config)
    iql.train(batch_size=config['batch_size'], num_epochs=config['value_epochs'])
    iql.extract_policy(batch_size=config['batch_size'], num_epochs=config['policy_epochs'])

    # save model
    if config['enable_save']:
        iql.save(os.path.join(check_point_path, 'iql.pth'))

    # test
    if config['enable_load']:
        iql.load(os.path.join(check_point_path, 'iql.pth'))
    for i in range(10):
        obs = env.reset()
        print(f"Episode: {i}")
        while True:
            env.render()
            action = iql.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                print(f"Episode: {i}, Reward: {reward}")
                break