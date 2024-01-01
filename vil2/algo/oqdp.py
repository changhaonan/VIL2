"""Online Q-value Diffusion Policy (OQDF)"""
from __future__ import annotations
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import vil2.utils.misc_utils as misc_utils
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from vil2.algo.model import QNetwork, VNetwork, PolicyNetwork
import gymnasium as gym
from collections import namedtuple, deque
import random
from tqdm.auto import tqdm

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))


class DictConcatWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        if isinstance(obs, dict):
            return np.concatenate([obs[obs_name] for obs_name in obs.keys()])
        return obs


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def push(self, obs, action, next_obs, reward, terminated):
        """Save a transition"""
        obs = torch.from_numpy(obs[None, :]).float().to(self.device)
        action = torch.from_numpy(action[None, :]).float().to(self.device)
        next_obs = torch.from_numpy(next_obs[None, :]).float().to(self.device)
        reward = torch.tensor([[reward]]).float().to(self.device)
        terminated = torch.tensor([[terminated]]).float().to(self.device)
        self.memory.append(Transition(
            obs, action, next_obs, reward, terminated))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OQDP:
    """Online Q-value Diffusion policy"""

    def __init__(self, env: gym.Env, config: dict) -> None:
        # parameters
        q_lr = config.get('q_lr', 1e-3)
        v_lr = config.get('v_lr', 1e-3)
        weight_decay = config.get('weight_decay', 0.0)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # parameters to search
        self.action_noise_type = config.get('action_noise_type', 'gaussian')
        self.policy_std = config.get('policy_std', 0.1)
        self.lambda_ = config.get('lambda', 10.0)  # balancing parameter
        self.sample_size = config.get('sample_size', 1)
        self.num_diffusion_iters = config.get('num_diffusion_iters', 10)
        self.a_lazy_init = config.get('a_lazy_init', True)
        # optimization related
        self.q_update_freq = config.get('q_update_freq', 1)
        self.target_update_freq = config.get('target_update_freq', 10)
        self.alpha = config.get('alpha', 0.1)
        self.gamma = config.get('gamma', 0.99)  # reward discount
        self.tau = config.get('tau', 0.7)
        # beta scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )
        # memory
        memory_capcity = config.get('max_memory_size', 100000)
        self.memory = ReplayMemory(capacity=memory_capcity)
        # path related
        self.log_path = config.get('log_path', None)
        # env
        self.env = DictConcatWrapper(env)  # concat observations
        # action space
        self.action_space = self.env.action_space
        self.action_dim = self.env.action_space.shape[0]
        self.action_range = np.array(
            [self.env.action_space.low, self.env.action_space.high]).transpose()
        # observation space
        self.observation_space = self.env.observation_space
        self.state_dim = 0
        for obs_name in self.observation_space.spaces.keys():
            self.state_dim += self.observation_space[obs_name].shape[0]

        # init networks
        q_input_dim = self.state_dim + self.action_dim
        q_hidden_dim = config['q_hidden_dim']
        q_output_dim = 1
        v_input_dim = self.state_dim
        v_hidden_dim = config['v_hidden_dim']
        v_output_dim = 1

        self.qf1 = QNetwork(input_dim=q_input_dim,
                            hidden_dim=q_hidden_dim, output_dim=q_output_dim)
        self.qf2 = QNetwork(input_dim=q_input_dim,
                            hidden_dim=q_hidden_dim, output_dim=q_output_dim)
        self.target_qf1 = QNetwork(
            input_dim=q_input_dim, hidden_dim=q_hidden_dim, output_dim=q_output_dim)
        self.target_qf2 = QNetwork(
            input_dim=q_input_dim, hidden_dim=q_hidden_dim, output_dim=q_output_dim)
        self.vf = VNetwork(input_dim=v_input_dim,
                           hidden_dim=v_hidden_dim, output_dim=v_output_dim)

        self.qf1.to(self.device)
        self.qf2.to(self.device)
        self.target_qf1.to(self.device)
        self.target_qf2.to(self.device)
        self.vf.to(self.device)

        # init optimizer
        self.qf1_optimizer = torch.optim.Adam(
            self.qf1.parameters(), lr=q_lr, weight_decay=weight_decay)
        self.qf2_optimizer = torch.optim.Adam(
            self.qf2.parameters(), lr=q_lr, weight_decay=weight_decay)
        self.v_optimizer = torch.optim.Adam(
            self.vf.parameters(), lr=v_lr, weight_decay=weight_decay)

    def update_q_target_network(self, alpha: float = 0.1):
        """Update Q target network with Polyak averaging"""
        for param, target_param in zip(self.qf1.parameters(), self.target_qf1.parameters()):
            target_param.data.copy_(
                alpha * param.data + (1.0 - alpha) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.target_qf2.parameters()):
            target_param.data.copy_(
                alpha * param.data + (1.0 - alpha) * target_param.data)

    def train(self, batch_size: int, num_epochs: int):
        """Train OQDF"""
        enable_render = True

        t_epoch = tqdm(range(num_epochs))
        for epoch in t_epoch:
            obs, _ = self.env.reset(seed=epoch)
            step_epoch = 0
            reward_epoch = 0.0
            actions = None
            epoch_dir = os.path.join(self.log_path, f"epoch-{epoch}")
            while True:
                actions = self.predict(
                    obs, prev_actions=actions, noise_level=self.policy_std)
                action = actions[np.random.choice(actions.shape[0])]
                next_obs, reward, terminated, truncated, info = self.env.step(
                    action)

                if reward > 0:
                    terminated = True  # force early termination
                    reward = 10.0  # amplify reward
                # move to gpu
                # add to memory
                self.memory.push(obs, action, next_obs, reward, terminated)
                # update info
                obs = next_obs
                reward_epoch += reward
                step_epoch += 1
                if terminated or truncated:
                    break
                # do visualization
                if enable_render and epoch % 50 == 0:
                    image = self.env.render()
                    if self.log_path is not None:
                        if not os.path.exists(epoch_dir):
                            os.makedirs(epoch_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(
                            epoch_dir, f"step-{step_epoch}.png"), image)
                # optimize model
                self.optimize_model(epoch, batch_size=batch_size)
                if terminated or truncated:
                    if epoch == 0:
                        print(f"========== Epoch: {epoch} ==========")
                    print(
                        f"Episode: {epoch}, Reward: {reward}, Step: {step_epoch}")
                    break
            t_epoch.set_description(
                f"Epoch: {epoch}, Reward: {reward_epoch}, Step: {step_epoch}")
            # log
            if epoch % 1 == 0:
                if self.log_path is not None:
                    with open(os.path.join(self.log_path, "eval.txt"), "a") as f:
                        f.write(
                            f"Epoch: {epoch}| reward: {reward_epoch}, step: {step_epoch}\n")

    def optimize_model(self, epoch: int, batch_size: int = 32):
        """Optimize model"""
        if len(self.memory) < batch_size:
            return
        # sample a batch of data
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        observations = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_observations = torch.cat(batch.next_state)
        terminals = torch.cat(batch.terminated)

        # update using IQL loss
        # QF loss
        q1_pred = self.qf1(torch.cat([observations, actions], dim=1))
        q2_pred = self.qf2(torch.cat([observations, actions], dim=1))
        # don't update V network with Q loss
        next_vf_pred = self.vf(next_observations).detach()
        q_target = rewards + self.gamma * (1 - terminals) * next_vf_pred
        qf1_loss = nn.MSELoss()(q1_pred, q_target)
        qf2_loss = nn.MSELoss()(q2_pred, q_target)

        # VF loss
        q1_target_pred = self.target_qf1(
            torch.cat([observations, actions], dim=1))
        q2_target_pred = self.target_qf2(
            torch.cat([observations, actions], dim=1))
        # don't update Q network with V loss
        q_target_pred = torch.min(q1_target_pred, q2_target_pred).detach()
        v_pred = self.vf(observations)
        vf_loss = self.expectile(q_target_pred - v_pred, tau=self.tau).mean()

        # update Networks
        if epoch % self.q_update_freq == 0:
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

        if epoch % self.target_update_freq == 0:
            # update target Q network
            self.update_q_target_network(alpha=self.alpha)

    def expectile(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """Expectile function"""
        return torch.where(x < 0, tau * x.square(), (1 - tau) * x.square())

    def predict(self, observations: np.ndarray | dict[str, np.ndarray],  **kwargs) -> np.ndarray:
        """a_i = 2 * beta_i / lambda * nabla_{a_{i-1}}Q(s, a_{i-1}) + a_{i-1}"""
        noise_level = kwargs.get('noise_level', 0.0)
        # Step 1: Sample a batch of actions from pure guassian/uniform noise or previous actions
        if self.a_lazy_init and (prev_actions := kwargs.get('prev_actions', None)) is not None:
            a_T = prev_actions + \
                np.random.normal(loc=0.0, scale=self.policy_std, size=(
                    self.sample_size, self.action_dim))
        else:
            if self.action_noise_type == "gaussian":
                a_T = np.random.normal(
                    loc=0.0, scale=1.0/3.0, size=(self.sample_size, self.action_dim))
            elif self.action_noise_type == "uniform":
                a_T = np.random.uniform(
                    low=-1.0, high=1.0, size=(self.sample_size, self.action_dim))
            else:
                raise ValueError(
                    "Unknown action noise type: {}".format(self.action_noise_type))
        # Step 2: Do denoising process
        if isinstance(observations, np.ndarray):
            s = torch.from_numpy(observations).float().to(self.device)
        else:
            s = observations
        s = torch.repeat_interleave(
            s[None, :], self.sample_size, dim=0).detach()  # no-gradient
        for t in range(self.num_diffusion_iters):
            if t == 0:
                a_t = torch.from_numpy(a_T).float().to(
                    self.device).requires_grad_(True)
            else:
                a_t = a_t.clone().detach()
                a_t.requires_grad_(True)
            a_t = self.denoise(a_t, s, t)
        # Step 3: Scale back to action range
        # add noise (for exploration)
        if noise_level > 0.0:
            a_t = a_t + \
                torch.from_numpy(np.random.normal(
                    loc=0.0, scale=noise_level, size=a_t.shape)).float().to(self.device)
        actions = torch.clamp(a_t, -1, 1)
        actions = actions.detach().cpu().numpy()
        if self.action_range is not None:
            actions = (actions + 1) / 2 * (
                self.action_range[:, 1] - self.action_range[:, 0]) + self.action_range[:, 0]
        return actions

    def denoise(self, a: torch.Tensor, s: torch.Tensor, t: int):
        """Denoising step"""
        q1_target_pred = self.target_qf1(
            torch.cat([s, a], dim=1))
        q2_target_pred = self.target_qf2(
            torch.cat([s, a], dim=1))
        q_target_pred = torch.min(q1_target_pred, q2_target_pred)

        # compute gradient
        q_target_pred.backward(torch.ones_like(q_target_pred))
        a_grad = a.grad

        # update a
        a = 2 * self.noise_scheduler.betas[t] * a_grad / self.lambda_ + a
        return a

    def save(self, path: str):
        """Save model"""
        torch.save({
            'qf1_network': self.qf1.state_dict(),
            'qf2_network': self.qf2.state_dict(),
            'target_qf1_network': self.target_qf1.state_dict(),
            'target_qf2_network': self.target_qf2.state_dict(),
            'v_network': self.vf.state_dict(),
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
        print(f"Load model from {path}")
