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
from vil2.algo.replay_buffer import ReplayBuffer, HERSampler


Transition = namedtuple(
    'Transition', ('desired_goal', 'achieved_goal', 'state', 'action', 'next_state', 'reward', 'terminated', 'steps_remain'))


class DictConcatWrapper(gym.ObservationWrapper):
    """Concatenate all observations into one vector"""

    def observation(self, obs):
        """override"""
        if isinstance(obs, dict):
            return np.concatenate([obs[obs_name] for obs_name in obs.keys()])
        return obs


class BDNPolicy:
    """Bellman Diffusion Network Policy"""

    def __init__(self, env: gym.Env, config: dict) -> None:
        # Parameters
        self.config: dict = config
        self.env: gym.Env = env
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        obs, _ = self.env.reset()
        self.goal_dim = obs["desired_goal"].shape[0]
        self.state_dim = obs["observation"].shape[0]
        self.action_dim = env.action_space.shape[0]
        buffer_size = config.get('buffer_size', 100000)

        # memory structure
        self.finite_horizon = config['finite_horizon']
        self.her_sampler = HERSampler(replay_strategy='future', replay_k=4, reward_func=self.her_reward_func)
        self.replay_buffer = ReplayBuffer(
            env_params={
                'max_timesteps': self.finite_horizon,
                'obs': self.state_dim,
                'goal': self.goal_dim,
                'action': self.action_dim,
            },
            buffer_size=buffer_size,
            sample_func=lambda x, y: self.her_sampler.sample_her_transitions(x, y),
        )

        # training related
        self.log_path = config.get('log_path', None)
        self.max_epoch_per_episode = config.get('max_epoch_per_episode', 50)
        self.log_period = config.get('log_period', 100)
        self.alpha = config.get('alpha', 0.1)
        self.her_tolerance = config.get('her_tolerance', 0.05)
        self.target_update_period = config.get('target_update_period', 1)

        # Policy: mapping from (g, s, n) to action
        policy_input_dim = self.goal_dim + self.state_dim + 1
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
        # n is the step remain
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

    def train(self, batch_size: int, num_episode: int):
        """Train BDN with goal_pi as the goal"""
        enable_render = True
        t_episode = tqdm(range(num_episode))
        for idx_episode in t_episode:
            # ----------------- collect data -----------------
            mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
            reward_episode = 0

            for iter_epoch in range(self.max_epoch_per_episode):
                obs, _ = self.env.reset()
                obs_epoch, ag_epoch, g_epoch, actions_epoch = [obs["observation"]], [obs["achieved_goal"]], [], []
                step_epoch = 0
                while True:
                    desired_goal = obs["desired_goal"]
                    observation = obs["observation"]
                    step_remain = self.finite_horizon - step_epoch
                    # step the environment
                    action = self.predict(observation, desired_goal, step_remain, is_deterministic=False)
                    obs, reward, terminated, truncated, _ = self.env.step(action)

                    if reward > 0:
                        terminated = True  # force early termination
                    if step_epoch >= self.finite_horizon:
                        terminated = True  # force early termination

                    # save transition
                    observation = obs["observation"]
                    achieved_goal = obs["achieved_goal"]
                    obs_epoch.append(observation.copy())
                    ag_epoch.append(achieved_goal.copy())
                    g_epoch.append(desired_goal.copy())
                    actions_epoch.append(action.copy())

                    # update info
                    reward_episode += reward
                    step_epoch += 1

                    # reset
                    if terminated or truncated:
                        break
                # collect epoch
                mb_obs.append(np.stack(obs_epoch))
                mb_ag.append(np.stack(ag_epoch))
                mb_g.append(np.stack(g_epoch))
                mb_actions.append(np.stack(actions_epoch))

            # ----------------- store data -----------------
            # finished an episode
            self.replay_buffer.store_episode(
                (np.stack(mb_obs), np.stack(mb_ag), np.stack(mb_g), np.stack(mb_actions)))

            # log
            t_episode.set_description(f"Episode: {idx_episode}, Reward: {reward_episode}")
            if idx_episode % self.log_period == 0:
                if self.log_path is not None:
                    with open(os.path.join(self.log_path, "eval.txt"), "a") as f:
                        f.write(
                            f"Episode: {idx_episode}| reward: {reward_episode}\n")

            # ----------------- optimize model -----------------
            # optimize som
            self.optimize_som(idx_episode, batch_size=batch_size)

            # optimize policy
            self.optimize_policy(idx_episode, batch_size=batch_size)

            # update som target network
            if idx_episode % self.target_update_period == 0:
                self.update_som_target_network(alpha=self.alpha)

    def predict(self, obs, goal, step_remain, is_deterministic=False):
        """Predict action given observation & goal"""
        self.policy.eval()
        # move to gpu
        obs = torch.from_numpy(obs[None, :]).float().to(self.device)
        goal = torch.from_numpy(goal[None, :]).float().to(self.device)
        step_remain = torch.tensor([[step_remain]]).float().to(self.device)
        # predict action
        mean = self.policy(torch.cat([goal, obs, step_remain], dim=1))
        if not is_deterministic:
            # build distribution
            dist = torch.distributions.Normal(mean, self.policy_std)
            # sample action
            action = dist.sample()
            return action.detach().cpu().numpy().squeeze()
        else:
            return mean.detach().cpu().numpy().squeeze()

    def her_reward_func(self, achieved_goal, desired_goal, info):
        """HER reward function"""
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > self.her_tolerance).astype(np.float32)

    def optimize_som(self, epoch: int, batch_size: int):
        """Optimize SOM (State Occupancy Measure) Model"""
        # sample transitions
        transitions = self.replay_buffer.sample(batch_size)
        pass

    def optimize_policy(self, epoch: int, batch_size: int):
        """Optimize Policy"""
        pass

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
