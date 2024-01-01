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
from vil2.utils.normalizer import normalizer
from diffusers.training_utils import EMAModel


Transition = namedtuple(
    'Transition', ('desired_goal', 'achieved_goal', 'state', 'action', 'next_state', 'reward', 'terminated', 't_remaining'))


class BDNPolicy:
    """Bellman Diffusion Network Policy"""

    def __init__(self, env: gym.Env, cfg, vision_encoder, noise_pred_net, policy_net=None, policy_fn=None) -> None:
        # Parameters
        self.cfg = cfg
        self.env: gym.Env = env
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        obs, _ = self.env.reset()
        self.goal_dim = obs["desired_goal"].shape[0]
        self.state_dim = obs["observation"].shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_range = np.array([env.action_space.low, env.action_space.high])  # (2, action_dim)
        buffer_size = cfg.get('buffer_size', 100000)

        # memory structure
        self.finite_horizon = cfg.get('finite_horizon', 50)
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
        self.log_path = cfg.get('log_path', None)
        self.max_epoch_per_episode = cfg.get('max_epoch_per_episode', 50)
        self.log_period = cfg.get('log_period', 100)
        self.alpha = cfg.get('alpha', 0.1)
        self.her_tolerance = cfg.get('her_tolerance', 0.05)
        self.target_update_period = cfg.get('target_update_period', 1)

        # Policy: mapping from (g, s, n) to action
        self.policy_fn = policy_fn
        # SOM: state occupancy measure
        # epsilon(x| s', a', t, n) predict the distribution of future states
        # here s' is the next state
        # a' = pi(s') is the action taken by the policy
        # t is the time step
        # n is the step remain
        self.nets = nn.ModuleDict({
            'noise_pred_net': noise_pred_net,
            'noise_pred_net_target': noise_pred_net,
        }).to(self.device)
        # self.nets['noise_pred_net_target'].requires_grad_(False)
        # noise scheduler
        self.num_diffusion_iters = cfg.MODEL.NUM_DIFFUSION_ITERS
        beta_schedule = cfg.MODEL.BETA_SCHEDULE
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

        # Normalizer
        self.o_norm = normalizer(size=self.state_dim)
        self.g_norm = normalizer(size=self.goal_dim)

    def update_som_target_network(self, alpha: float = 0.1):
        """Update som target network with Polyak averaging"""
        for param, target_param in zip(self.nets['noise_pred_net'].parameters(), self.nets['noise_pred_net_target'].parameters()):
            target_param.data.copy_(alpha * param.data +
                                    (1 - alpha) * target_param.data)

    def train(self, batch_size: int, num_episode: int, train_policy: bool = False):
        """Train BDN with goal_pi as the goal"""
        enable_render = True
        self.nets['noise_pred_net'].train()
        # self.policy.train()
        # Standard ADAM optimizer
        som_optimizier = torch.optim.AdamW(params=self.nets['noise_pred_net'].parameters(), lr=1e-4, weight_decay=1e-6)

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
                    if self.policy_fn is not None and train_policy is False:
                        action = self.policy_fn(obs).squeeze()
                    else:
                        action = self.predict(observation, desired_goal, step_remain, is_deterministic=True)
                        action = action.detach().cpu().numpy().squeeze()
                        # unnormalize
                        action = np.clip(action, -1.0, 1.0)
                        action = (action + 1.0) / 2.0 * (self.action_range[1] - self.action_range[0]) + self.action_range[0]
                    obs, reward, terminated, truncated, _ = self.env.step(action)

                    if reward > 0:
                        terminated = True  # force early termination
                    if step_epoch >= self.finite_horizon - 1:
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
            mb_obs = np.stack(mb_obs)
            mb_ag = np.stack(mb_ag)
            mb_g = np.stack(mb_g)
            mb_actions = np.stack(mb_actions)
            self.replay_buffer.store_episode((mb_obs, mb_ag, mb_g, mb_actions))
            self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

            # ----------------- optimize model -----------------
            # optimize som
            som_loss = self.optimize_som(idx_episode, batch_size=batch_size)
            som_loss.backward()
            som_optimizier.step()
            som_optimizier.zero_grad()

            # optimize policy
            if train_policy:
                pi_loss = self.optimize_policy(idx_episode, batch_size=batch_size)
            else:
                pi_loss = 0.0

            # update som target network
            if idx_episode % self.target_update_period == 0:
                self.update_som_target_network(alpha=self.alpha)

            # ----------------- logging -----------------
            t_episode.set_description(f"Episode: {idx_episode}, reward: {reward_episode}; som_loss: {som_loss.sum().item()}; pi_loss: {pi_loss}")
            # if idx_episode % self.log_period == 0:
            #     if self.log_path is not None:
            #         with open(os.path.join(self.log_path, "eval.txt"), "a") as f:
            #             f.write(
            #                 f"Episode: {idx_episode}| reward: {reward_episode}; som_loss: {som_loss}; pi_loss: {pi_loss}\n")

    def predict(self, obs, goal, t_remaining, is_deterministic=False):
        """Predict action given observation & goal"""
        # normalize
        obs = self.o_norm.normalize(obs)
        goal = self.g_norm.normalize(goal)
        # move to gpu
        obs = torch.from_numpy(obs[None, :]).float().to(self.device)
        goal = torch.from_numpy(goal[None, :]).float().to(self.device)
        t_remaining = torch.tensor([[t_remaining]]).float().to(self.device)
        # predict action
        mean = self.policy(torch.cat([goal, obs, t_remaining], dim=1))
        if not is_deterministic:
            # build distribution
            dist = torch.distributions.Normal(mean, self.policy_std)
            # sample action
            action = dist.sample()
            return action
        else:
            return mean

    def her_reward_func(self, achieved_goal, desired_goal, info):
        """HER reward function"""
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > self.her_tolerance).astype(np.float32)

    def optimize_som(self, epoch: int, batch_size: int, use_bellman: bool = False):
        """Optimize SOM (State Occupancy Measure) Model"""
        # ---------------- sample transitions ----------------
        transitions = self.replay_buffer.sample(batch_size)
        transitions = self._normalize_transition(transitions)
        obs = torch.from_numpy(transitions['obs']).float().to(self.device)
        obs_next = torch.from_numpy(transitions['obs_next']).float().to(self.device)
        ag_next = torch.from_numpy(transitions['ag_next']).float().to(self.device)
        g = torch.from_numpy(transitions['g']).float().to(self.device)
        t_remaining = torch.from_numpy(transitions['t_remaining']).float().to(self.device)
        future_g = torch.from_numpy(transitions['future_g']).float().to(self.device)

        # ---------------- sample steps & noise ----------------
        timesteps = torch.randint(0, self.num_diffusion_iters, (batch_size,)).long().to(self.device)
        noise = torch.randn(batch_size, self.goal_dim, device=self.device)
        noisy_ag_next = self.noise_scheduler.add_noise(ag_next, noise, timesteps)  # apply noise to g_next
        noisy_future_g = self.noise_scheduler.add_noise(future_g, noise, timesteps)  # apply noise to future_g

        # ---------------- Diffusion loss ----------------
        obs_dict = {"observation": transitions['obs'], "desired_goal": transitions['g']}
        if self.policy_fn is not None:
            pi_obs = self.policy_fn(obs_dict)
            pi_obs = torch.from_numpy(pi_obs).float().to(self.device)
        else:
            raise NotImplementedError
        # som_cond = torch.cat([obs, pi_obs, t_remaining], dim=1)
        som_cond = None
        noise_g_g_next = self.nets['noise_pred_net'](noisy_ag_next, timesteps, global_cond=som_cond)  # SOM from g to g'
        loss_diffusion = nn.functional.mse_loss(noise_g_g_next, noise)

        # ---------------- Bellman loss ----------------
        # if use_bellman:
        #     pi_obs_next = self.policy(torch.cat([g, obs_next, t_remaining - 1], dim=1)).detach()  # pi(obs_next)
        #     som_cond_next = torch.cat([obs_next, pi_obs_next, t_remaining - 1], dim=1)
        #     noise_g_next_g_f = self.som_noise_target(noisy_future_g, som_cond_next, timesteps)  # SOM from g' to g_f
        #     noise_g_g_f = self.som_noise(noisy_future_g, som_cond, timesteps)  # SOM from g to g_f
        #     loss_bellman = nn.functional.mse_loss(noise_g_next_g_f, noise_g_g_f)

        # ---------------- loss ----------------
        return loss_diffusion

    def optimize_policy(self, epoch: int, batch_size: int):
        """Optimize Policy"""
        # ---------------- sample transitions ----------------
        transitions = self.replay_buffer.sample(batch_size)
        transitions = self._normalize_transition(transitions)
        obs = torch.from_numpy(transitions['obs']).float().to(self.device)
        policy_g = torch.from_numpy(transitions['policy_g']).float().to(self.device)
        t_remaining = torch.from_numpy(transitions['t_remaining']).float().to(self.device)

        # ---------------- sample steps & noise ----------------
        timesteps = torch.randint(0, self.num_diffusion_iters, (batch_size, 1)).long().to(self.device)
        noise = torch.randn(batch_size, self.goal_dim, device=self.device)
        noisy_policy_g = self.noise_scheduler.add_noise(policy_g, noise, timesteps)  # apply noise to policy_g
        pi_obs = self.policy(torch.cat([policy_g, obs, t_remaining], dim=1))  # pi(obs)
        som_cond = torch.cat([obs, pi_obs, t_remaining], dim=1)
        noise_g_goal = self.som_noise_target(noisy_policy_g, som_cond, timesteps)  # SOM from g to goal

        # ---------------- loss ----------------
        loss = nn.functional.mse_loss(noise_g_goal, noise)
        loss.backward()
        self.policy.optimizer.step()
        self.policy.optimizer.zero_grad()
        return loss.item()

    def save(self, path):
        """Save model to path"""
        torch.save(self.nets.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model from path"""
        state_dict = torch.load(path, map_location=self.device)
        self.nets.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

     # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_sampler.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -np.inf, np.inf)
        g = np.clip(g, -np.inf, np.inf)
        return o, g

    def _normalize_transition(self, transitions: dict):
        """Normalize the observation and goal"""
        o, o_next = transitions['obs'], transitions['obs_next']
        g, future_g, policy_g, ag_next = transitions['g'], transitions['future_g'], transitions['policy_g'], transitions['ag_next']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        _, transitions['future_g'] = self._preproc_og(o, future_g)
        _, transitions['policy_g'] = self._preproc_og(o, policy_g)
        transitions['obs_next'], transitions['ag_next'] = self._preproc_og(o_next, ag_next)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_norm = self.g_norm.normalize(transitions['g'])
        future_g_norm = self.g_norm.normalize(transitions['future_g'])
        policy_g_norm = self.g_norm.normalize(transitions['policy_g'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])
        # update the transition
        transitions['obs'] = obs_norm
        transitions['g'] = g_norm
        transitions['future_g'] = future_g_norm
        transitions['policy_g'] = policy_g_norm
        transitions['obs_next'] = obs_next_norm
        transitions['ag_next'] = ag_next_norm

        return transitions
