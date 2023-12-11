"""Q-guided Diffusion Policy"""
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
from vil2.algo.offline_policy import OfflinePolicy
from vil2.algo.iql import IQL
import gymnasium as gym


class QGDP(IQL):
    """Q-guided Diffusion Policy"""

    def __init__(self, env: gym.Env, dataset: dict, config: dict):
        super().__init__(env=env, dataset=dataset, config=config)

        # override
        self.sample_size = 64
        self.num_diffusion_iters = 10
        # parameters to search
        self.action_noise_type = config.get('action_noise_type', 'gaussian')
        self.policy_std = config.get('policy_std', 0.1)
        self.lambda_ = config.get('lambda_', 10.0)  # balancing parameter
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

    def update_policy(self, observations, actions, beta: float = 1):
        """IQDF don't need to update policy"""
        return 0.0

    def predict(self, observations: np.ndarray,  **kwargs) -> np.ndarray:
        """a_i = 2 * beta_i / lambda * nabla_{a_{i-1}}Q(s, a_{i-1}) + a_{i-1}"""
        track_a = kwargs.get('track_a', True)
        # Step 1: Sample a batch of actions from pure guassian/uniform noise
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
        s = torch.from_numpy(observations).float().to(self.device)
        s = torch.repeat_interleave(s[None, :], self.sample_size, dim=0)
        a_ts = []
        for t in range(self.num_diffusion_iters):
            if t == 0:
                a_t = torch.from_numpy(a_T).float().to(
                    self.device).requires_grad_(True)
            else:
                a_t = a_t.clone().detach()
                a_t.requires_grad_(True)
            if track_a:
                a_t_numpy = a_t.detach().cpu().numpy()
                # sorted by x
                a_t_numpy = a_t_numpy[a_t_numpy[:, 0].argsort()]
                a_ts.append(a_t_numpy)
            a_t = self.denoise(a_t, s, t)
        # Step 3: Scale back to action range
        action = torch.clamp(a_t, -1, 1)
        if self.action_range is not None:
            action = (action + 1) / 2 * (
                self.action_range[:, 1] - self.action_range[:, 0]) + self.action_range[:, 0]
        # Step 4: Sample one from the batch
        action = action[0].detach().cpu().numpy()
        # Step 5: (Optional) Visualize the diffusion process
        if track_a:
            a_ts = np.array(a_ts)
            a_ts = a_ts.transpose(1, 0, 2)
            a_ts = a_ts * 0.5 + 0.5
            a_ts = a_ts * 255
            a_ts = a_ts.astype(np.uint8)
            # cv2.imwrite(os.path.join(self.log_path, "diffusion.png"), a_ts)
            # scale vis image to height 100
            vis_height = 400
            vis_width = int(vis_height * a_ts.shape[2] / a_ts.shape[1])
            a_ts = cv2.resize(a_ts, (vis_width, vis_height))
            # visualize
            cv2.imshow("diffusion", a_ts)
            cv2.waitKey(1)
        return action

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
