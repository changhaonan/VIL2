"""Miscellaneous utilities."""
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch

# -----------------------------------------------------------------------------
# RL Utils
# -----------------------------------------------------------------------------

def draw_V_map(observations: np.ndarray, V_f: nn.Module, sample_ratio: float = 0.01, output_path: str = None):
    obs = torch.from_numpy(observations).float().to("cuda")
    V = V_f(obs).detach().cpu().numpy()
    num_samples = int(obs.shape[0] * sample_ratio)
    obs_samples = obs[np.random.choice(obs.shape[0], num_samples, replace=False)]
    value = V_f(obs_samples).detach().cpu().numpy()
    obs_samples = obs_samples.detach().cpu().numpy()
    plt.clf()
    plt.scatter(obs_samples[:, 0], obs_samples[:, 1], s=0.01, c=value)
    plt.colorbar()
    plt.title("V")
    # set axis equal
    plt.gca().set_aspect('equal', adjustable='box')
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def draw_Q_map(observations: np.ndarray, Q_f: nn.Module, action: np.ndarray, sample_ratio: float = 0.01, output_path: str = None):
    obs = torch.from_numpy(observations).float().to("cuda")
    num_samples = int(obs.shape[0] * sample_ratio)
    obs_samples = obs[np.random.choice(obs.shape[0], num_samples, replace=False)]
    action_samples = torch.from_numpy(np.tile(action, (num_samples, 1))).float().to("cuda")
    Q = Q_f(torch.cat([obs_samples, action_samples], axis=1)).detach().cpu().numpy()
    obs_samples = obs_samples.detach().cpu().numpy()
    plt.clf()
    plt.scatter(obs_samples[:, 0], obs_samples[:, 1], s=0.01, c=Q)
    plt.colorbar()
    plt.title(f"Q under: {action}")
    # set axis equal
    plt.gca().set_aspect('equal', adjustable='box')
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def draw_A_map(observations: np.ndarray, Q_f1: nn.Module, Q_f2: nn.Module, V_f: nn.Module, action: np.ndarray, sample_ratio: float = 0.01, output_path: str = None):
    obs = torch.from_numpy(observations).float().to("cuda")
    num_samples = int(obs.shape[0] * sample_ratio)
    obs_samples = obs[np.random.choice(obs.shape[0], num_samples, replace=False)]
    action_samples = torch.from_numpy(np.tile(action, (num_samples, 1))).float().to("cuda")
    Q1 = Q_f1(torch.cat([obs_samples, action_samples], axis=1)).detach().cpu().numpy()
    Q2 = Q_f2(torch.cat([obs_samples, action_samples], axis=1)).detach().cpu().numpy()
    Q = np.minimum(Q1, Q2)
    V = V_f(obs_samples).detach().cpu().numpy()
    A = Q - V
    obs_samples = obs_samples.detach().cpu().numpy()
    plt.clf()
    plt.scatter(obs_samples[:, 0], obs_samples[:, 1], s=0.01, c=A)
    plt.colorbar()
    plt.title(f"A under: {action}")
    # set axis equal
    plt.gca().set_aspect('equal', adjustable='box')
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()