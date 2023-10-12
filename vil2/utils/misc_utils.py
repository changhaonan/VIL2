"""Miscellaneous utilities."""
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch


def position_encoding(x: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e2):
    """Sinusoidal Position Encoding."""
    length = x.shape[-1]
    channels = x.shape[-2]
    position = torch.arange(length, dtype=torch.float32, device=x.device)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32, device=x.device) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    signal = signal.unsqueeze(0).repeat(x.shape[:-1] + (1,))
    return signal

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


# -----------------------------------------------------------------------------
# Visualization Utils
# -----------------------------------------------------------------------------


def plot_hist_scatter(data, title: str = "hist and scatter plot visualization of data", fig_name: str = "dataset.png", save_path: str = "output"):
    """
    Args: 
        data: (batch_size, state_dim)
    """
    plt.title(title)
    fig, axes = plt.subplots(nrows=data.shape[1], ncols=data.shape[1], figsize=(20, 20))  # Adjust the figsize as needed
    # We'll create an aggregated version of data to compute statistics across all data
    all_data = np.stack(data, axis=0)
    # Compute max and min for the data for consistent axis limits
    data_min = all_data.min(axis=0)
    data_max = all_data.max(axis=0)
    # Loop over all pairs of features (including pairs with themselves)
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            # If diagonal, plot a histogram
            if i == j:
                axes[i, j].hist(data[:, i], bins=30, color='gray', alpha=0.7)
                axes[i, j].set_xlim([-1.0, 1.0])  # Set x-axis range
                axes[i, j].set_ylim([-1.0, 1.0])  # Set y-axis range
                axes[i, j].set_yticklabels([])
                axes[i, j].set_xticklabels([])
            # If off-diagonal, plot a scatter plot
            else:
                axes[i, j].scatter(data[:, j], data[:, i], alpha=0.5, s=5, color='blue')
                axes[i, j].set_xlim([-1.0, 1.0])  # Set x-axis range
                axes[i, j].set_ylim([-1.0, 1.0]) # Set y-axis range
                axes[i, j].set_yticklabels([])
                axes[i, j].set_xticklabels([])

    # Set tighter layout
    plt.tight_layout()
    # Save the figure if desired
    plt.savefig(f"{save_path}/{fig_name}")