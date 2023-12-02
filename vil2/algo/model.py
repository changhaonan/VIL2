from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import functional as F
from vil2.algo.model_utils import PositionalEmbedding


# Utils
def noise_scheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
    """Noise scheduler for diffusion model"""
    if beta_schedule == "linear":
        betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32)
    elif beta_schedule == "quadratic":
        betas = torch.linspace(
            beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
    else:
        raise ValueError("Unknown beta schedule: {}".format(beta_schedule))
    return betas


# Networks
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


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class NoiseNetwork(nn.Module):
    """Diffusion noise network; naive version"""

    def __init__(self, input_size: int, condition_size: int, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128, time_emb: str = "learnable", input_emb: str = "learnable"):
        super(NoiseNetwork, self).__init__()
        self.time_mlp = PositionalEmbedding(1, emb_size, time_emb)
        self.input_mlp = PositionalEmbedding(input_size, emb_size, input_emb)
        self.condition_mlp = PositionalEmbedding(condition_size, emb_size, input_emb)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp.layer) + len(self.condition_mlp.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, input_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x being input, c being condition, t being timestep"""
        x_emb = self.input_mlp(x)
        c_emb = self.condition_mlp(c)
        t_emb = self.time_mlp(t)
        x = torch.cat([x_emb, c_emb, t_emb], dim=-1)
        x = self.joint_mlp(x)
        return x


class NoiseScheduler(nn.Module):
    """Diffusion model"""

    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float):
        super(NoiseScheduler, self).__init__()
        # constants
        betas = noise_scheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear"
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5
        sqrt_inv_alphas_cumprod = torch.sqrt(1 / alphas_cumprod)
        sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / alphas_cumprod - 1)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        # register
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_inv_alphas_cumprod", sqrt_inv_alphas_cumprod)
        self.register_buffer("sqrt_inv_alphas_cumprod_minus_one", sqrt_inv_alphas_cumprod_minus_one)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise
