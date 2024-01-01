import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import functools
from vil2.model.embeddings.pct import PointTransformerEncoderSmall
from vil2.model.embeddings.pct import  DropoutSampler
from vil2.model.embeddings.pct import EncoderMLP

def ve_marginal_prob(x, t, sigma_min=0.01, sigma_max=200):
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std

def perturb_pose(gt_pose: torch.Tensor = None, eps: float = 1e-5):
    batch_size = gt_pose.shape[0]
    random_t = torch.rand(batch_size, device=gt_pose.device) * (1 - eps) + eps
    random_t = random_t.unsqueeze(-1)
    assert len(random_t.shape) == len(gt_pose.shape)
    # Obtaining the standard deviation
    mu, std = ve_marginal_prob(gt_pose, random_t)
    std = std.view(-1, 1)
    # Perturb the data and get estimated score
    z = torch.randn_like(gt_pose)
    perturbed_pose = mu + std * z
    target_score = (-z * std) / (std ** 2)
    return perturbed_pose, std, random_t, target_score

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x
    
class RotHead(nn.Module):
    # Sample Usage:
    # points = torch.rand(2, 1350, 1024)  # batchsize x feature x numofpoint
    # rot_head = RotHead(in_feat_dim=1350, out_dim=3)
    # rot = rot_head(points)
    # print(rot.shape)
    def __init__(self, in_feat_dim, out_dim=3):
        super(RotHead, self).__init__()
        self.f = in_feat_dim
        self.k = out_dim

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()

        return x
    
class TransHead(nn.Module):
    # Sample Usage:
    # feature = torch.rand(10, 1896, 1000)
    # net = TransHead(in_feat_dim=1896, out_dim=3)
    # out = net(feature)
    # print(out.shape)

    def __init__(self, in_feat_dim, out_dim=3):
        super(TransHead, self).__init__()
        self.f = in_feat_dim
        self.k = out_dim

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        return x