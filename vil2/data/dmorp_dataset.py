import zarr
import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, Dataset
from vil2.utils.data_utils import get_data_stats, normalize_data, unnormalize_data, create_sample_indices, sample_sequence


class DmorpDataset(torch.utils.data.Dataset):
    """Object DiffusionPolicy Dataset"""

    def __init__(self, dataset_path: str, max_scene_size: int = 10, padding_method: str = "zero"):
        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, "r")
        self.max_scene_size = max_scene_size
        # (N, D)
        train_data = {
            "data_stamp": dataset_root["scene_id"][:],
            "pose": dataset_root["pose"][:],
            "sem_id": dataset_root["sem_id"][:],
            "sem_feat": dataset_root["sem_feat"][:],
            "geo_feat": dataset_root["geo_feat"][:],
        }
        self.episode_ends = dataset_root["meta"]["episode_ends"][:]

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            if key == "data_stamp" or key == "sem_id":
                normalized_train_data[key] = data
                continue
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.data_size = self.episode_ends.shape[0]
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.padding_method = padding_method

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if idx == 0:
            start_idx = 0
        else:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        nsample = dict()
        for key, data in self.normalized_train_data.items():
            if self.padding_method == "zero":
                padded_value = np.zeros((self.max_scene_size, data.shape[-1]), dtype=data.dtype)
                padded_value[:end_idx - start_idx] = data[start_idx:end_idx]
            else:
                raise NotImplementedError
            nsample[key] = padded_value
        return nsample
