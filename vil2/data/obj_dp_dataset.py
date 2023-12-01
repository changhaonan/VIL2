"""Objwise Diffusion Policy Dataset"""
from __future__ import annotations
import numpy as np
import torch
import zarr
import pickle
from tqdm.auto import tqdm
import os

# ------------  utils ------------


def create_sample_indices(
    episode_ends: np.ndarray, sequence_length: int, pad_before: int = 0, pad_after: int = 0
):
    """Buffer refers to the chunk that with padding;
    Sample refers to the chunk that contains real data"""
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence_by_key(
    train_data, key, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx
):
    input_arr = train_data[key]
    sample = input_arr[buffer_start_idx:buffer_end_idx]
    data = sample
    if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
        data = np.zeros(shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
        if sample_start_idx > 0:
            data[:sample_start_idx] = sample[0]
        if sample_end_idx < sequence_length:
            data[sample_end_idx:] = sample[-1]
        data[sample_start_idx:sample_end_idx] = sample
    return data


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data

# ------------ geometry encoder ------------


def encode_geometry(geometry: np.ndarray, encoder: str, aggretator: str):
    """Encode geometry
    Args:
        geometry: (num_vertices, 3)
        encoder: "pointnet"
    Return:
        encoded_geometry: (num_vertices, 128)
    """
    if encoder == "fake":
        point_feats = np.ones((geometry.shape[0], 128))
    else:
        raise ValueError(f"Unknown encoder: {encoder}")
    if aggretator == "mean":
        aggr_point_feats = point_feats.mean(axis=0)
    elif aggretator == "max":
        aggr_point_feats = point_feats.max(axis=0)
    else:
        raise ValueError(f"Unknown aggretator: {aggretator}")
    return aggr_point_feats


# build dataset


def parser_obs(obs: dict, predict_type: str, carrier_type: str, geometry_encoder: str, aggretator: str, horizon_length: int = 10):
    """Parse observation
    Return:
        img: (H, W, 3)
        depth: (H, W)
        active_obj_super_voxel_traj: (horizon_length, 3/6)
        active_obj_geometry_feat: (num_super_voxel/1, feat_dim)
        active_obj_voxel_center: (num_super_voxel/1, 3)
    """
    active_obj_id = obs["active_obj_id"][0]  # only one object

    # Get image
    img = obs["image"]
    depth = obs["depth"]

    # Get active obj geometry
    if carrier_type == "super_voxel":
        # get active obj super voxel
        active_obj_super_voxels = obs["super_voxel"][active_obj_id]
    elif carrier_type == "rigid_body":
        # get active obj geometry
        active_obj_super_voxels = obs["geometry"][active_obj_id][None, :, :]
    else:
        raise ValueError(f"Unknown carrier_type: {carrier_type}")
    # encode geometry
    active_obj_geometry_feat = []
    for super_voxel in active_obj_super_voxels:
        geometry_feat = encode_geometry(super_voxel, geometry_encoder, aggretator)
        active_obj_geometry_feat.append(geometry_feat)
    active_obj_geometry_feat = np.stack(active_obj_geometry_feat)

    # Get active obj voxel center
    if carrier_type == "super_voxel":
        active_obj_voxel_center = obs["voxel_center"][active_obj_id]
    elif carrier_type == "rigid_body":
        active_obj_voxel_center = np.mean(active_obj_super_voxels, axis=1)
    else:
        raise ValueError(f"Unknown carrier_type: {carrier_type}")

    # Get active obj trajectory
    active_obj_super_voxel_traj = np.vstack(obs["trajectory"][active_obj_id])
    if predict_type == "horizon":
        if active_obj_super_voxel_traj.shape[0] < horizon_length:
            active_obj_super_voxel_traj = np.vstack(
                [
                    active_obj_super_voxel_traj,
                    np.tile(
                        active_obj_super_voxel_traj[-1][None, ...],
                        (horizon_length - active_obj_super_voxel_traj.shape[0], 1, 1),
                    ),
                ]
            )
        else:
            active_obj_super_voxel_traj = active_obj_super_voxel_traj[:horizon_length]
    else:
        raise ValueError(f"Unknown predict_type: {predict_type}")

    return img, depth, active_obj_super_voxel_traj, active_obj_geometry_feat, active_obj_voxel_center


def build_objdp_dataset(data_path: str, export_path: str, predict_type: str, carrier_type: str, geometry_encoder: str, aggretator: str, horizon_length: int = 10):
    """Build dataset for Objwise Diffusion Policy; Transfer raw data to a standard zarr dataset
    """
    # data buffer
    img_buffer = []
    depth_buffer = []
    obj_super_voxel_traj_buffer = []
    obj_geometry_feat_buffer = []
    obj_voxel_center_buffer = []
    dones_buffer = []
    for epoch_dir in tqdm(os.listdir(data_path), desc="Building OBJDP dataset..."):
        epoch_path = os.path.join(data_path, epoch_dir)
        if not os.path.isdir(epoch_path):
            continue
        for file_name in os.listdir(epoch_path):
            if file_name.endswith(".pkl"):
                with open(os.path.join(epoch_path, file_name), "rb") as f:
                    obs = pickle.load(f)
                # parse obs
                img, depth, active_obj_super_voxel_traj, active_obj_geometry_feat, active_obj_voxel_center = parser_obs(
                    obs, predict_type, carrier_type, geometry_encoder, aggretator, horizon_length
                )
                # append to buffer
                img_buffer.append(img)
                depth_buffer.append(depth)
                obj_super_voxel_traj_buffer.append(active_obj_super_voxel_traj)
                obj_geometry_feat_buffer.append(active_obj_geometry_feat)
                obj_voxel_center_buffer.append(active_obj_voxel_center)
                dones_buffer.append(False)
        # append done
        dones_buffer[-1] = True
    # convert to numpy & save to zarr
    img_buffer = np.stack(img_buffer)
    depth_buffer = np.stack(depth_buffer)
    obj_super_voxel_traj_buffer = np.stack(obj_super_voxel_traj_buffer)
    obj_geometry_feat_buffer = np.stack(obj_geometry_feat_buffer)
    obj_voxel_center_buffer = np.stack(obj_voxel_center_buffer)
    dones_buffer = np.array(dones_buffer)
    # compute episode ends: index of the last element of each episode
    eposide_ends = np.where(dones_buffer)[0] + 1
    # save to zarr
    root = zarr.open(f"{export_path}/obj_dp_dataset.zarr", mode="w")
    root.create_dataset("img", data=img_buffer)
    root.create_dataset("depth", data=depth_buffer)
    root.create_dataset("obj_voxel_traj", data=obj_super_voxel_traj_buffer)
    root.create_dataset("obj_voxel_feat", data=obj_geometry_feat_buffer)
    root.create_dataset("obj_voxel_center", data=obj_voxel_center_buffer)
    root.create_dataset("eposide_ends", data=eposide_ends)
    # save meta data
    dim_geometry_feat = obj_geometry_feat_buffer.shape[-1]
    num_super_voxel = obj_geometry_feat_buffer.shape[1]
    meta_data = {
        "predict_type": predict_type,
        "carrier_type": carrier_type,
        "geometry_encoder": geometry_encoder,
        "aggretator": aggretator,
        "horizon_length": horizon_length,
        "dim_geometry_feat": dim_geometry_feat,
        "num_super_voxel": num_super_voxel,
    }
    root.attrs.update(meta_data)


class ObjDPDataset(torch.utils.data.Dataset):
    """Object DiffusionPolicy Dataset"""

    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int, action_horizon: int):
        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, "r")

        # float32, [0,1], (N, width, height, 3)
        train_image_data = dataset_root["img"][:]
        train_image_data = np.moveaxis(train_image_data, -1, 1)
        # (N, 3, width, height)

        # float32, [0,1], (N, width, height)
        train_depth_data = dataset_root["depth"][:]
        train_depth_data = train_depth_data[:, None, :, :]
        # (N, 1, width, height)

        # (N, D)
        train_data = {
            "obj_voxel_traj": dataset_root["obj_voxel_traj"][:],
            "obj_voxel_feat": dataset_root["obj_voxel_feat"][:],
        }
        episode_ends = dataset_root["eposide_ends"][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data["image"] = train_image_data
        normalized_train_data["depth"] = train_depth_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        # images
        nimage = sample_sequence_by_key(
            self.normalized_train_data, "image", self.pred_horizon, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx
        )
        ndepth = sample_sequence_by_key(
            self.normalized_train_data, "depth", self.pred_horizon, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx
        )
        # obj traj
        nobj_voxel_traj = self.normalized_train_data["obj_voxel_traj"][sample_start_idx, :self.obs_horizon:, ...]  # (obs_horizon, num_voxel, 3/6)
        nobj_voxel_feat = self.normalized_train_data["obj_voxel_feat"][sample_start_idx]  # (num_voxel, feat_dim)
        # action
        naction = self.normalized_train_data["obj_voxel_traj"][sample_start_idx, -self.action_horizon:, ...]  # (action_horizon, num_voxel, 3/6)
        # assemble
        nsample = dict()
        nsample["image"] = nimage  # (obs_horizon, 3, H, W)
        nsample["depth"] = ndepth  # (obs_horizon, 1, H, W)
        nsample["obj_voxel_traj"] = nobj_voxel_traj  # (obs_horizon, num_voxel, 3/6)
        nsample["obj_voxel_feat"] = nobj_voxel_feat  # (num_voxel, feat_dim)  # only have current feat
        nsample["action"] = naction  # (action_horizon, num_voxel, 3/6)
        return nsample


if __name__ == "__main__":
    build_objdp_dataset(
        data_path="/home/robot-learning/Projects/VIL2/vil2/test_data/ObjSim/raw_data",
        export_path="/home/robot-learning/Projects/VIL2/vil2/test_data/ObjSim",
        predict_type="horizon",
        carrier_type="super_voxel",
        geometry_encoder="fake",
        aggretator="mean",
        horizon_length=8)

    # Build & test dataset
    dataset = ObjDPDataset(
        dataset_path="/home/robot-learning/Projects/VIL2/vil2/test_data/ObjSim/obj_dp_dataset.zarr",
        pred_horizon=8,
        obs_horizon=6,
        action_horizon=2,
    )

    data = dataset[0]
    pass