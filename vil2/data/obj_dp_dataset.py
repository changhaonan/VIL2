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


def sample_sequence(
    train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


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


def parser_obs(obs: dict, carrier_type: str, geometry_encoder: str, aggretator: str):
    """Parse observation
    Return:
        img: (H, W, 3)
        depth: (H, W)
        active_obj_super_voxel_pose: (num_super_voxel/1, 3/6)
        active_obj_geometry_feat: (num_super_voxel/1, feat_dim)
        active_obj_voxel_center: (num_super_voxel/1, 3)
    """
    active_obj_id = obs["active_obj_id"][0]  # only one object
    img = obs["image"]
    depth = obs["depth"]
    t = obs["t"]

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

    # Get active obj voxel_pose
    active_obj_super_voxel_pose = obs["voxel_pose"][active_obj_id]

    return t, img, depth, active_obj_super_voxel_pose, active_obj_geometry_feat, active_obj_voxel_center


def build_objdp_dataset(data_path: str, export_path: str, carrier_type: str, geometry_encoder: str, aggretator: str, action_horizon: int = 2, obs_horizon: int = 8):
    """Build dataset for Objwise Diffusion Policy; Transfer raw data to a standard zarr dataset
    """
    # data buffer
    t_buffer = []
    img_buffer = []
    depth_buffer = []
    obj_super_voxel_pose_buffer = []
    obj_geometry_feat_buffer = []
    obj_voxel_center_buffer = []
    dones_buffer = []
    for epoch_dir in tqdm(os.listdir(data_path), desc="Building OBJDP dataset..."):
        epoch_path = os.path.join(data_path, epoch_dir)
        if not os.path.isdir(epoch_path):
            continue
        for file_name in sorted(os.listdir(epoch_path)):
            if file_name.endswith(".pkl"):
                with open(os.path.join(epoch_path, file_name), "rb") as f:
                    obs = pickle.load(f)
                # parse obs
                t, img, depth, active_obj_super_voxel_pose, active_obj_geometry_feat, active_obj_voxel_center = parser_obs(
                    obs, carrier_type, geometry_encoder, aggretator
                )
                # append to buffer
                t_buffer.append(t)
                img_buffer.append(img)
                depth_buffer.append(depth)
                obj_super_voxel_pose_buffer.append(active_obj_super_voxel_pose)
                obj_geometry_feat_buffer.append(active_obj_geometry_feat)
                obj_voxel_center_buffer.append(active_obj_voxel_center)
                dones_buffer.append(False)
        # append done
        dones_buffer[-1] = True
    # convert to numpy & save to zarr
    t_buffer = np.array(t_buffer)
    img_buffer = np.stack(img_buffer)
    depth_buffer = np.stack(depth_buffer)
    obj_super_voxel_pose_buffer = np.vstack(obj_super_voxel_pose_buffer)
    obj_geometry_feat_buffer = np.stack(obj_geometry_feat_buffer)
    obj_voxel_center_buffer = np.stack(obj_voxel_center_buffer)
    dones_buffer = np.array(dones_buffer)
    # compute episode ends: index of the last element of each episode
    eposide_ends = np.where(dones_buffer)[0] + 1
    # save to zarr
    root = zarr.open(f"{export_path}/obj_dp_dataset.zarr", mode="w")
    root.create_dataset("t", data=t_buffer.reshape(-1, 1))  # (N, 1)
    root.create_dataset("img", data=img_buffer)  # (N, H, W, 3)
    root.create_dataset("depth", data=depth_buffer)  # (N, H, W)
    root.create_dataset("obj_voxel_pose", data=obj_super_voxel_pose_buffer)  # (N, num_super_voxel, 3/6)
    root.create_dataset("obj_voxel_feat", data=obj_geometry_feat_buffer)  # (N, num_super_voxel, feat_dim)
    root.create_dataset("obj_voxel_center", data=obj_voxel_center_buffer)  # (N, num_super_voxel, 3)
    root.create_dataset("eposide_ends", data=eposide_ends)  # (N,)
    # save meta data
    dim_geometry_feat = obj_geometry_feat_buffer.shape[-1]
    num_super_voxel = obj_geometry_feat_buffer.shape[1]
    meta_data = {
        "carrier_type": carrier_type,
        "geometry_encoder": geometry_encoder,
        "aggretator": aggretator,
        "dim_geometry_feat": dim_geometry_feat,
        "num_super_voxel": num_super_voxel,
    }
    root.attrs.update(meta_data)


class ObjDPDataset(torch.utils.data.Dataset):
    """Object DiffusionPolicy Dataset"""

    def __init__(self, dataset_path: str, obs_horizon: int, action_horizon: int, pred_horizon: int):
        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, "r")

        # read meta data from zarr
        self.carrier_type = dataset_root.attrs["carrier_type"]
        self.geometry_encoder = dataset_root.attrs["geometry_encoder"]
        self.aggretator = dataset_root.attrs["aggretator"]
        self.dim_geometry_feat = dataset_root.attrs["dim_geometry_feat"]
        self.num_super_voxel = dataset_root.attrs["num_super_voxel"]

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
            "obj_voxel_pose": dataset_root["obj_voxel_pose"][:],
            "obj_voxel_feat": dataset_root["obj_voxel_feat"][:],
            "obj_voxel_center": dataset_root["obj_voxel_center"][:],
            "t": dataset_root["t"][:],
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

        # compute statistics and normalized data to [-1, 1]
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
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        # discard unused observations
        nsample["t"] = nsample["t"].astype(np.float32)
        nsample["image"] = nsample["image"][: self.obs_horizon, :].astype(np.float32)
        nsample["depth"] = nsample["depth"][: self.obs_horizon, :].astype(np.float32)
        nsample["obj_voxel_feat"] = nsample["obj_voxel_feat"].astype(np.float32)
        nsample["obj_voxel_center"] = nsample["obj_voxel_center"].astype(np.float32)
        nsample["obj_voxel_pose"] = nsample["obj_voxel_pose"].astype(np.float32)
        return nsample


if __name__ == "__main__":
    build_objdp_dataset(
        data_path="/home/robot-learning/Projects/VIL2/vil2/test_data/ObjSim/raw_data",
        export_path="/home/robot-learning/Projects/VIL2/vil2/test_data/ObjSim",
        carrier_type="super_voxel",
        geometry_encoder="fake",
        aggretator="mean")

    # Build & test dataset
    dataset = ObjDPDataset(
        dataset_path="/home/robot-learning/Projects/VIL2/vil2/test_data/ObjSim/obj_dp_dataset.zarr",
        pred_horizon=8,
        obs_horizon=2,
        action_horizon=4,
    )

    data = dataset[0]
    pass
