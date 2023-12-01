"""Objwise Diffusion Policy Dataset"""
from __future__ import annotations
import numpy as np
import torch
import zarr
import pickle
from tqdm.auto import tqdm
import os

# geometry encoder


def encode_geometry(geometry: np.ndarray, encoder: str, aggretator: str):
    """Encode geometry
    Args:
        geometry: (num_vertices, 3)
        encoder: "pointnet"
    Return:
        encoded_geometry: (num_vertices, 128)
    """
    if encoder == "fake":
        point_feats = np.zeros((geometry.shape[0], 128))
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
        active_obj_traj: (horizon_length, 3/6)
        active_obj_geometry_feat: (num_super_voxel/1, feat_dim)
        active_obj_voxel_center: (num_super_voxel/1, 3)
    """
    active_obj_id = obs["active_obj_id"][0]  # only one object
    # Get active obj traj
    active_obj_traj = np.vstack(obs["trajectory"][active_obj_id]).reshape(-1, 4, 4)
    if predict_type == "horizon":
        if active_obj_traj.shape[0] < horizon_length:
            active_obj_traj = np.vstack(
                [
                    active_obj_traj,
                    np.tile(
                        active_obj_traj[-1][None, ...],
                        (horizon_length - active_obj_traj.shape[0], 1, 1),
                    ),
                ]
            )
        else:
            active_obj_traj = active_obj_traj[:horizon_length]
    else:
        raise ValueError(f"Unknown predict_type: {predict_type}")

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
    return img, depth, active_obj_traj, active_obj_geometry_feat, active_obj_voxel_center


def build_objdp_dataset(data_path: str, predict_type: str, carrier_type: str, geometry_encoder: str, aggretator: str, horizon_length: int = 10):
    """Build dataset for Objwise Diffusion Policy; Transfer raw data to a standard zarr dataset
    """
    # data buffer
    img_buffer = []
    depth_buffer = []
    obj_traj_buffer = []
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
                img, depth, active_obj_traj, active_obj_geometry_feat, active_obj_voxel_center = parser_obs(
                    obs, predict_type, carrier_type, geometry_encoder, aggretator, horizon_length
                )
                # append to buffer
                img_buffer.append(img)
                depth_buffer.append(depth)
                obj_traj_buffer.append(active_obj_traj)
                obj_geometry_feat_buffer.append(active_obj_geometry_feat)
                obj_voxel_center_buffer.append(active_obj_voxel_center)
                dones_buffer.append(False)
        # append done
        dones_buffer[-1] = True
    # convert to numpy & save to zarr
    img_buffer = np.stack(img_buffer)
    depth_buffer = np.stack(depth_buffer)
    obj_traj_buffer = np.stack(obj_traj_buffer)
    obj_geometry_feat_buffer = np.stack(obj_geometry_feat_buffer)
    obj_voxel_center_buffer = np.stack(obj_voxel_center_buffer)
    dones_buffer = np.array(dones_buffer)
    # save to zarr
    root = zarr.open(f"{data_path}/obj_dp_dataset.zarr", mode="w")
    root.create_dataset("img", data=img_buffer)
    root.create_dataset("depth", data=depth_buffer)
    root.create_dataset("obj_traj", data=obj_traj_buffer)
    root.create_dataset("obj_geometry_feat", data=obj_geometry_feat_buffer)
    root.create_dataset("obj_voxel_center", data=obj_voxel_center_buffer)
    root.create_dataset("dones", data=dones_buffer)
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


if __name__ == "__main__":
    build_objdp_dataset(
        data_path="/home/robot-learning/Projects/VIL2/vil2/test_data/ObjSim",
        predict_type="horizon",
        carrier_type="super_voxel",
        geometry_encoder="fake",
        aggretator="mean",
        horizon_length=4)
