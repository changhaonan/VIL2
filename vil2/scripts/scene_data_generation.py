"""Generate test data; Self-contained script for generating test data"""
# Imports
from vil2.data.dmorp_data import ObjData, SceneData
import os
from tqdm.auto import tqdm
import numpy as np
import random
from dataclasses import dataclass
import pickle
import copy
from sklearn.neighbors import NearestNeighbors
import argparse
import yaml
from box import Box
from detectron2.config import LazyConfig, instantiate
import clip
import torch
import zarr
import json

# Parameters
d_max_structure_per_scene = 2
d_min_obj_in_structure = 4
d_max_obj_in_structure = 6
d_max_rel_trans = 0.3  # maximum shift within structures
d_min_structure_shift = 0.5  # shift between structures
d_max_structure_shift = 1.0
d_num_scenes = 1000
d_num_obj = 8  # number of objects
d_num_structure = 4  # total number of structures
d_knn = 5
d_noise_level = 0.02
d_semantic_feat_noise_level = 0.05
d_fix_z = True

# Objects define


def gen_random_structure_pose(rng: np.random.Generator, num_objs: int, max_dist: float, fix_z: bool = True,  enable_rot: bool = False):
    """Generate random poses strcuture
    Args:
        max_dist (float): maximum distance between two objects
        enable_rot (bool, optional): enable rotation. Defaults to False.
    """
    trans = rng.uniform(-max_dist, max_dist, size=(num_objs, 3))
    if fix_z:
        trans[:, 2] = 0.0
    if enable_rot:
        # quaternion
        rot = rng.uniform(-1.0, 1.0, size=(num_objs, 4))
        rot = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    else:
        rot = np.zeros((num_objs, 4))
        rot[:, 0] = 1.0
    poses = np.hstack([trans, rot])
    return poses


if __name__ == "__main__":
    # Load config
    task_name = "Dmorp"
    root_path = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    cfg_file = os.path.join(root_path, "config", "dmorp_simplify.py")
    cfg = LazyConfig.load(cfg_file)

    # Override parameters
    d_num_obj = cfg.ENV.NUM_OBJ
    d_num_structure = cfg.ENV.NUM_STRUCTURE
    semantic_feat_type = cfg.ENV.SEMANTIC_FEAT_TYPE

    export_dir = os.path.join(root_path, f"test_data/{task_name}/raw_data")
    # Clean old data
    if os.path.exists(export_dir):
        os.system(f"rm -rf {export_dir}")
    os.makedirs(export_dir, exist_ok=True)
    obj_dict = {
        "bowl": {
            "semantic_str": "Bowl",
            "semantic_id": 1,
        },
        "coffee_mug": {
            "semantic_str": "Coffee Mug",
            "semantic_id": 2,
        },
        "plate": {
            "semantic_str": "Plate",
            "semantic_id": 3,
        },
        "spoon": {
            "semantic_str": "Spoon",
            "semantic_id": 4,
        },
        "tea_pot": {
            "semantic_str": "Tea Pot",
            "semantic_id": 5,
        },
        "wine_glass": {
            "semantic_str": "Wine Glass",
            "semantic_id": 6,
        },
        "wine_bottle": {
            "semantic_str": "Wine Bottle",
            "semantic_id": 7,
        },
        "fork": {
            "semantic_str": "Fork",
            "semantic_id": 8,
        },
        "knife": {
            "semantic_str": "Knife",
            "semantic_id": 9,
        },
        "spatula": {
            "semantic_str": "Spatula",
            "semantic_id": 10,
        },
    }

    # Init CLIP model
    if semantic_feat_type == "clip":
        clip_model, preprocess = clip.load("ViT-B/32", device="cuda:0")
        for obj_name, obj_data in obj_dict.items():
            text_inputs = torch.cat([clip.tokenize(f"A picture of {obj_name}")]).to("cuda:0")
            obj_data["semantic_feature"] = clip_model.encode_text(
                text_inputs).detach().cpu().numpy()[0]
    elif semantic_feat_type == "one_hot":
        for obj_name, obj_data in obj_dict.items():
            obj_data["semantic_feature"] = np.zeros(
                (cfg.ENV.SEMANTIC_FEAT_DIM,), dtype=np.float32)
            obj_data["semantic_feature"][obj_data["semantic_id"]-1] = 1.0

    # Build objects data
    obj_data_list = []
    for obj_name, obj_data in obj_dict.items():
        obj_data_list.append(
            ObjData(
                pose=np.zeros(7),
                semantic_str=obj_data["semantic_str"],
                semantic_id=obj_data["semantic_id"],
                semantic_feature=obj_data["semantic_feature"],
                pcd=np.zeros(3),
                geometry=np.zeros(3),
            )
        )

    # Set seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate structure pool
    structure_list = [
        {
            "id": np.array([1, 2, 3, 4]),
        },
        {
            "id": np.array([2, 3, 4, 5]),
        },
        {
            "id": np.array([5, 6, 7, 8]),
        }
    ]
    for structure in structure_list:
        structure["poses"] = gen_random_structure_pose(
            rng, structure["id"].shape[0], d_max_rel_trans, fix_z=d_fix_z, enable_rot=False)

    # Generate scene data
    scene_id_buffer = []
    sem_id_buffer = []
    sem_feat_buffer = []
    geo_feat_buffer = []
    pose_buffer = []
    episode_ends = []
    for scene_id in tqdm(range(d_num_scenes), desc="Generating scene data"):
        # Randomly sample a structure
        struc_id = rng.choice(len(structure_list), size=(1), replace=True).item()
        scene_obj_id_list = np.array([], dtype=np.int32)
        scene_pose_list = np.array([], dtype=np.float32).reshape(0, 7)
        scene_struc_ids = []

        struc_data = structure_list[struc_id]
        structure_pose = struc_data["poses"].copy()
        # Apply a random shift
        shift = rng.uniform(d_min_structure_shift, d_max_structure_shift, size=(
            3)) * rng.choice([-1, 1], size=(3))
        if d_fix_z:
            shift[2] = 0.0
        structure_pose[:, :3] += shift
        # Append
        scene_obj_id_list = struc_data["id"]
        scene_pose_list = structure_pose

        # Build scene data
        for obj_id, obj_pose in zip(scene_obj_id_list, scene_pose_list):
            # obj_id starts from 1
            scene_id_buffer.append(scene_id)
            pose_buffer.append(obj_pose)
            sem_id_buffer.append(obj_data_list[obj_id-1].semantic_id)
            sem_feat = obj_data_list[obj_id-1].semantic_feature.copy()
            sem_feat += rng.normal(0.0, d_semantic_feat_noise_level, size=sem_feat.shape)
            sem_feat_buffer.append(sem_feat)
            geo_feat_buffer.append(obj_data_list[obj_id-1].geometry)
        # episode ends
        episode_ends.append(scene_obj_id_list.shape[0])

    # Convert & Save
    scene_id_buffer = np.stack(scene_id_buffer).astype(np.int32).reshape(-1, 1)
    pose_buffer = np.stack(pose_buffer).astype(np.float32)
    sem_id_buffer = np.stack(sem_id_buffer).astype(np.int32).reshape(-1, 1)
    sem_feat_buffer = np.stack(sem_feat_buffer).astype(np.float32)
    geo_feat_buffer = np.stack(geo_feat_buffer).astype(np.float32)
    episode_ends = np.array(episode_ends).astype(np.int32)
    episode_ends = np.cumsum(episode_ends)
    # Save
    root = zarr.open(os.path.join(export_dir, "dmorp_data.zarr"), "w")
    root.create_dataset("scene_id", data=scene_id_buffer)
    root.create_dataset("pose", data=pose_buffer)
    root.create_dataset("sem_id", data=sem_id_buffer)
    root.create_dataset("sem_feat", data=sem_feat_buffer)
    root.create_dataset("geo_feat", data=geo_feat_buffer)
    root.create_dataset("meta/episode_ends", data=episode_ends)

    # Save vocabulary in data
    obj_name_list = list(obj_dict.keys())
    obj_name_dict = {obj_name: i for i, obj_name in enumerate(obj_name_list)}
    with open(os.path.join(export_dir, "vocab.json"), "w") as f:
        json.dump(obj_name_dict, f)
