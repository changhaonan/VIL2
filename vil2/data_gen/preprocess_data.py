import json
import os
import numpy as np
import h5py
import open3d as o3d
import vil2.utils.misc_utils as utils
import torch
import pickle
from tqdm import tqdm
import copy
from detectron2.config import LazyConfig


def read_hdf5(file_name):
    """Read HDF5 file and return data."""
    with h5py.File(file_name, "r") as file:
        return np.array(file["colors"]), np.array(file["depth"])


def visualize_pcd_with_open3d(
    pcd_with_color1, pcd_with_color2, transform1: np.ndarray = None, shift_transform: np.ndarray = None
):
    assert pcd_with_color1.shape[1] == 9 and pcd_with_color2.shape[1] == 9
    # Create an Open3D point cloud object
    pcd1 = o3d.geometry.PointCloud()

    # Set the points of the point cloud
    pcd1.points = o3d.utility.Vector3dVector(pcd_with_color1[:, :3])

    # Set the normals of the point cloud
    pcd1.normals = o3d.utility.Vector3dVector(pcd_with_color1[:, 3:6])

    # Assuming the colors are in the range [0, 255], normalize them to [0, 1]
    pcd1.colors = o3d.utility.Vector3dVector(pcd_with_color1[:, 6:9] / 255.0)

    # Create an Open3D point cloud object
    pcd2 = o3d.geometry.PointCloud()

    # Set the points of the point cloud
    pcd2.points = o3d.utility.Vector3dVector(pcd_with_color2[:, :3])

    # Set the normals of the point cloud
    pcd2.normals = o3d.utility.Vector3dVector(pcd_with_color2[:, 3:6])

    # Assuming the colors are in the range [0, 255], normalize them to [0, 1]
    pcd2.colors = o3d.utility.Vector3dVector(pcd_with_color2[:, 6:9] / 255.0)

    if shift_transform is not None:
        pcd1.transform(shift_transform)
        pcd2.transform(shift_transform)

    if transform1 is not None:
        pcd1.transform(transform1)

    o3d.visualization.draw_geometries([pcd1, pcd2], point_show_normal=True)


class DiffDataset(torch.utils.data.Dataset):
    def __init__(self, dtset: dict):
        self.dtset = dtset
        self.called = 0

    def __len__(self):
        return len(self.dtset)

    def __getitem__(self, idx):
        dt_entry = copy.deepcopy(self.dtset[idx])

        return self.dtset[idx]


def perform_gram_schmidt_transform(trans_matrix):
    trans_matrix = copy.deepcopy(trans_matrix)
    translation = trans_matrix[:3, 3]
    rotation = trans_matrix[:3, :3]
    v1 = rotation[:, 0]
    v1_normalized = v1 / np.linalg.norm(v1)
    v2 = rotation[:, 1]
    v2_orthogonal = v2 - np.dot(v2, v1_normalized) * v1_normalized
    v2_normalized = v2_orthogonal / np.linalg.norm(v2_orthogonal)
    return copy.deepcopy(np.concatenate((translation, v1_normalized, v2_normalized)))


def build_dataset(scene_info_path, camera_info_path, cfg):
    """Build the dataset from the given scene and camera info"""
    # Parse number of scenes and cameras
    with open(os.path.join(scene_info_path, "scene_info.json"), "r") as f:
        scene_info = json.load(f)
    scene_info_list = list(os.listdir(camera_info_path))
    num_init_scenes = len(scene_info_list)
    render_file_list = list(os.listdir(os.path.join(camera_info_path, scene_info_list[0])))
    render_file_list = [f for f in render_file_list if f.endswith(".hdf5")]
    num_cameras = len(render_file_list) // 2
    pcd_size = cfg.MODEL.PCD_SIZE
    print(f"Number of scenes: {num_init_scenes}; Number of cameras: {num_cameras}...")

    for i in tqdm(range(num_init_scenes), desc="Processing scenes"):
        h5file_dir = os.path.join(camera_info_path, f"{i:06d}")
        intrinsic_json = json.load(open(os.path.join(h5file_dir, "camera.json"), "r"))
        intrinsic = np.array(
            [
                [-intrinsic_json["fx"], 0.0, intrinsic_json["cx"]],
                [0.0, intrinsic_json["fy"], intrinsic_json["cy"]],
                [0.0, 0.0, 1.0],
            ]
        )
        camera_pose_json = json.load(open(os.path.join(h5file_dir, "poses.json"), "r"))
        target_transform_world = np.array(scene_info["transform_list"][i])
        initial_pose_A_world = np.array(scene_info["init_pose_1_list"][i])
        for j in tqdm(range(num_cameras), desc=f"Processing cameras for scene {i}", leave=False):
            # Read the target and fixed pointcloud
            target_color, target_depth = read_hdf5(os.path.join(h5file_dir, f"{j}.hdf5"))
            fixed_color, fixed_depth = read_hdf5(os.path.join(h5file_dir, f"{j + num_cameras}.hdf5"))

            target_depth = target_depth.astype(np.float32)
            target_depth[target_depth > 1000.0] = 0.0
            target_depth = -target_depth
            target_pcd, tpointcloud_size = utils.get_pointcloud(target_color, target_depth, intrinsic)

            if tpointcloud_size >= pcd_size:
                target_pcd = target_pcd.farthest_point_down_sample(pcd_size)
                target_pcd_arr = np.hstack(
                    (np.array(target_pcd.points), np.array(target_pcd.normals), np.array(target_pcd.colors))
                )

                fixed_depth = fixed_depth.astype(np.float32)
                fixed_depth[fixed_depth > 1000.0] = 0.0
                fixed_depth = -fixed_depth
                fixed_pcd, fpointcloud_size = utils.get_pointcloud(fixed_color, fixed_depth, intrinsic)

                if fpointcloud_size >= pcd_size:
                    fixed_pcd = fixed_pcd.farthest_point_down_sample(pcd_size)

                    camera_pose_inv = np.linalg.inv(np.array(camera_pose_json["cam2world"][j]))
                    camera_pose = np.array(camera_pose_json["cam2world"][j])

                    fixed_pcd_arr = np.hstack(
                        (np.array(fixed_pcd.points), np.array(fixed_pcd.normals), np.array(fixed_pcd.colors))
                    )
                    # Calculate centroid for fixed_pcd
                    fixed_centroid = np.mean(fixed_pcd_arr, axis=0)

                    target_transform_camera = camera_pose_inv @ target_transform_world @ camera_pose
                    # visualize_pcd_with_open3d(target_pcd_arr, fixed_pcd_arr, target_transform_camera)
                    translation = target_transform_camera[:3, 3]
                    rotation = target_transform_camera[:3, :3]
                    v1 = rotation[:, 0]
                    v1_normalized = v1 / np.linalg.norm(v1)
                    v2 = rotation[:, 1]
                    v2_orthogonal = v2 - np.dot(v2, v1_normalized) * v1_normalized
                    v2_normalized = v2_orthogonal / np.linalg.norm(v2_orthogonal)
                    v3 = np.cross(v1_normalized, v2_normalized)

                    target_pcd_shifted = copy.deepcopy(target_pcd_arr)
                    target_pcd_shifted[:, :3] -= fixed_centroid[:3]
                    fixed_pcd_shifted = copy.deepcopy(fixed_pcd_arr)
                    fixed_pcd_shifted[:, :3] -= fixed_centroid[:3]

                    centroid_shift_transform = np.eye(4)
                    centroid_shift_transform[:3, 3] = -fixed_centroid[:3]
                    target_transform_camera_shifted = (
                        centroid_shift_transform @ target_transform_camera @ np.linalg.inv(centroid_shift_transform)
                    )
                    # visualize_pcd_with_open3d(target_pcd_shifted, fixed_pcd_shifted, target_transform_camera_shifted)
                    dtset.append(
                        {
                            "original": {
                                "target": target_pcd_arr,
                                "fixed": fixed_pcd_arr,
                                "transform": target_transform_camera,
                                "9dpose": perform_gram_schmidt_transform(target_transform_camera),
                                "cam_pose": camera_pose,
                            },
                            "shifted": {
                                "target": target_pcd_shifted,
                                "fixed": fixed_pcd_shifted,
                                "transform": target_transform_camera_shifted,
                                "9dpose": perform_gram_schmidt_transform(target_transform_camera_shifted),
                                "cam_pose": camera_pose,
                            },
                        }
                    )

    print("Len of dtset:", len(dtset))
    print(f"Saving dataset to {os.path.join(root_dir, 'test_data', 'dmorp_augmented')}...")
    # Save the dtset into a .pkl file
    with open(
        os.path.join(
            root_dir, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(dtset, f)
    print("Done!")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_file = os.path.join(root_dir, "config", "pose_transformer.py")
    cfg = LazyConfig.load(cfg_file)
    data_id = [0]
    dtset = []
    for did in data_id:
        print(f"Processing data {did}...")
        scene_info_path = os.path.join(
            root_dir, "test_data", "dmorp_augmented", f"{did:06d}-{cfg.MODEL.DATASET_CONFIG}"
        )
        camera_info_path = os.path.join(root_dir, "test_data", "dmorp", f"{did:06d}-{cfg.MODEL.DATASET_CONFIG}")
        build_dataset(scene_info_path, camera_info_path, cfg)
