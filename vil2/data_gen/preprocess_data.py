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
import volumentations as V

# def farthest_point_sampling_with_color(pcd, n_points):
#     """
#     Perform Farthest Point Sampling on a point cloud with color information.

#     :param pcd: numpy array of shape (N, 6), where N is the number of points in the point cloud.
#                 The first three columns are spatial coordinates and the last three are color information.
#     :param n_points: number of points to sample.
#     :return: sampled point cloud of shape (n_points, 6).
#     """
#     # farthest_points = np.zeros((n_points, 6))
#     farthest_points = np.zeros((n_points, 9))
#     # Initialize an array to store the shortest distance of each point to any selected point
#     shortest_distances = np.full(len(pcd), np.inf)
#     # Randomly choose the first point and update the distances
#     first_index = np.random.randint(len(pcd))
#     farthest_points[0] = pcd[first_index]
#     shortest_distances = np.linalg.norm(pcd[:, :3] - farthest_points[0, :3], axis=1)

#     for i in range(1, n_points):
#         # Select the point that is farthest to any point in the farthest_points set
#         farthest_index = np.argmax(shortest_distances)
#         farthest_points[i] = pcd[farthest_index]
#         # Update shortest_distances array
#         distances_to_new_point = np.linalg.norm(pcd[:, :3] - farthest_points[i, :3], axis=1)
#         shortest_distances = np.minimum(shortest_distances, distances_to_new_point)

#     return farthest_points


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


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_file = os.path.join(root_dir, "config", "dmorp_simplify.py")
    cfg = LazyConfig.load(cfg_file)
    data_id = [0, 1]
    dtset = []
    for did in data_id:
        print(f"Processing data {did}...")
        scene_info_path = os.path.join(
            root_dir, "test_data", "dmorp_augmented", f"{did:06d}-{cfg.MODEL.DATASET_CONFIG}"
        )
        # scene_info_path = os.path.join(root_dir, "test_data", "dmorp_augmented", f"{did:06d}")
        camera_info_path = os.path.join(root_dir, "test_data", "dmorp", f"{did:06d}-{cfg.MODEL.DATASET_CONFIG}")
        with open(os.path.join(scene_info_path, "scene_info.json"), "r") as f:
            scene_info = json.load(f)

        # Parse number of scenes and cameras
        scene_info_list = list(os.listdir(camera_info_path))
        num_init_scenes = len(scene_info_list)
        render_file_list = list(os.listdir(os.path.join(camera_info_path, scene_info_list[0])))
        render_file_list = [f for f in render_file_list if f.endswith(".hdf5")]
        num_cameras = len(render_file_list) // 2
        pcd_size = cfg.MODEL.PCD_SIZE

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
                target_color, target_depth = read_hdf5(os.path.join(h5file_dir, f"{j}.hdf5"))
                fixed_color, fixed_depth = read_hdf5(os.path.join(h5file_dir, f"{j + num_cameras}.hdf5"))

                target_depth = target_depth.astype(np.float32)
                target_depth[target_depth > 1000.0] = 0.0
                target_depth = -target_depth
                target_pcd, tpointcloud_size = utils.get_pointcloud(target_color, target_depth, intrinsic)

                if tpointcloud_size >= pcd_size:
                    # target_pcd = farthest_point_sampling_with_color(target_pcd, pcd_size)

                    target_pcd = target_pcd.farthest_point_down_sample(pcd_size)

                    target_pcd_arr = np.hstack((np.array(target_pcd.points), np.array(target_pcd.normals) , np.array(target_pcd.colors)))
                    # tpcd = o3d.geometry.PointCloud()
                    # tpcd.points = o3d.utility.Vector3dVector(target_pcd[:, :3])

                    # # Estimate normals
                    # tpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                    # # Optional: Orient the normals to be consistent
                    # tpcd.orient_normals_consistent_tangent_plane(k=50)

                    # # Convert normals to numpy array
                    # normals = np.array(tpcd.normals)

                    # target_pcd = np.concatenate((target_pcd[:, :3], normals, target_pcd[:, 3:6]), axis=1)

                    fixed_depth = fixed_depth.astype(np.float32)
                    fixed_depth[fixed_depth > 1000.0] = 0.0
                    fixed_depth = -fixed_depth
                    fixed_pcd, fpointcloud_size = utils.get_pointcloud(fixed_color, fixed_depth, intrinsic)
                    
                    if fpointcloud_size >= pcd_size:
                        fixed_pcd = fixed_pcd.farthest_point_down_sample(pcd_size)

                        # fpcd = o3d.geometry.PointCloud()
                        # fpcd.points = o3d.utility.Vector3dVector(fixed_pcd[:, :3])

                        # # Estimate normals
                        # fpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                        # # Optional: Orient the normals to be consistent
                        # fpcd.orient_normals_consistent_tangent_plane(k=50)

                        # # Convert normals to numpy array
                        # normals = np.array(fpcd.normals)

                        # fixed_pcd = np.concatenate((fixed_pcd[:, :3], normals, fixed_pcd[:, 3:6]), axis=1)
                        
                        camera_pose_inv = np.linalg.inv(np.array(camera_pose_json["cam2world"][j]))
                        camera_pose = np.array(camera_pose_json["cam2world"][j])

                        fixed_pcd_arr = np.hstack((np.array(fixed_pcd.points), np.array(fixed_pcd.normals) , np.array(fixed_pcd.colors)))
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
    # Save the dtset into a .pkl file
    with open(
        os.path.join(
            root_dir, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(dtset, f)
