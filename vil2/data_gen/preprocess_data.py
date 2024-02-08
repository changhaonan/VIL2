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
import argparse
from scipy.spatial.transform import Rotation as R


def pose7d_to_mat(pose7d):
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = R.from_quat(pose7d[3:]).as_matrix()
    pose_mat[:3, 3] = pose7d[:3]
    return pose_mat


def read_hdf5(file_name):
    """Read HDF5 file and return data."""
    with h5py.File(file_name, "r") as file:
        return np.array(file["colors"]), np.array(file["depth"])


def read_scene_hdf5(fixed_hdf5, target_hdf5, intrinsic_file):
    with open(intrinsic_file, "r") as f:
        intrinsic_json = json.load(f)
    intrinsic = np.array(
        [
            [-intrinsic_json["fx"], 0.0, intrinsic_json["cx"]],
            [0.0, intrinsic_json["fy"], intrinsic_json["cy"]],
            [0.0, 0.0, 1.0],
        ]
    )
    target_color, target_depth = read_hdf5(target_hdf5)
    fixed_color, fixed_depth = read_hdf5(fixed_hdf5)
    # Filter depth
    target_depth = target_depth.astype(np.float32)
    target_depth[target_depth > 1000.0] = 0.0
    target_depth = -target_depth
    fixed_depth = fixed_depth.astype(np.float32)
    fixed_depth[fixed_depth > 1000.0] = 0.0
    fixed_depth = -fixed_depth
    return target_color, target_depth, fixed_color, fixed_depth, intrinsic


def normalize_pcd(pcd_anchor, pcd_list):
    # Normalize to unit cube
    pcd_center = (pcd_anchor.get_max_bound() + pcd_anchor.get_min_bound()) / 2
    pcd_anchor = pcd_anchor.translate(-pcd_center)
    scale_xyz = pcd_anchor.get_max_bound() - pcd_anchor.get_min_bound()
    scale_xyz = np.max(scale_xyz)
    pcd_anchor = pcd_anchor.scale(1 / scale_xyz, center=np.array([0, 0, 0]))

    # Normalize the child point clouds
    normalized_pcd_list = []
    for pcd in pcd_list:
        pcd = pcd.translate(-pcd_center)
        pcd = pcd.scale(1 / scale_xyz, center=np.array([0, 0, 0]))
        normalized_pcd_list.append(pcd)
    return pcd_anchor, normalized_pcd_list, pcd_center, scale_xyz


def visualize_pcd_with_open3d(
    pcd_with_color1,
    pcd_with_color2,
    transform1: np.ndarray = None,
    shift_transform: np.ndarray = None,
    camera_pose=None,
):
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    pcd1.points = o3d.utility.Vector3dVector(pcd_with_color1[:, :3])
    pcd2.points = o3d.utility.Vector3dVector(pcd_with_color2[:, :3])

    if pcd_with_color1.shape[1] >= 6:
        pcd1.normals = o3d.utility.Vector3dVector(pcd_with_color1[:, 3:6])
        pcd2.normals = o3d.utility.Vector3dVector(pcd_with_color2[:, 3:6])

    if pcd_with_color1.shape[1] >= 9:
        color_scale = 1.0 if pcd_with_color1[:, 6:9].max() <= 1.0 else 255.0
        pcd1.colors = o3d.utility.Vector3dVector(pcd_with_color1[:, 6:9] / color_scale)
        pcd2.colors = o3d.utility.Vector3dVector(pcd_with_color2[:, 6:9] / color_scale)
    else:
        pcd1.paint_uniform_color([0.0, 0.651, 0.929])
        pcd2.paint_uniform_color([1.0, 0.706, 0.0])

    if shift_transform is not None:
        pcd1.transform(shift_transform)
        pcd2.transform(shift_transform)

    if transform1 is not None:
        pcd1.transform(transform1)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    if camera_pose is not None:
        camera_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        camera_origin.transform(camera_pose)
        o3d.visualization.draw_geometries([pcd1, pcd2, origin, camera_origin])
    else:
        o3d.visualization.draw_geometries([pcd1, pcd2, origin])


def assemble_tmorp_data(
    fixed_color,
    fixed_depth,
    fixed_label,
    target_color,
    target_depth,
    target_label,
    intrinsic,
    camera_pose,
    target_transform_world,
    pcd_size,
    data_id: int = 0,
    is_opengl: bool = False,
):
    # Filter depth
    target_depth = target_depth.astype(np.float32)
    target_depth[target_depth > 1000.0] = 0.0
    target_depth = -target_depth
    fixed_depth = fixed_depth.astype(np.float32)
    fixed_depth[fixed_depth > 1000.0] = 0.0
    fixed_depth = -fixed_depth

    if is_opengl:
        flip_x = True
        flip_y = False
    else:
        flip_x = False
        flip_y = True
    target_pcd, tpointcloud_size = utils.get_o3d_pointcloud(target_color, target_depth, intrinsic, flip_x, flip_y)
    if tpointcloud_size < pcd_size:
        return None
    fixed_pcd, fpointcloud_size = utils.get_o3d_pointcloud(fixed_color, fixed_depth, intrinsic, flip_x, flip_y)
    if fpointcloud_size < pcd_size:
        return None
    # Transform pcd to world frame
    fixed_pcd.transform(camera_pose)
    target_pcd.transform(camera_pose)

    target_pcd = target_pcd.farthest_point_down_sample(pcd_size)
    target_pcd_arr = np.hstack((np.array(target_pcd.points), np.array(target_pcd.normals), np.array(target_pcd.colors)))
    fixed_pcd = fixed_pcd.farthest_point_down_sample(pcd_size)
    fixed_pcd_arr = np.hstack((np.array(fixed_pcd.points), np.array(fixed_pcd.normals), np.array(fixed_pcd.colors)))

    rotation = target_transform_world[:3, :3]
    v1 = rotation[:, 0]
    v1_normalized = v1 / np.linalg.norm(v1)
    v2 = rotation[:, 1]
    v2_orthogonal = v2 - np.dot(v2, v1_normalized) * v1_normalized
    v2_normalized = v2_orthogonal / np.linalg.norm(v2_orthogonal)
    v3 = np.cross(v1_normalized, v2_normalized)
    return {
        "target": target_pcd_arr,
        "fixed": fixed_pcd_arr,
        "target_label": target_label,
        "fixed_label": fixed_label,
        "transform": target_transform_world,
        "9dpose": utils.perform_gram_schmidt_transform(target_transform_world),
        "cam_pose": camera_pose,
        "data_id": data_id,
    }


# def build_dataset_blender(scene_info_path, camera_info_path, cfg):
#     """Build the dataset from the given scene and camera info"""
#     # Parse number of scenes and cameras
#     with open(os.path.join(scene_info_path, "scene_info.json"), "r") as f:
#         scene_info = json.load(f)
#     scene_info_list = list(os.listdir(camera_info_path))
#     num_init_scenes = len(scene_info_list)
#     render_file_list = list(os.listdir(os.path.join(camera_info_path, scene_info_list[0])))
#     render_file_list = [f for f in render_file_list if f.endswith(".hdf5")]
#     num_cameras = len(render_file_list) // 2
#     pcd_size = cfg.MODEL.PCD_SIZE
#     print(f"Number of scenes: {num_init_scenes}; Number of cameras: {num_cameras}...")

#     dtset = []
#     for i in tqdm(range(num_init_scenes), desc="Processing scenes"):
#         target_transform_world = np.array(scene_info["transform_list"][i])
#         for j in tqdm(range(num_cameras), desc=f"Processing cameras for scene {i}", leave=False):
#             # Read the target and fixed pointcloud
#             h5file_dir = os.path.join(camera_info_path, f"{i:06d}")
#             intrinsic_file = os.path.join(h5file_dir, "camera.json")
#             camera_pose_file = os.path.join(h5file_dir, "poses.json")
#             target_hdf5 = os.path.join(h5file_dir, f"{j}.hdf5")
#             fixed_hdf5 = os.path.join(h5file_dir, f"{j + num_cameras}.hdf5")
#             target_color, target_depth, fixed_color, fixed_depth, intrinsic = read_scene_hdf5(
#                 fixed_hdf5, target_hdf5, intrinsic_file
#             )
#             with open(camera_pose_file, "r") as f:
#                 camera_pose_json = json.load(f)
#             camera_pose = np.array(camera_pose_json["cam2world"][j])

#             # Assemble data
#             data = assemble_tmorp_data(
#                 fixed_color,
#                 fixed_depth,
#                 target_color,
#                 target_depth,
#                 intrinsic,
#                 camera_pose,
#                 target_transform_world,
#                 pcd_size,
#             )
#             if data is None:
#                 continue
#             # visualize_pcd_with_open3d(target_pcd_arr, fixed_pcd_arr, np.eye(4, dtype=np.float32))
#             # visualize_pcd_with_open3d(target_pcd_arr, fixed_pcd_arr, target_transform_world)
#             dtset.append(data)

#     print("Len of dtset:", len(dtset))
#     print(f"Saving dataset to {os.path.join(root_dir, 'test_data', 'dmorp_augmented')}...")
#     # Save the dtset into a .pkl file
#     with open(
#         os.path.join(
#             root_dir, "test_data", "dmorp_augmented", f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}.pkl"
#         ),
#         "wb",
#     ) as f:
#         pickle.dump(dtset, f)
#     print("Done!")


def build_dataset_pyrender(data_path, cfg, data_id: int = 0, vis: bool = False):
    """Build the dataset from the pyrender render engine"""
    data_file_list = os.listdir(data_path)
    data_file_list = [f for f in data_file_list if f.endswith(".npz")]
    # Sorted by numerical order
    data_file_list = sorted(data_file_list, key=lambda x: int(x.split(".")[0]))
    dtset = []
    for data_file in tqdm(data_file_list, desc="Processing data"):
        data_list = np.load(os.path.join(data_path, data_file), allow_pickle=True)["data"]
        assert len(data_list) % 2 == 0
        for i in tqdm(range(0, len(data_list), 2), desc="Processing frames", leave=False):
            data_0 = data_list[i]  # target
            data_1 = data_list[i + 1]  # fixed
            tmorp_data = assemble_tmorp_data(
                data_1["color"][..., :3],
                data_1["depth"],
                data_1["semantic"],
                data_0["color"][..., :3],
                data_0["depth"],
                data_0["semantic"],
                data_0["intrinsic"],
                data_0["camera_pose"],
                data_0["transform"],
                cfg.MODEL.PCD_SIZE,
                data_id=data_id,
                is_opengl=True,
            )
            if tmorp_data is None:
                continue
            if vis:
                # Visualize & Check
                visualize_pcd_with_open3d(
                    tmorp_data["target"],
                    tmorp_data["fixed"],
                    np.eye(4, dtype=np.float32),
                    camera_pose=tmorp_data["cam_pose"],
                )
                visualize_pcd_with_open3d(
                    tmorp_data["target"],
                    tmorp_data["fixed"],
                    tmorp_data["transform"],
                    camera_pose=tmorp_data["cam_pose"],
                )
            dtset.append(tmorp_data)
    print("Len of dtset:", len(dtset))
    print(f"Saving dataset to {os.path.join(root_dir, 'test_data', 'dmorp_faster')}...")
    # Save the dtset into a .pkl file
    with open(
        os.path.join(
            root_dir,
            "test_data",
            "dmorp_faster",
            f"diffusion_dataset_{data_id}_{cfg.MODEL.PCD_SIZE}_{cfg.MODEL.DATASET_CONFIG}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(dtset, f)
    print("Done!")


def build_dataset_real(data_path, cfg, data_id: int = 0, vis: bool = False, filter_key: str = None):
    """Build the dataset from the real data"""
    data_file_list = os.listdir(data_path)
    data_file_list = [f for f in data_file_list if f.endswith(".pkl")]
    # Filter by key
    if filter_key is not None:
        data_file_list = [f for f in data_file_list if filter_key in f]
    dtset = []
    pcd_size = cfg.MODEL.PCD_SIZE
    rot_axis = cfg.MODEL.ROT_AXIS
    for data_file in tqdm(data_file_list, desc="Processing data"):
        # Load data
        pcd_dict = pickle.load(open(os.path.join(data_path, data_file), "rb"))
        data_len = len(pcd_dict["object_0"])
        for i in tqdm(range(data_len), desc="Processing frames", leave=False):
            # Assemble data
            target_pcd_arr = pcd_dict["object_0"][i]
            fixed_pcd_arr = pcd_dict["object_1"][i]
            target_label = pcd_dict["object_0_semantic"][i]
            fixed_label = pcd_dict["object_1_semantic"][i]
            if target_pcd_arr.shape[0] < pcd_size or fixed_pcd_arr.shape[0] < pcd_size:
                continue

            data_id = data_id
            # Shift all points to the origin
            target_pcd_center = np.mean(target_pcd_arr[:, :3], axis=0)
            fixed_pcd_center = np.mean(fixed_pcd_arr[:, :3], axis=0)
            target_pcd_arr[:, :3] -= target_pcd_center
            fixed_pcd_arr[:, :3] -= fixed_pcd_center
            shift_transform = np.eye(4, dtype=np.float32)
            shift_transform[:3, 3] = target_pcd_center - fixed_pcd_center
            pose_9d = utils.mat_to_pose9d(shift_transform, rot_axis=rot_axis)
            tmorp_data = {
                "target": target_pcd_arr,
                "fixed": fixed_pcd_arr,
                "target_label": target_label,
                "fixed_label": fixed_label,
                "transform": shift_transform,
                "9dpose": pose_9d,
                "cam_pose": np.eye(4, dtype=np.float32),
                "data_id": data_id,
            }
            # Visualize & Check
            if vis:
                visualize_pcd_with_open3d(
                    tmorp_data["target"],
                    tmorp_data["fixed"],
                    np.eye(4, dtype=np.float32),
                    camera_pose=tmorp_data["cam_pose"],
                )
                visualize_pcd_with_open3d(
                    tmorp_data["target"],
                    tmorp_data["fixed"],
                    tmorp_data["transform"],
                    camera_pose=tmorp_data["cam_pose"],
                )
            dtset.append(tmorp_data)
    print("Len of dtset:", len(dtset))
    print(f"Saving dataset to {os.path.join(root_dir, 'test_data', 'dmorp_real')}...")
    # Save the dtset into a .pkl file
    if filter_key is not None:
        with open(
            os.path.join(
                root_dir,
                "test_data",
                "dmorp_real",
                f"diffusion_dataset_{data_id}_{cfg.MODEL.PCD_SIZE}_{cfg.MODEL.DATASET_CONFIG}_{filter_key}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(dtset, f)
    else:
        with open(
            os.path.join(
                root_dir,
                "test_data",
                "dmorp_real",
                f"diffusion_dataset_{data_id}_{cfg.MODEL.PCD_SIZE}_{cfg.MODEL.DATASET_CONFIG}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(dtset, f)
    print("Done!")


def build_dataset_rdiff(data_dir, cfg, data_id: int = 0, vis: bool = False, normalize: bool = False):
    """Build the dataset from the rdiff data"""
    data_file_list = os.listdir(data_dir)
    data_file_list = [f for f in data_file_list if f.endswith(".npz")]
    grid_size = cfg.PREPROCESS.GRID_SIZE
    target_rescale = cfg.PREPROCESS.TARGET_RESCALE
    rot_axis = cfg.DATALOADER.AUGMENTATION.ROT_AXIS
    num_point_lower_bound = cfg.PREPROCESS.NUM_POINT_LOW_BOUND
    # Split info
    split_dict = {}
    train_split_info = os.path.join(data_dir, "split_info", "train_split.txt")
    val_split_info = os.path.join(data_dir, "split_info", "train_val_split.txt")
    test_split_info = os.path.join(data_dir, "split_info", "test_split.txt")
    with open(train_split_info, "r") as f:
        train_split_list = f.readlines()
        train_split_info = [x.split("\n")[0] for x in train_split_list]
    with open(val_split_info, "r") as f:
        val_split_list = f.readlines()
        val_split_info = [x.split("\n")[0] for x in val_split_list]
    with open(test_split_info, "r") as f:
        test_split_list = f.readlines()
        test_split_info = [x.split("\n")[0] for x in test_split_list]
    split_dict["train"] = train_split_info
    split_dict["val"] = val_split_info
    split_dict["test"] = test_split_info

    # Read data
    def parse_child_parent(arr):
        pcd_dict = arr[()]
        parent_val = pcd_dict["parent"]
        child_val = pcd_dict["child"]
        return parent_val, child_val

    data_dict = {}
    data_dict["train"] = []
    data_dict["val"] = []
    data_dict["test"] = []
    for data_file in tqdm(data_file_list, desc="Processing data"):
        data = np.load(os.path.join(data_dir, data_file), allow_pickle=True)
        # parent_pcd_s, child_pcd_s = parse_child_parent(data["multi_obj_start_pcd"])
        parent_pcd_f, child_pcd_f = parse_child_parent(data["multi_obj_final_pcd"])
        parent_pose_s, child_pose_s = parse_child_parent(data["multi_obj_start_obj_pose"])
        # Transform pose to matrix
        parent_pose_s = pose7d_to_mat(parent_pose_s)

        if child_pcd_f.shape[0] <= num_point_lower_bound or parent_pcd_f.shape[0] <= num_point_lower_bound:
            continue
        # Filter outliers
        parent_pcd_f = parent_pcd_f[np.linalg.norm(parent_pcd_f, axis=1) <= 2.0]
        child_pcd_f = child_pcd_f[np.linalg.norm(child_pcd_f, axis=1) <= 2.0]

        # Rescale the target pcd in case that there are not enough points after voxel downsampling
        # Shift all points to the origin
        target_pcd = o3d.geometry.PointCloud()
        # target_pcd.points = o3d.utility.Vector3dVector(child_pcd_s)
        target_pcd.points = o3d.utility.Vector3dVector(child_pcd_f)

        fixed_pcd = o3d.geometry.PointCloud()
        fixed_pcd.points = o3d.utility.Vector3dVector(parent_pcd_f)
        fixed_pcd.transform(np.linalg.inv(parent_pose_s))

        # Sample & Compute normal
        target_pcd.transform(np.linalg.inv(parent_pose_s))
        fixed_pcd, [target_pcd], _, __ = normalize_pcd(fixed_pcd, [target_pcd])

        # Compute normal
        target_pcd_center = (target_pcd.get_max_bound() + target_pcd.get_min_bound()) / 2
        target_pcd.translate(-target_pcd_center)
        # Rescale the target pcd in case that there are not enough points after voxel downsampling
        # target_pcd.scale(target_rescale, center=np.array([0, 0, 0]))  # FIXME: will this bring systematic error?

        target_pcd = target_pcd.voxel_down_sample(grid_size)
        fixed_pcd = fixed_pcd.voxel_down_sample(grid_size)
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        target_pcd_arr = np.hstack((np.array(target_pcd.points), np.array(target_pcd.normals)))
        fixed_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        fixed_pcd_arr = np.hstack((np.array(fixed_pcd.points), np.array(fixed_pcd.normals)))

        # Move target to center
        target_transform = np.eye(4, dtype=np.float32)
        target_transform[:3, 3] = target_pcd_center

        if (
            vis
            or target_pcd_arr.shape[0] < (num_point_lower_bound / 2)
            or fixed_pcd_arr.shape[0] < num_point_lower_bound
        ):
            visualize_pcd_with_open3d(target_pcd_arr, fixed_pcd_arr, np.eye(4, dtype=np.float32))
            visualize_pcd_with_open3d(target_pcd_arr, fixed_pcd_arr, target_transform)
            # Visualize & Check
            raw_target_pcd = o3d.geometry.PointCloud()
            raw_target_pcd.points = o3d.utility.Vector3dVector(child_pcd_f)
            raw_target_pcd.transform(np.linalg.inv(parent_pose_s))
            raw_target_pcd.paint_uniform_color([0.0, 0.651, 0.929])
            raw_fixed_pcd = o3d.geometry.PointCloud()
            raw_fixed_pcd.points = o3d.utility.Vector3dVector(parent_pcd_f)
            raw_fixed_pcd.transform(np.linalg.inv(parent_pose_s))
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([raw_target_pcd, raw_fixed_pcd, origin])

        # DEBUG: sanity check
        if np.max(np.abs(target_pcd_arr[:, :3])) == 0 or np.max(np.abs(fixed_pcd_arr[:, :3])) == 0:
            print("Zero pcd found")
            # Check raw pcd
            vis_list = []
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(child_pcd_f)
            vis_list.append(target_pcd)
            fixed_pcd = o3d.geometry.PointCloud()
            fixed_pcd.points = o3d.utility.Vector3dVector(parent_pcd_f)
            vis_list.append(fixed_pcd)
            o3d.visualization.draw_geometries(vis_list)
            continue

        tmorp_data = {
            "target": target_pcd_arr,
            "fixed": fixed_pcd_arr,
            "target_label": np.array([0]),
            "fixed_label": np.array([1]),
            "9dpose": utils.mat_to_pose9d(target_transform, rot_axis=rot_axis),
            "data_id": data_id,
        }
        for split, split_list in split_dict.items():
            if data_file in split_list:
                data_dict[split].append(tmorp_data)
                break

    print("Len of dtset:", len(data_dict["train"]), len(data_dict["val"]), len(data_dict["test"]))
    # Save the dtset into a .pkl file
    os.makedirs(os.path.join(root_dir, "test_data", "dmorp_rdiff"), exist_ok=True)
    for split, split_list in split_dict.items():
        print(f"Saving dataset to {os.path.join(root_dir, 'test_data', 'dmorp_rdiff')}...")
        with open(
            os.path.join(
                root_dir,
                "test_data",
                "dmorp_rdiff",
                f"diffusion_dataset_{data_id}_{cfg.MODEL.PCD_SIZE}_{cfg.MODEL.DATASET_CONFIG}_{split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(data_dict[split], f)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_id", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="/home/harvey/Data/rdiff/can_in_cabinet_stack/task_name_stack_can_in_cabinet",
    )
    parser.add_argument("--data_type", type=str, default="rdiff")
    parser.add_argument("--filter_key", type=str, default=None)
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    data_root_dir = args.data_root_dir
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_file = os.path.join(root_dir, "config", "pose_transformer_rdiff.py")
    cfg = LazyConfig.load(cfg_file)
    data_id = args.data_id
    filter_key = args.filter_key
    vis = args.vis
    # vis = True

    dtset = []
    for did in data_id:
        print(f"Processing data {did}...")
        if args.data_type == "pyrender":
            data_path = os.path.join(root_dir, "test_data", "dmorp_faster", f"{did:06d}")
            build_dataset_pyrender(data_path, cfg, data_id=did, vis=vis)
        elif args.data_type == "real":
            data_path = os.path.join(root_dir, "test_data", "dmorp_real", f"{did:06d}")
            build_dataset_real(data_path, cfg, data_id=did, vis=vis, filter_key=filter_key)
        elif args.data_type == "rdiff":
            build_dataset_rdiff(data_root_dir, cfg, data_id=did, vis=vis)
        else:
            raise NotImplementedError
