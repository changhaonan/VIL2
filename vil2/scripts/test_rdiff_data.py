"""Test the performance of RDiff"""

import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from vil2.utils.pcd_utils import check_pcd_pyramid


def parse_child_parent(arr):
    pcd_dict = arr[()]
    parent_val = pcd_dict["parent"]
    child_val = pcd_dict["child"]
    return parent_val, child_val


def pose7d_to_mat(pose7d):
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = R.from_quat(pose7d[3:]).as_matrix()
    pose_mat[:3, 3] = pose7d[:3]
    return pose_mat


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


def gen_bbox_grid(box_size: float, grid_size: int):
    bbox_list = []
    for i in range(-grid_size, grid_size):
        for j in range(-grid_size, grid_size):
            for k in range(-grid_size, grid_size):
                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=[i * box_size, j * box_size, k * box_size],
                    max_bound=[(i + 1) * box_size, (j + 1) * box_size, (k + 1) * box_size],
                )
                bbox.color = (1, 0, 0)
                bbox_list.append(bbox)
    return bbox_list


def compute_nearby_pointcloud(pcd_anchor, pcd_goal, radius=0.1):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_anchor)
    indices = []
    for point in pcd_goal.points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, radius)
        indices.extend(idx)
    return indices


if __name__ == "__main__":
    data_root_folder = "/home/harvey/Data/rdiff"
    data_name = "can_in_cabinet_stack"
    task_name = "task_name_stack_can_in_cabinet"
    # data_name = "mug_on_rack_multi_large_proc_gen_demos"
    # task_name = "task_name_mug_on_rack_multi"
    data_folder = os.path.join(data_root_folder, data_name, task_name)
    data_list = os.listdir(data_folder)
    data_list = [os.path.join(data_folder, data) for data in data_list]

    for idx in range(len(data_list)):
        # Load data
        data = np.load(data_list[idx], allow_pickle=True)
        data_dict = {}
        for key in data.keys():
            data_dict[key] = data[key]
        parent_pcd_s, child_pcd_s = parse_child_parent(data["multi_obj_start_pcd"])
        parent_pcd_f, child_pcd_f = parse_child_parent(data["multi_obj_final_pcd"])
        parent_pose_s_7d, child_pose_s_7d = parse_child_parent(data["multi_obj_start_obj_pose"])
        parent_pose_f_7d, child_pose_f_7d = parse_child_parent(data["multi_obj_final_obj_pose"])

        # Jump empty data
        if parent_pcd_s.size == 0 or child_pcd_s.size == 0:
            continue

        # Move the child point cloud to the origin
        child_pcd_center = (child_pcd_s.max(axis=0) + child_pcd_s.min(axis=0)) / 2
        child_pcd_s[:, :3] -= child_pcd_center

        if type(parent_pose_s_7d[0]) != np.ndarray:
            # Unify the type
            parent_pose_s_7d = [parent_pose_s_7d]
            child_pose_s_7d = [child_pose_s_7d]
            child_pose_f_7d = [child_pose_f_7d]
        num_poses = len(parent_pose_s_7d)
        if num_poses > 1:
            continue
        for idx_pose in range(num_poses):
            parent_pose_s = pose7d_to_mat(parent_pose_s_7d[idx_pose])
            child_pose_s = pose7d_to_mat(child_pose_s_7d[0])
            child_pose_f = pose7d_to_mat(child_pose_f_7d[0])

            # Shift all points to the origin
            parent_pose_s_inv = np.linalg.inv(parent_pose_s)
            parent_pcd_s_origin = parent_pcd_s @ parent_pose_s_inv[:3, :3].T + parent_pose_s_inv[:3, 3]

            target_shift = np.eye(4, dtype=np.float32)
            target_shift[:3, 3] = child_pcd_center

            child_movement = child_pose_f @ np.linalg.inv(child_pose_s)
            child_movement = np.linalg.inv(parent_pose_s) @ child_movement @ target_shift

            # Visualize the start and final point clouds
            start_parent_pcd_o3d = o3d.geometry.PointCloud()
            start_parent_pcd_o3d.points = o3d.utility.Vector3dVector(parent_pcd_s_origin)
            start_parent_pcd_o3d.paint_uniform_color([1, 0, 0])
            start_child_pcd_o3d = o3d.geometry.PointCloud()
            start_child_pcd_o3d.points = o3d.utility.Vector3dVector(child_pcd_s)
            start_child_pcd_o3d.paint_uniform_color([0, 1, 0])
            start_child_pcd_o3d.transform(child_movement)

            start_parent_pcd_o3d, normalized_pcd_list, start_parent_pcd_center, scale_xyz = normalize_pcd(
                start_parent_pcd_o3d, [start_child_pcd_o3d]
            )
            unit_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-0.5, -0.5, -0.5], max_bound=[0.5, 0.5, 0.5])
            unit_bbox.color = (1, 0, 0)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

            # Compute nearby point cloud
            nearby_radius = 0.02
            nearby_indices = compute_nearby_pointcloud(
                start_parent_pcd_o3d, normalized_pcd_list[0], radius=nearby_radius
            )
            # Paint the nearby points to a different color
            colors = np.asarray(start_parent_pcd_o3d.colors)
            colors[nearby_indices] = (0, 0, 1)

            nearby_indices = compute_nearby_pointcloud(
                normalized_pcd_list[0], start_parent_pcd_o3d, radius=nearby_radius
            )
            # Paint the nearby points to a different color
            colors = np.asarray(normalized_pcd_list[0].colors)
            colors[nearby_indices] = (0, 1, 1)
            
            # Check point pyramid
            pcd_list = check_pcd_pyramid(start_parent_pcd_o3d, [0.015, 0.03, 0.06, 0.09])

            o3d.visualization.draw_geometries([start_parent_pcd_o3d, unit_bbox, origin] + normalized_pcd_list)

        # # Move goal center
        # child_translation = child_movement[:3, 3]
        # child_translation = (child_translation - start_parent_pcd_center) / scale_xyz

        # # Voxelization
        # N = len(parent_pcd_s)
        # start_parent_pcd_o3d.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
        # start_parent_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(start_parent_pcd_o3d, voxel_size=0.05)

        # voxelized_grid_list = []
        # for pcd in normalized_pcd_list:
        #     voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.08)
        #     voxelized_grid_list.append(voxel_grid)

        # # Generate a grid of bounding boxes
        # box_size = 0.16
        # grid_size = np.ceil(0.5 / box_size).astype(int)
        # bbox_list = gen_bbox_grid(box_size, grid_size)
        # o3d.visualization.draw_geometries(
        #     [start_parent_voxel_grid, origin, unit_bbox] + voxelized_grid_list + bbox_list
        # )

        # start_child_pcd_o3d = o3d.geometry.PointCloud()
        # start_child_pcd_o3d.points = o3d.utility.Vector3dVector(child_pcd_s)
        # start_child_pcd_o3d.paint_uniform_color([0, 1, 0])
        # # start_child_pcd_o3d.transform(cild_movement)
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([start_parent_pcd_o3d, start_child_pcd_o3d, origin])

        # start_child_pcd_o3d.transform(cild_movement)
        # # add a sphere
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # sphere.compute_vertex_normals()
        # sphere.paint_uniform_color([1, 0, 0])
        # sphere.transform(cild_movement)
        # o3d.visualization.draw_geometries([start_parent_pcd_o3d, sphere, origin])

        # # print(f"Success: {data_dict['success']}")
        # # Final pcd
        # final_parent_pcd_o3d = o3d.geometry.PointCloud()
        # final_parent_pcd_o3d.points = o3d.utility.Vector3dVector(parent_pcd_f)
        # final_child_pcd_o3d = o3d.geometry.PointCloud()
        # final_child_pcd_o3d.points = o3d.utility.Vector3dVector(child_pcd_f)
        # final_child_pcd_o3d.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([final_parent_pcd_o3d, final_child_pcd_o3d])
