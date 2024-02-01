"""Test the performance of RDiff"""

import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


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


if __name__ == "__main__":
    data_root_folder = "/home/harvey/Data/rdiff"
    data_name = "can_in_cabinet_stack"
    task_name = "task_name_stack_can_in_cabinet"
    data_folder = os.path.join(data_root_folder, data_name, task_name)
    data_list = os.listdir(data_folder)
    data_list = [os.path.join(data_folder, data) for data in data_list]

    for idx in range(len(data_list)):
        # Load data
        data = np.load(data_list[idx], allow_pickle=True)
        data_dict = {}
        for key in data.keys():
            data_dict[key] = data[key]
        start_parent_pcd, start_child_pcd = parse_child_parent(data["multi_obj_start_pcd"])
        final_parent_pcd, final_child_pcd = parse_child_parent(data["multi_obj_final_pcd"])
        start_parent_obj_pose, start_child_obj_pose = parse_child_parent(data["multi_obj_start_obj_pose"])
        final_parent_obj_pose, final_child_obj_pose = parse_child_parent(data["multi_obj_final_obj_pose"])
        start_parent_obj_pose = pose7d_to_mat(start_parent_obj_pose)
        start_child_obj_pose = pose7d_to_mat(start_child_obj_pose)
        final_parent_obj_pose = pose7d_to_mat(final_parent_obj_pose)
        final_child_obj_pose = pose7d_to_mat(final_child_obj_pose)

        # Compute movement
        cild_movement = final_child_obj_pose @ np.linalg.inv(start_child_obj_pose)

        # Visualize the start and final point clouds
        start_parent_pcd_o3d = o3d.geometry.PointCloud()
        start_parent_pcd_o3d.points = o3d.utility.Vector3dVector(start_parent_pcd)
        start_child_pcd_o3d = o3d.geometry.PointCloud()
        start_child_pcd_o3d.points = o3d.utility.Vector3dVector(start_child_pcd)
        start_child_pcd_o3d.paint_uniform_color([0, 1, 0])
        start_child_pcd_o3d.transform(cild_movement)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([start_parent_pcd_o3d, start_child_pcd_o3d, origin])

        final_parent_pcd_o3d = o3d.geometry.PointCloud()
        final_parent_pcd_o3d.points = o3d.utility.Vector3dVector(final_parent_pcd)
        final_child_pcd_o3d = o3d.geometry.PointCloud()
        final_child_pcd_o3d.points = o3d.utility.Vector3dVector(final_child_pcd)
        final_child_pcd_o3d.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([final_parent_pcd_o3d, final_child_pcd_o3d, origin])

        print(f"Success: {data_dict['success']}")
