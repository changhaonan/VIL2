from __future__ import annotations
import open3d as o3d
import copy
import torch
import numpy as np


def check_pcd_pyramid(pcd: o3d.PointCloud, grid_sizes: list[int]):
    """Visualize the pcd pyramid."""
    pcd_list = [pcd]
    current_pcd = pcd
    print(f"Pcd size: {len(current_pcd.points)}")
    for grid_size in grid_sizes:
        pcd_down = current_pcd.voxel_down_sample(voxel_size=grid_size)
        pcd_list.append(pcd_down)
        print(f"Pcd size: {len(pcd_down.points)}")
        o3d.visualization.draw_geometries([pcd_down])
        current_pcd = pcd_down
    return pcd_list


def visualize_tensor_pcd(pcd: torch.Tensor):
    """Visualize the tensor pcd."""
    pcd = pcd.cpu().numpy()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    o3d.visualization.draw_geometries([pcd])


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
