"""Utility function for 3D operations.
"""

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


def normalize_pcd(pcd_anchor, pcd_target, do_scaling: bool = True):
    # Normalize to unit cube
    pcd_center = (pcd_anchor.get_max_bound() + pcd_anchor.get_min_bound()) / 2
    pcd_anchor = pcd_anchor.translate(-pcd_center)
    scale_xyz = pcd_anchor.get_max_bound() - pcd_anchor.get_min_bound()
    scale_xyz = np.max(scale_xyz)
    if not do_scaling:
        scale_xyz = 1.0
    pcd_anchor = pcd_anchor.scale(1 / scale_xyz, center=np.array([0, 0, 0]))

    # Normalize the child point clouds
    pcd_target = pcd_target.translate(-pcd_center)
    normalize_pcd_target = pcd_target.scale(1 / scale_xyz, center=np.array([0, 0, 0]))
    return pcd_anchor, normalize_pcd_target, pcd_center, scale_xyz


def check_collision(pcd_anchor: np.ndarray, pcd_target: np.ndarray, threshold=0.01):
    """Check if there existing collision between two point clouds."""
    dists = np.linalg.norm(pcd_anchor[:, None, :] - pcd_target[None, :, :], axis=-1)
    min_dists = np.min(dists, axis=1)
    return np.any(min_dists < threshold)


def visualize_point_pyramid(pos: np.ndarray | torch.Tensor, normal: np.ndarray | torch.Tensor | None, cluster_indices: list[np.ndarray] | list[torch.Tensor]):
    """Visualize the point pyramid."""
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    if isinstance(normal, torch.Tensor):
        normal = normal.cpu().numpy()
    if isinstance(cluster_indices[0], torch.Tensor):
        cluster_indices = [c.cpu().numpy() for c in cluster_indices]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)
    num_clusters = [np.unique(c).size for c in cluster_indices]
    num_color = max(num_clusters)
    color = np.random.rand(num_color, 3)
    cum_cluster_index = cluster_indices[0]
    for i, cluster_index in enumerate(cluster_indices):
        cluster_index_mask = cluster_index != 0
        cluster_index[cluster_index_mask] = cluster_index[cluster_index_mask] - np.min(cluster_index[cluster_index_mask])
        # Map the cluster index
        if i != 0:
            for j in range(len(cluster_index)):
                if cluster_index[j] != 0:
                    cum_cluster_index[cum_cluster_index == j] = cluster_index[j]
        pcd.colors = o3d.utility.Vector3dVector(color[cum_cluster_index])
        o3d.visualization.draw_geometries([pcd])
