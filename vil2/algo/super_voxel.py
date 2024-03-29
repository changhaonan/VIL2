"""For a given geometry, generate a super-patch."""
from __future__ import annotations
import numpy as np
import open3d as o3d
from copy import deepcopy


def furthest_point_sampling(points, num_samples):
    # Initialize an array to store the indices of sampled points
    sampled_indices = []
    # Start with a random index
    current_index = np.random.choice(len(points))
    sampled_indices.append(current_index)

    # Compute the initial distances from the first point to all other points
    all_distances = np.linalg.norm(points - points[current_index], axis=1)

    for _ in range(1, num_samples):
        # Select the point that is furthest away from all previously selected points
        current_index = np.argmax(all_distances)
        sampled_indices.append(current_index)

        # Update the distances based on the new point
        distances_from_new_point = np.linalg.norm(points - points[current_index], axis=1)
        all_distances = np.minimum(all_distances, distances_from_new_point)

    # Return the sampled points
    return points[sampled_indices]


def generate_random_voxel(geometry: o3d.geometry.TriangleMesh, num_points: int, seed: int = 0):
    np.random.seed(seed)
    # sample point using furthest point sampling
    sampled_points = o3d.geometry.PointCloud()
    sampled_points.points = o3d.utility.Vector3dVector(
        furthest_point_sampling(np.asarray(geometry.vertices), num_points)
    )

    # compute radius, the min distance between points
    radius = np.inf
    for i in range(len(sampled_points.points)):
        for j in range(i + 1, len(sampled_points.points)):
            radius = min(
                radius,
                np.linalg.norm(
                    np.asarray(sampled_points.points)[i]
                    - np.asarray(sampled_points.points)[j]
                ),
            )
    # cluster points w.r.t distance towards samples points
    pcd_tree = o3d.geometry.KDTreeFlann(geometry)
    cluster_labels = np.zeros(len(geometry.vertices), dtype=np.int32)
    for i in range(len(sampled_points.points)):
        _, indices, _ = pcd_tree.search_radius_vector_3d(
            sampled_points.points[i], radius
        )
        cluster_labels[indices] = i + 1  # 0 is background
    # generate super voxel
    super_voxel = []
    for i in range(len(sampled_points.points)):
        indices = np.where(cluster_labels == (i + 1))[0]
        patch = deepcopy(geometry)
        # select points by indices
        patch.vertices = o3d.utility.Vector3dVector(
            np.asarray(patch.vertices)[indices]
        )
        patch.triangles = o3d.utility.Vector3iVector(
            np.asarray(patch.triangles)[indices]
        )
        # remove duplicated vertices
        patch.remove_duplicated_vertices()
        #
        patch_pcd = deepcopy(np.asarray(patch.vertices))
        super_voxel.append(patch_pcd)
    return super_voxel, np.asarray(sampled_points.points)


def generate_circle_yz_voxel(radius: float, num_points, seed: int = 0):
    """Generate `num_points` points on a circle with radius `radius`. on y-z plane"""
    np.random.seed(seed)
    points = []
    for i in range(num_points):
        theta = i * 2 * np.pi / num_points
        x = 0
        y = radius * np.cos(theta)
        z = radius * np.sin(theta)
        points.append(np.array([x, y, z]))
    return points, np.stack(points, axis=0)
