"""Dataset definition for point cloud like data.
Code refered from Mask3D: https://github.com/JonasSchult/Mask3D/blob/11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5/datasets/semseg.py#L486
"""
from __future__ import annotations
from typing import Optional
from torch.utils.data import Dataset
import numpy as np
import scipy
import albumentations as A

# import volumentations as V
import yaml
from pathlib import Path
import pickle
from copy import deepcopy
from random import random, sample, uniform
import vil2.utils.misc_utils as utils
from scipy.spatial.transform import Rotation as R
from yaml import load
from box import Box
import copy
from scipy.spatial.transform import Rotation as R
import open3d as o3d


class PointCloudDataset(Dataset):
    """Dataset definition for point cloud like data."""

    def __init__(
        self,
        data_file_list: list[str],
        dataset_name: str,
        indices: list = None,
        color_mean_std: str = "color_mean_std.yaml",
        add_colors: bool = False,
        add_normals: bool = False,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        is_elastic_distortion: bool = False,
        is_random_distortion: bool = False,
        random_distortion_rate: float = 0.2,
        random_distortion_mag: float = 0.01,
        random_segment_drop_rate: float = 0.15,
    ):
        # Set parameters
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.is_elastic_distortion = is_elastic_distortion
        self.is_random_distortion = is_random_distortion
        self.random_distortion_rate = random_distortion_rate
        self.random_distortion_mag = random_distortion_mag
        self.random_segment_drop_rate = random_segment_drop_rate
        if volume_augmentations_path is not None:
            self.volume_augmentations = Box(yaml.load(open(volume_augmentations_path, "r"), Loader=yaml.FullLoader))
        else:
            self.volume_augmentations = None
        if image_augmentations_path is not None:
            self.image_augmentations = A.load(image_augmentations_path)
        else:
            self.image_augmentations = None
        # Load data
        data_list = []
        for data_file in data_file_list:
            raw_data = pickle.load(open(data_file, "rb"))  # A list of (coordinates, color, normals, labels, pose)
            if indices is not None:
                data_list += [raw_data[i] for i in indices]
            else:
                data_list += raw_data
        self._data = data_list
        # Color normalization
        # if Path(str(color_mean_std)).exists():
        #     color_mean_std = self._load_yaml(color_mean_std)
        #     color_mean, color_std = (
        #         tuple(color_mean_std["mean"]),
        #         tuple(color_mean_std["std"]),
        #     )
        # if add_colors:
        #     self.normalize_color = A.Normalize(mean=color_mean, std=color_std)
        self.mode = "train"  # train, val, test

    def set_mode(self, mode: str):
        self.mode = mode

    def parse_pcd_data(self, batch_idx):
        """Parse data from the dataset."""
        data = self._data[batch_idx]
        target_pcd = data["target"]
        fixed_pcd = data["fixed"]
        pose = data["9dpose"]
        target_label = data["target_label"]
        fixed_label = data["fixed_label"]
        return target_pcd, fixed_pcd, target_label, fixed_label, pose

    def augment_pcd_instance(self, coordinate, normal, color, label, pose):
        # FIXME: add augmentation
        # label = np.concatenate((label, aug["labels"]))
        if self.is_elastic_distortion:
            coordinate = elastic_distortion(copy.deepcopy(coordinate), 0.1, 0.1)
        if self.is_random_distortion:
            coordinate, color, normal, label = random_around_points(
                copy.deepcopy(coordinate),
                copy.deepcopy(color),
                copy.deepcopy(normal),
                copy.deepcopy(label),
                rate=self.random_distortion_rate,
                noise_level=self.random_distortion_mag,
            )

        if self.volume_augmentations is not None:
            # do the below only with some probability
            if "rotation" in self.volume_augmentations.keys():
                if random() < self.volume_augmentations.rotation.prob:
                    coordinate, normal, color, pose = rotate_around_axis(
                        coordinate=copy.deepcopy(coordinate),
                        normal=copy.deepcopy(normal),
                        pose=copy.deepcopy(pose),
                        color=copy.deepcopy(color),
                        axis=np.random.rand(
                            3,
                        ),
                        angle=np.random.rand(1) * 2 * np.pi,
                        center_point=None,
                    )
            if "translation" in self.volume_augmentations.keys():
                if random() < self.volume_augmentations.translation.prob:
                    random_offset = np.random.rand(1, 3)
                    random_offset[0, 0] = np.random.uniform(
                        self.volume_augmentations.translation.min_x,
                        self.volume_augmentations.translation.max_x,
                        size=(1,),
                    )
                    random_offset[0, 1] = np.random.uniform(
                        self.volume_augmentations.translation.min_y,
                        self.volume_augmentations.translation.max_y,
                        size=(1,),
                    )
                    random_offset[0, 2] = np.random.uniform(
                        self.volume_augmentations.translation.min_z,
                        self.volume_augmentations.translation.max_z,
                        size=(1,),
                    )
                    coordinate, normal, color, pose = random_translation(
                        coordinate=coordinate,
                        normal=normal,
                        pose=pose,
                        color=color,
                        offset_type="given",
                        offset=random_offset,
                    )
            if "segment_drop" in self.volume_augmentations.keys():
                if random() < self.volume_augmentations.segment_drop.prob:
                    coordinate, normal, color, pose = random_segment_drop(
                        coordinate=coordinate,
                        normal=normal,
                        pose=pose,
                        color=color,
                        random_segment_drop_rate=self.random_segment_drop_rate,
                    )

        return coordinate, normal, color, label, pose

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        idx = idx % len(self._data)
        # Parse data from the dataset
        target_pcd, fixed_pcd, target_label, fixed_label, pose = self.parse_pcd_data(idx)
        # Convert to float32
        target_pcd = target_pcd.astype(np.float32)
        fixed_pcd = fixed_pcd.astype(np.float32)
        target_label = np.array([target_label]).astype(np.int64)
        fixed_label = np.array([fixed_label]).astype(np.int64)
        target_pose = pose.astype(np.float32)

        # Prepare data
        target_coord = target_pcd[:, :3]
        fixed_coord = fixed_pcd[:, :3]
        if self.add_normals:
            target_normal = target_pcd[:, 3:6]
            fixed_normal = fixed_pcd[:, 3:6]
        else:
            target_normal = None
            fixed_normal = None
        if self.add_colors:
            target_color = target_pcd[:, 6:9]
            fixed_color = fixed_pcd[:, 6:9]
        else:
            target_color = None
            fixed_color = None
        target_pose = utils.pose9d_to_mat(target_pose)
        fixed_pose = np.eye(4)

        if self.mode == "train" or self.mode == "val":
            # Augment data
            target_coord, target_normal, target_color, _, target_pose = self.augment_pcd_instance(
                target_coord, target_normal, target_color, None, target_pose
            )
            fixed_coord, fixed_normal, fixed_color, _, fixed_pose = self.augment_pcd_instance(
                fixed_coord, fixed_normal, fixed_color, None, fixed_pose
            )

        target_pose = utils.mat_to_pose9d(np.linalg.inv(fixed_pose) @ target_pose)
        fixed_pose = utils.mat_to_pose9d(fixed_pose)
        return {
            "target_coord": target_coord.astype(np.float32),
            "target_normal": target_normal.astype(np.float32),
            "target_color": target_color.astype(np.float32),
            "target_pose": target_pose.astype(np.float32),
            "target_label": target_label.astype(np.int64),
            "fixed_coord": fixed_coord.astype(np.float32),
            "fixed_normal": fixed_normal.astype(np.float32),
            "fixed_color": fixed_color.astype(np.float32),
            "fixed_pose": fixed_pose.astype(np.float32),
            "fixed_label": fixed_label.astype(np.int64),
        }

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f)
        return file

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def random_around_points(
    coordinates,
    color,
    normals,
    labels,
    rate=0.2,
    noise_level=0.01,
    ignore_label=255,
):
    # coordinate
    coord_indexes = sample(list(range(len(coordinates))), k=int(len(coordinates) * rate))
    coordinate_noises = np.random.rand(len(coord_indexes), 3) * 2 - 1
    coordinates[coord_indexes] += coordinate_noises * noise_level

    # normals
    if normals is not None:
        normal_noises = np.random.rand(len(coord_indexes), 3) * 2 - 1
        normals[coord_indexes] += normal_noises * noise_level
    return coordinates, color, normals, labels


def random_on_pose(
    pose,
    noise_level=0.01,
):
    tran_noise = (np.random.rand(3) * 2 - 1) * noise_level
    rot_vector = np.random.rand(3) * 2 - 1
    rot_vector = rot_vector / np.linalg.norm(rot_vector)
    rot_angle = (np.random.rand(1) * 2 - 1) * np.pi / 2 * noise_level
    pose[:3, 3] += tran_noise
    pose[:3, :3] = R.from_rotvec(rot_vector * rot_angle).as_matrix() @ pose[:3, :3]
    return pose


def rotate_around_axis(coordinate, normal, color, pose, axis, angle, center_point=None):
    axis = axis / np.linalg.norm(axis)
    rotation_matrix = R.from_rotvec(axis * angle).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix

    # Manually apply the transformation to each point and normal
    transformed_points = []
    transformed_normals = []
    for point, normal in zip(coordinate, normal):
        # Convert to homogeneous coordinate and transform
        point_homogeneous = np.append(point, 1)
        transformed_point_homogeneous = np.dot(transformation_matrix, point_homogeneous)
        transformed_point = transformed_point_homogeneous[:3]
        transformed_points.append(transformed_point)

        # Transform the normal (only rotation)
        transformed_normal = np.dot(rotation_matrix, normal)
        transformed_normals.append(transformed_normal)

    pose_transform = np.eye(4)
    pose_transform[:3, :3] = np.linalg.inv(rotation_matrix)
    pose = pose @ pose_transform
    return np.array(transformed_points), np.array(transformed_normals), copy.deepcopy(color), copy.deepcopy(pose)


def random_translation(coordinate, normal, color, pose, offset_type: str = "given", offset=None):
    """
    Return the translated coordinates, normals and the pose
    """
    if offset_type == "center":
        offset = coordinate.mean(axis=0).astype(coordinate.dtype)
        offset = -offset
    else:
        assert offset is not None

    transformation_matrix = np.eye(4)
    rotation_matrix = np.eye(3)
    transformation_matrix[:3, 3] = offset

    # Manually apply the transformation to each point and normal
    transformed_points = []
    transformed_normals = []
    for point, normal in zip(coordinate, normal):
        # Convert to homogeneous coordinate and transform
        point_homogeneous = np.append(point, 1)
        transformed_point_homogeneous = np.dot(transformation_matrix, point_homogeneous)
        transformed_point = transformed_point_homogeneous[:3]
        transformed_points.append(transformed_point)

        # Transform the normal (only rotation)
        transformed_normal = np.dot(rotation_matrix, normal)
        transformed_normals.append(transformed_normal)

    pose_transform = np.eye(4)
    pose_transform[:3, :3] = np.linalg.inv(rotation_matrix)
    pose_transform[:3, 3] = -offset
    pose = pose @ pose_transform
    return np.array(transformed_points), np.array(transformed_normals), copy.deepcopy(color), copy.deepcopy(pose)


def random_segment_drop(coordinate, normal, color, pose, random_segment_drop_rate: float = 0.15):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinate)
    pcd.normals = o3d.utility.Vector3dVector(normal)
    pcd.colors = o3d.utility.Vector3dVector(color)
    print(f"Before segment drop with {coordinate.shape[0]} points")
    # o3d.visualization.draw_geometries([pcd])
    # Heuristic: center of the sphere is the mean of the points
    center = coordinate.mean(axis=0)
    # randomly shift the center but keep it inside the point cloud
    center += np.random.uniform(low=-0.02, high=0.02, size=(3,))
    # Heuristic: start with a small sphere and increase until k% of points are inside
    total_points = len(coordinate)
    target_points = total_points * (random_segment_drop_rate)
    radius = 0.02  # initial radius
    while True:
        # Calculate distances from the center
        distances = np.linalg.norm(coordinate - center, axis=1)
        # Count points inside the sphere
        inside_count = np.sum(distances < radius)
        if inside_count >= target_points:
            break
        else:
            radius += 0.02

    # Create a mask for points outside the sphere
    mask = distances >= radius
    assert mask.sum() > 0, "No points outside the sphere"
    if mask.sum() < target_points:
        mask = distances < radius
        print("Masking inside points instead")
    # Create a new point cloud without the points inside the sphere
    new_points = coordinate[mask]
    print(f"After segment drop with {new_points.shape[0]} points")
    # Duplicate points to make up for the dropped points
    num_points_to_add = total_points - len(new_points)
    indices_to_duplicate = np.random.choice(len(new_points), num_points_to_add)
    duplicated_points = new_points[indices_to_duplicate]
    coordinate = copy.deepcopy(np.concatenate((new_points, duplicated_points)))
    # Load it to open3d pcd object and recomputes normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinate)
    pcd.estimate_normals()
    normal = copy.deepcopy(np.asarray(pcd.normals))
    pcd.colors = o3d.utility.Vector3dVector(color)
    print(f"After duplication with {coordinate.shape[0]} points")
    # o3d.visualization.draw_geometries([pcd])
    return coordinate, normal, copy.deepcopy(color), copy.deepcopy(pose)


if __name__ == "__main__":
    import os

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Test data loader
    dataset = PointCloudDataset(
        data_file_list=[f"{root_dir}/test_data/dmorp_real/diffusion_dataset_0_512_s25000-c1-r0.5.pkl"],
        dataset_name="dmorp",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=True,
        is_random_distortion=True,
        volume_augmentations_path=f"{root_dir}/config/va_rotation.yaml",
    )
    dataset.set_mode("train")

    # Test data augmentation
    for i in range(10):
        random_idx = np.random.randint(0, len(dataset))
        data = dataset[random_idx]
        target_coord = data["target_coord"]
        target_normal = data["target_normal"]
        target_color = data["target_color"]
        target_pose = data["target_pose"]
        fixed_coord = data["fixed_coord"]
        fixed_normal = data["fixed_normal"]
        fixed_color = data["fixed_color"]
        fixed_pose = data["fixed_pose"]

        utils.visualize_pcd_list(
            [target_coord, fixed_coord],
            [target_normal, fixed_normal],
            [target_color, fixed_color],
            [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)],
        )

        utils.visualize_pcd_list(
            [target_coord, fixed_coord],
            [target_normal, fixed_normal],
            [target_color, fixed_color],
            [utils.pose9d_to_mat(target_pose), np.eye(4, dtype=np.float32)],
        )
