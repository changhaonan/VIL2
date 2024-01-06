"""Dataset definition for point cloud like data.
Code refered from Mask3D: https://github.com/JonasSchult/Mask3D/blob/11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5/datasets/semseg.py#L486
"""
from __future__ import annotations
from typing import Optional
from torch.utils.data import Dataset
import numpy as np
import scipy
import albumentations as A
import volumentations as V
import yaml
from pathlib import Path
import pickle
from copy import deepcopy
from random import random, sample, uniform
import vil2.utils.misc_utils as utils
from scipy.spatial.transform import Rotation as R


class PointCloudDataset(Dataset):
    """Dataset definition for point cloud like data."""

    def __init__(
        self,
        data_file: str,
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
    ):
        # Set parameters
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.is_elastic_distortion = is_elastic_distortion
        self.is_random_distortion = is_random_distortion
        self.random_distortion_rate = random_distortion_rate
        self.random_distortion_mag = random_distortion_mag
        if volume_augmentations_path is not None:
            self.volume_augmentations = V.load(volume_augmentations_path)
        else:
            self.volume_augmentations = None
        if image_augmentations_path is not None:
            self.image_augmentations = A.load(image_augmentations_path)
        else:
            self.image_augmentations = None
        # Load data
        raw_data = pickle.load(open(data_file, "rb"))  # A list of (coordinates, color, normals, labels, pose)
        if indices is not None:
            self._data = [raw_data[i] for i in indices]
        else:
            self._data = raw_data
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
        target_pcd = data["shifted"]["target"]
        fixed_pcd = data["shifted"]["fixed"]
        pose = data["shifted"]["9dpose"]
        return target_pcd, fixed_pcd, pose

    def augment_pcd_instance(self, coordinate, normal, color, label, pose):
        # FIXME: add augmentation
        # label = np.concatenate((label, aug["labels"]))
        if self.is_elastic_distortion:
            coordinate = elastic_distortion(coordinate, 0.1, 0.1)
        if self.is_random_distortion:
            coordinate, color, normal, label = random_around_points(
                coordinate,
                color,
                normal,
                label,
                rate=self.random_distortion_rate,
                noise_level=self.random_distortion_mag,
            )
            # apply random to pose
            pose = random_on_pose(pose, noise_level=self.random_distortion_mag)
        return coordinate, normal, color, label, pose

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        idx = idx % len(self._data)
        # Parse data from the dataset
        target_pcd, fixed_pcd, target_pose = self.parse_pcd_data(idx)
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

        if self.mode == "train":
            # Augment data
            target_coord, target_normal, target_color, _, target_pose = self.augment_pcd_instance(
                target_coord, target_normal, target_color, None, target_pose
            )
            fixed_coord, fixed_normal, fixed_color, _, fixed_pose = self.augment_pcd_instance(
                fixed_coord, fixed_normal, fixed_color, None, fixed_pose
            )
        target_pose = utils.mat_to_pose9d(fixed_pose @ target_pose)
        fixed_pose = utils.mat_to_pose9d(fixed_pose)
        return {
            "target_coord": target_coord,
            "target_normal": target_normal,
            "target_color": target_color,
            "target_pose": target_pose,
            "fixed_coord": fixed_coord,
            "fixed_normal": fixed_normal,
            "fixed_color": fixed_color,
            "fixed_pose": fixed_pose,
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


def rotate_around_axis(points, axis, angle, center_point=None):
    """
    Note: Code borrowed from volumentations
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by angle in radians.
    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    if center_point is None:
        center_point = points[:, :3].mean(axis=0).astype(points[:, :3].dtype)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
    points[:, :3] = points[:, :3] - center_point
    points[:, :3] = np.dot(points[:, :3], rotation_matrix.T)
    points[:, :3] = points[:, :3] + center_point
    return points


if __name__ == "__main__":
    import os

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Test data loader
    dataset = PointCloudDataset(
        data_file=f"{root_dir}/test_data/dmorp_augmented/diffusion_dataset_512_s1000-c200-r0.5.pkl",
        dataset_name="dmorp",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=True,
        is_random_distortion=True,
    )

    # Test data augmentation
    for i in range(10):
        data = dataset[i]
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
            [utils.pose9d_to_mat(target_pose), utils.pose9d_to_mat(fixed_pose)],
        )
