"""Dataset definition for point cloud like data.
Code refered from Mask3D: https://github.com/JonasSchult/Mask3D/blob/11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5/datasets/semseg.py#L486
"""

from __future__ import annotations
from typing import Optional
from torch.utils.data import Dataset
import numpy as np
import scipy
import albumentations as A
import random

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
from sklearn.neighbors import NearestNeighbors


class PcdPairDataset(Dataset):
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
        crop_pcd: bool = False,
        crop_size: float = 0.2,
        crop_noise: float = 0.1,
        crop_strategy: str = "knn",
        noise_level: float = 0.1,
        rot_axis: str = "xy",
        **kwargs,
    ):
        # Set parameters
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.is_elastic_distortion = is_elastic_distortion
        self.is_random_distortion = is_random_distortion
        self.random_distortion_rate = random_distortion_rate
        self.random_distortion_mag = random_distortion_mag
        self.crop_pcd = crop_pcd
        self.crop_size = crop_size
        self.crop_noise = crop_noise
        self.crop_strategy = crop_strategy
        if self.crop_strategy == "knn":
            self.knn_k = kwargs.get("knn_k", 20)
        self.noise_level = noise_level  # number of noise levels
        if volume_augmentations_path is not None:
            self.volume_augmentations = Box(yaml.load(open(volume_augmentations_path, "r"), Loader=yaml.FullLoader))
        else:
            self.volume_augmentations = None
        if image_augmentations_path is not None:
            self.image_augmentations = A.load(image_augmentations_path)
        else:
            self.image_augmentations = None
        self.rot_axis = rot_axis
        # Load data
        data_list = []
        for data_file in data_file_list:
            raw_data = pickle.load(open(data_file, "rb"))  # A list of (coordinates, color, normals, labels, pose)
            if indices is not None:
                data_list += [raw_data[i] for i in indices]
            else:
                data_list += raw_data
        self._data = data_list
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

    def augment_pcd_instance(self, coordinate, normal, color, label, pose, disable_rot: bool = False):
        """Augment a single point cloud instance."""
        if self.is_elastic_distortion:
            # coordinate = elastic_distortion(coordinate, 0.05, 0.05)
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

        if self.volume_augmentations is not None:
            # do the below only with some probability
            if "rotation" in self.volume_augmentations.keys() and not disable_rot:
                max_angle = np.pi * self.noise_level
                if random() < self.volume_augmentations.rotation.prob:
                    angle = np.random.uniform(-max_angle, max_angle)
                    coordinate, normal, color, pose = rotate_around_axis(
                        coordinate=coordinate,
                        normal=normal,
                        pose=pose,
                        color=color,
                        axis=np.random.rand(
                            3,
                        ),
                        angle=angle,
                        center_point=None,
                    )
            if "translation" in self.volume_augmentations.keys():
                if random() < self.volume_augmentations.translation.prob:
                    random_offset = np.random.rand(1, 3)
                    random_offset[0, 0] = np.random.uniform(
                        self.volume_augmentations.translation.min_x * self.noise_level,
                        self.volume_augmentations.translation.max_x * self.noise_level,
                        size=(1,),
                    )
                    random_offset[0, 1] = np.random.uniform(
                        self.volume_augmentations.translation.min_y * self.noise_level,
                        self.volume_augmentations.translation.max_y * self.noise_level,
                        size=(1,),
                    )
                    random_offset[0, 2] = np.random.uniform(
                        self.volume_augmentations.translation.min_z * self.noise_level,
                        self.volume_augmentations.translation.max_z * self.noise_level,
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
        target_pose = utils.pose9d_to_mat(target_pose, rot_axis=self.rot_axis)
        fixed_pose = np.eye(4)

        if self.mode == "train" or self.mode == "val":
            # Augment data
            target_coord, target_normal, target_color, _, target_pose = self.augment_pcd_instance(
                target_coord, target_normal, target_color, None, target_pose
            )
            fixed_coord, fixed_normal, fixed_color, _, fixed_pose = self.augment_pcd_instance(
                fixed_coord, fixed_normal, fixed_color, None, fixed_pose, disable_rot=True
            )  # Disable rotation for fixed pcd

        # Crop pcd to focus
        if self.crop_pcd:
            max_crop_attempts = 10
            for i in range(max_crop_attempts):
                crop_center = target_pose[:3, 3] + np.random.rand(3) * 2 * self.crop_noise - self.crop_noise
                crop_size = (0.5 * np.random.rand(3) + 0.5) * self.crop_size
                x_min, y_min, z_min = crop_center - crop_size
                x_max, y_max, z_max = crop_center + crop_size
                if self.crop_strategy == "bbox":
                    fixed_indices = crop_bbox(fixed_coord, x_min, y_min, z_min, x_max, y_max, z_max)
                elif self.crop_strategy == "radius":
                    fixed_indices = crop_radius(fixed_coord, crop_center, self.crop_size)
                elif self.crop_strategy == "knn":
                    fixed_indices = crop_knn(fixed_coord, target_coord, crop_center, k=self.knn_k)
                else:
                    raise ValueError("Invalid crop strategy")
                if fixed_indices.sum() > 0:  # Make sure there are points in the crop
                    break
                if i == max_crop_attempts - 1:
                    print("Warning: Failed to find a crop")
                    fixed_indices = np.arange(len(fixed_coord))
            raw_fixed_coord = copy.deepcopy(fixed_coord)
            fixed_coord = fixed_coord[fixed_indices]
            if self.add_normals:
                fixed_normal = fixed_normal[fixed_indices]
            if self.add_colors:
                fixed_color = fixed_color[fixed_indices]
            # Update fixed pose
            fixed_shift = np.eye(4)
            fixed_shift[:3, 3] = crop_center
            fixed_pose = fixed_pose @ fixed_shift
            fixed_coord -= crop_center
            # DEBUG:
            raw_fixed_coord -= crop_center

        target_pose = utils.mat_to_pose9d(np.linalg.inv(fixed_pose) @ target_pose, rot_axis=self.rot_axis)
        fixed_pose = utils.mat_to_pose9d(fixed_pose, rot_axis=self.rot_axis)

        # Concat feat
        target_feat = [copy.deepcopy(target_coord)]
        fixed_feat = [copy.deepcopy(fixed_coord)]
        if self.add_colors:
            target_feat.append(target_color)
            fixed_feat.append(fixed_color)
        if self.add_normals:
            target_feat.append(target_normal)
            fixed_feat.append(fixed_normal)
        target_feat = np.concatenate(target_feat, axis=-1)
        fixed_feat = np.concatenate(fixed_feat, axis=-1)

        # # Visualize
        # vis_list = []
        # raw_fixed_pcd_o3d = o3d.geometry.PointCloud()
        # raw_fixed_pcd_o3d.points = o3d.utility.Vector3dVector(raw_fixed_coord)
        # raw_fixed_pcd_o3d.paint_uniform_color([0.1, 0.1, 0.7])
        # vis_list.append(raw_fixed_pcd_o3d)
        # target_pcd_o3d = o3d.geometry.PointCloud()
        # target_pcd_o3d.points = o3d.utility.Vector3dVector(target_coord)
        # target_pcd_o3d.paint_uniform_color([0.7, 0.1, 0.1])
        # vis_list.append(target_pcd_o3d)
        # fixed_pcd_o3d = o3d.geometry.PointCloud()
        # fixed_pcd_o3d.points = o3d.utility.Vector3dVector(fixed_coord)
        # fixed_pcd_o3d.paint_uniform_color([0.1, 0.7, 0.1])
        # vis_list.append(fixed_pcd_o3d)
        # target_pose_mat = utils.pose9d_to_mat(target_pose, rot_axis=self.rot_axis)
        # target_pcd_shift_o3d = o3d.geometry.PointCloud()
        # target_pcd_shift_o3d.points = o3d.utility.Vector3dVector(target_coord)
        # target_pcd_shift_o3d.paint_uniform_color([0.7, 0.1, 0.7])
        # target_pcd_shift_o3d.transform(target_pose_mat)
        # vis_list.append(target_pcd_shift_o3d)
        # o3d.visualization.draw_geometries(vis_list)

        # DEBUG: sanity check
        if fixed_coord.shape[0] == 0 or np.max(np.abs(fixed_coord)) == 0:
            print("Fixed coord is zero")
            # After crop this becomes empty
            vis_list = []
            fixed_pcd_o3d = o3d.geometry.PointCloud()
            fixed_pcd_o3d.points = o3d.utility.Vector3dVector(fixed_pcd[:, :3])
            vis_list.append(fixed_pcd_o3d)
            # Add a sphere to visualize the crop center
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.crop_size, resolution=20)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([0.1, 0.1, 0.7])
            sphere.translate(crop_center)
            vis_list.append(sphere)
            o3d.visualization.draw_geometries(vis_list)

        if target_coord.shape[0] == 0 or np.max(np.abs(target_coord)) == 0:
            print("Target coord is zero")

        return {
            "target_coord": target_coord.astype(np.float32),
            "target_feat": target_feat.astype(np.float32),
            "target_pose": target_pose.astype(np.float32),
            "target_label": target_label.astype(np.int64),
            "fixed_coord": fixed_coord.astype(np.float32),
            "fixed_feat": fixed_feat.astype(np.float32),
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


def crop_bbox(points, x_min, y_min, z_min, x_max, y_max, z_max):
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


def crop_radius(points, center, radius):
    inds = np.linalg.norm(points - center, axis=1) < radius
    return inds


def crop_knn(points, ref_points, crop_center, k=20):
    if points.shape[0] < k:
        raise ValueError("The number of points should be larger than k")
    points_shifted = points - crop_center
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points_shifted)
    indices = neigh.kneighbors(ref_points, return_distance=False)
    return indices.flatten()


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
    return np.array(transformed_points), np.array(transformed_normals), color, pose


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
        transformed_normal = normal
        transformed_normals.append(transformed_normal)

    pose_transform = np.eye(4)
    pose_transform[:3, 3] = -offset
    pose = pose @ pose_transform
    return np.array(transformed_points), np.array(transformed_normals), color, pose


def random_segment_drop(coordinate, normal, color, pose):
    # Heuristic: center of the sphere is the mean of the points
    center = coordinate.mean(axis=0)
    # randomly shift the center but keep it inside the point cloud
    center += np.random.uniform(low=-0.02, high=0.02, size=(3,))
    # Heuristic: start with a small sphere and increase until k% of points are inside
    total_points = len(coordinate)
    half_points = total_points * 0.5
    radius = round(np.random.uniform(low=0.02, high=0.05), 2)
    distances = np.linalg.norm(coordinate - center, axis=1)
    # Count points inside the sphere
    inside_count = np.sum(distances < radius)
    if inside_count >= half_points:
        mask = distances >= radius  # keep mask
        if mask.sum() == 0:
            return coordinate, normal, color, pose
        if mask.sum() < half_points:
            mask = distances < radius
        # Create a new point cloud without the points inside the sphere
        new_points = coordinate[mask]
        new_normals = normal[mask]
        # Duplicate points to make up for the dropped points
        num_points_to_add = total_points - len(new_points)
        indices_to_duplicate = np.random.choice(len(new_points), num_points_to_add)
        duplicated_points = new_points[indices_to_duplicate]
        duplicated_normals = new_normals[indices_to_duplicate]
        coordinate = np.concatenate((new_points, duplicated_points))
        normal = np.concatenate((new_normals, duplicated_normals))
    return coordinate, normal, color, pose


if __name__ == "__main__":
    import os
    from detectron2.config import LazyConfig

    dataset_name = "dmorp_rdiff"
    split = "test"
    task_name = "Dmorp"
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", "pose_transformer_rdiff.py")
    cfg = LazyConfig.load(cfg_file)

    # Test data loader
    pcd_size = cfg.MODEL.PCD_SIZE
    is_elastic_distortion = cfg.DATALOADER.AUGMENTATION.IS_ELASTIC_DISTORTION
    is_random_distortion = cfg.DATALOADER.AUGMENTATION.IS_RANDOM_DISTORTION
    random_distortion_rate = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_RATE
    random_distortion_mag = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_MAG
    volume_augmentation_file = cfg.DATALOADER.AUGMENTATION.VOLUME_AUGMENTATION_FILE
    crop_pcd = cfg.DATALOADER.AUGMENTATION.CROP_PCD
    crop_size = cfg.DATALOADER.AUGMENTATION.CROP_SIZE
    crop_strategy = cfg.DATALOADER.AUGMENTATION.CROP_STRATEGY
    crop_noise = cfg.DATALOADER.AUGMENTATION.CROP_NOISE
    noise_level = cfg.DATALOADER.AUGMENTATION.NOISE_LEVEL
    rot_axis = cfg.DATALOADER.AUGMENTATION.ROT_AXIS
    knn_k = cfg.DATALOADER.AUGMENTATION.KNN_K

    # Load dataset & data loader
    if cfg.ENV.GOAL_TYPE == "multimodal":
        dataset_folder = "dmorp_multimodal"
    elif "real" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "dmorp_real"
    elif "struct" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "dmorp_struct"
    elif "rdiff" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "dmorp_rdiff"
    else:
        dataset_folder = "dmorp_faster"

    # Get different split
    splits = ["train", "val", "test"]
    data_file_dict = {}
    for split in splits:
        data_file_dict[split] = os.path.join(
            root_path,
            "test_data",
            dataset_folder,
            f"diffusion_dataset_0_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_{split}.pkl",
        )
    print("Data loaded from: ", data_file_dict)

    volume_augmentations_path = (
        os.path.join(root_path, "config", volume_augmentation_file) if volume_augmentation_file is not None else None
    )
    dataset = PcdPairDataset(
        data_file_list=[data_file_dict["test"]],
        dataset_name="dmorp",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=True,
        is_random_distortion=is_random_distortion,
        random_distortion_rate=random_distortion_rate,
        random_distortion_mag=random_distortion_mag,
        volume_augmentations_path=volume_augmentations_path,
        crop_pcd=crop_pcd,
        crop_size=crop_size,
        crop_noise=crop_noise,
        crop_strategy=crop_strategy,
        noise_level=noise_level,
        rot_axis=rot_axis,
        knn_k=knn_k,
    )

    # Test data augmentation
    for i in range(10):
        # random_idx = np.random.randint(0, len(dataset))
        random_idx = i
        data = dataset[random_idx]
        target_coord = data["target_coord"]
        target_feat = data["target_feat"]
        fixed_coord = data["fixed_coord"]
        fixed_feat = data["fixed_feat"]
        target_pose = data["target_pose"]
        print("Target pose: ", target_pose)
        print(f"Number of target points: {len(target_coord)}, Number of fixed points: {len(fixed_coord)}")

        target_color = np.zeros_like(target_coord)
        target_color[:, 0] = 1
        fixed_color = np.zeros_like(fixed_coord)
        fixed_color[:, 1] = 1

        fixed_normal = fixed_feat[:, 3:6]
        target_normal = target_feat[:, 3:6]
        target_pose_mat = utils.pose9d_to_mat(target_pose, rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)

        # utils.visualize_pcd_list(
        #     coordinate_list=[target_coord, fixed_coord],
        #     normal_list=[target_normal, fixed_normal],
        #     color_list=[target_color, fixed_color],
        #     pose_list=[np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)],
        # )

        # utils.visualize_pcd_list(
        #     coordinate_list=[target_coord, fixed_coord],
        #     normal_list=[target_normal, fixed_normal],
        #     color_list=[target_color, fixed_color],
        #     pose_list=[target_pose_mat, np.eye(4, dtype=np.float32)],
        # )
