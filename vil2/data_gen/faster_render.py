"""Faster render using pyrender"""
import argparse
import os
import json
import pyrender
import trimesh
import numpy as np
import cv2
from tqdm import tqdm


def sample_camera_pose(cam_radius_min: float, cam_radius_max: float, look_at: np.ndarray):
    # Sample a radius within the given range
    radius = np.random.uniform(cam_radius_min, cam_radius_max)

    # Sample spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)

    # Convert spherical coordinates to Cartesian coordinates for the camera position
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Calculate camera position
    cam_position = look_at + np.array([x, y, z])

    # Calculate camera orientation vectors
    z_axis = -(look_at - cam_position)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(np.array([0, 1, 0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Construct the camera-to-world transformation matrix
    camera_to_world_matrix = np.eye(4)
    camera_to_world_matrix[0:3, 0] = x_axis
    camera_to_world_matrix[0:3, 1] = y_axis
    camera_to_world_matrix[0:3, 2] = z_axis
    camera_to_world_matrix[0:3, 3] = cam_position

    return camera_to_world_matrix


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--scene_path", type=str, default="test_data/dmorp_augmented")
    argparser.add_argument("--data_id", type=int, default=0)
    argparser.add_argument("--output_dir", type=str, default="test_data/dmorp_faster")
    argparser.add_argument("--num_cam_poses", type=int, default=2)
    argparser.add_argument("--light_radius_min", type=float, default=1.0)
    argparser.add_argument("--light_radius_max", type=float, default=2.0)
    argparser.add_argument("--cam_radius_min", type=float, default=1.0)
    argparser.add_argument("--cam_radius_max", type=float, default=1.3)
    argparser.add_argument("--fix_cam", action="store_true")
    argparser.add_argument("--chunk_size", type=int, default=1000)
    argparser.add_argument("--start_idx", type=int, default=0)
    argparser.add_argument("--end_idx", type=int, default=-1)
    args = argparser.parse_args()

    # Set parameters
    output_dir = args.output_dir
    scene_path = args.scene_path
    data_id = args.data_id
    start_idx = args.start_idx
    end_idx = args.end_idx

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, output_dir)
    scene_path = os.path.join(root_dir, scene_path)
    scene_path = os.path.join(scene_path, f"{data_id:06d}")
    export_dir = os.path.join(output_dir, f"{data_id:06d}")
    os.makedirs(export_dir, exist_ok=True)
    num_cam_poses = args.num_cam_poses if not args.fix_cam else 1
    cam_radius_min = args.cam_radius_min
    cam_radius_max = args.cam_radius_max
    chunk_size = args.chunk_size

    # Read scene info
    with open(os.path.join(scene_path, "scene_info.json"), "r") as f:
        scene_info = json.load(f)
    init_pose_1_list = scene_info["init_pose_1_list"][start_idx:end_idx]
    init_pose_2_list = scene_info["init_pose_2_list"][start_idx:end_idx]
    transform_list = scene_info["transform_list"][start_idx:end_idx]

    # Create objects
    mesh_files = [scene_info["mesh_1"], scene_info["mesh_2"]]
    objs = []

    # Load meshs
    for mesh_file in mesh_files:
        mesh = trimesh.load(mesh_file)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        objs.append(mesh)

    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0)

    scene = pyrender.Scene()
    light_node = scene.add(light)
    obj0_node = scene.add(objs[0])
    obj1_node = scene.add(objs[1])

    # Set-up intrinsic camera
    f_x = 500
    f_y = 500
    c_x = 320
    c_y = 240
    viewport_width = 640
    viewport_height = 480
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    ic = pyrender.IntrinsicsCamera(fx=f_x, fy=f_y, cx=c_x, cy=c_y)
    camera_node = scene.add(ic, pose=np.eye(4))

    r = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height, point_size=1.0)
    # Render
    idx_chunk = 0
    chunk_data = []
    for scene_idx, (init_pose_1, init_pose_2, transform) in tqdm(
        enumerate(zip(init_pose_1_list, init_pose_2_list, transform_list)), desc="Generating data", leave=False
    ):
        # init_pose_1_t = np.array(transform) @ np.array(init_pose_1)
        # Update object poses
        scene.set_pose(obj0_node, init_pose_1)
        scene.set_pose(obj1_node, init_pose_2)
        # Sample camera poses
        for cam_idx in range(num_cam_poses):
            camera_pose = sample_camera_pose(
                cam_radius_min,
                cam_radius_max,
                np.zeros(
                    3,
                ),
            )
            # Update camera pose
            scene.set_pose(camera_node, camera_pose)
            # Update light pose
            scene.set_pose(light_node, camera_pose)

            # Render
            flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            # Set obj0 to be invisible
            obj1_node.mesh.is_visible = False
            color, depth = r.render(scene, flags=flags)
            scene_data_0 = {
                "color": color,
                "depth": depth,
                "camera_pose": camera_pose,
                "intrinsic": K,
                "scene_idx": scene_idx,
                "transform": np.array(transform),  # target object 0
            }
            # cv2.imshow("color", color)
            # cv2.waitKey(0)
            # Set obj0 to be visible and obj1 to be invisible
            obj0_node.mesh.is_visible = False
            obj1_node.mesh.is_visible = True
            color, depth = r.render(scene, flags=flags)
            scene_data_1 = {
                "color": color,
                "depth": depth,
                "camera_pose": camera_pose,
                "intrinsic": K,
                "scene_idx": scene_idx,
                "transform": np.eye(4, dtype=np.float32),  # fix object 1
            }
            # Set obj0 to be visible
            obj0_node.mesh.is_visible = True
            # Update chunk data
            chunk_data.append(scene_data_0)
            chunk_data.append(scene_data_1)
            if len(chunk_data) == chunk_size or scene_idx == len(init_pose_1_list) - 1:
                # Save chunk data
                np.savez_compressed(
                    os.path.join(export_dir, f"{start_idx + idx_chunk * chunk_size:06d}.npz"),
                    data=chunk_data,
                )
                # Reset chunk data
                chunk_data = []
                idx_chunk += 1
