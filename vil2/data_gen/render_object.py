import blenderproc as bproc
import numpy as np
import argparse
import bpy
import os
from blenderproc.python.writer.BopWriterUtility import _BopWriterUtility
import json
# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

# -------------------------- Main -------------------------- #
argparser = argparse.ArgumentParser()
argparser.add_argument('--scene_path', type=str, default='../test_data/dmorp_augmented')
argparser.add_argument('--data_id', type=int, default=0)
argparser.add_argument('--output_dir', type=str, default='../test_data/dmorp')
argparser.add_argument('--num_cam_poses', type=int, default=3)
argparser.add_argument('--light_radius_min', type=float, default=1.0)
argparser.add_argument('--light_radius_max', type=float, default=2.0)
argparser.add_argument('--cam_radius_min', type=float, default=1.0)
argparser.add_argument('--cam_radius_max', type=float, default=1.5)
argparser.add_argument('--fix_cam', action='store_true')
args = argparser.parse_args()

# Set parameters
output_dir = args.output_dir
scene_path = args.scene_path
data_id = args.data_id
scene_path = os.path.join(scene_path, f"{data_id:06d}")
export_dir = os.path.join(output_dir, f"{data_id:06d}")

# Read scene info
with open(os.path.join(scene_path, "scene_info.json"), "r") as f:
    scene_info = json.load(f)
init_pose_1_list = scene_info["init_pose_1_list"]
init_pose_2_list = scene_info["init_pose_2_list"]
transform_list = scene_info["transform_list"]

mesh_files = [scene_info["mesh_1"], scene_info["mesh_2"]]
fix_cam = args.fix_cam
num_cam_poses = args.num_cam_poses if not fix_cam else 1
light_radius_min = args.light_radius_min
light_radius_max = args.light_radius_max
cam_radius_min = args.cam_radius_min
cam_radius_max = args.cam_radius_max

# -------------------------- Init & Load -------------------------- #
bproc.init()
cam_poses = []
obj_poses = {}
# Create a simple object:
objs = []
for id, data_file in enumerate(mesh_files):
    obj = bproc.loader.load_obj(data_file, forward_axis='Y', up_axis='Z')[0]
    obj.set_cp("category_id", id+1)
    # obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
    obj.set_shading_mode('auto')
    objs.append(obj)

# enable depth output
bproc.renderer.enable_depth_output(activate_antialiasing=False)
# enable segmentation masks (per class and per instance)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])

# Background light
light_background = bproc.types.Light(light_type="SUN")
# Point light
light_point = bproc.types.Light(light_type="POINT")

for sample_idx, (init_pose_1, init_pose_2, transform) in enumerate(zip(init_pose_1_list, init_pose_2_list, transform_list)):
    print(f"Generating sample {sample_idx}....")
    # -------------------------- Object Pose -------------------------- #
    init_pose_1 = np.array(init_pose_1)
    init_pose_2 = np.array(init_pose_2)
    # transform = np.array(transform)  # How to move object 1 (target)
    # Set pose for the two objects
    objs[0].set_location(init_pose_1[:3, 3])
    objs[0].set_rotation_mat(init_pose_1[:3, :3])
    objs[1].set_location(init_pose_2[:3, 3])
    objs[1].set_rotation_mat(init_pose_2[:3, :3])
    # -------------------------- Light -------------------------- #
    light_background.set_energy(5)
    light_background.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    light_background.set_location([np.random.random(), np.random.random(), 10])
    light_point.set_energy(200)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    location = bproc.sampler.shell(center=[0, 0, 0], radius_min=light_radius_min, radius_max=light_radius_max,
                                   elevation_min=5, elevation_max=89, uniform_volume=False)
    light_point.set_location(location)

    # -------------------------- Camera -------------------------- #
    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

    count_cam_poses = 0
    # Render two camera poses
    while count_cam_poses < num_cam_poses:
        if not fix_cam:
            # Sample location
            location = bproc.sampler.shell(center=[0, 0, 0],
                                        radius_min=cam_radius_min,
                                        radius_max=cam_radius_max,
                                        elevation_min=1,
                                        elevation_max=89,
                                        uniform_volume=False)
            # Determine point of interest in scene as the object closest to the mean of a subset of objects
            poi = bproc.object.compute_poi(objs)
        else:
            location = np.array([1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]) * (cam_radius_min + cam_radius_max) / 2.0
            poi = np.array([0, 0, 0])
        # Compute rotation based on vector going from location towards poi
        # rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
        #                                                          inplane_rot=np.random.uniform(-0.7854, 0.7854))
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                                 inplane_rot=0.0)
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix,
                                         frame=count_cam_poses)
            count_cam_poses += 1
            cam_poses.append(cam2world_matrix.tolist())

    # -------------------------- Render & Save -------------------------- #
    sample_export_dir = os.path.join(export_dir, f"{sample_idx:06d}")
    for obj_idx in range(len(objs)):
        for _idx, obj in enumerate(objs):
            if _idx != obj_idx:
                obj.blender_obj.hide_render = True
        # Render the segmentation under Cycles engine
        bpy.context.scene.render.engine = 'CYCLES'
        data = bproc.renderer.render(load_keys={'segmap', 'depth'})
        # Render color and depth under Eevee engine
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        data.update(bproc.renderer.render(load_keys={'colors'}))
        bpy.context.scene.render.engine = 'CYCLES'

        # Write the rendering into an hdf5 file
        bproc.writer.write_hdf5(sample_export_dir, data, append_to_existing_output=True)

        # Recover the visibility
        for _idx, obj in enumerate(objs):
            if _idx != obj_idx:
                obj.blender_obj.hide_render = False

    # -------------------------- Export information -------------------------- #
    # Write camera info
    _BopWriterUtility.write_camera(os.path.join(sample_export_dir, 'camera.json'))
    with open(os.path.join(sample_export_dir, 'poses.json'), 'w') as f:
        json.dump({"cam2world": cam_poses, "transform": transform, "num_cam_poses": num_cam_poses}, f)
