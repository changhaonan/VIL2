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
argparser.add_argument('--object_pair', type=str, default='tea_pot,tea_mug')
argparser.add_argument('--output_dir', type=str, default='../test_data/dmorp')
argparser.add_argument('--num_poses', type=int, default=10)
argparser.add_argument('--light_radius_min', type=float, default=1.0)
argparser.add_argument('--light_radius_max', type=float, default=2.0)
argparser.add_argument('--cam_radius_min', type=float, default=1.0)
argparser.add_argument('--cam_radius_max', type=float, default=1.5)
args = argparser.parse_args()

# Set parameters
output_dir = args.output_dir
object_pair = args.object_pair
export_dir = os.path.join(output_dir, object_pair.replace(',', '-'))
object_names = object_pair.split(',')
data_files = [f'../assets/google_scanned_objects/{object_name}/meshes/model.obj' for object_name in object_names]
num_poses = args.num_poses
light_radius_min = args.light_radius_min
light_radius_max = args.light_radius_max
cam_radius_min = args.cam_radius_min
cam_radius_max = args.cam_radius_max

# -------------------------- Init & Load -------------------------- #
bproc.init()
cam_poses = []
# Create a simple object:
objs = []
for id, data_file in enumerate(data_files):
    obj = bproc.loader.load_obj(data_file, forward_axis='Y', up_axis='Z')[0]
    obj.set_cp("category_id", id+1)
    obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
    obj.set_shading_mode('auto')
    objs.append(obj)

# -------------------------- Create Room -------------------------- #
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)
    plane.set_cp("category_id", 0)

# -------------------------- Object Pose Random -------------------------- #
# Define a function that samples 6-DoF poses


def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


bproc.object.sample_poses(objects_to_sample=objs,
                          sample_pose_func=sample_pose_func,
                          max_tries=1000)
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                  max_simulation_time=10,
                                                  check_object_interval=1,
                                                  substeps_per_frame=20,
                                                  solver_iters=25)

# -------------------------- Light -------------------------- #
# Background light
light_background = bproc.types.Light(light_type="SUN")
light_background.set_energy(5)
light_background.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
light_background.set_location([np.random.random(), np.random.random(), 10])

# Point light
light_point = bproc.types.Light(light_type="POINT")
light_point.set_energy(200)
light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
location = bproc.sampler.shell(center=[0, 0, 0], radius_min=light_radius_min, radius_max=light_radius_max,
                               elevation_min=5, elevation_max=89, uniform_volume=False)
light_point.set_location(location)

# -------------------------- Camera -------------------------- #
# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

poses = 0
# Render two camera poses
while poses < num_poses:
    # Sample location
    location = bproc.sampler.shell(center=[0, 0, 0],
                                   radius_min=cam_radius_min,
                                   radius_max=cam_radius_max,
                                   elevation_min=1,
                                   elevation_max=89,
                                   uniform_volume=False)
    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    poi = bproc.object.compute_poi(objs)
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
                                     frame=poses)
        poses += 1
        cam_poses.append(cam2world_matrix.tolist())


# -------------------------- Render & Save -------------------------- #
bproc.renderer.enable_depth_output(activate_antialiasing=False)
# enable segmentation masks (per class and per instance)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])

# Render the segmentation under Cycles engine
bpy.context.scene.render.engine = 'CYCLES'
data = bproc.renderer.render(load_keys={'segmap'})
# Render color and depth under Eevee engine
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
data.update(bproc.renderer.render(load_keys={'colors', 'depth'}))

# # Write the rendering into an hdf5 file
bproc.writer.write_hdf5(export_dir, data)

# Write in coco format
bproc.writer.write_coco_annotations(os.path.join(export_dir, 'coco_data'),
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG")
# Write camera info
_BopWriterUtility.write_camera(os.path.join(export_dir, 'coco_data', 'camera.json'))
with open(os.path.join(export_dir, 'coco_data', 'cam_poses.json'), 'w') as f:
    json.dump({"cam_poses": cam_poses}, f)
