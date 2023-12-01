from __future__ import annotations
from vil2.env.obj_sim.obj import ObjData
import gym
import numpy as np
from vil2.utils import load_utils
from vil2.utils import misc_utils
import open3d as o3d
from copy import deepcopy
from vil2.algo.super_voxel import generate_random_voxel


class ObjSim(gym.Env):
    """Simulate object movement in the environment."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_iter = cfg.ENV.max_iter
        # load objs
        self.objs: list[ObjData] = []
        self._obj_names = dict()
        self._obj_init_poses = dict()
        obj_source = cfg.ENV.obj_source
        if obj_source == "google_scanned_objects":
            self._load_objs_google_scanned_objects()
        else:
            raise ValueError(f"Unknown obj_source: {obj_source}")
        self._super_voxel_traj = dict()
        self._t = 0
        self._active_obj_id = []
        # generate super voxel
        for obj in self.objs:
            obj.super_voxel, obj.voxel_center = self.generate_obj_super_voxel(
                obj.id, cfg.ENV.super_patch_size
            )
        self.super_patch_size = cfg.ENV.super_patch_size

    def _load_objs_google_scanned_objects(self):
        obj_names = self.cfg.ENV.obj_names
        obj_mesh_dir = self.cfg.ENV.obj_mesh_dir
        for i, obj_name in enumerate(obj_names):
            # semantic id
            if obj_name in self._obj_names:
                semantic_id = self._obj_names[obj_name]
            else:
                semantic_id = len(self._obj_names)
                self._obj_names[obj_name] = semantic_id
            # pose
            pose = misc_utils.quat_to_mat(
                np.array(self.cfg.ENV.obj_init_poses[i])
            )
            # load mesh
            geometry = load_utils.load_google_scan_obj(
                obj_name, obj_mesh_dir
            )
            self._obj_init_poses[obj_name] = pose
            obj_data = ObjData(
                pose=pose,
                semantic_str=obj_name,
                id=semantic_id,
                semantic_feature=None,
                pcd=geometry.vertices,
                geometry=geometry,
            )

            self.objs.append(obj_data)

    def reset(self):
        # set obj back to init pose
        for obj in self.objs:
            obj.pose = self._obj_init_poses[obj.semantic_str]
        # set obj traj to empty
        self._super_voxel_traj = dict()
        self._t = 0
        self._active_obj_id = []

    def step(self, action):
        """Step the environment."""
        # step each object
        self._active_obj_id = []
        if isinstance(action, dict):
            for obj_id, action in action.items():
                self._step_single_obj(obj_id, action)
                self._active_obj_id.append(obj_id)
        # record traj
        for obj in self.objs:
            if obj.id not in self._super_voxel_traj:
                self._super_voxel_traj[obj.id] = []
            rot_vec = misc_utils.mat_to_rotvec(obj.pose)
            obj_pose = np.concatenate([obj.pose[:3, 3], rot_vec])
            super_voxel_pose = np.tile(obj_pose[None, :], (self.super_patch_size, 1))
            self._super_voxel_traj[obj.id].append(
                super_voxel_pose[None, ...]
            )
        # compute observation
        obs = self._compute_obs()
        # update time
        self._t += 1
        # check done
        done = self._t >= self.max_iter
        # compute reward
        reward = self._compute_reward()
        # compute info
        info = self._compute_info()
        return obs, reward, done, info

    def _compute_reward(self):
        """Compute reward for each object."""
        return 0.0

    def _compute_info(self):
        """Compute info for each object."""
        return {}

    def _step_single_obj(self, obj_id, action):
        delta_pos = action[:3]
        delta_quat = action[3:]
        pose = self.objs[obj_id].pose.copy()
        pose[:3, 3] += delta_pos
        pose[:3, :3] = misc_utils.quat_to_mat(delta_quat) @ pose[:3, :3]
        self.objs[obj_id].pose = pose

    def generate_obj_super_voxel(self, obj_id, patch_size):
        """Generate super-voxel for a given object."""
        obj = self.objs[obj_id]
        super_voxel, voxel_centers = generate_random_voxel(obj.geometry, num_points=patch_size)
        return super_voxel, voxel_centers

    def compute_pairwise_patch_distance(self, obj_id1, obj_id2):
        """Compute pairwise distance between all patches of two objects."""
        distances = np.zeros((len(self.objs[obj_id1].super_voxel), len(self.objs[obj_id2].super_voxel)))
        voxel_centers1 = self.objs[obj_id1].voxel_center
        voxel_centers2 = self.objs[obj_id2].voxel_center
        for i, voxel_center1 in enumerate(voxel_centers1):
            for j, voxel_center2 in enumerate(voxel_centers2):
                distances[i, j] = np.linalg.norm(voxel_center1 - voxel_center2)
        return distances

    def render(self, show_super_patch=False, return_image=False):
        """Render object in open3d."""
        # render in open3d
        vis_list = []
        for obj in self.objs:
            if not show_super_patch:
                vis_obj = deepcopy(obj.geometry)
                vis_obj.transform(obj.pose)
                vis_list.append(vis_obj)
            else:
                for i, voxel in enumerate(obj.super_voxel):
                    vis_patch = o3d.geometry.PointCloud()
                    vis_patch.points = o3d.utility.Vector3dVector(voxel)
                    vis_patch.transform(obj.pose)
                    # random color
                    color = np.random.rand(3)
                    vis_patch.paint_uniform_color(color)
                    vis_list.append(vis_patch)
                    # voxel center
                    voxel_center = obj.voxel_center[i]
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    sphere.translate(voxel_center)
                    sphere.transform(obj.pose)
                    sphere.paint_uniform_color(color)
                    vis_list.append(sphere)

        # add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis_list.append(coord_frame)
        if not return_image:
            o3d.visualization.draw_geometries(vis_list)
            return None, None
        else:
            # render image
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=640, height=480, visible=False)
            vis.add_geometry(coord_frame)
            for obj in self.objs:
                vis_obj = deepcopy(obj.geometry)
                vis_obj.transform(obj.pose)
                vis.add_geometry(vis_obj)
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            depth_image = vis.capture_depth_float_buffer(do_render=True)  # Capture depth buffer
            # Convert the image to a numpy array
            image = np.asarray(image)
            # Convert the image from BGR to RGB
            image = image[:, :, [2, 1, 0]]
            image = (image * 255).astype(np.uint8)
            vis.destroy_window()
            return np.asarray(image).copy(), np.asarray(depth_image).copy()

    def _compute_obs(self):
        # compute observation
        obs = {}

        # record traj
        obs["trajectory"] = deepcopy(self._super_voxel_traj)

        # record geometry
        obs["geometry"] = [deepcopy(np.asarray(obj.geometry.vertices)) for obj in self.objs]

        # record super voxel
        obs["super_voxel"] = [deepcopy(obj.super_voxel) for obj in self.objs]

        # record voxel center
        obs["voxel_center"] = [deepcopy(obj.voxel_center) for obj in self.objs]

        # record active obj id
        obs["active_obj_id"] = deepcopy(self._active_obj_id)

        # record image
        obs["image"], obs["depth"] = self.render(return_image=True)
        return obs