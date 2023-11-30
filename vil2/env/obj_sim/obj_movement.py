from __future__ import annotations
from vil2.env.obj_sim.obj import ObjData
import gym
import numpy as np
from vil2.utils import load_utils
from vil2.utils import misc_utils
import open3d as o3d
from copy import deepcopy
from vil2.algo.super_patch import generate_random_patch


class ObjSim(gym.Env):
    """Simulate object movement in the environment."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        # load objs
        self.objs: list[ObjData] = []
        self._obj_names = dict()
        self._obj_init_poses = dict()
        obj_source = cfg.ENV.obj_source
        if obj_source == "google_scanned_objects":
            self._load_objs_google_scanned_objects()
        else:
            raise ValueError(f"Unknown obj_source: {obj_source}")
        # generate super patch
        for obj in self.objs:
            obj.super_patch, obj.patch_center = self.generate_obj_super_patch(
                obj.semantic_id, cfg.ENV.super_patch_size
            )

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
                semantic_id=semantic_id,
                semantic_feature=None,
                pcd=geometry.vertices,
                geometry=geometry,
            )

            self.objs.append(obj_data)

    def reset(self):
        # set obj back to init pose
        for obj in self.objs:
            obj.pose = self._obj_init_poses[obj.semantic_str]

    def step(self, action):
        if isinstance(action, dict):
            for obj_name, action in action.items():
                self._step_single_obj(obj_name, action)
        # Compute observation
        obs = {}
        # compute pairwise distance
        for i, obj1 in enumerate(self.objs):
            for j, obj2 in enumerate(self.objs):
                if i == j:
                    continue
                obs[f"dist_{i}_{j}"] = self.compute_pairwise_patch_distance(
                    obj1.semantic_id, obj2.semantic_id
                )
        return obs, None, None, None

    def _step_single_obj(self, obj_id, action):
        delta_pos = action[:3]
        delta_quat = action[3:]
        pose = self.objs[obj_id].pose.copy()
        pose[:3, 3] += delta_pos
        pose[:3, :3] = misc_utils.quat_to_mat(delta_quat) @ pose[:3, :3]
        self.objs[obj_id].pose = pose

    def generate_obj_super_patch(self, obj_id, patch_size):
        """Generate super-patch for a given object."""
        obj = self.objs[obj_id]
        super_patch, patch_centers = generate_random_patch(obj.geometry, num_points=patch_size)
        return super_patch, patch_centers

    def compute_pairwise_patch_distance(self, obj_id1, obj_id2):
        """Compute pairwise distance between all patches of two objects."""
        distances = np.zeros((len(self.objs[obj_id1].super_patch), len(self.objs[obj_id2].super_patch)))
        patch_centers1 = self.objs[obj_id1].patch_center
        patch_centers2 = self.objs[obj_id2].patch_center
        for i, patch_center1 in enumerate(patch_centers1):
            for j, patch_center2 in enumerate(patch_centers2):
                distances[i, j] = np.linalg.norm(patch_center1 - patch_center2)
        return distances

    def render(self, show_super_patch=False):
        """Render object in open3d."""
        # render in open3d
        vis_list = []
        for obj in self.objs:
            if not show_super_patch:
                vis_obj = deepcopy(obj.geometry)
                vis_obj.transform(obj.pose)
                vis_list.append(vis_obj)
            else:
                for i, patch in enumerate(obj.super_patch):
                    vis_patch = deepcopy(patch)
                    vis_patch.transform(obj.pose)
                    # random color
                    color = np.random.rand(3)
                    vis_patch.paint_uniform_color(color)
                    vis_list.append(vis_patch)
                    # patch center
                    patch_center = obj.patch_center[i]
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    sphere.translate(patch_center)
                    sphere.transform(obj.pose)
                    sphere.paint_uniform_color(color)
                    vis_list.append(sphere)

        # add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis_list.append(coord_frame)
        o3d.visualization.draw_geometries(vis_list)
