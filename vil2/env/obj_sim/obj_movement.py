from __future__ import annotations
from vil2.env.obj_sim.obj import ObjData
import gym
import numpy as np
from vil2.utils import load_utils
from vil2.utils import misc_utils
import open3d as o3d
from copy import deepcopy


class ObjSim(gym.Env):
    """Simulate object movement in the environment."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        # load objs
        self.objs = []
        self._obj_names = dict()
        self._obj_init_poses = dict()
        obj_source = cfg.ENV.obj_source
        if obj_source == "google_scanned_objects":
            self._load_objs_google_scanned_objects()
        else:
            raise ValueError(f"Unknown obj_source: {obj_source}")

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
        return None, None, None, None

    def _step_single_obj(self, obj_id, action):
        delta_pos = action[:3]
        delta_quat = action[3:]
        pose = self.objs[obj_id].pose.copy()
        pose[:3, 3] += delta_pos
        pose[:3, :3] = misc_utils.quat_to_mat(delta_quat) @ pose[:3, :3]
        self.objs[obj_id].pose = pose

    def render(self):
        """Render object in open3d."""
        # render in open3d
        vis_list = []
        for obj in self.objs:
            vis_obj = deepcopy(obj.geometry)
            vis_obj.transform(obj.pose)
            vis_list.append(vis_obj)
        # add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis_list.append(coord_frame)
        o3d.visualization.draw_geometries(vis_list)
