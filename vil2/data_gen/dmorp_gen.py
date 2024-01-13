import os
import numpy as np
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import argparse


class DmorpSceneAugmentor:
    """Scene Augmentor for DMORP dataset"""

    def __init__(
        self,
        mesh_1,
        mesh_2,
        semantic_1,
        semantic_2,
        pose_1_list,
        pose_2_list,
        mesh_type: str = "obj",
    ) -> None:
        """Dmorp scene is constructed by collecting a series of mesh pairs and their poses.
        Each pose is bounded to their mesh, and the pose is defined as the transformation from the mesh's local frame
        """
        self.mesh_type = mesh_type
        self.mesh_1 = mesh_1
        self.mesh_2 = mesh_2
        self.pose_1_list = pose_1_list
        self.pose_2_list = pose_2_list
        self.semantic_1 = semantic_1
        self.semantic_2 = semantic_2

    def augment(
        self, data_idx: int, num_augment: int, export_dir, random_region: float = 0.5, fix_anchor: bool = False
    ):
        """Augment the scene with the given data_idx
        2 is anchor, 1 is target
        """
        scene_info = {}
        scene_info["random_region"] = random_region
        scene_info["mesh_1"] = self.mesh_1
        scene_info["mesh_2"] = self.mesh_2
        scene_info["semantic_1"] = self.semantic_1
        scene_info["semantic_2"] = self.semantic_2
        # Sample a series of random poses within the random region
        init_pose_1_list = []
        init_pose_2_list = []
        target_pose_1_list = []
        transform_list = []
        num_poses = len(self.pose_1_list)
        for i in range(num_augment):
            # Sample a goal pose pair
            pose_1 = np.array(self.pose_1_list[np.random.randint(num_poses)])
            pose_2 = np.array(self.pose_2_list[np.random.randint(num_poses)])
            # Sample pose for anchor and target
            random_init_pose_1 = np.eye(4)
            random_init_pose_1[:3, 3] = np.random.uniform(-random_region / 2.0, random_region / 2.0, size=3)
            random_init_pose_1[:3, :3] = R.from_euler("xyz", np.random.uniform(-np.pi, np.pi, size=3)).as_matrix()
            random_init_pose_2 = np.eye(4)  # Anchor object will always be at center
            if not fix_anchor:
                random_init_pose_2[:3, :3] = R.from_euler("xyz", np.random.uniform(-np.pi, np.pi, size=3)).as_matrix()
            # Compute the required pose to reach the target pose
            target_goal_pose = random_init_pose_2 @ (np.linalg.inv(pose_2) @ pose_1)
            transform = target_goal_pose @ np.linalg.inv(random_init_pose_1)
            init_pose_1_list.append(random_init_pose_1.tolist())
            init_pose_2_list.append(random_init_pose_2.tolist())
            target_pose_1_list.append(target_goal_pose.tolist())
            transform_list.append(transform.tolist())
            # TODO: remove overlap poses
        scene_info["init_pose_1_list"] = init_pose_1_list
        scene_info["init_pose_2_list"] = init_pose_2_list
        scene_info["transform_list"] = transform_list
        # Export the scene info
        export_path = os.path.join(export_dir, f"{data_idx:06d}")
        os.makedirs(export_path, exist_ok=True)
        json_file = os.path.join(export_path, "scene_info.json")
        with open(json_file, "w") as f:
            json.dump(scene_info, f, indent=4)

    def visualize(self, data_idx: int, export_dir, sample_idx: int = 0):
        """Visualize the scene with the given data_idx"""
        # Read mesh
        scene_info_file = os.path.join(export_dir, f"{data_idx:06d}", "scene_info.json")
        with open(scene_info_file, "r") as f:
            scene_info = json.load(f)
        if self.mesh_type == "obj":
            mesh_1 = o3d.io.read_triangle_mesh(scene_info["mesh_1"])
            mesh_2 = o3d.io.read_triangle_mesh(scene_info["mesh_2"])
            # Compute normals
            mesh_1.compute_vertex_normals()
            mesh_2.compute_vertex_normals()
        else:
            raise NotImplementedError
        random_region = scene_info["random_region"]
        # Read pose
        init_pose_1 = scene_info["init_pose_1_list"][sample_idx]
        init_pose_2 = scene_info["init_pose_2_list"][sample_idx]
        transform = scene_info["transform_list"][sample_idx]
        # Visualize
        mesh_1_goal = deepcopy(mesh_1)
        mesh_1.transform(init_pose_1)
        mesh_2.transform(init_pose_2)
        mesh_1_goal.transform(np.array(transform) @ np.array(init_pose_1))
        mesh_1_goal.paint_uniform_color([0.5, 0.1, 0.1])
        # Draw a transparent unit sphere
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=random_region)
        mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
        # Convert to wireframe
        mesh_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_sphere)

        # Origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_1)
        vis.add_geometry(mesh_2)
        vis.add_geometry(mesh_1_goal)
        vis.add_geometry(mesh_sphere)
        vis.add_geometry(origin)
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    obj_list = ["tea_pot", "tea_mug", "spoon"]
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--data_id", type=int, default=0)
    argparser.add_argument("--num_samples", type=int, default=10000)
    argparser.add_argument("--target_object", type=int, default=0)
    argparser.add_argument("--anchor_object", type=int, default=1)
    argparser.add_argument("--random_region", type=float, default=0.5)
    argparser.add_argument("--fix_anchor", action="store_true")
    argparser.add_argument("--visualize", action="store_true")
    args = argparser.parse_args()
    args.fix_anchor = True
    data_id = args.data_id
    num_samples = args.num_samples
    random_region = args.random_region
    target_object = args.target_object
    anchor_object = args.anchor_object
    fix_anchor = args.fix_anchor
    visualize = args.visualize

    # Prepare data
    root_path = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    mesh_1_file = os.path.join(
        root_path, "assets", "google_scanned_objects", obj_list[target_object], "meshes", "model.obj"
    )
    mesh_2_file = os.path.join(
        root_path, "assets", "google_scanned_objects", obj_list[anchor_object], "meshes", "model.obj"
    )

    # Goal poses pair
    pose_1_list = []
    pose_2_list = []
    if target_object == 0 and anchor_object == 1:
        # Pose pair 1
        pose_1 = np.eye(4)
        pose_1[:3, 3] = np.array([0.15, 0.15, 0.0])
        pose_1[:3, :3] = R.from_euler("xyz", [0.0, 0.0, np.pi / 3.0]).as_matrix()
        pose_2 = np.eye(4)
        pose_2[:3, 3] = np.array([0.0, 0.0, 0.0])
        pose_2[:3, :3] = R.from_euler("xyz", [0.0, 0.0, np.pi / 2.0]).as_matrix()
        pose_1_list.append(pose_1)
        pose_2_list.append(pose_2)
        # Pose pair 2
        pose_1 = np.eye(4)
        pose_1[:3, 3] = np.array([0.15, -0.15, 0.0])
        pose_1[:3, :3] = R.from_euler("xyz", [0.0, 0.0, -np.pi / 3.0]).as_matrix()
        pose_2 = np.eye(4)
        pose_2[:3, 3] = np.array([0.0, 0.0, 0.0])
        pose_2[:3, :3] = R.from_euler("xyz", [0.0, 0.0, np.pi / 2.0]).as_matrix()
        pose_1_list.append(pose_1)
        pose_2_list.append(pose_2)
    elif target_object == 1 and anchor_object == 2:
        # Pose pair 1
        pose_1 = np.eye(4)
        pose_1[:3, 3] = np.array([0.2, 0.0, 0.0])
        pose_1[:3, :3] = R.from_euler("xyz", [0.0, 0.0, np.pi / 2.0]).as_matrix()
        pose_2 = np.eye(4)
        pose_2[:3, 3] = np.array([0.0, 0.0, 0.0])
        pose_2[:3, :3] = R.from_euler("xyz", [0.0, 0.0, np.pi / 2.0]).as_matrix()
        pose_1_list.append(pose_1)
        pose_2_list.append(pose_2)
        # Pose pair 2
        pose_1 = np.eye(4)
        pose_1[:3, 3] = np.array([-0.2, 0.0, 0.0])
        pose_1[:3, :3] = R.from_euler("xyz", [0.0, 0.0, -np.pi / 2.0]).as_matrix()
        pose_2 = np.eye(4)
        pose_2[:3, 3] = np.array([0.0, 0.0, 0.0])
        pose_2[:3, :3] = R.from_euler("xyz", [0.0, 0.0, np.pi / 2.0]).as_matrix()
        pose_1_list.append(pose_1)
        pose_2_list.append(pose_2)
    # Set seed
    np.random.seed(args.seed)

    augmentor = DmorpSceneAugmentor(mesh_1_file, mesh_2_file, target_object, anchor_object, pose_1_list, pose_2_list)
    export_dir = os.path.join(root_path, "test_data", "dmorp_augmented")
    augmentor.augment(data_id, num_samples, export_dir, random_region=random_region, fix_anchor=fix_anchor)

    # Check the result
    if visualize:
        for i in range(10):
            augmentor.visualize(data_id, export_dir, sample_idx=i)
