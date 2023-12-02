import open3d as o3d
import os
import numpy as np
import importlib_resources


def load_google_scan_obj(obj_name: str, obj_mesh_dir: str) -> o3d.geometry.TriangleMesh:
    """Load google scanned object mesh.

    Args:
        obj_name (str): object name
        obj_mesh_dir (str): object mesh directory

    Returns:
        o3d.geometry.TriangleMesh: object mesh
    """
    resource_dir = importlib_resources.files("vil2").joinpath("assets")
    obj_path = os.path.join(resource_dir, obj_mesh_dir, obj_name, "meshes", "model.obj")
    # load with texture
    obj_mesh = o3d.io.read_triangle_mesh(obj_path, True)
    return obj_mesh
