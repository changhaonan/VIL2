"""Check superpoint data"""

import os
import pickle
import numpy as np
import open3d as o3d


def visualize_superpoint(superpoint_data):
    pos = superpoint_data["pos"]
    normal = superpoint_data["normal"]
    super_indexes = superpoint_data["super_index"]
    num_color = np.max(super_indexes[0]) + 1
    # Generate random color
    color = np.random.rand(num_color, 3)
    for i, super_index in enumerate(super_indexes):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos)
        pcd.normals = o3d.utility.Vector3dVector(normal)
        pcd.colors = o3d.utility.Vector3dVector(color[super_index])
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    data_path_dict = {
        "stack_can_in_cabinet": "/home/harvey/Project/VIL2/vil2/external/rpdiff/data/task_demos/can_in_cabinet_stack/task_name_stack_can_in_cabinet",
        "book_in_bookshelf": "/home/harvey/Data/rpdiff_V3/book_in_bookshelf",
        "mug_on_rack_multi": "/home/harvey/Project/VIL2/vil2/external/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos/task_name_mug_on_rack_multi",
    }
    task_name = "book_in_bookshelf"
    data_dir = data_path_dict[task_name]
    data_file_list = os.listdir(data_dir)
    data_file_list = [f for f in data_file_list if f.endswith(".npz")]

    # Check superpoint data
    export_dir = os.path.join(data_dir, "superpoint_data")
    os.makedirs(export_dir, exist_ok=True)
    superpoint_file = os.path.join(export_dir, "superpoint_dict.pkl")
    superpoint_dict = pickle.load(open(superpoint_file, "rb"))

    for key, superpoint_data in superpoint_dict.items():
        c_superpoint_data = superpoint_data["child"]
        p_superpoint_data = superpoint_data["parent"]

        visualize_superpoint(c_superpoint_data)
        visualize_superpoint(p_superpoint_data)
