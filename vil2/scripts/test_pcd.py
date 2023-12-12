import h5py
import open3d as o3d
import numpy as np
import vil2.utils.misc_utils as utils
import matplotlib.pyplot as plt


def read_hdf5(file_name):
    """Read HDF5 file and return data."""
    with h5py.File(file_name, 'r') as file:
        return np.array(file['colors']), np.array(file['depth'])


hdf5_file = "/home/robot-learning/Projects/VIL2/vil2/test_data/dmorp/000000/000000/4.hdf5"
color, depth = read_hdf5(hdf5_file)
depth = depth.astype(np.float32)
depth[depth > 1000.0] = 0.0
# visulize depth
plt.imshow(depth)
plt.colorbar()
plt.show()

intrinsic = np.array([[711.0, 0.0, 255.5], [0.0, 711.0, 255.5], [0.0, 0.0, 1.0]])
pcd = utils.get_pointcloud(depth, intrinsic)

# Remove 0 points
pcd = pcd[pcd[:, 0] != 0.0, :]

# Visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=0.1)

# Keep each axis on the same scale
max_range = np.array([pcd[:, 0].max() - pcd[:, 0].min(), pcd[:, 1].max() - pcd[:, 1].min(),
                      pcd[:, 2].max() - pcd[:, 2].min()]).max() / 2.0
mean_x = pcd[:, 0].mean()
mean_y = pcd[:, 1].mean()
mean_z = pcd[:, 2].mean()
ax.set_aspect('equal')
plt.show()
