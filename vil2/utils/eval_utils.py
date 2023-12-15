"""Evaluate diffusion model"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def draw_pose_distribution(
    poses: np.ndarray,
    color: np.ndarray,
    title: str = None,
    scale: float = 1.0,
    xyz_range: np.ndarray | None = None
):
    """Draw poses using scatter plot"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        poses[:, 0],
        poses[:, 1],
        poses[:, 2],
        c=color,
        cmap="jet",
        s=scale,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title is not None:
        plt.title(title)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Intensity')  # You can change 'Intensity' to whatever label is appropriate for your data
    # axis equal
    if xyz_range is None:
        max_range = np.array(
            [
                poses[:, 0].max() - poses[:, 0].min(),
                poses[:, 1].max() - poses[:, 1].min(),
                poses[:, 2].max() - poses[:, 2].min(),
            ]
        ).max()
        mean_x = poses[:, 0].mean()
        mean_y = poses[:, 1].mean()
        mean_z = poses[:, 2].mean()
        ax.set_xlim(mean_x - max_range / 2, mean_x + max_range / 2)
        ax.set_ylim(mean_y - max_range / 2, mean_y + max_range / 2)
        ax.set_zlim(mean_z - max_range / 2, mean_z + max_range / 2)
    else:
        ax.set_xlim(xyz_range[0], xyz_range[1])
        ax.set_ylim(xyz_range[2], xyz_range[3])
        ax.set_zlim(xyz_range[4], xyz_range[5])
    plt.show()


def compare_distribution(data_0, data_1=None, dim_start: int = 0, dim_end: int = 1, title: str = None, plot_type: str = "histogram", save_path: str = None, save_fig: bool = False, visualize: bool = True):
    """Compare two distribution: data_0 and data_1; Compare the first num_dim dimensions
    We draw them using histogram
    """
    num_dim = dim_end - dim_start
    if plot_type == "histogram":
        fig, axs = plt.subplots(1, num_dim, figsize=(20, 5))
        plt.suptitle(title)
        for i in range(num_dim):
            # draw histogram for each dimension in ratio
            if data_0 is not None:
                axs[i].hist(data_0[:, dim_start+i], bins=100, alpha=0.5, label="data_0")
            if data_1 is not None:
                axs[i].hist(data_1[:, dim_start+i], bins=100, alpha=0.5, label="data_1")
            axs[i].legend(loc="upper right")
    elif plot_type == "scatter":
        # draw only one figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')
        if num_dim == 2:
            # draw scatter plot for each dimension
            if data_0 is not None:
                ax.scatter(data_0[:, dim_start], data_0[:, dim_start+1], alpha=0.5, label="data_0")
            if data_1 is not None:
                ax.scatter(data_1[:, dim_start], data_1[:, dim_start+1], alpha=0.5, label="data_1")
            ax.legend(loc="upper right")
        elif num_dim == 3:
            # draw scatter plot for each dimension
            if data_0 is not None:
                ax.scatter(data_0[:, dim_start], data_0[:, dim_start+1], data_0[:, dim_start+2], marker='o', label="data_0")
            if data_1 is not None:
                ax.scatter(data_1[:, dim_start], data_1[:, dim_start+1], data_1[:, dim_start+2], marker='o', label="data_1")
            ax.legend(loc="upper right")
        else:
            raise NotImplementedError
    if save_fig:
        plt.savefig(save_path)
    if visualize:
        plt.show()
