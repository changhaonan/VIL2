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
