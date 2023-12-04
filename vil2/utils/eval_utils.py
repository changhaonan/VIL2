"""Evaluate diffusion model"""
import numpy as np
import matplotlib.pyplot as plt


def draw_pose_distribution(
    poses: np.ndarray,
    color: np.ndarray,
    title: str = None,
    scale: float = 1.0,
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
    plt.show()
