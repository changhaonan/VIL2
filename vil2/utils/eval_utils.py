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
    plt.show()
