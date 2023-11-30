from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ObjData:
    """Data class for object"""
    pose: np.ndarray
    semantic_str: str = "Object"
    semantic_id: int = 0  # for close-vocab
    semantic_feature: np.ndarray | None = None  # for open-vocab
    pcd: np.ndarray | None = None
    geometry: np.ndarray | None = None
    super_patch: list | None = None
    patch_center: np.ndarray | None = None
