from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ObjData:
    """Data class for object"""
    pose: np.ndarray
    semantic_str: str = "Object"
    semantic_id: int = 0  # for close-vocab
    semantic_feature: np.ndarray | None = None  # for open-vocab
    pcd: np.ndarray | None = None
    geometry: np.ndarray | None = None


@dataclass
class SceneData:
    """Data class for scene"""
    scene_id: int
    poses: np.ndarray = np.array([])  # (M, 7) / (M, 3)
    # features
    sem_feats: np.ndarray = np.array([])  # (M, D)
    geo_feats: np.ndarray = np.array([]) | None  # (M, D)
    meta: dict[str, str] | None = None
