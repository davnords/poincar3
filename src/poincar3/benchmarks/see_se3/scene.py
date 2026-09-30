from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scene:
    scene_id: str
    frame_ids: list[int]
    image_paths: list[str]
    poses: np.ndarray  # (N,4,4) camera-to-world, float64
