from __future__ import annotations

import json
import os

import numpy as np

from poincar3.benchmarks.see_se3.scene import Scene

_SPLIT_FILES = {"train": "nvs_sem_train.txt", "val": "nvs_sem_val.txt", "test": "nvs_test.txt"}

# Same constant as `poincar3.data.sequence_data._GL_TO_COLMAP` -- duplicated
# (rather than imported) to avoid pulling that module's ~20 training-dataset
# imports into this otherwise-lightweight eval path just for one 4x4 matrix.
_GL_TO_COLMAP = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
)


def list_scenes(root_dir: str = "data/scannet++/data_download/scannetpp/data", split: str = "val") -> list[str]:
    """Scene ids in `<root>/../splits/<split>.txt`, restricted to those that
    actually exist under `root_dir` (mirrors `scannet.list_scenes`'s return
    shape: a sorted list of scene-id strings)."""
    split_file = os.path.join(root_dir, "..", "splits", _SPLIT_FILES[split])
    split_scenes = set(open(split_file).read().splitlines())
    return sorted(d for d in os.listdir(root_dir) if d in split_scenes)


def load_scene(root_dir: str, scene_id: str) -> Scene | None:
    """Loads every DSLR frame's pose for `scene_id`, ordered by filename
    (ScanNet++'s DSLR filenames are a sequential shutter counter, e.g.
    `DSC00925.JPG`, so this closely follows capture order). Drops frames
    whose image file is missing or whose pose is non-finite. Returns `None`
    if fewer than 2 valid frames remain."""
    transforms_path = os.path.join(root_dir, scene_id, "dslr", "nerfstudio", "transforms_undistorted.json")
    if not os.path.exists(transforms_path):
        return None
    with open(transforms_path) as f:
        meta = json.load(f)

    image_dir = os.path.join(root_dir, scene_id, "dslr", "resized_undistorted_images")
    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])

    kept_paths, kept_poses = [], []
    for frame in frames:
        image_path = os.path.join(image_dir, frame["file_path"])
        if not os.path.exists(image_path):
            continue
        pose_gl = np.array(frame["transform_matrix"], dtype=np.float64)
        pose_c2w = pose_gl @ _GL_TO_COLMAP
        if pose_c2w.shape != (4, 4) or not np.all(np.isfinite(pose_c2w)):
            continue
        kept_paths.append(image_path)
        kept_poses.append(pose_c2w)

    if len(kept_paths) < 2:
        return None
    return Scene(
        scene_id=scene_id,
        frame_ids=list(range(len(kept_paths))),
        image_paths=kept_paths,
        poses=np.stack(kept_poses, axis=0),
    )
