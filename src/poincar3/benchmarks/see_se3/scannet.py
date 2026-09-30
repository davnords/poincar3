from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from poincar3.benchmarks.see_se3.scene import Scene


def list_scenes(
    root_dir: str = "data/scannet/scans/scans_train",
    primary_scan_only: bool = True,
) -> list[str]:
    """Scene directory names under `root_dir`, sorted. `primary_scan_only`
    (default True) keeps only each room's primary scan (`sceneXXXX_00`),
    dropping `_01`/`_02` re-scans of the same physical room -- otherwise a
    small `max_scenes` would oversample a handful of rooms scanned multiple
    times rather than sampling diverse scenes."""
    scene_ids = sorted(
        d for d in os.listdir(root_dir) if d.startswith("scene") and os.path.isdir(os.path.join(root_dir, d))
    )
    if primary_scan_only:
        scene_ids = [s for s in scene_ids if s.endswith("_00")]
    return scene_ids


def load_scene(root_dir: str, scene_id: str) -> Scene | None:
    """Loads every frame's pose for `scene_id`, sorted by (integer) frame id.
    Drops individual frames with a non-finite pose (ScanNet's BundleFusion
    tracker occasionally loses tracking on a handful of frames per scene).
    Returns `None` if fewer than 2 valid frames remain."""
    color_dir = os.path.join(root_dir, scene_id, "color")
    pose_dir = os.path.join(root_dir, scene_id, "pose")
    if not (os.path.isdir(color_dir) and os.path.isdir(pose_dir)):
        return None

    frame_ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(pose_dir))
    kept_ids, kept_paths, kept_poses = [], [], []
    for frame_id in frame_ids:
        pose_path = os.path.join(pose_dir, f"{frame_id}.txt")
        pose = np.loadtxt(pose_path)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            continue
        image_path = os.path.join(color_dir, f"{frame_id}.jpg")
        if not os.path.exists(image_path):
            continue
        kept_ids.append(frame_id)
        kept_paths.append(image_path)
        kept_poses.append(pose)

    if len(kept_ids) < 2:
        return None
    return Scene(
        scene_id=scene_id,
        frame_ids=kept_ids,
        image_paths=kept_paths,
        poses=np.stack(kept_poses, axis=0),
    )


def load_image(path: str, image_size: tuple[int, int]) -> torch.Tensor:
    """Returns (3,H,W) float32 in [0,1], RGB. `image_size`: (H, W)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def load_scene_images(scene: Scene, image_size: tuple[int, int]) -> torch.Tensor:
    """Loads every frame of `scene` once. Returns (N,3,H,W) uint8 (RGB) on
    CPU -- kept as uint8 rather than float32 to hold a whole (~500-2000
    frame) scene in memory at once without excessive RAM (e.g. ~1500 frames
    at 480x640 is ~1.4GB as uint8 vs ~5.5GB as float32); callers convert
    whatever chunk/window they need to float just before feeding the model."""
    imgs = torch.empty(len(scene.image_paths), 3, image_size[0], image_size[1], dtype=torch.uint8)
    for i, path in enumerate(scene.image_paths):
        imgs[i] = (load_image(path, image_size) * 255.0).round().to(torch.uint8)
    return imgs
