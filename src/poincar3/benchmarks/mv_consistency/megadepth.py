from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class _Sequence:
    scene_name: str
    frame_indices: list[int]


def _build_adjacency(pairs: np.ndarray, overlaps: np.ndarray, min_overlap: float) -> dict[int, list[int]]:
    adj: dict[int, set[int]] = {}
    for (a, b), overlap in zip(pairs, overlaps):
        if overlap < min_overlap:
            continue
        a, b = int(a), int(b)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return {k: sorted(v) for k, v in adj.items()}


def _sample_cluster(
    adj: dict[int, list[int]], seed_node: int, n: int, rng: np.random.Generator
) -> list[int] | None:
    """Random-walk an `n`-frame covisible chain from `seed_node`. `None` if the
    local connected component is too small (rather than padding with
    unrelated frames, unlike training's sampler -- a stray, non-covisible
    frame would just show up as "always invisible" in every eval track)."""
    seq = [seed_node]
    frontier = set(adj[seed_node])
    while len(seq) < n and frontier:
        nxt = int(rng.choice(sorted(frontier)))
        seq.append(nxt)
        frontier.discard(nxt)
        frontier.update(c for c in adj.get(nxt, []) if c not in seq)
    return seq if len(seq) == n else None


class MegaDepth(Dataset):
    """`num_views`-length covisible clusters sampled from MegaDepth's two
    held-out test scenes."""

    test_scenes = ["0015", "0022"]

    def __init__(
        self,
        root_dir: str = "data/megadepth",
        num_views: int = 8,
        image_size: tuple[int, int] = (448, 448),
        min_overlap: float = 0.1,
        max_sequences_per_scene: int = 50,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size

        rng = np.random.default_rng(seed)
        self.scene_info: dict[str, dict] = {}
        self.sequences: list[_Sequence] = []
        for scene_name in self.test_scenes:
            info = np.load(
                os.path.join(root_dir, "prep_scene_info", f"{scene_name}.npy"), allow_pickle=True
            ).item()
            self.scene_info[scene_name] = info
            adj = _build_adjacency(info["pairs"], info["overlaps"], min_overlap)
            seed_nodes = list(adj.keys())
            rng.shuffle(seed_nodes)
            for seed_node in seed_nodes[:max_sequences_per_scene]:
                cluster = _sample_cluster(adj, seed_node, num_views, rng)
                if cluster is not None:
                    self.sequences.append(_Sequence(scene_name, cluster))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict:
        seq = self.sequences[index]
        info = self.scene_info[seq.scene_name]
        H, W = self.image_size
        images, depths, intrinsics, poses = [], [], [], []

        for frame_idx in seq.frame_indices:
            img_path = os.path.join(self.root_dir, info["image_paths"][frame_idx])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
            orig_h, orig_w = img.shape[:2]
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(torch.from_numpy(img_rgb.copy()).permute(2, 0, 1).float() / 255.0)

            depth_path = os.path.join(self.root_dir, info["depth_paths"][frame_idx])
            with h5py.File(depth_path, "r") as f:
                depth = f["depth"][:].astype(np.float32)
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            depths.append(torch.from_numpy(depth.copy()).unsqueeze(0))

            K = info["intrinsics"][frame_idx].astype(np.float32).copy()
            sx, sy = W / orig_w, H / orig_h
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            intrinsics.append(torch.from_numpy(K))

            # MegaDepth's `poses` are already world-to-camera, matching NAVI's
            # and ScanNet's (post-inversion) convention.
            poses.append(torch.from_numpy(info["poses"][frame_idx].astype(np.float32)))

        depths = torch.stack(depths)
        return {
            "image": torch.stack(images),
            "depth": depths,
            "intrinsics": torch.stack(intrinsics),
            "Rt": torch.stack(poses),
            "mask": (depths > 0).float(),
        }
