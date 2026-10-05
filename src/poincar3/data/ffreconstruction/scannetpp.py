from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import OpenEXR
import torch
from tqdm import tqdm

from poincar3.logging import logger


def read_exr_depth(path: Path) -> np.ndarray:
    with OpenEXR.File(path.as_posix()) as file:
        depth = file.channels()["Y"].pixels.astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth


class ScanNetPlusPlusReconScene(torch.utils.data.Dataset):
    def __init__(self, data_root: Path, *, use_pairs_and_overlaps: bool) -> None:
        super().__init__()
        self.data_root = data_root
        with open(data_root / "dslr" / "nerfstudio" / "transforms_undistorted.json") as f:
            self.meta_info = json.load(f)
        self.frame_map = {frame["file_path"]: frame for frame in self.meta_info["frames"]}
        if use_pairs_and_overlaps:
            scene_name = data_root.name
            pairs_and_overlaps = np.load(
                data_root / ".." / ".." / "overlaps" / f"{scene_name}.npy",
                allow_pickle=True,
            ).item()
            pairs = pairs_and_overlaps["pairs"]
            broken = np.logical_or(pairs[:, 0] == 0, pairs[:, 1] == 0)
            self.pairs = pairs[~broken]
            self.overlaps = pairs_and_overlaps["overlaps"][~broken]
        else:
            self.pairs = None
            self.overlaps = None

    def __len__(self) -> int:
        return len(self.meta_info["frames"]) ** 2 if self.pairs is None else len(self.pairs)

    def load_depth(self, frame: dict) -> np.ndarray:
        depth_path = self.data_root / "rendered_depth" / (frame["file_path"] + ".exr")
        return read_exr_depth(depth_path)


class ScanNetPlusPlusRecon(torch.utils.data.Dataset):
    @dataclass(frozen=True)
    class Cfg:
        split: Literal["train", "val", "test"]
        data_root: str = "data/scannet++/data_download/scannetpp/data"
        weight: int = 1_000_000
        use_pairs_and_overlaps: bool = True
        fraction: float = 1.0

    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        split_root = self.data_root / ".." / "splits"
        split_file = {
            "train": "nvs_sem_train.txt",
            "val": "nvs_sem_val.txt",
            "test": "nvs_test.txt",
        }[cfg.split]
        split_scenes = set((split_root / split_file).read_text().splitlines())
        logger.info(f"Loading {type(self).__name__} {cfg.split=} scenes.")
        self.scenes: list[ScanNetPlusPlusReconScene] = []
        for scene_root in tqdm(
            sorted(self.data_root.iterdir()),
            desc=f"Loading {type(self).__name__} {cfg.split} scenes...",
        ):
            if scene_root.name not in split_scenes:
                continue
            try:
                self.scenes.append(
                    ScanNetPlusPlusReconScene(
                        scene_root, use_pairs_and_overlaps=cfg.use_pairs_and_overlaps
                    )
                )
            except Exception as e:  # noqa: BLE001 -- one broken scene must not abort loading
                logger.warning(f"Error loading scene {scene_root.name}: {e}")
        if cfg.fraction < 1.0:
            self.scenes = self.scenes[: max(1, round(len(self.scenes) * cfg.fraction))]
        logger.info(
            f"Built {len(self.scenes)} ScanNet++Recon scenes with a total of "
            f"{sum(len(scene) for scene in self.scenes)} pairs."
        )

    def __len__(self) -> int:
        return self.cfg.weight
