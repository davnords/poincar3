from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class _Sequence:
    scene_id: str
    frame_ids: list[str]
    intrinsics: torch.Tensor


class ScanNet(Dataset):
    """`num_views`-length sequences of consecutive ScanNet test-scene frames."""

    def __init__(
        self,
        root_dir: str = "data/scannet_test_1500",
        num_views: int = 8,
        image_size: tuple[int, int] = (480, 640),
        max_frame_skip: int = 35,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_views = num_views
        self.image_size = image_size
        self.max_frame_skip = max_frame_skip
        self.transform = transforms.ToTensor()

        self.intrinsics_dict = {
            k: torch.from_numpy(v).float()
            for k, v in np.load(os.path.join(root_dir, "intrinsics.npz")).items()
        }
        self.sequences = self._build_sequences()

    def _build_sequences(self) -> list[_Sequence]:
        sequences = []
        scene_ids = sorted(
            d for d in os.listdir(self.root_dir)
            if d.startswith("scene") and os.path.isdir(os.path.join(self.root_dir, d))
        )
        for scene_id in scene_ids:
            pose_dir = os.path.join(self.root_dir, scene_id, "pose")
            if not os.path.exists(pose_dir):
                continue

            frame_ids = sorted(
                (os.path.splitext(f)[0] for f in os.listdir(pose_dir)), key=int
            )
            frame_numbers = [int(f) for f in frame_ids]
            if len(frame_numbers) < self.num_views:
                continue

            for i in range(len(frame_numbers) - self.num_views + 1):
                skips = (
                    frame_numbers[i + j + 1] - frame_numbers[i + j]
                    for j in range(self.num_views - 1)
                )
                if all(skip <= self.max_frame_skip for skip in skips):
                    sequences.append(
                        _Sequence(
                            scene_id=scene_id,
                            frame_ids=frame_ids[i : i + self.num_views],
                            intrinsics=self.intrinsics_dict[scene_id],
                        )
                    )
                    break  # first valid window per scene, matching upstream
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict:
        seq = self.sequences[index]
        images, depths, poses = [], [], []

        for frame_id in seq.frame_ids:
            img_path = os.path.join(self.root_dir, seq.scene_id, "color", f"{frame_id}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
            img = cv2.resize(
                img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR
            )
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(self.transform(img_rgb))

            depth_path = os.path.join(self.root_dir, seq.scene_id, "depth", f"{frame_id}.png")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Depth not found at {depth_path}")
            depth = cv2.resize(
                depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST
            )
            depths.append(torch.from_numpy(depth.astype(np.float32) / 1000.0).unsqueeze(0))

            pose_path = os.path.join(self.root_dir, seq.scene_id, "pose", f"{frame_id}.txt")
            pose = torch.from_numpy(np.loadtxt(pose_path)).float()
            poses.append(pose.inverse())  # ScanNet pose files are c2w; convert to w2c.

        depths = torch.stack(depths)
        # Pre-expand the scene's single intrinsics matrix to one per view, so
        # MvConsistencyBenchmark can treat every dataset's `intrinsics` the
        # same way (NAVI already gives one per view).
        intrinsics = seq.intrinsics.unsqueeze(0).expand(self.num_views, -1, -1).contiguous()
        return {
            "image": torch.stack(images),
            "depth": depths,
            "intrinsics": intrinsics,
            "Rt": torch.stack(poses),
            "mask": (depths > 0).float(),
        }
