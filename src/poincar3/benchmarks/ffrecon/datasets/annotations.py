from __future__ import annotations

import gzip
import json
import os.path as osp
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tvf

_to_tensor = tvf.ToTensor()


@dataclass
class RelposeAnnotationDataset:
    """`sequences[scene_name]` is a list of sequences with at least
    `min_num_images` frames (shorter ones are dropped, matching
    `test_relpose.py`'s `process_sequence` early-return)."""

    data_dir: str
    sequences: dict[str, list[list[dict]]]

    @classmethod
    def load(cls, data_dir: str, anno_path: str, min_num_images: int = 10) -> "RelposeAnnotationDataset":
        with gzip.open(anno_path, "r") as f:
            annotation: dict[str, list[list[dict]]] = json.loads(f.read())
        sequences = {
            scene_name: [seq for seq in scene_data if len(seq) >= min_num_images]
            for scene_name, scene_data in annotation.items()
        }
        sequences = {k: v for k, v in sequences.items() if v}
        return cls(data_dir=data_dir, sequences=sequences)

    def sample_sequence(
        self, scene_name: str, seq_idx: int, num_frames: int, rng: np.random.Generator
    ) -> tuple[list[str], np.ndarray]:
        """Returns `(image_paths, gt_extrinsics (num_frames, 4, 4))`, `num_frames`
        images sampled without replacement from the sequence."""
        seq_data = self.sequences[scene_name][seq_idx]
        ids = rng.choice(len(seq_data), min(num_frames, len(seq_data)), replace=False)

        image_paths = [osp.join(self.data_dir, seq_data[i]["filepath"]) for i in ids]
        extri_3x4 = np.stack([np.array(seq_data[i]["extri"], dtype=np.float64) for i in ids], axis=0)

        extri_4x4 = np.tile(np.eye(4), (len(ids), 1, 1))
        extri_4x4[:, :3, :] = extri_3x4
        return image_paths, extri_4x4


def load_images(image_paths: list[str], height: int, width: int) -> torch.Tensor:
    """Loads + resizes to a fixed `(height, width)` (no aspect-ratio
    preservation -- matches this repo's other fixed-shape eval loaders,
    e.g. `experiments/ffrecon/train_poincar3.py`'s eval loader). Returns
    `(N, 3, height, width)`, raw `[0, 1]`."""
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        images.append(_to_tensor(img))
    return torch.stack(images, dim=0)


def _load_rgb_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("RGB")


def _crop_to_supported_aspect_ratio(
    image: Image.Image, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2.0
) -> Image.Image:
    width, height = image.size
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height))
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height))
    return image


def _round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def _balanced_target_shape(aspect_ratio: float, resolution: int, patch_size: int) -> tuple[int, int]:
    token_number = (resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _pad_images_to_common_size(images: list[torch.Tensor], shapes: set[tuple[int, int]]) -> list[torch.Tensor]:
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)
    padded = []
    for image in images:
        h_padding = max_height - image.shape[1]
        w_padding = max_width - image.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top, pad_left = h_padding // 2, w_padding // 2
            image = torch.nn.functional.pad(
                image, (pad_left, w_padding - pad_left, pad_top, h_padding - pad_top), mode="constant", value=1.0
            )
        padded.append(image)
    return padded


def load_images_aspect_preserving(image_paths: list[str], resolution: int, patch_size: int = 16) -> torch.Tensor:
    """Aspect-preserving load, "balanced" mode: keeps each image's total
    patch-token count close to `(resolution // patch_size) ** 2` rather than
    resizing its long side to `resolution` -- what MegaDepth's reference
    protocol uses (`aspect_preserving=True`, see `RelposeBenchmark.Cfg`).
    Extreme aspect
    ratios are first center-cropped into `[0.5, 2.0]`; images that still end
    up with different shapes after their own per-image resize are then
    center-padded (white) to a common batch shape. Returns `(N, 3, H, W)`,
    raw `[0, 1]` -- `H`/`W` are whatever shape this batch's images settled
    on, not `resolution` itself.
    """
    images: list[torch.Tensor] = []
    shapes: set[tuple[int, int]] = set()
    for path in image_paths:
        image = _crop_to_supported_aspect_ratio(_load_rgb_image(path))
        width, height = image.size
        aspect_ratio = height / max(width, 1)
        target_h, target_w = _balanced_target_shape(aspect_ratio, resolution, patch_size)
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensor = _to_tensor(image)
        shapes.add((tensor.shape[1], tensor.shape[2]))
        images.append(tensor)

    if len(shapes) > 1:
        warnings.warn(f"Found images with different shapes: {shapes}; padding to a common size.", stacklevel=2)
        images = _pad_images_to_common_size(images, shapes)
    return torch.stack(images, dim=0)
