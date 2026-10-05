from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch


class MaskingGenerator:
    """Random rectangular block mask over an `(H, W)` patch grid."""

    def __init__(
        self,
        input_size: tuple[int, int],
        min_num_patches: int = 4,
        max_num_patches: int | None = None,
        min_aspect: float = 0.3,
        max_aspect: float | None = None,
    ) -> None:
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.min_num_patches = min_num_patches
        self.max_num_patches = self.num_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                num_masked = mask[top : top + h, left : left + w].sum()
                delta_here = h * w - num_masked
                if 0 < delta_here <= max_mask_patches:
                    mask[top : top + h, left : left + w] = True
                    delta = int(delta_here)
                    break
        return delta

    def __call__(self, num_masking_patches: int) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = min(num_masking_patches - mask_count, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return mask


@dataclass(frozen=True)
class MaskingCfg:
    # Fraction of the student's frames that get masked at all (the rest are
    # left fully unmasked, mirroring iBOT/DINOv2's per-sample masking split).
    mask_probability: float = 0.5
    # [min, max] fraction of patches to mask, sampled uniformly, for frames
    # chosen to be masked.
    mask_ratio: tuple[float, float] = (0.1, 0.5)
    min_num_patches: int = 4


def build_student_mask(
    num_frames: int, patch_grid: tuple[int, int], cfg: MaskingCfg
) -> torch.Tensor:
    """Returns a `[num_frames, H*W]` bool mask, True = masked.

    Each frame is independently left unmasked with probability
    `1 - cfg.mask_probability`, so an entirely-unmasked batch (every frame,
    every sample) is possible, just increasingly unlikely as `num_frames`
    grows -- and `Poincar3Loss.forward` hard-fails on it (`teacher_masked.shape[0]
    == 0`, "No masked patches in batch"), which is fatal to the whole
    distributed job since only the unlucky rank(s) crash while others
    continue. So: if every frame here came back unmasked, force-mask one
    (random.randrange, matching the RNG already used above) rather than let
    that non-zero probability ever surface as a crash.
    """
    generator = MaskingGenerator(patch_grid, min_num_patches=cfg.min_num_patches)
    height, width = patch_grid
    masks = []
    for _ in range(num_frames):
        if random.random() < cfg.mask_probability:
            ratio = random.uniform(*cfg.mask_ratio)
            num_masking_patches = int(height * width * ratio)
            mask = generator(num_masking_patches).reshape(-1)
        else:
            mask = np.zeros(height * width, dtype=bool)
        masks.append(torch.from_numpy(mask))

    if not any(mask.any() for mask in masks):
        ratio = random.uniform(*cfg.mask_ratio)
        num_masking_patches = int(height * width * ratio)
        forced = generator(num_masking_patches).reshape(-1)
        if not forced.any():
            # Patch grid too small for `generator` to place any block (e.g. a
            # 1-patch-wide grid from an extreme aspect ratio) -- fall back to
            # unconditionally masking a prefix of patches directly.
            forced = np.zeros(height * width, dtype=bool)
            forced[: max(cfg.min_num_patches, 1)] = True
        masks[random.randrange(num_frames)] = torch.from_numpy(forced)

    return torch.stack(masks)
