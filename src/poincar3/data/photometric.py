from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torchvision import transforms as TF


class PhotometricAug:
    @dataclass(frozen=True)
    class Cfg:
        # These defaults are intentionally a bit heavier than VGGT's. For
        # reference, VGGT's first dataset block (VGGT's `training/config/
        # default_dataset.yaml) uses: ColorJitter(0.5, 0.5, 0.5, 0.1) p=0.9,
        # RandomGrayscale p=0.05, GaussianBlur OFF, and cojitter=True/ratio=0.3
        # (i.e. one shared jitter per sequence ~70% of the time, per-frame 30%).
        enabled: bool = True
        brightness: float = 0.5  # VGGT: 0.5
        contrast: float = 0.5  # VGGT: 0.5
        saturation: float = 0.5  # VGGT: 0.5
        hue: float = 0.1  # VGGT: 0.1
        jitter_prob: float = 0.8  # VGGT: 0.9
        grayscale_prob: float = 0.1  # VGGT: 0.05
        blur_prob: float = 0.1  # VGGT: 0.0 (off)
        # Probability of applying ONE shared jitter to all frames in a sequence
        # (vs. independent per-frame jitter, the default). VGGT: ~0.7 (cojitter).
        cojitter_prob: float = 0.0

    def __init__(self, cfg: Cfg | None = None) -> None:
        self.cfg = cfg or PhotometricAug.Cfg()
        ops: list = []
        if self.cfg.jitter_prob > 0:
            ops.append(
                TF.RandomApply(
                    [
                        TF.ColorJitter(
                            brightness=self.cfg.brightness,
                            contrast=self.cfg.contrast,
                            saturation=self.cfg.saturation,
                            hue=self.cfg.hue,
                        )
                    ],
                    p=self.cfg.jitter_prob,
                )
            )
        if self.cfg.grayscale_prob > 0:
            ops.append(TF.RandomGrayscale(p=self.cfg.grayscale_prob))
        if self.cfg.blur_prob > 0:
            ops.append(
                TF.RandomApply([TF.GaussianBlur(5, sigma=(0.1, 2.0))], p=self.cfg.blur_prob)
            )
        self.transform = TF.Compose(ops) if ops else None

    def __call__(self, frames: list[torch.Tensor]) -> list[torch.Tensor]:
        """Augment a list of ``(3, H, W)`` float frames in ``[0, 1]``."""
        if not self.cfg.enabled or self.transform is None:
            return frames
        # Shared jitter for the whole sequence: one transform call on the stack
        # applies identical random params to every frame.
        if self.cfg.cojitter_prob > 0 and random.random() < self.cfg.cojitter_prob:
            return list(self.transform(torch.stack(frames)).unbind(0))
        # Independent per-frame jitter (re-samples params each call).
        return [self.transform(f) for f in frames]
