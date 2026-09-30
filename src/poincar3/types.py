from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Batch:
    """One multi-view training batch.

    imgs: `[batch, teacher_frames, 3, H, W]` -- the teacher's view of the full
        sequence.
    student_imgs: `[batch, student_frames, 3, H, W]` -- the same leading frames,
        independently photometrically augmented, as the student sees them. Both
        branches drawing from the same augmentation pool keeps student/teacher
        alignment a plain slice of the teacher's outputs.
    mask: `[batch, student_frames, patches]` bool, True = masked. Applies to the
        student only, and implicitly defines `student_frames`.
    poses / K / image_paths: optional per-frame cam-from-world pose
        `[batch, teacher_frames, 4, 4]`, intrinsics `[..., 3, 3]` already
        rescaled to match `imgs`, and source paths. Carried through by the
        datasets that have them; unused by the SSL objective.
    """

    imgs: torch.Tensor
    student_imgs: torch.Tensor
    mask: torch.Tensor
    poses: torch.Tensor | None = None
    K: torch.Tensor | None = None
    image_paths: list[str] | list[list[str]] | None = None

    def to(self, device: torch.device) -> Batch:
        return Batch(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.__dict__.items()})

    @classmethod
    def collate(cls, samples: list[Batch]) -> Batch:
        batch = {}
        for key, value in samples[0].__dict__.items():
            if isinstance(value, torch.Tensor):
                batch[key] = torch.stack([s.__dict__[key] for s in samples])
            else:
                batch[key] = [s.__dict__[key] for s in samples]
        return Batch(**batch)


class Model(nn.Module):
    @property
    def name(self) -> str:
        return self.__class__.__name__
