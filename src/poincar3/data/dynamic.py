from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import Sampler

from poincar3.distrib import get_rank, get_world_size
from poincar3.data.seq_preprocess import get_target_shape


class DynamicSampler(Sampler[int]):
    """A reshuffled, rank-sharded stream of base dataset indices.

    Like ``DistributedSampler`` but standalone (works single- or multi-GPU).
    ``set_epoch`` reshuffles deterministically.
    """

    def __init__(self, dataset_len: int, *, shuffle: bool = True, seed: int = 0) -> None:
        self.dataset_len = dataset_len
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.rank = get_rank()
        self.world_size = get_world_size()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_len, generator=g).tolist()
        else:
            indices = list(range(self.dataset_len))
        # Drop the ragged tail *before* sharding so every rank gets the same
        # count -- indices[rank::world_size] alone is uneven when dataset_len
        # isn't divisible by world_size (the first dataset_len % world_size
        # ranks would get one extra index), which desyncs DDP step counts and
        # can hang an all_reduce once the short ranks exhaust their epoch.
        even_len = (self.dataset_len // self.world_size) * self.world_size
        return iter(indices[: even_len][self.rank :: self.world_size])

    def __len__(self) -> int:
        return self.dataset_len // self.world_size


class DynamicBatchSampler(Sampler[list[tuple[int, int, int, int, int]]]):
    """Groups a ``DynamicSampler`` into ``[B, N+T, H, W]``-shaped batches.

    Per batch: sample one student length ``N``, one extra teacher-only frame
    count ``T``, and one aspect ratio; derive a target shape
    ``(short = img_size * aspect, img_size)`` rounded to the patch size,
    optionally transpose to portrait, and emit ``B = floor(num_frames / (N+T))``
    tuples ``(idx, N, T, H, W)`` sharing that shape (so the batch is collatable).
    The dataset then aspect-preservingly resizes+crops to ``(H, W)``.
    """

    @dataclass(frozen=True)
    class Cfg:
        # [min, max] student sequence length N, sampled uniformly per batch.
        seq_len: tuple[int, int] = (2, 24)
        # [min, max] extra teacher-only frame count T, sampled uniformly per
        # batch, independently of N. The teacher sees N+T frames, the student
        # sees the leading N of them.
        teacher_len: tuple[int, int] = (0, 12)
        # Per-GPU image budget; B = floor(num_frames / (N+T)).
        num_frames: int = 48
        # Long side of the target shape.
        img_size: int = 256
        # [min, max] aspect ratio (H/W), sampled uniformly per batch. Spans both
        # orientations (<1 landscape, >1 portrait).
        aspects: tuple[float, float] = (0.33, 1.33)
        patch_size: int = 16
        seed: int = 0

    def __init__(self, cfg: Cfg, sampler: DynamicSampler) -> None:
        lo, hi = cfg.seq_len
        if not (1 <= lo <= hi):
            raise ValueError(f"seq_len must be [min, max] with 1 <= min <= max, got {cfg.seq_len}")
        t_lo, t_hi = cfg.teacher_len
        if not (0 <= t_lo <= t_hi):
            raise ValueError(f"teacher_len must be [min, max] with 0 <= min <= max, got {cfg.teacher_len}")
        if cfg.num_frames < hi + t_hi:
            raise ValueError(
                f"num_frames ({cfg.num_frames}) must be >= max seq_len + max teacher_len "
                f"({hi + t_hi}) so at least one full sequence fits per batch."
            )
        a_lo, a_hi = cfg.aspects
        if not (0 < a_lo <= a_hi):
            raise ValueError(f"aspects must be [min, max] with 0 < min <= max, got {cfg.aspects}")
        if cfg.img_size % cfg.patch_size != 0:
            raise ValueError(f"img_size ({cfg.img_size}) must be a multiple of patch_size ({cfg.patch_size})")
        self.cfg = cfg
        self.sampler = sampler
        self.possible_n = list(range(lo, hi + 1))
        self.possible_t = list(range(t_lo, t_hi + 1))
        self.rng = random.Random()
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)
        # Seeded identically across ranks -> all ranks agree on N/T/H/W/B per batch.
        self.rng.seed(self.cfg.seed + epoch)

    def __iter__(self) -> Iterator[list[tuple[int, int, int, int, int]]]:
        sampler_iter = iter(self.sampler)
        while True:
            n = self.rng.choice(self.possible_n)
            t = self.rng.choice(self.possible_t)
            aspect = self.rng.uniform(*self.cfg.aspects)
            h, w = get_target_shape(aspect, self.cfg.img_size, self.cfg.patch_size)
            batch_size = max(1, self.cfg.num_frames // (n + t))

            batch: list[tuple[int, int, int, int, int]] = []
            for _ in range(batch_size):
                try:
                    batch.append((next(sampler_iter), n, t, h, w))
                except StopIteration:
                    break
            if not batch:
                return
            yield batch

    def __len__(self) -> int:
        # Approximate batches per epoch using the mean sequence length.
        avg_n = sum(self.possible_n) / len(self.possible_n)
        avg_t = sum(self.possible_t) / len(self.possible_t)
        avg_batch_size = max(1.0, self.cfg.num_frames / (avg_n + avg_t))
        return int(len(self.sampler) / avg_batch_size)
