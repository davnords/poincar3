from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from poincar3.data.datasets.scannetpp import ScanNetPlusPlus
from poincar3.data.masking import MaskingCfg, build_student_mask
from poincar3.data.photometric import PhotometricAug
from poincar3.data.seq_preprocess import process_frame
from poincar3.logging import logger
from poincar3.types import Batch


def _build_adjacency(pairs: np.ndarray) -> dict[int, list[int]]:
    """Undirected covisibility adjacency from a scene's (M, 2) pair array.

    Vectorized rather than a per-pair Python loop: dense all-pairs scenes push
    M into the hundreds of millions, where the loop takes minutes.
    """
    sym = np.concatenate([pairs, pairs[:, ::-1]], axis=0).astype(np.int64)
    order = np.lexsort((sym[:, 1], sym[:, 0]))
    sym = sym[order]
    nodes, starts = np.unique(sym[:, 0], return_index=True)
    groups = np.split(sym[:, 1], starts[1:])
    return {int(node): np.unique(neighbors).tolist() for node, neighbors in zip(nodes, groups)}


def _dense_adjacency(n: int) -> dict[int, list[int]]:
    """Adjacency for the complete graph on ``n`` nodes, built directly rather
    than by materializing an (n^2, 2) pair array and running it through
    `_build_adjacency`.
    """
    all_nodes = list(range(n))
    return {i: all_nodes[:i] + all_nodes[i + 1 :] for i in range(n)}


def _sample_sequence_indices(
    adj: dict[int, list[int]], n: int, rng: np.random.Generator
) -> list[int]:
    """Grow an ``n``-frame covisible chain by random-walking the graph.

    Starts at a random node and repeatedly adds a random unused neighbour of any
    already-collected frame, yielding a connected covisible cluster. Falls back
    to filling with random graph nodes if the local component is too small.
    """
    nodes = list(adj.keys())
    seed = int(rng.choice(nodes))
    seq = [seed]
    frontier = set(adj[seed])
    while len(seq) < n and frontier:
        nxt = int(rng.choice(sorted(frontier)))
        seq.append(nxt)
        frontier.discard(nxt)
        frontier.update(c for c in adj[nxt] if c not in seq)
    if len(seq) < n:
        remaining = [x for x in nodes if x not in seq]
        rng.shuffle(remaining)
        seq.extend(remaining[: n - len(seq)])
    return seq[:n]


# A sequence index: either a plain base index (uses the dataset's default
# N/T/H/W) or a (idx, N, T, H, W) tuple emitted by DynamicBatchSampler.
SeqIndex = int | tuple[int, int, int, int, int]


class NoEligibleScene(RuntimeError):
    """No scene could supply the requested N+T-frame sequence after retrying."""


class SequenceDataset(torch.utils.data.Dataset):
    """Base for multi-view covisible-sequence datasets.

    A subclass provides ``self.scenes`` (each scene exposing a covisibility
    ``pairs`` array indexing its frames) and ``self.cum_weights`` (per-scene
    sampling CDF), plus per-frame loading via ``_load_frame``. This base owns
    the covisibility-graph sequence sampling, the ``(idx, N, T, H, W)`` tuple
    index, the teacher/student frame split, the student's iBOT-style masking,
    and the aspect-preserving resize+crop -- everything dataset-agnostic.
    """

    scenes: list[Any]
    cum_weights: torch.Tensor

    def __init__(
        self,
        *,
        weight: int,
        default_seq_len: int,
        default_height: int,
        default_width: int,
        patch_size: int = 16,
        rescale_aug: bool = True,
        photometric: PhotometricAug.Cfg | None = None,
        masking: MaskingCfg | None = None,
    ) -> None:
        super().__init__()
        self._weight = weight
        self._default_seq_len = default_seq_len
        self._default_height = default_height
        self._default_width = default_width
        self._patch_size = patch_size
        self._rescale_aug = rescale_aug
        # Geometry-preserving photometric aug applied to every frame.
        self._photometric = PhotometricAug(photometric)
        self._masking_cfg = masking or MaskingCfg()
        self._adj: dict[int, dict[int, list[int]] | None] = {}
        self.max_seq_frames: int = 0

    def __len__(self) -> int:
        return self._weight

    def _load_frame(self, scene: Any, frame_idx: int) -> tuple[Image.Image, torch.Tensor, torch.Tensor, str]:
        """Return ``(pil_img, K (3,3), pose cam-from-world (4,4), path)`` for a frame."""
        raise NotImplementedError

    def _scene_pairs(self, scene: Any) -> np.ndarray:
        """``(M, 2)`` covisible frame-index pairs for a scene.

        Defaults to the scene's ``pairs`` array. Override for datasets that
        express covisibility differently. Returned indices must match those
        passed to ``_load_frame``.
        """
        return np.asarray(scene.pairs)

    def _scene_dense_size(self, scene: Any) -> int | None:
        """Frame count for scenes with dense (complete-graph) covisibility.

        Override this instead of `_scene_pairs` for datasets where every frame
        is covisible with every other; `_scene_adjacency` then builds adjacency
        directly. Returning `None` (the default) keeps the
        `_scene_pairs`/`_build_adjacency` path.
        """
        return None

    def _scene_adjacency(self, scene_index: int, scene: Any):
        adj = self._adj.get(scene_index, 0)
        if adj == 0:
            n = self._scene_dense_size(scene)
            if n is not None:
                adj = _dense_adjacency(n) if n >= 2 else None
            else:
                pairs = self._scene_pairs(scene)
                adj = _build_adjacency(pairs) if len(pairs) else None
            self._adj[scene_index] = adj
        return adj

    def _precompute_adjacency(self) -> None:
        """Build every scene's adjacency in the main process, so it is shared
        copy-on-write across DataLoader workers rather than duplicated per
        worker. Also sets `max_seq_frames`, the largest covisible-frame count
        any scene offers, which `ComposedSequenceDataset` uses to skip datasets
        that cannot satisfy a batch's requested N+T."""
        for scene_index, scene in enumerate(self.scenes):
            adj = self._scene_adjacency(scene_index, scene)
            if adj is not None:
                self.max_seq_frames = max(self.max_seq_frames, len(adj))

    # Cap same-dataset resample attempts; without it a batch requesting more
    # frames (N+T) than any scene has recurses unboundedly into a
    # RecursionError, crashing the worker (and, under DDP, the whole job).
    _MAX_RETRIES = 50

    def __getitem__(self, index: SeqIndex, _retries: int = 0) -> Batch:
        if isinstance(index, tuple):
            idx, n, t, height, width = index
        else:
            idx = index
            n, t, height, width = self._default_seq_len, 0, self._default_height, self._default_width
        total = n + t

        if _retries >= self._MAX_RETRIES:
            raise NoEligibleScene(
                f"{type(self).__name__}: no scene has >= {total} covisible frames "
                f"(N+T) after {_retries} retries; longest scene has {self.max_seq_frames}."
            )

        rng = np.random.default_rng()
        scene_index = int(torch.searchsorted(self.cum_weights, (idx + 0.5) / len(self)))
        scene = self.scenes[scene_index]
        adj = self._scene_adjacency(scene_index, scene)
        if adj is None or len(adj) < total:
            # No covisible pairs, or too few covisible frames to form an
            # (n+t)-view sequence: skip and retry elsewhere. Without this a
            # small scene yields a short sequence, breaking the batch's
            # torch.stack collate.
            return self.__getitem__((int(rng.integers(len(self))), n, t, height, width), _retries + 1)
        indices = _sample_sequence_indices(adj, total, rng)
        try:
            return self._load_sequence(scene, indices, n, height, width, rng)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{type(self).__name__} failed on scene {scene_index}: {e}")
            return self.__getitem__((int(rng.integers(len(self))), n, t, height, width), _retries + 1)

    def _load_sequence(
        self,
        scene: Any,
        indices: list[int],
        num_student_frames: int,
        height: int,
        width: int,
        rng: np.random.Generator,
    ) -> Batch:
        images = []
        Ks = []
        poses = []
        paths = []
        for idx in indices:
            im, K, pose, path = self._load_frame(scene, idx)
            # Aspect-preserving resize-to-cover + principal-point crop to
            # (height, width); no geometry-warping square resize. K is updated.
            arr, K_new = process_frame(
                im, K.numpy(), (height, width), rng=rng, rescale_aug=self._rescale_aug
            )
            images.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0)
            Ks.append(torch.from_numpy(K_new).float())
            poses.append(pose)
            paths.append(path)

        # Independent photometric augmentation per branch (colour jitter /
        # grayscale / blur) -- DINOv2/BYOL-style: teacher and student each get
        # their own draw from the same pool for the frames they share, rather
        # than seeing identical pixels, so there's an appearance-invariance
        # signal on top of the student's masking. Two calls on the same raw
        # `images` list, not one chained into the other.
        student_images = self._photometric(images[:num_student_frames])
        images = self._photometric(images)

        patch_grid = (height // self._patch_size, width // self._patch_size)
        mask = build_student_mask(num_student_frames, patch_grid, self._masking_cfg)

        return Batch(
            imgs=torch.stack(images),  # (N+T, 3, H, W)
            student_imgs=torch.stack(student_images),  # (N, 3, H, W)
            mask=mask,  # (N, H//patch_size * W//patch_size)
            poses=torch.stack(poses),
            K=torch.stack(Ks),
            image_paths=paths,
        )


# OpenGL (cam-from-world) -> COLMAP axis flip, shared by ScanNet++ frames.
_GL_TO_COLMAP = torch.tensor(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float
)


class ScanNetPlusPlusSequence(SequenceDataset):
    """Multi-view sequences over ScanNet++ indoor DSLR scenes.

    Each scene's covisibility ``pairs`` is an ``(M, 2)`` array of frame *name*
    strings, and frames are addressed via ``frame_map`` / ``meta_info``. Names
    are mapped to contiguous indices in ``_scene_pairs``; the depth EXRs are
    skipped, since self-distillation does not need them.
    """

    @dataclass(frozen=True)
    class Cfg:
        split: Literal["train", "val", "test"] = "train"
        seq_len: int = 8
        height: int = 256
        width: int = 256
        patch_size: int = 16
        weight: int = 1_000_000
        data_root: str = "data/scannet++/data_download/scannetpp/data"
        fraction: float = 1.0
        rescale_aug: bool = True
        aug: PhotometricAug.Cfg = PhotometricAug.Cfg()
        masking: MaskingCfg = MaskingCfg()

    def __init__(self, cfg: Cfg) -> None:
        super().__init__(
            weight=cfg.weight,
            default_seq_len=cfg.seq_len,
            default_height=cfg.height,
            default_width=cfg.width,
            patch_size=cfg.patch_size,
            rescale_aug=cfg.rescale_aug,
            photometric=cfg.aug,
            masking=cfg.masking,
        )
        self.cfg = cfg
        base = ScanNetPlusPlus(
            ScanNetPlusPlus.Cfg(
                split=cfg.split,
                data_root=cfg.data_root,
                weight=cfg.weight,
                use_pairs_and_overlaps=True,
                fraction=cfg.fraction,
            )
        )
        self.scenes = base.scenes
        # Underlying dataset samples scenes uniformly; mirror that as a CDF.
        n = len(self.scenes)
        self.cum_weights = torch.linspace(1 / n, 1, n)
        self._precompute_adjacency()
        logger.info(
            f"ScanNetPlusPlusSequence ready: {n} scenes "
            f"(default seq_len={cfg.seq_len}, {cfg.height}x{cfg.width})."
        )

    def _frame_names(self, scene: Any) -> list[str]:
        """Contiguous node ordering for a scene, cached on the scene object.

        Nodes are the frame names that both appear in covisible pairs and exist
        in ``frame_map`` (so ``_load_frame`` can always resolve them).
        """
        names = getattr(scene, "_seq_frame_names", None)
        if names is None:
            valid = scene.frame_map
            names = sorted({str(n) for pair in scene.pairs for n in pair if str(n) in valid})
            scene._seq_frame_names = names
            scene._seq_name_to_idx = {n: i for i, n in enumerate(names)}
        return names

    def _scene_pairs(self, scene: Any) -> np.ndarray:
        if scene.pairs is None or len(scene.pairs) == 0:
            return np.empty((0, 2), dtype=np.int64)
        self._frame_names(scene)
        n2i = scene._seq_name_to_idx
        idx_pairs = [
            (n2i[a], n2i[b])
            for a, b in ((str(x), str(y)) for x, y in scene.pairs)
            if a in n2i and b in n2i
        ]
        if not idx_pairs:
            return np.empty((0, 2), dtype=np.int64)
        return np.asarray(idx_pairs, dtype=np.int64)

    def _load_frame(self, scene: Any, frame_idx: int):
        name = scene._seq_frame_names[frame_idx]
        frame = scene.frame_map[name]
        meta = scene.meta_info
        im_path = scene.data_root / "dslr" / "resized_undistorted_images" / frame["file_path"]
        im = Image.open(im_path)
        K = torch.zeros((3, 3))
        K[0, 0] = meta["fl_x"]
        K[1, 1] = meta["fl_y"]
        K[0, 2] = meta["cx"]
        K[1, 2] = meta["cy"]
        K[2, 2] = 1.0
        pose_c2w_opengl = torch.tensor(frame["transform_matrix"]).float()
        pose_c2w_colmap = pose_c2w_opengl @ _GL_TO_COLMAP
        pose_w2c_colmap = pose_c2w_colmap.inverse()  # cam-from-world
        return im, K, pose_w2c_colmap, str(im_path)


def _frame_sort_key(p: Path) -> tuple[int, int | str]:
    """Numeric filenames (e.g. RE10K's timestamps) sort numerically; anything
    else falls back to lexical, so this never raises on an unexpected naming
    scheme."""
    try:
        return (0, int(p.stem))
    except ValueError:
        return (1, p.stem)


class VideoFrameSequence(SequenceDataset):
    """Multi-view sequences over pre-extracted video-clip frames (RE10K, and
    any other dataset of short single-shot clips with no covisibility, pose or
    intrinsics on disk).

    Each scene is a directory of one clip's extracted frames, laid out as
    `data_root/[split/]<scene_id>/*.{jpg,png}`. Every frame pair in a clip is
    treated as covisible, on the assumption that a short single-shot clip stays
    within one covisible space. `K`/`poses` are identity placeholders: the SSL
    objective never reads them, and this keeps the per-frame shape contract the
    same as every other dataset.

    Point a new `Cfg(data_root=..., split=...)` at another extracted-frames
    directory to add a dataset; no subclassing needed.
    """

    class _Scene:
        def __init__(self, scene_path: Path) -> None:
            self.scene_path = scene_path
            self.frame_paths = sorted(
                (p for p in scene_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
                key=_frame_sort_key,
            )

    @dataclass(frozen=True)
    class Cfg:
        # Set to "" for datasets with no split subdirectory (data_root's
        # immediate children are scene directories).
        split: str = "train"
        seq_len: int = 8
        height: int = 256
        width: int = 256
        patch_size: int = 16
        weight: int = 1_000_000
        data_root: str = "data/re10k"
        fraction: float = 1.0
        rescale_aug: bool = True
        aug: PhotometricAug.Cfg = PhotometricAug.Cfg()
        masking: MaskingCfg = MaskingCfg()

    def __init__(self, cfg: Cfg) -> None:
        super().__init__(
            weight=cfg.weight,
            default_seq_len=cfg.seq_len,
            default_height=cfg.height,
            default_width=cfg.width,
            patch_size=cfg.patch_size,
            rescale_aug=cfg.rescale_aug,
            photometric=cfg.aug,
            masking=cfg.masking,
        )
        self.cfg = cfg
        data_root = Path(cfg.data_root) / cfg.split if cfg.split else Path(cfg.data_root)
        scene_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
        if cfg.fraction < 1.0:
            scene_dirs = scene_dirs[: max(1, round(len(scene_dirs) * cfg.fraction))]
        self.scenes: list[VideoFrameSequence._Scene] = []
        for scene_path in scene_dirs:
            scene = self._Scene(scene_path)
            if len(scene.frame_paths) >= 2:
                self.scenes.append(scene)
        n = len(self.scenes)
        self.cum_weights = torch.linspace(1 / n, 1, n)
        self._precompute_adjacency()
        logger.info(
            f"VideoFrameSequence ready ({cfg.data_root}): {n} scenes "
            f"(default seq_len={cfg.seq_len}, {cfg.height}x{cfg.width})."
        )

    def _scene_dense_size(self, scene: Any) -> int:
        return len(scene.frame_paths)

    def _load_frame(self, scene: Any, frame_idx: int):
        im_path = scene.frame_paths[frame_idx]
        im = Image.open(im_path)
        K = torch.eye(3, dtype=torch.float)
        pose = torch.eye(4, dtype=torch.float)
        return im, K, pose, str(im_path)


class ComposedSequenceDataset(torch.utils.data.Dataset):
    """Weighted union of several sequence datasets.

    Each ``__getitem__`` independently picks a sub-dataset by weight and
    forwards the (tuple) index to it, so a single ``DynamicBatchSampler`` batch
    can mix sequences from different datasets. The per-call random pick uses
    fresh entropy, so it is safe across workers.
    """

    def __init__(self, datasets: list[tuple[torch.utils.data.Dataset, float]]) -> None:
        if not datasets:
            raise ValueError("ComposedSequenceDataset needs at least one dataset.")
        self.datasets = [d for d, _ in datasets]
        weights = np.asarray([w for _, w in datasets], dtype=np.float64)
        if (weights <= 0).any():
            raise ValueError("Dataset weights must be positive.")
        self.probs = weights / weights.sum()
        self._lengths = [len(d) for d in self.datasets]  # type: ignore[arg-type]
        # Longest covisible sequence each dataset can supply (`inf` if
        # untracked) -- excludes short-scene datasets from the weighted pick
        # when a batch's N+T exceeds what they can satisfy.
        self._max_seq_frames = np.array(
            [getattr(d, "max_seq_frames", np.inf) or np.inf for d in self.datasets]
        )

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, index: SeqIndex) -> Batch:
        if isinstance(index, tuple):
            idx, rest = index[0], index[1:]
        else:
            idx, rest = index, ()
        rng = np.random.default_rng()
        probs = self.probs
        if len(rest) >= 2:
            eligible = self._max_seq_frames >= rest[0] + rest[1]  # N + T
            if eligible.any() and not eligible.all():
                probs = np.where(eligible, self.probs, 0.0)
        probs = probs / probs.sum()
        # A dataset can still be "eligible" per its longest scene yet exhaust
        # its own resample budget (most scenes too short): drop it and redraw
        # instead of propagating `NoEligibleScene` and killing the batch.
        for _ in range(len(self.datasets)):
            d_idx = int(rng.choice(len(self.datasets), p=probs))
            dataset = self.datasets[d_idx]
            sub_idx = idx % self._lengths[d_idx]
            try:
                return dataset[(sub_idx, *rest) if rest else sub_idx]
            except NoEligibleScene:
                probs[d_idx] = 0.0
                if probs.sum() == 0:
                    break
                probs = probs / probs.sum()
        raise NoEligibleScene(f"No dataset can supply the requested sequence (index={index}).")
