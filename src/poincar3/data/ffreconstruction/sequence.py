from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from poincar3.data.ffreconstruction.scannetpp import ScanNetPlusPlusRecon
from poincar3.data.photometric import PhotometricAug
from poincar3.data.seq_preprocess import process_frame_with_depth
from poincar3.data.sequence_data import _build_adjacency, _sample_sequence_indices
from poincar3.logging import logger


@dataclass
class ReconBatch:
    """
    imgs: [batch_size, num_views, 3, height, width], in [0, 1].
    depths: [batch_size, num_views, height, width], z-depth in meters (0 where invalid).
    depth_masks: [batch_size, num_views, height, width] bool, True = valid (finite, > 0) depth.
    poses: [batch_size, num_views, 4, 4] cam-from-world.
    K: [batch_size, num_views, 3, 3] intrinsics, already matching `imgs`'s resolution.
    image_paths: source paths, for debugging.
    """

    imgs: torch.Tensor
    depths: torch.Tensor
    depth_masks: torch.Tensor
    poses: torch.Tensor
    K: torch.Tensor
    image_paths: list[list[str]]

    def to(self, device: torch.device) -> "ReconBatch":
        data = {}
        for k, v in self.__dict__.items():
            data[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return ReconBatch(**data)

    @classmethod
    def collate(cls, samples: list["ReconBatch"]) -> "ReconBatch":
        keys = samples[0].__dict__.keys()
        batch = {}
        for k in keys:
            if isinstance(samples[0].__dict__[k], torch.Tensor):
                batch[k] = torch.stack([s.__dict__[k] for s in samples])
            else:
                batch[k] = [s.__dict__[k] for s in samples]
        return cls(**batch)


# A sequence index: either a plain base index (uses the dataset's default
# num_views/H/W) or a (idx, N, T, H, W) tuple emitted by DynamicBatchSampler
# (see module docstring -- T is always 0 here).
SeqIndex = int | tuple[int, int, int, int, int]


class ReconSequenceDataset(torch.utils.data.Dataset):
    """Base for multi-view covisible-sequence datasets with depth GT.

    A subclass provides `self.scenes`/`self.cum_weights` (per-scene sampling
    CDF, as in `poincar3.data.sequence_data.SequenceDataset`) plus `_load_frame`
    (here: also returning depth) and `_scene_pairs`.
    """

    scenes: list[Any]
    cum_weights: torch.Tensor

    def __init__(
        self,
        *,
        weight: int,
        default_num_views: int,
        default_height: int,
        default_width: int,
        rescale_aug: bool = True,
        photometric: PhotometricAug.Cfg | None = None,
    ) -> None:
        super().__init__()
        self._weight = weight
        self._default_num_views = default_num_views
        self._default_height = default_height
        self._default_width = default_width
        self._rescale_aug = rescale_aug
        # Geometry-preserving photometric aug applied to every frame (see
        # `poincar3.data.photometric`'s docstring) -- safe here even though depth/
        # pose are GT-supervised, since it never touches pixel geometry.
        self._photometric = PhotometricAug(photometric)
        self._adj: dict[int, dict[int, list[int]] | None] = {}

    def __len__(self) -> int:
        return self._weight

    def _load_frame(self, scene: Any, frame_idx: int):
        """Return `(pil_img, depth (H, W) f32, K (3, 3), pose cam-from-world (4, 4), path)`."""
        raise NotImplementedError

    def _scene_pairs(self, scene: Any) -> np.ndarray:
        """`(M, 2)` covisible frame-index pairs for a scene (see
        `poincar3.data.sequence_data.SequenceDataset._scene_pairs`)."""
        raise NotImplementedError

    def _scene_adjacency(self, scene_index: int, scene: Any):
        adj = self._adj.get(scene_index, 0)
        if adj == 0:
            pairs = self._scene_pairs(scene)
            adj = _build_adjacency(pairs) if len(pairs) else None
            self._adj[scene_index] = adj
        return adj

    def _precompute_adjacency(self) -> None:
        """Build every scene's adjacency now, in the main process (see
        `SequenceDataset._precompute_adjacency`'s docstring for why)."""
        for scene_index, scene in enumerate(self.scenes):
            self._scene_adjacency(scene_index, scene)

    def __getitem__(self, index: SeqIndex) -> ReconBatch:
        if isinstance(index, tuple):
            idx, n, t, height, width = index
        else:
            idx = index
            n, t = self._default_num_views, 0
            height, width = self._default_height, self._default_width
        num_views = n + t  # t is always 0 here -- no teacher/student split.

        rng = np.random.default_rng()
        scene_index = int(torch.searchsorted(self.cum_weights, (idx + 0.5) / len(self)))
        scene = self.scenes[scene_index]
        adj = self._scene_adjacency(scene_index, scene)
        if adj is None or len(adj) < num_views:
            return self[(int(rng.integers(len(self))), n, t, height, width)]
        indices = _sample_sequence_indices(adj, num_views, rng)
        try:
            return self._load_sequence(scene, indices, height, width, rng)
        except Exception as e:  # noqa: BLE001 -- mirror SequenceDataset robustness
            logger.warning(f"{type(self).__name__} failed on scene {scene_index}: {e}")
            return self[(int(rng.integers(len(self))), n, t, height, width)]

    def _load_sequence(
        self, scene: Any, indices: list[int], height: int, width: int, rng: np.random.Generator
    ) -> ReconBatch:
        images, depths, depth_masks, Ks, poses, paths = [], [], [], [], [], []
        for idx in indices:
            im, depth, K, pose, path = self._load_frame(scene, idx)
            arr, K_new, depth_new = process_frame_with_depth(
                im, K, depth, (height, width), rng=rng, rescale_aug=self._rescale_aug
            )
            images.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0)
            depth_t = torch.from_numpy(depth_new)
            mask_t = torch.isfinite(depth_t) & (depth_t > 0)
            depths.append(torch.nan_to_num(depth_t, nan=0.0, posinf=0.0, neginf=0.0))
            depth_masks.append(mask_t)
            Ks.append(torch.from_numpy(K_new).float())
            poses.append(torch.as_tensor(np.asarray(pose)).float())
            paths.append(path)

        images = self._photometric(images)
        return ReconBatch(
            imgs=torch.stack(images),
            depths=torch.stack(depths),
            depth_masks=torch.stack(depth_masks),
            poses=torch.stack(poses),
            K=torch.stack(Ks),
            image_paths=paths,
        )


# OpenGL (cam-from-world) -> COLMAP axis flip, shared by ScanNet++ frames.
_GL_TO_COLMAP = torch.tensor(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float
)


class ScanNetPlusPlusReconSequence(ReconSequenceDataset):
    """Multi-view depth+pose sequences over ScanNet++ indoor DSLR scenes.

    See `poincar3.data.sequence_data.ScanNetPlusPlusSequence`'s docstring for
    the frame-name-pair covisibility pattern this mirrors; `_load_frame`
    additionally reads the rendered depth EXR via `ScanNetPlusPlusReconScene
    .load_depth`.
    """

    @dataclass(frozen=True)
    class Cfg:
        split: Literal["train", "val", "test"] = "train"
        # Fallback N/H/W used only when indexed by a plain int (the dynamic
        # sampler overrides these per batch via a tuple index).
        num_views: int = 8
        height: int = 256
        width: int = 256
        weight: int = 1_000_000
        data_root: str = "data/scannet++/data_download/scannetpp/data"
        fraction: float = 1.0
        rescale_aug: bool = True
        # Fully independent per-frame jitter (`cojitter_prob=0.0`), rather
        # than jitter mostly shared across a sequence's frames.
        aug: PhotometricAug.Cfg = PhotometricAug.Cfg()

    def __init__(self, cfg: Cfg) -> None:
        super().__init__(
            weight=cfg.weight,
            default_num_views=cfg.num_views,
            default_height=cfg.height,
            default_width=cfg.width,
            rescale_aug=cfg.rescale_aug,
            photometric=cfg.aug,
        )
        self.cfg = cfg
        base = ScanNetPlusPlusRecon(
            ScanNetPlusPlusRecon.Cfg(
                split=cfg.split,
                data_root=cfg.data_root,
                weight=cfg.weight,
                use_pairs_and_overlaps=True,
                fraction=cfg.fraction,
            )
        )
        self.scenes = base.scenes
        n = len(self.scenes)
        self.cum_weights = torch.linspace(1 / n, 1, n)
        self._precompute_adjacency()
        logger.info(f"ScanNetPlusPlusReconSequence ready: {n} scenes.")

    def _frame_names(self, scene: Any) -> list[str]:
        names = getattr(scene, "_recon_frame_names", None)
        if names is None:
            valid = scene.frame_map
            names = sorted({str(n) for pair in scene.pairs for n in pair if str(n) in valid})
            scene._recon_frame_names = names
            scene._recon_name_to_idx = {n: i for i, n in enumerate(names)}
        return names

    def _scene_pairs(self, scene: Any) -> np.ndarray:
        if scene.pairs is None or len(scene.pairs) == 0:
            return np.empty((0, 2), dtype=np.int64)
        self._frame_names(scene)
        n2i = scene._recon_name_to_idx
        idx_pairs = [
            (n2i[a], n2i[b])
            for a, b in ((str(x), str(y)) for x, y in scene.pairs)
            if a in n2i and b in n2i
        ]
        if not idx_pairs:
            return np.empty((0, 2), dtype=np.int64)
        return np.asarray(idx_pairs, dtype=np.int64)

    def _load_frame(self, scene: Any, frame_idx: int):
        name = scene._recon_frame_names[frame_idx]
        frame = scene.frame_map[name]
        meta = scene.meta_info
        im_path = scene.data_root / "dslr" / "resized_undistorted_images" / frame["file_path"]
        im = Image.open(im_path)
        depth = scene.load_depth(frame)
        K = np.zeros((3, 3), dtype=np.float32)
        K[0, 0] = meta["fl_x"]
        K[1, 1] = meta["fl_y"]
        K[0, 2] = meta["cx"]
        K[1, 2] = meta["cy"]
        K[2, 2] = 1.0
        pose_c2w_opengl = torch.tensor(frame["transform_matrix"]).float()
        pose_c2w_colmap = pose_c2w_opengl @ _GL_TO_COLMAP
        pose_w2c_colmap = pose_c2w_colmap.inverse().numpy()  # cam-from-world
        return im, depth, K, pose_w2c_colmap, str(im_path)


class ComposedReconDataset(torch.utils.data.Dataset):
    """Weighted union of several `ReconSequenceDataset`s (mirrors
    `poincar3.data.sequence_data.ComposedSequenceDataset`, including its
    tuple-index passthrough so `DynamicBatchSampler` can drive the union
    directly: each `__getitem__` independently picks a sub-dataset by
    weight, using fresh entropy so it's safe across workers)."""

    def __init__(self, datasets: list[tuple[torch.utils.data.Dataset, float]]) -> None:
        if not datasets:
            raise ValueError("ComposedReconDataset needs at least one dataset.")
        self.datasets = [d for d, _ in datasets]
        weights = np.asarray([w for _, w in datasets], dtype=np.float64)
        if (weights <= 0).any():
            raise ValueError("Dataset weights must be positive.")
        self.probs = weights / weights.sum()
        self._lengths = [len(d) for d in self.datasets]  # type: ignore[arg-type]

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, index: SeqIndex) -> ReconBatch:
        if isinstance(index, tuple):
            idx, rest = index[0], index[1:]
        else:
            idx, rest = index, ()
        rng = np.random.default_rng()
        d_idx = int(rng.choice(len(self.datasets), p=self.probs))
        dataset = self.datasets[d_idx]
        sub_idx = idx % self._lengths[d_idx]
        return dataset[(sub_idx, *rest) if rest else sub_idx]
