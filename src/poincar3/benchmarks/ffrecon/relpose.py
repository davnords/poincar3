from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import torch
from tqdm import tqdm

from poincar3.heads.rotation import mat_to_quat

from .datasets.annotations import RelposeAnnotationDataset, load_images, load_images_aspect_preserving
from .inference import ReconModel


class _DatasetDefaults(NamedTuple):
    data_dir: str
    anno_path: str
    # Whether this dataset's *reference* protocol loads images
    # aspect-preservingly. Per-dataset rather than one shared default, because
    # the reference protocols disagree -- and this is not cosmetic: scoring
    # MegaDepth with a square resize instead reads AUC@30 0.46 vs 0.75
    # (RRA@15 0.71 vs 0.99), because squashing its varied aspect ratios into a
    # square is a geometry the model never sees in training.
    aspect_preserving: bool


# Per-dataset defaults, so `experiments/ffrecon/eval.py` reproduces a whole
# protocol from `--relpose.dataset` alone. Annotations are built by
# `experiments/ffrecon/build_relpose_annotations.py` for
# eth3d/co3d/scannet/scannetpp; megadepth and re10k ship their own.
_DATASET_DEFAULTS: dict[str, _DatasetDefaults] = {
    "megadepth": _DatasetDefaults("data/megadepth", "data/megadepth/annotations/test.jgz", True),
    "re10k": _DatasetDefaults("data/re10k_test", "data/re10k_test/annotations/test.jgz", False),
    "eth3d": _DatasetDefaults("data/eth3d", "data/eth3d/annotations/test.jgz", False),
    "scannet": _DatasetDefaults("data/scannet_test_1500", "data/scannet_test_1500/annotations/test.jgz", False),
    "co3d": _DatasetDefaults("data/CO3D", "data/CO3D/annotations/co3dv2/relpose_test.jgz", False),
    # 50 held-out ScanNet++ `nvs_sem_val` scenes, short-baseline sequences:
    # each is drawn from one seed frame's nearest neighbours by viewing
    # direction and camera position. Median GT relative rotation ~17 degrees,
    # easier than MegaDepth's 26, and it separates training runs much more
    # cleanly than the whole-scene variant below. Those frames are picked
    # using the GT poses, part of what the metric scores, so treat this as a
    # within-suite signal rather than an externally comparable number.
    # ScanNet++ is also in the ffrecon training mixture, but val is
    # scene-disjoint from train, so this is an in-distribution check.
    "scannetpp": _DatasetDefaults(
        "data/scannet++/data_download/scannetpp/data",
        "data/scannet++/data_download/scannetpp/annotations/relpose_val_local.jgz",
        True,
    ),
    # Whole-scene variant of the same 50 scenes: one sequence per scene, 10
    # frames sampled from anywhere in it. Much harder -- median GT relative
    # rotation 88 degrees.
    "scannetpp_reference": _DatasetDefaults(
        "data/scannet++/data_download/scannetpp/data",
        "data/scannet++/data_download/scannetpp/annotations/relpose_val.jgz",
        True,
    ),
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_pair_index(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    i1, i2 = torch.combinations(torch.arange(n), 2, with_replacement=False).unbind(-1)
    return i1, i2


def rotation_angle(rot_gt: torch.Tensor, rot_pred: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    return err_q * 180 / np.pi


def compare_translation_by_angle(
    t_gt: torch.Tensor, t: torch.Tensor, eps: float = 1e-15, default_err: float = 1e6
) -> torch.Tensor:
    t = t / (torch.norm(t, dim=1, keepdim=True) + eps)
    t_gt = t_gt / (torch.norm(t_gt, dim=1, keepdim=True) + eps)
    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt: torch.Tensor, tvec_pred: torch.Tensor, ambiguity: bool = True) -> torch.Tensor:
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred) * 180.0 / np.pi
    if ambiguity:  # essential-matrix-style translation-direction sign ambiguity
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())
    return rel_tangle_deg


def closed_form_inverse_se3(se3: torch.Tensor) -> torch.Tensor:
    R, T = se3[:, :3, :3], se3[:, :3, 3:]
    R_t = R.transpose(1, 2)
    top_right = -torch.bmm(R_t, T)
    inv = torch.eye(4, device=se3.device, dtype=se3.dtype)[None].repeat(len(R), 1, 1)
    inv[:, :3, :3] = R_t
    inv[:, :3, 3:] = top_right
    return inv


def se3_to_relative_pose_error(
    pred_se3: torch.Tensor, gt_se3: torch.Tensor, num_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """pred_se3/gt_se3: `(N, 4, 4)`, cam-from-world (w2c). Returns per-pair
    `(rel_rangle_deg, rel_tangle_deg)` over every `C(N, 2)` frame pair."""
    i1, i2 = build_pair_index(num_frames)
    relative_gt = gt_se3[i1].bmm(closed_form_inverse_se3(gt_se3[i2]))
    relative_pred = pred_se3[i1].bmm(closed_form_inverse_se3(pred_se3[i2]))
    rel_rangle_deg = rotation_angle(relative_gt[:, :3, :3], relative_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_gt[:, :3, 3], relative_pred[:, :3, 3])
    return rel_rangle_deg, rel_tangle_deg


def calculate_auc_np(r_error: np.ndarray, t_error: np.ndarray, max_threshold: int = 30) -> tuple[float, np.ndarray]:
    """AUC of the cumulative-accuracy curve over `max(r_error, t_error)`,
    binned at 1-degree resolution up to `max_threshold`."""
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    normalized_histogram = histogram.astype(float) / float(len(max_errors))
    return float(np.mean(np.cumsum(normalized_histogram))), normalized_histogram


def calculate_rra_rta_maa(r_error: np.ndarray, t_error: np.ndarray) -> dict[str, float]:
    """`RRA@15`/`RTA@15` (fraction of pairs under a 15-degree rotation/
    translation error) and `mAA30` (mean of the pointwise-min
    rotation/translation accuracy curves, thresholds 0-30 degrees)."""
    num_pairs = len(r_error)
    if num_pairs == 0:
        return {"RRA@15": 0.0, "RTA@15": 0.0, "mAA30": 0.0}

    tau = 15
    rra_15 = float(np.mean(r_error < tau))
    rta_15 = float(np.mean(t_error < tau))

    max_threshold = 30
    bins = np.arange(max_threshold + 1)
    r_hist, _ = np.histogram(r_error, bins=bins)
    r_curve = np.cumsum(r_hist.astype(float) / num_pairs)
    t_hist, _ = np.histogram(t_error, bins=bins)
    t_curve = np.cumsum(t_hist.astype(float) / num_pairs)
    maa_30 = float(np.mean(np.minimum(r_curve, t_curve)))

    return {"RRA@15": rra_15, "RTA@15": rta_15, "mAA30": maa_30}


class RelposeBenchmark:
    @dataclass(frozen=True)
    class Cfg:
        # Which dataset's `_DATASET_DEFAULTS` entry to fall back to for
        # whichever of `data_dir`/`anno_path`/`aspect_preserving` is left
        # `None` -- set those explicitly to point at a dataset/split not in
        # that table.
        dataset: Literal[
            "megadepth", "re10k", "eth3d", "scannet", "co3d", "scannetpp", "scannetpp_reference"
        ] = "megadepth"
        data_dir: str | None = None
        anno_path: str | None = None
        num_frames: int = 10
        min_num_images: int = 10
        # First N sequences per scene (None = every sequence) -- the standard
        # "fast eval" protocol, closer to what a quick per-checkpoint eval
        # wants than exhaustively sampling every sequence of every scene.
        max_sequences_per_scene: int | None = 10
        # First N *scenes* (None = every scene). Unlike `max_sequences_per_scene`,
        # this actually bounds total work for datasets with many one-sequence
        # scenes (e.g. RE10K's test set: 1386 scenes x 1 sequence each --
        # `max_sequences_per_scene` alone can't cap that). Scene order is
        # whatever `dict` iteration gives (insertion order from the
        # annotation file), so this is a fixed, reproducible subset, not a
        # random sample.
        max_scenes: int | None = None
        # Hard cap on the *total* number of sequences scored across every
        # scene combined -- `max_scenes`/`max_sequences_per_scene` cap
        # per-scene or scene count, not the combined total.
        max_sequences: int | None = None
        height: int = 256
        width: int = 256
        # Aspect-preserving load instead of a naive square resize to
        # `(height, width)`. `None` takes `dataset`'s own reference-protocol
        # value from `_DATASET_DEFAULTS`. Override it only deliberately: the
        # two are not comparable numbers (see `_DatasetDefaults`).
        aspect_preserving: bool | None = None
        seed: int = 0

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        defaults = _DATASET_DEFAULTS[cfg.dataset]
        data_dir = cfg.data_dir or defaults.data_dir
        anno_path = cfg.anno_path or defaults.anno_path
        self.aspect_preserving = defaults.aspect_preserving if cfg.aspect_preserving is None else cfg.aspect_preserving
        self.dataset = RelposeAnnotationDataset.load(data_dir, anno_path, min_num_images=cfg.min_num_images)

    @torch.no_grad()
    def benchmark(
        self, recon_model: ReconModel, step: int | None = None, wandb_prefix: str = "relpose"
    ) -> dict[str, float]:
        """Scores every scene independently, then reports the *mean across
        scenes* of each metric, following VGGT's own `evaluation/test_relpose.py`.
        Pooling every pair from every scene into one flat array instead would
        implicitly weight scenes by how many pairs they contribute -- a
        materially different number whenever scene sizes are uneven, as in
        RE10K's test set (1386 scenes, wildly different pair counts)."""
        _set_seed(self.cfg.seed)
        recon_model.eval()
        device = next(recon_model.parameters()).device

        per_scene_metrics: list[dict[str, float]] = []
        total_seqs = 0

        scene_items = list(self.dataset.sequences.items())
        if self.cfg.max_scenes is not None:
            scene_items = scene_items[: self.cfg.max_scenes]

        for scene_name, sequences in tqdm(scene_items, desc="relpose"):
            if self.cfg.max_sequences is not None and total_seqs >= self.cfg.max_sequences:
                break
            seq_indices = range(len(sequences))
            if self.cfg.max_sequences_per_scene is not None:
                seq_indices = list(seq_indices)[: self.cfg.max_sequences_per_scene]

            scene_r_errors: list[np.ndarray] = []
            scene_t_errors: list[np.ndarray] = []
            for seq_idx in seq_indices:
                if self.cfg.max_sequences is not None and total_seqs >= self.cfg.max_sequences:
                    break
                rng = np.random.default_rng(self.cfg.seed + seq_idx)
                image_paths, gt_extri = self.dataset.sample_sequence(scene_name, seq_idx, self.cfg.num_frames, rng)
                num_frames = len(image_paths)
                if num_frames < 2:  # need at least one pair
                    continue
                total_seqs += 1

                if self.aspect_preserving:
                    images = load_images_aspect_preserving(image_paths, resolution=self.cfg.height).to(device)
                else:
                    images = load_images(image_paths, self.cfg.height, self.cfg.width).to(device)
                pred = recon_model.infer(images)

                pred_se3 = torch.eye(4, device=device, dtype=pred.extrinsics.dtype)[None].repeat(num_frames, 1, 1)
                pred_se3[:, :3, :] = pred.extrinsics
                gt_se3 = torch.from_numpy(gt_extri).to(device=device, dtype=pred_se3.dtype)

                rel_r, rel_t = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)
                scene_r_errors.append(rel_r.cpu().numpy())
                scene_t_errors.append(rel_t.cpu().numpy())

            if not scene_r_errors:
                continue
            r_errors, t_errors = np.concatenate(scene_r_errors), np.concatenate(scene_t_errors)
            metrics = {f"AUC@{t}": calculate_auc_np(r_errors, t_errors, max_threshold=t)[0] for t in (30, 15, 5, 3)}
            metrics.update(calculate_rra_rta_maa(r_errors, t_errors))
            per_scene_metrics.append(metrics)

        if not per_scene_metrics:
            return {}

        summary: dict[str, float] = {
            k: float(np.mean([m[k] for m in per_scene_metrics])) for k in per_scene_metrics[0]
        }
        summary["num_sequences"] = float(total_seqs)

        if step is not None:
            try:
                import wandb

                wandb.log({f"{wandb_prefix}/{k}": v for k, v in summary.items()}, step=step)
            except Exception:  # noqa: BLE001 -- wandb logging is best-effort
                pass
        return summary
