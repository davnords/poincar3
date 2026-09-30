from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from .datasets.dtu import DTU
from .datasets.eth3d import ETH3D
from .geometry import unproject_depth_map_to_point_map
from .inference import ReconModel
from .metrics import accuracy, apply_transform, completion, estimate_normals_pca, icp_point_to_point, umeyama

_SEQ_ID_MAPS_DIR = Path(__file__).parent / "datasets" / "seq_id_maps"


class PointcloudBenchmark:
    @dataclass(frozen=True)
    class Cfg:
        dataset: Literal["dtu", "eth3d"] = "eth3d"
        data_root: str | None = None  # None -> "data/dtu_test_mvsnet_release" / "data/eth3d"
        load_img_size: int = 518  # target width; both sides rounded to patch_multiple (518 -> 512 at patch 16)
        patch_multiple: int = 16  # Poincar3/MuM's patch_size -- see datasets/cropping.py
        # Max correspondence distance for ICP refinement, and the unit
        # accuracy/completion report in. None -> 100 for DTU (millimeter
        # scale depth) or 0.1 for ETH3D (meter scale), matching the original
        # script's per-dataset thresholds.
        icp_threshold: float | None = None
        cache_file: str | None = None
        max_sequences: int | None = None

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        seq_id_map_name = "DTU_mv-recon_seq-id-map-kf5.json" if cfg.dataset == "dtu" else "ETH3D_mv-recon_seq-id-map-kf5.json"
        with open(_SEQ_ID_MAPS_DIR / seq_id_map_name) as f:
            self.seq_id_map: dict[str, list[int]] = json.load(f)

        if cfg.dataset == "dtu":
            self.dataset: DTU | ETH3D = DTU(
                DTU_DIR=cfg.data_root or "data/dtu_test_mvsnet_release",
                load_img_size=cfg.load_img_size,
                patch_multiple=cfg.patch_multiple,
                cache_file=cfg.cache_file or "data/dataset_cache/dtu_mv_recon_cache.npy",
            )
        elif cfg.dataset == "eth3d":
            self.dataset = ETH3D(
                ETH3D_DIR=cfg.data_root or "data/eth3d",
                load_img_size=cfg.load_img_size,
                patch_multiple=cfg.patch_multiple,
                cache_file=cfg.cache_file or "data/dataset_cache/eth3d_mv_recon_cache.npy",
            )
        else:
            raise ValueError(f"Unknown pointcloud dataset: {cfg.dataset!r}")

        self.icp_threshold = cfg.icp_threshold if cfg.icp_threshold is not None else (100.0 if cfg.dataset == "dtu" else 0.1)

    @torch.no_grad()
    def benchmark(self, recon_model: ReconModel, step: int | None = None) -> dict[str, float]:
        recon_model.eval()
        device = next(recon_model.parameters()).device

        totals: dict[str, float] = {}
        num_scored = 0
        seq_items = list(self.seq_id_map.items())
        if self.cfg.max_sequences is not None:
            seq_items = seq_items[: self.cfg.max_sequences]

        for seq_name, ids in tqdm(seq_items, desc=f"pointcloud/{self.cfg.dataset}"):
            data = self.dataset.get_data(sequence_name=seq_name, ids=ids)
            images = data["images"].to(device)  # (S, 3, H, W)
            gt_pts: np.ndarray = data["pointclouds"]  # (S, H, W, 3)
            valid_mask: np.ndarray = data["valid_mask"]  # (S, H, W)

            pred = recon_model.infer(images)
            pred_pts = unproject_depth_map_to_point_map(pred.depth, pred.extrinsics, pred.intrinsics)

            if pred_pts.shape != gt_pts.shape:
                raise ValueError(
                    f"Predicted points shape {pred_pts.shape} != ground truth shape {gt_pts.shape} for "
                    f"{seq_name!r} -- check that `load_img_size`/`patch_multiple` match the model's patch size."
                )
            if valid_mask.sum() < 10:
                continue

            # Coarse Sim(3) alignment (predicted poses/depth live in the
            # model's own arbitrary scale/frame -- see `normalize_batch_scale`)...
            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts_aligned = c * np.einsum("nhwj,ij->nhwi", pred_pts, R) + t.T

            pred_valid = pred_pts_aligned[valid_mask].reshape(-1, 3).astype(np.float64)
            gt_valid = gt_pts[valid_mask].reshape(-1, 3).astype(np.float64)

            # ...then a rigid ICP refinement pass on top.
            transform = icp_point_to_point(pred_valid, gt_valid, max_correspondence_distance=self.icp_threshold)
            pred_refined = apply_transform(pred_valid, transform)

            pred_normals = estimate_normals_pca(pred_refined)
            gt_normals = estimate_normals_pca(gt_valid)

            acc, acc_med, nc1, nc1_med = accuracy(gt_valid, pred_refined, gt_normals, pred_normals)
            comp, comp_med, nc2, nc2_med = completion(gt_valid, pred_refined, gt_normals, pred_normals)

            for k, v in {
                "Acc-mean": acc,
                "Acc-med": acc_med,
                "Comp-mean": comp,
                "Comp-med": comp_med,
                "NC-mean": (nc1 + nc2) / 2,
                "NC-med": (nc1_med + nc2_med) / 2,
                "NC1-mean": nc1,
                "NC1-med": nc1_med,
                "NC2-mean": nc2,
                "NC2-med": nc2_med,
            }.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            num_scored += 1

        if num_scored == 0:
            return {}

        summary = {k: v / num_scored for k, v in totals.items()}
        summary["num_sequences"] = float(num_scored)

        if step is not None:
            try:
                import wandb

                wandb.log({f"pointcloud/{self.cfg.dataset}/{k}": v for k, v in summary.items()}, step=step)
            except Exception:  # noqa: BLE001 -- wandb logging is best-effort
                pass
        return summary
