from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from poincar3.benchmarks.dense_features import get_attention_probe, get_feature_probe, resolve_block_fraction
from poincar3.benchmarks.see_se3.poincare import fit_poincare_adapter
from poincar3.benchmarks.see_se3.pose import pose9d, relative_pose_targets
from poincar3.benchmarks.see_se3.scannet import load_scene_images
from poincar3.benchmarks.see_se3.scannet import list_scenes as _list_scannet_scenes
from poincar3.benchmarks.see_se3.scannet import load_scene as _load_scannet_scene
from poincar3.benchmarks.see_se3.scannetpp import list_scenes as _list_scannetpp_scenes
from poincar3.benchmarks.see_se3.scannetpp import load_scene as _load_scannetpp_scene
from poincar3.benchmarks.see_se3.scene import Scene


def _mutual_knn(features: torch.Tensor, poses_9d: torch.Tensor, k: int, min_temporal_gap: int) -> float | None:
    """`features`: (M,C), `poses_9d`: (M,9). Mean fraction of shared k-nn
    (cosine, feature space; Euclidean, pose space) -- `None` if fewer than
    `2 * min_temporal_gap + k + 1` frames are available (not enough
    candidates survive the temporal-neighbor exclusion for a well-defined
    top-k)."""
    m = features.shape[0]
    if m < 2 * min_temporal_gap + k + 1:
        return None

    feat = F.normalize(features, dim=-1)
    sim = feat @ feat.T
    pose_dist = torch.cdist(poses_9d, poses_9d)

    idx = torch.arange(m)
    excluded = (idx[:, None] - idx[None, :]).abs() <= min_temporal_gap
    sim = sim.masked_fill(excluded, float("-inf"))
    pose_dist = pose_dist.masked_fill(excluded, float("inf"))

    feat_knn = sim.topk(k, dim=-1).indices
    pose_knn = pose_dist.topk(k, dim=-1, largest=False).indices

    overlap = torch.tensor(
        [len(set(feat_knn[i].tolist()) & set(pose_knn[i].tolist())) for i in range(m)], dtype=torch.float32
    )
    return (overlap / k).mean().item()


def _subsample_scene(scene: Scene, max_frames: int | None) -> Scene:
    n = len(scene.frame_ids)
    if max_frames is None or n <= max_frames:
        return scene
    idx = sorted(set(np.linspace(0, n - 1, max_frames).round().astype(int).tolist()))
    return Scene(
        scene_id=scene.scene_id,
        frame_ids=[scene.frame_ids[i] for i in idx],
        image_paths=[scene.image_paths[i] for i in idx],
        poses=scene.poses[idx],
    )


def _extract_block_tokens(probe, images: torch.Tensor, run_forward=None) -> torch.Tensor:
    """(V, num_tokens, C): *every* token at `probe.block_module`'s output --
    prefix (register / camera / CLS) tokens included, unlike
    `dense_features.extract_dense_features_at_block`, which slices them off
    and reshapes the patch tokens into a spatial grid. Same forward-hook
    mechanism, and handles the same two possible block-output layouts (a
    joint `(1, V*num_tokens, C)` cross-view block, or a single-view block's
    own `(V, num_tokens, C)`) by deriving `num_tokens` from the captured
    tensor's element count.

    Needed because Poincar3's own camera token -- a learned per-frame token that
    passes through every cross-view attention block, and the most natural
    place for camera-pose information to live (`model.py`'s
    `return_camera_tokens` reads exactly this token) -- sits *before* the
    patch tokens, so the dense-features path can't see it at all."""
    v = images.shape[0]
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inputs: tuple, output) -> None:
        captured["out"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = probe.block_module.register_forward_hook(hook)
    try:
        (run_forward or probe.run_forward)(images)
    finally:
        handle.remove()

    out = captured["out"]
    channels = out.shape[-1]
    num_tokens = out.numel() // (v * channels)
    return out.reshape(v, num_tokens, channels).float()


def _global_token_index(model: nn.Module, num_prefix_tokens: int) -> int | None:
    """Which of the `num_prefix_tokens` non-patch tokens is this backbone's
    own per-frame "global summary" token, or `None` if it has none.

    Poincar3 lays its tokens out as `[register x num_register_tokens, camera,
    patches...]` (see `model.py`'s `patch_token_start` and its
    `camera_tokens = final_layer[:, :, num_register_tokens]`), so its camera
    token is at index `num_register_tokens`. Everything else with a prefix
    (DINOv3 and the other `EncoderOnlyBaseline`-wrapped ViTs) puts CLS first,
    then storage/register tokens -- index 0."""
    if num_prefix_tokens <= 0:
        return None
    if getattr(model, "camera_token", None) is not None:
        idx = getattr(model, "num_register_tokens", None)
        if idx is not None and 0 <= idx < num_prefix_tokens:
            return idx
    return 0


def _pool_tokens(
    tokens: torch.Tensor, num_patches: int, feature_source: str, global_index: int | None
) -> torch.Tensor:
    """(V, num_tokens, C) -> (V, C) (or (V, 2C) for the concat source)."""
    patch_mean = tokens[:, tokens.shape[1] - num_patches :, :].mean(dim=1)
    if feature_source == "patch_mean" or global_index is None:
        return patch_mean
    global_token = tokens[:, global_index, :]
    if feature_source == "global_token":
        return global_token
    return torch.cat([patch_mean, global_token], dim=-1)


def _window_indices(anchor: int, n: int, context_window: int, context_stride: int) -> tuple[list[int], int]:
    """Frame indices of the anchor's context window, and the anchor's own
    position within it. `context_stride > 1` spreads the window over a wider
    baseline (`anchor +/- k*context_stride`) instead of taking immediately
    adjacent frames -- adjacent frames in a dense capture are near-duplicate
    viewpoints, which give cross-view attention almost nothing new to work
    with. Indices are clipped to the scene, so windows near either end can
    repeat a frame (harmless -- just a redundant view in the cluster)."""
    half = context_window // 2
    offsets = [(k - half) * context_stride for k in range(context_window)]
    idxs = [min(max(anchor + o, 0), n - 1) for o in offsets]
    return idxs, offsets.index(0)


def _extract_features(
    model: nn.Module,
    images_u8: torch.Tensor,
    probe,
    patch_size: int,
    device: torch.device,
    context_window: int,
    context_stride: int,
    feature_source: str,
    feature_batch_size: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Returns (N,C) pooled features, one per frame of `images_u8`
    (N,3,H,W uint8, RGB, one scene's worth), read at whichever block `probe`
    points at and pooled per `feature_source`. See the module docstring for
    `context_window`/`context_stride`."""
    n = images_u8.shape[0]
    num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

    def pool(tokens: torch.Tensor) -> torch.Tensor:
        global_index = _global_token_index(model, tokens.shape[1] - num_patches)
        return _pool_tokens(tokens, num_patches, feature_source, global_index)

    if context_window <= 1:
        # Every frame its own independent forward pass (num_frames=1), no
        # cross-frame attention regardless of backbone -- batched
        # `feature_batch_size` at a time purely for throughput.
        feats = []
        for start in range(0, n, feature_batch_size):
            batch = images_u8[start : start + feature_batch_size].to(device).float() / 255.0
            tokens = _extract_block_tokens(probe, batch, run_forward=lambda imgs: model(imgs.unsqueeze(1), mask=None))
            feats.append(pool(tokens).cpu())
        return torch.cat(feats, dim=0)

    # Each frame's feature comes from a joint forward pass over its own
    # `context_window`-frame cluster -- one window at a time, keeping only the
    # anchor's own pooled feature. Slower than the independent path above (no
    # cross-anchor batching), but activates real cross-view attention for any
    # backbone that has it.
    feats = []
    for i in range(n):
        idxs, anchor_pos = _window_indices(i, n, context_window, context_stride)
        window = images_u8[idxs].to(device).float() / 255.0
        tokens = _extract_block_tokens(probe, window)  # probe's own run_forward: one joint cluster
        feats.append(pool(tokens)[anchor_pos].cpu())
    return torch.stack(feats, dim=0)


class SeeSE3Benchmark:
    @dataclass(frozen=True)
    class Cfg:
        # "scannetpp": ScanNet++ DSLR scenes, ~600+ frames/scene on the
        # held-out val split -- a dense photographic walkthrough rather than
        # continuous video, but dense enough for the long strides below
        # (M1's stride=21 needs >= 651 frames; 34% of val scenes qualify).
        # "scannet": dense ScanNet *video* scenes, closer in capture style to
        # the original paper but only ~163 frames/scene here, so stride=21 is
        # almost never computable (0.3% of scenes reach 651 frames) and M4's
        # per-scene R^2 fits are far less reliable for want of pairs.
        dataset: Literal["scannet", "scannetpp"] = "scannetpp"
        # None -> "data/scannet/scans/scans_train" or
        # "data/scannet++/data_download/scannetpp/data", depending on `dataset`.
        data_root: str | None = None
        # Keep only each room's primary scan (sceneXXXX_00) -- see
        # `scannet.list_scenes`. Ignored for `dataset="scannetpp"`.
        primary_scan_only: bool = True
        # Which ScanNet++ split to draw scenes from. Pretraining only ever
        # uses "train", so "val" (the default) and "test" are genuinely held
        # out. Ignored for `dataset="scannet"`.
        scannetpp_split: Literal["train", "val", "test"] = "val"
        max_scenes: int | None = 20
        # Evenly subsample each scene to at most this many frames before
        # anything else -- caps compute for `context_window > 1` (which
        # forwards one window per frame, not batched) on scenes with
        # thousands of frames. `None` (default) uses every valid frame.
        # Changes what a given stride means (subsampling widens the native
        # gap between consecutive frames), so keep it fixed within one sweep.
        max_frames_per_scene: int | None = None
        seed: int = 42
        image_size: tuple[int, int] = (480, 640)

        metric: Literal["m1", "m4", "both"] = "both"

        # Metric M1: mutual k-nn topological alignment (Appendix A.1).
        m1_num_frames: int = 256
        # Short vs. long stride, as in the SeeSE3 paper's Fig. 2: short
        # probes pixel-level overlap, long probes genuine spatial awareness.
        # Stride 21 needs >= 2*m1_min_temporal_gap + m1_k + 1 = 31 anchors,
        # i.e. >= 651 frames. Drop it to e.g. (1, 5) for `dataset="scannet"`,
        # whose scenes are too short.
        m1_strides: tuple[int, ...] = (1, 21)
        m1_k: int = 10
        m1_min_temporal_gap: int = 10

        # Metric M4: the Poincare Adapter (Sec. 4 / Appendix A.2), swept
        # across frame strides as in the paper's Fig. 4.
        m4_strides: tuple[int, ...] = (2, 4, 8, 16, 20, 24, 32, 40, 60)
        m4_train_frac: float = 0.8
        # Skip a (scene, stride) combo if either split has fewer pairs than
        # this -- avoids fitting/reporting R^2 on a handful of pairs.
        m4_min_pairs: int = 20
        m4_hidden_dim: int = 64
        m4_geo_dim: int = 20
        # Appendix A.2's base protocol. Appendix A.8's later-tuned 200
        # epochs overfits badly on per-scene pair counts this small (every
        # stride's raw R^2 mean gets *more* negative), so 10 it is.
        m4_epochs: int = 10
        m4_batch_size: int = 512
        m4_lr: float = 1e-3
        m4_weight_decay: float = 1e-2
        m4_grad_clip: float = 1.0
        # How many independent adapter fits (different inits) per
        # (scene, stride). M4 is seed-sensitive -- the paper averages 15
        # seeds for its own ablation numbers -- and a single unseeded fit is
        # not reproducible. Nearly free: the features are already in memory,
        # and extraction is ~99.8% of a multi-view run's wall clock, so 10
        # seeds cost ~1% more. Every reported statistic is aggregated within
        # each seed and then summarized as mean +/- std across seeds.
        m4_num_seeds: int = 10
        m4_seed_base: int = 0

        # Which decoder block to read dense features from -- at most one of
        # these two may be set; `None`/`None` reads each backbone's own
        # default (last) block. See `poincar3.benchmarks.dense_features`.
        feature_block_index: int | None = None
        block_fraction: float | None = None

        # See the module docstring: >1 activates genuine cross-view
        # attention (a joint forward over this many native frames per
        # anchor) instead of the paper-faithful independent-frame default.
        context_window: int = 1
        # Spacing between the context window's frames. 1 (default) takes
        # immediately adjacent frames, which in a dense capture are
        # near-duplicate viewpoints -- little new information for cross-view
        # attention to exploit. >1 spreads the same number of frames over a
        # wider baseline (`anchor +/- k*context_stride`), which is where a
        # genuinely multi-view model's cross-view reasoning should actually
        # pay off. Only meaningful when `context_window > 1`.
        context_stride: int = 1
        # Which token to read per frame. "patch_mean" (default) mean-pools
        # the patch tokens, the paper's own fallback for encoders without a
        # CLS token. "global_token" instead reads the backbone's own
        # per-frame global summary token -- Poincar3's dedicated *camera token*
        # (the one `model.py`'s `return_camera_tokens` exposes, which passes
        # through every cross-view attention block and is the natural home
        # for camera-pose information), or a plain ViT's CLS token -- which
        # is what the paper itself uses whenever a model has one (Appendix
        # A.1: "for encoders that contain a CLS token, we take features from
        # that token"). "patch_mean+global_token" concatenates both.
        # Falls back to patch-mean for any backbone with no prefix tokens.
        feature_source: Literal["patch_mean", "global_token", "patch_mean+global_token"] = "patch_mean"
        feature_batch_size: int = 32

        def __post_init__(self) -> None:
            assert self.feature_block_index is None or self.block_fraction is None, (
                "set at most one of feature_block_index/block_fraction"
            )

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg

    @torch.no_grad()
    def benchmark(self, model: nn.Module, step: int | None = None) -> dict[str, float]:
        del step  # unused -- `experiments/eval.py`'s `_report` handles wandb logging generically.
        cfg = self.cfg
        model.eval()
        device = next(model.parameters()).device

        # Some backbones expose their readable blocks only through
        # `attention_probe`, not `feature_probe` -- MuM v1 in decoder mode
        # defines a `feature_probe` method that deliberately returns `None`
        # ("attention_probe's cross-view blocks already serve both
        # purposes", see `baselines/mum/adapter.py`), and
        # `dense_features.get_feature_probe` only falls back to
        # `attention_probe` when the *method* is missing, not when it
        # *returns* None -- so that backbone otherwise has no probe here at
        # all. Fall back explicitly, for both the fraction resolution and
        # the probe itself, rather than patching the shared helper that
        # other benchmarks depend on.
        use_attention_blocks = get_feature_probe(model, None) is None
        block_index = (
            resolve_block_fraction(model, cfg.block_fraction, use_attention=use_attention_blocks)
            if cfg.block_fraction is not None
            else cfg.feature_block_index
        )
        probe = get_feature_probe(model, block_index) or get_attention_probe(model, block_index)
        if probe is None:
            raise RuntimeError(f"{type(model).__name__} has no feature probe at block_index={block_index!r}")
        patch_size = model.patch_size

        if cfg.dataset == "scannet":
            data_root = cfg.data_root or "data/scannet/scans/scans_train"
            scene_ids = _list_scannet_scenes(data_root, cfg.primary_scan_only)
            load_one_scene = _load_scannet_scene
        else:
            data_root = cfg.data_root or "data/scannet++/data_download/scannetpp/data"
            scene_ids = _list_scannetpp_scenes(data_root, cfg.scannetpp_split)
            load_one_scene = _load_scannetpp_scene
        random.Random(cfg.seed).shuffle(scene_ids)
        if cfg.max_scenes is not None:
            scene_ids = scene_ids[: cfg.max_scenes]

        # M4 accumulators are `{stride: [[per-scene value] for each seed]}` --
        # the seed axis is kept separate (rather than averaged per scene) so
        # the summary can aggregate over scenes *within* each seed and report
        # the spread of the headline number itself across seeds, which is
        # what "would this table look the same if I reran it" actually asks.
        m1_scores: dict[int, list[float]] = {s: [] for s in cfg.m1_strides}
        seeds = range(cfg.m4_num_seeds)
        m4_r2: dict[int, list[list[float]]] = {s: [[] for _ in seeds] for s in cfg.m4_strides}
        m4_r2_trans: dict[int, list[list[float]]] = {s: [[] for _ in seeds] for s in cfg.m4_strides}
        m4_r2_rot: dict[int, list[list[float]]] = {s: [[] for _ in seeds] for s in cfg.m4_strides}
        m4_mse: dict[int, list[list[float]]] = {s: [[] for _ in seeds] for s in cfg.m4_strides}

        for scene_id in tqdm(scene_ids, desc="see_se3"):
            scene = load_one_scene(data_root, scene_id)
            if scene is None:
                print(f"see_se3: skipping {scene_id} (too few valid-pose frames)")
                continue
            scene = _subsample_scene(scene, cfg.max_frames_per_scene)
            n = len(scene.frame_ids)

            images_u8 = load_scene_images(scene, cfg.image_size)
            features = _extract_features(
                model,
                images_u8,
                probe,
                patch_size,
                device,
                cfg.context_window,
                cfg.context_stride,
                cfg.feature_source,
                cfg.feature_batch_size,
                cfg.image_size,
            )
            del images_u8

            if cfg.metric in ("m1", "both"):
                for stride in cfg.m1_strides:
                    anchor_idx = np.arange(0, n, stride)[: cfg.m1_num_frames]
                    feats = features[anchor_idx]
                    poses9 = torch.from_numpy(pose9d(scene.poses[anchor_idx])).float()
                    score = _mutual_knn(feats, poses9, cfg.m1_k, cfg.m1_min_temporal_gap)
                    if score is None:
                        print(f"see_se3: {scene_id} stride={stride}: too few frames ({len(anchor_idx)}) for M1, skipping")
                        continue
                    m1_scores[stride].append(score)

            if cfg.metric in ("m4", "both"):
                n_train = int(cfg.m4_train_frac * n)
                for stride in cfg.m4_strides:
                    train_pairs = [(i, i + stride) for i in range(n_train - stride)] if n_train > stride else []
                    test_pairs = [(i, i + stride) for i in range(n_train, n - stride)] if n - stride > n_train else []
                    if len(train_pairs) < cfg.m4_min_pairs or len(test_pairs) < cfg.m4_min_pairs:
                        print(
                            f"see_se3: {scene_id} stride={stride}: too few pairs "
                            f"(train={len(train_pairs)}, test={len(test_pairs)}) for M4, skipping"
                        )
                        continue
                    train_pairs_np = np.array(train_pairs)
                    test_pairs_np = np.array(test_pairs)

                    y_train = torch.from_numpy(relative_pose_targets(scene.poses, train_pairs_np)).float().to(device)
                    y_test = torch.from_numpy(relative_pose_targets(scene.poses, test_pairs_np)).float().to(device)
                    z1_train = features[train_pairs_np[:, 0]].to(device)
                    z2_train = features[train_pairs_np[:, 1]].to(device)
                    z1_test = features[test_pairs_np[:, 0]].to(device)
                    z2_test = features[test_pairs_np[:, 1]].to(device)

                    # Refit `m4_num_seeds` times from different inits. The
                    # features are already extracted and in memory, and
                    # extraction is ~99.8% of a multi-view run's wall clock
                    # (measured: 1:27:00 total vs ~15s for the whole fitting
                    # stage), so extra seeds are nearly free here -- and M4
                    # is seed-sensitive enough that a single fit isn't a
                    # reproducible number (see `fit_poincare_adapter`).
                    for seed_idx in range(cfg.m4_num_seeds):
                        with torch.enable_grad():
                            result = fit_poincare_adapter(
                                z1_train, z2_train, y_train, z1_test, z2_test, y_test,
                                hidden_dim=cfg.m4_hidden_dim, geo_dim=cfg.m4_geo_dim, epochs=cfg.m4_epochs,
                                batch_size=cfg.m4_batch_size, lr=cfg.m4_lr, weight_decay=cfg.m4_weight_decay,
                                grad_clip=cfg.m4_grad_clip, seed=cfg.m4_seed_base + seed_idx,
                            )
                        m4_r2[stride][seed_idx].append(result.r2)
                        m4_r2_trans[stride][seed_idx].append(result.r2_trans)
                        m4_r2_rot[stride][seed_idx].append(result.r2_rot)
                        m4_mse[stride][seed_idx].append(result.mse)

        # Different strides get support from very different numbers of
        # scenes (a longer stride needs a longer scene to produce enough
        # anchors/pairs at all) -- reported per stride so a result isn't
        # mistaken for one drawn from the full `scene_ids` sample when it's
        # actually a biased-towards-longer-scenes subset of it.
        summary: dict[str, float] = {}
        if cfg.metric in ("m1", "both"):
            for stride, scores in m1_scores.items():
                summary[f"m1_stride{stride}_num_scenes"] = float(len(scores))
                if not scores:
                    print(f"see_se3: no scene produced a valid M1 score at stride={stride}")
                    continue
                summary[f"m1_stride{stride}_mutual_knn"] = float(np.mean(scores))
        if cfg.metric in ("m4", "both"):
            summary["m4_num_seeds"] = float(cfg.m4_num_seeds)
            for stride in cfg.m4_strides:
                summary[f"m4_stride{stride}_num_scenes"] = float(len(m4_r2[stride][0]))
                if not m4_r2[stride][0]:
                    print(f"see_se3: no scene produced a valid M4 fit at stride={stride}")
                    continue
                # (num_seeds, num_scenes). R^2 is unbounded below, so the raw
                # per-scene mean is a misleading headline number: a handful of
                # catastrophically overfit scenes (tiny per-scene pair counts,
                # an unlucky adapter init) can drag it to e.g. -20 even when
                # most scenes fit fine. The paper's own Fig. 5 / Table 2
                # instead report a "Clipped Mean R^2 = mean(max(0, R^2))" and
                # the fraction-of-scenes-positive/above-0.3 as the headline
                # success-rate stats -- reported here too, alongside the raw
                # mean (kept only for reference/debugging).
                #
                # Each statistic is computed *per seed* (aggregating over
                # scenes), then summarized as mean/std over seeds -- so the
                # `_std` entries say how much the reported number itself
                # would move on a rerun, which is the quantity a table's
                # error bars should show.
                r2 = np.array(m4_r2[stride])  # (seeds, scenes)

                def _seedwise(values: np.ndarray, key: str) -> None:
                    summary[f"m4_stride{stride}_{key}"] = float(values.mean())
                    summary[f"m4_stride{stride}_{key}_std"] = float(values.std())

                _seedwise(r2.mean(axis=1), "r2_mean")
                _seedwise(np.clip(r2, 0, None).mean(axis=1), "r2_clipped_mean")
                # Best *single scene* at this stride -- the paper's own
                # "Max R^2" column (Table 2), which is a max over scenes,
                # not over strides. Reported so our tables can use the same
                # column the paper does rather than a similarly-named but
                # different quantity.
                _seedwise(r2.max(axis=1), "r2_max_scene")
                _seedwise((r2 > 0).mean(axis=1), "r2_frac_positive")
                _seedwise((r2 > 0.3).mean(axis=1), "r2_frac_above_0.3")
                _seedwise(np.array(m4_r2_trans[stride]).mean(axis=1), "r2_trans_mean")
                _seedwise(np.array(m4_r2_rot[stride]).mean(axis=1), "r2_rot_mean")
                _seedwise(np.array(m4_mse[stride]).mean(axis=1), "mse")
        return summary
