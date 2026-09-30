from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from poincar3.benchmarks.dense_features import (
    extract_dense_features,
    extract_dense_features_at_block,
    extract_global_attention_qk,
    extract_global_attention_qk_multilayer,
    get_all_attention_probes,
    get_attention_probe,
    get_feature_probe,
    resolve_block_fraction,
)
from poincar3.distrib import is_main_process

from .megadepth import MegaDepth
from .navi import NAVI
from .scannet import ScanNet


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_knn(query: torch.Tensor, target: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Brute-force k-NN (L2 distance). query: (Nq, D), target: (Nt, D)."""
    query = query.contiguous()
    target = target.contiguous()
    query_sq = torch.sum(query**2, dim=1, keepdim=True)
    target_sq = torch.sum(target**2, dim=1, keepdim=True).t()
    cross_term = -2 * torch.matmul(query, target.t())
    dist_sq = query_sq + cross_term + target_sq
    dists_sq, indices = torch.topk(dist_sq, k, dim=-1, largest=False)
    return torch.sqrt(dists_sq.clamp(min=0)), indices


def transform_points_Rt(points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    R = viewpoint[..., :3, :3]
    t = viewpoint[..., None, :3, 3]
    if inverse:  # viewpoint is world-to-camera
        return (points - t) @ R
    return points @ R.transpose(-2, -1) + t  # viewpoint is camera-to-world


def project_3dto2d(xyz: torch.Tensor, K_mat: torch.Tensor) -> torch.Tensor:
    uvd = xyz @ K_mat.transpose(-1, -2)
    return uvd[:, :2] / uvd[:, 2:3].clamp(min=1e-9)


def unproject_2d_to_3d(uv_points: torch.Tensor, depth_map: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    N = uv_points.shape[0]
    if uv_points.numel() == 0:
        return torch.empty((0, 3), device=uv_points.device)
    K_inv = torch.inverse(intrinsics)
    uv_h = F.pad(uv_points, (0, 1), "constant", 1.0)
    H, W = depth_map.shape
    norm_u = 2.0 * uv_points[:, 0] / (W - 1) - 1.0
    norm_v = 2.0 * uv_points[:, 1] / (H - 1) - 1.0
    grid = torch.stack([norm_u, norm_v], dim=1).view(1, 1, -1, 2)
    sampled_depths = F.grid_sample(depth_map.unsqueeze(0).unsqueeze(0), grid, align_corners=True)
    depth_values = sampled_depths.view(N)
    xyz_points = (K_inv @ uv_h.T).T * depth_values[:, None]
    return xyz_points


def is_visible(u_proj, v_proj, depth_proj, target_depth_map: torch.Tensor, W: int, H: int) -> bool:
    if u_proj is None or v_proj is None or depth_proj is None:
        return False
    if not (depth_proj > 0 and 0 <= u_proj < W and 0 <= v_proj < H):
        return False
    u_int, v_int = int(round(u_proj)), int(round(v_proj))
    if not (0 <= u_int < W and 0 <= v_int < H):
        return False
    depth_from_map = target_depth_map[v_int, u_int].item()
    epsilon = 0.05
    return not (depth_from_map > 0 and depth_proj > (depth_from_map + epsilon))


def project_point(
    u: int, v: int, depth_source: torch.Tensor, intrinsics_source: torch.Tensor, Rt_source: torch.Tensor,
    intrinsics_target: torch.Tensor, Rt_target: torch.Tensor, source_inverse: bool, target_inverse: bool,
):
    """Projects a point from the source view into the target view. Returns
    `(u, v, depth)` in the target, or `(None, None, None)` if the source has no depth there."""
    depth_value = depth_source[v, u].item()
    if depth_value <= 0:
        return None, None, None
    u_center, v_center = u + 0.5, v + 0.5
    uv_one = torch.tensor([u_center, v_center, 1.0], dtype=torch.float32, device=depth_source.device)
    K_inv = torch.inverse(intrinsics_source)
    point_cam_source = ((K_inv @ uv_one) * depth_value).unsqueeze(0)
    point_world = transform_points_Rt(point_cam_source, Rt_source, inverse=source_inverse)
    point_cam_target = transform_points_Rt(point_world, Rt_target, inverse=target_inverse)
    depth_in_target = point_cam_target[0, 2].item()
    uv_target = project_3dto2d(point_cam_target, intrinsics_target).squeeze()
    return uv_target[0].item() - 0.5, uv_target[1].item() - 0.5, depth_in_target


def calculate_gt_tracks(
    start_points: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    Rts: torch.Tensor,
    source_inverse: bool,
    target_inverse: bool,
    invalidate_if_never_visible: bool = True,
) -> torch.Tensor:
    """Reprojects `start_points` (in view 0) into every other view, marking
    visibility. Returns `(num_tracks, num_views, 3)`: `(u, v, visible>0)`."""
    num_tracks, _ = start_points.shape
    V, _, H, W = depths.shape
    gt_tracks = torch.full((num_tracks, V, 3), -1.0, device=start_points.device)
    gt_tracks[:, 0, 0:2] = start_points
    gt_tracks[:, 0, 2] = 1.0

    source_depth, source_intrinsics, source_Rt = depths[0, 0], intrinsics[0], Rts[0]
    for i in range(num_tracks):
        start_u, start_v = int(start_points[i, 0]), int(start_points[i, 1])
        if not (0 <= start_u < W and 0 <= start_v < H):
            gt_tracks[i, :, 2] = -1.0
            continue
        for v_idx in range(1, V):
            u_proj, v_proj, depth_proj = project_point(
                start_u, start_v, source_depth, source_intrinsics, source_Rt,
                intrinsics[v_idx], Rts[v_idx], source_inverse, target_inverse,
            )
            if is_visible(u_proj, v_proj, depth_proj, depths[v_idx, 0], W, H):
                gt_tracks[i, v_idx, 0] = u_proj
                gt_tracks[i, v_idx, 1] = v_proj
                gt_tracks[i, v_idx, 2] = 1.0
        if invalidate_if_never_visible and not torch.any(gt_tracks[i, 1:, 2] > 0):
            gt_tracks[i, 0, 2] = -1.0
    return gt_tracks


def find_correspondences_feature_based(
    uv_s: torch.Tensor, feat_s: torch.Tensor, feat_t: torch.Tensor
) -> torch.Tensor:
    """1-NN feature match of source points (in `feat_s`'s grid) into `feat_t`'s grid."""
    _, C, Hf, Wf = feat_s.shape
    N = uv_s.shape[0]
    norm_u = 2.0 * uv_s[:, 0] / (Wf - 1) - 1.0
    norm_v = 2.0 * uv_s[:, 1] / (Hf - 1) - 1.0
    grid_query = torch.stack([norm_u, norm_v], dim=1).view(1, 1, -1, 2)
    sampled_feats = F.grid_sample(feat_s, grid_query, align_corners=False)
    query_feats = sampled_feats.permute(0, 3, 2, 1).reshape(N, C)
    target_feats = feat_t.reshape(C, -1).T
    _, nn_indices = torch_knn(query_feats, target_feats, k=1)
    matched_indices = nn_indices.squeeze(-1)
    matched_v = matched_indices // Wf
    matched_u = matched_indices % Wf
    return torch.stack([matched_u, matched_v], dim=1).float()


def find_correspondences_attention_based(
    uv_s: torch.Tensor, q_source: torch.Tensor, k_all: torch.Tensor, scale: float, target_start: int, Hf: int, Wf: int
) -> torch.Tensor:
    """Attention-based analogue of `find_correspondences_feature_based`: bilinearly
    interpolates the source view's *query* vectors (`q_source`, dense over its
    patch grid) at `uv_s`'s sub-patch positions, then reads off the softmax
    cross-view attention (over `k_all`, every view's keys) restricted to the
    target view's patch-token slice (`[target_start, target_start + Hf*Wf)`),
    averaged over heads -- argmax is the matched patch, same `(u, v)`
    feature-grid convention as the feature-based match. Softmax is taken over
    *all* of `k_all` (not just the target slice) so the result is a real
    attention distribution, matching what the model itself computes."""
    h, Hf_s, Wf_s, d = q_source.shape
    N = uv_s.shape[0]
    norm_u = 2.0 * uv_s[:, 0] / (Wf_s - 1) - 1.0
    norm_v = 2.0 * uv_s[:, 1] / (Hf_s - 1) - 1.0
    grid_query = torch.stack([norm_u, norm_v], dim=1).view(1, 1, -1, 2)
    q_flat = q_source.permute(0, 3, 1, 2).reshape(1, h * d, Hf_s, Wf_s)
    q_sampled = F.grid_sample(q_flat, grid_query, align_corners=False).view(h, d, N).permute(2, 0, 1)  # (N,h,d)

    logits = torch.einsum("nhd,hkd->nhk", q_sampled, k_all) * scale  # (N, h, N_total)
    probs = logits.softmax(dim=-1)  # softmax over every key, per head -- a real attention distribution
    target_probs = probs[:, :, target_start : target_start + Hf * Wf].mean(dim=1)  # (N, Hf*Wf), avg over heads
    matched_indices = target_probs.argmax(dim=-1)
    matched_v = matched_indices // Wf
    matched_u = matched_indices % Wf
    return torch.stack([matched_u, matched_v], dim=1).float()


def _dense_attention_multilayer_scores(
    q_alls: list[torch.Tensor], k_alls: list[torch.Tensor], scales: list[float], source_start: int, target_start: int, Hf: int, Wf: int
) -> torch.Tensor:
    """Dense source-patch x target-patch raw (pre-softmax) attention score
    matrix for one view pair, bidirectionally combined and averaged over
    every cross-view block.
    Returns `(Hf*Wf, Hf*Wf)`, row = source patch, column = target patch."""
    combined = None
    for q_all, k_all, scale in zip(q_alls, k_alls, scales):
        q_source, q_target = q_all[:, source_start : source_start + Hf * Wf], q_all[:, target_start : target_start + Hf * Wf]
        k_source, k_target = k_all[:, source_start : source_start + Hf * Wf], k_all[:, target_start : target_start + Hf * Wf]
        s_to_t = torch.einsum("hnd,hmd->nmh", q_source, k_target).mean(dim=-1) * scale  # (Ns, Nt), avg over heads
        t_to_s = torch.einsum("hnd,hmd->nmh", q_target, k_source).mean(dim=-1) * scale  # (Nt, Ns), avg over heads
        layer_combined = (s_to_t + t_to_s.transpose(0, 1)) / 2
        combined = layer_combined if combined is None else combined + layer_combined
    return combined / len(q_alls)  # avg over blocks


def find_correspondences_attention_multilayer_based(
    uv_s: torch.Tensor,
    q_alls: list[torch.Tensor],
    k_alls: list[torch.Tensor],
    scales: list[float],
    source_start: int,
    target_start: int,
    Hf: int,
    Wf: int,
    H: int,
    W: int,
    temperature: float,
) -> torch.Tensor:
    """Multi-layer, bidirectional analogue of `find_correspondences_attention_based`
    -- a soft-argmax over the attention scores,
    generalized via `get_all_attention_probes` to any backbone with multiple
    cross-view blocks. Builds the dense score matrix once (no query
    interpolation), soft-argmaxes every row into a `(Hf, Wf, 2)` target-grid
    field, then bilinearly upsamples that field to pixel resolution (`H, W`)
    and samples it at `uv_s`'s exact positions -- sub-pixel precision comes
    from interpolating the output field, not the input query. Returns
    pixel-space `(u, v)`, unlike the other correspondence functions' patch-grid
    space."""
    scores = _dense_attention_multilayer_scores(q_alls, k_alls, scales, source_start, target_start, Hf, Wf)
    weights = (scores / temperature).softmax(dim=-1)  # (Ns, Nt) soft-argmax over every source patch's row
    grid_u = torch.arange(Wf, device=uv_s.device, dtype=weights.dtype)
    grid_v = torch.arange(Hf, device=uv_s.device, dtype=weights.dtype)
    vv, uu = torch.meshgrid(grid_v, grid_u, indexing="ij")
    matched_u = (weights * uu.reshape(1, -1)).sum(dim=-1)
    matched_v = (weights * vv.reshape(1, -1)).sum(dim=-1)
    flow_field = torch.stack([matched_u, matched_v], dim=-1).view(1, Hf, Wf, 2).permute(0, 3, 1, 2)  # (1,2,Hf,Wf)

    field_px = F.interpolate(flow_field, size=(H, W), mode="bilinear", align_corners=False)[0]
    field_px = field_px * torch.tensor([W / Wf, H / Hf], device=uv_s.device).view(2, 1, 1)  # patch coords -> pixel coords

    norm_u = 2.0 * uv_s[:, 0] / (W - 1) - 1.0
    norm_v = 2.0 * uv_s[:, 1] / (H - 1) - 1.0
    grid_query = torch.stack([norm_u, norm_v], dim=1).view(1, 1, -1, 2)
    sampled = F.grid_sample(field_px.unsqueeze(0), grid_query, align_corners=False)  # (1, 2, 1, N)
    return sampled.view(2, -1).T  # (N, 2), pixel-space (u, v) in the target view


def compute_errors_for_sample(
    pred_tracks: torch.Tensor, gt_tracks: torch.Tensor, depths: torch.Tensor, intrinsics: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-visible-point 2D pixel error and 3D metric error (after unprojection)."""
    _, num_views, _ = gt_tracks.shape
    all_2d_errors, all_3d_errors = [], []
    for v_idx in range(num_views):
        visibility_mask = gt_tracks[:, v_idx, 2] > 0
        if not visibility_mask.any():
            continue
        visible_gt_uv = gt_tracks[visibility_mask, v_idx, :2]
        visible_pred_uv = pred_tracks[visibility_mask, v_idx, :]
        all_2d_errors.append(torch.linalg.norm(visible_pred_uv - visible_gt_uv, dim=1))
        gt_xyz = unproject_2d_to_3d(visible_gt_uv, depths[v_idx, 0], intrinsics[v_idx])
        pred_xyz = unproject_2d_to_3d(visible_pred_uv, depths[v_idx, 0], intrinsics[v_idx])
        all_3d_errors.append(torch.linalg.norm(pred_xyz - gt_xyz, dim=1))
    if not all_2d_errors:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(all_2d_errors), torch.cat(all_3d_errors)


def _relative_pose_gt(Rt_source: torch.Tensor, Rt_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """GT relative pose mapping a point in the source view's camera frame into
    the target view's: `X_target = R_rel @ X_source + t_rel`."""
    R_s, t_s = Rt_source[:3, :3], Rt_source[:3, 3]
    R_t, t_t = Rt_target[:3, :3], Rt_target[:3, 3]
    R_rel = R_t @ R_s.T
    return R_rel, t_t - R_rel @ t_s


def estimate_relative_pose_error(
    uv_source: torch.Tensor,
    uv_target: torch.Tensor,
    K_source: torch.Tensor,
    K_target: torch.Tensor,
    Rt_source: torch.Tensor,
    Rt_target: torch.Tensor,
    min_inliers: int = 8,
) -> tuple[float, float] | None:
    """Estimates relative pose from feature correspondences (essential matrix +
    RANSAC) and scores it against the GT relative pose. Returns `(rotation_error_deg,
    translation_angular_error_deg)`, or `None` if estimation was degenerate/failed.
    `translation_angular_error_deg` is `nan` if the GT baseline is ~zero (pure
    rotation, translation direction undefined)."""
    ones = torch.ones((uv_source.shape[0], 1), device=uv_source.device, dtype=uv_source.dtype)
    rays_source = torch.cat([uv_source, ones], dim=1) @ torch.inverse(K_source).T
    rays_target = torch.cat([uv_target, ones], dim=1) @ torch.inverse(K_target).T
    pts1 = (rays_source[:, :2] / rays_source[:, 2:3]).cpu().double().numpy()
    pts2 = (rays_target[:, :2] / rays_target[:, 2:3]).cpu().double().numpy()

    # RANSAC threshold in normalized-camera-ray units: ~1px reprojection error
    # rescaled by focal length, since points are pre-normalized by each view's
    # own (possibly different, e.g. MegaDepth) intrinsics.
    focal_avg = (K_source[0, 0].item() + K_target[0, 0].item()) / 2
    E, mask = cv2.findEssentialMat(
        pts1, pts2, cameraMatrix=np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.0 / focal_avg
    )
    if E is None or mask is None or mask.sum() < min_inliers:
        return None
    _, R_est, t_est, _ = cv2.recoverPose(E, pts1, pts2, cameraMatrix=np.eye(3), mask=mask)

    R_rel, t_rel = _relative_pose_gt(Rt_source, Rt_target)
    trace = np.clip((np.trace(R_rel.cpu().double().numpy().T @ R_est) - 1) / 2, -1.0, 1.0)
    rot_err_deg = float(np.degrees(np.arccos(trace)))

    t_gt_norm = torch.linalg.norm(t_rel).item()
    if t_gt_norm < 1e-6:
        return rot_err_deg, float("nan")
    t_gt_dir = (t_rel / t_gt_norm).cpu().double().numpy()
    cos_sim = np.clip(np.dot(t_gt_dir, t_est.flatten()), -1.0, 1.0)
    return rot_err_deg, float(np.degrees(np.arccos(cos_sim)))


def _summarize_errors(
    errors_2d: list[torch.Tensor], errors_3d: list[torch.Tensor], metric_scale: bool = True
) -> dict[str, float]:
    """`metric_scale=False` drops the `*cm` keys -- e.g. MegaDepth's depth/poses
    come from unscaled SfM, so a "cm" error there doesn't mean actual centimeters."""
    if not errors_2d:
        return {}
    e2d = torch.cat(errors_2d)
    summary = {
        "ate_2d_px": e2d.mean().item(),
    }
    for th in (1, 2, 5, 10, 25, 50):
        summary[f"acc_{th}px"] = (e2d < th).float().mean().item() * 100
    if metric_scale:
        e3d = torch.cat(errors_3d)
        summary["ate_3d_cm"] = e3d.mean().item() * 100
        for th_cm, th_m in ((1, 0.01), (2, 0.02), (5, 0.05), (10, 0.10)):
            summary[f"acc_{th_cm}cm"] = (e3d < th_m).float().mean().item() * 100
    return summary


def _get_boustrophedon_grid_pos(v_idx, u_coord, v_coord, H, W, nrow, padding):
    grid_row = v_idx // nrow
    v_col_in_row = v_idx % nrow
    grid_col = (nrow - 1) - v_col_in_row if grid_row % 2 == 1 else v_col_in_row
    return int(u_coord + grid_col * (W + padding)), int(v_coord + grid_row * (H + padding))


def visualize_comparison_tracks(
    images: list[torch.Tensor], pred_tracks: torch.Tensor, gt_tracks: torch.Tensor, save_path: str, nrow: int = 4
) -> None:
    import os

    images_unnorm = [img.cpu().clamp(0, 1) for img in images]
    num_views = len(images_unnorm)
    images_reordered = []
    for i in range(0, num_views, nrow):
        chunk = images_unnorm[i : i + nrow]
        images_reordered.extend(chunk[::-1] if (i // nrow) % 2 == 1 else chunk)

    padding = 4
    grid_img_tensor = torchvision.utils.make_grid(images_reordered, nrow=nrow, padding=padding)
    vis_img = cv2.cvtColor(
        (grid_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )

    _, H, W = images[0].shape
    pred_color, gt_color = (0, 0, 255), (0, 255, 0)

    for i in range(pred_tracks.shape[0]):
        for v in range(num_views):
            if gt_tracks[i, v, 2].item() <= 0:
                continue
            u, v_coord = pred_tracks[i, v].cpu().numpy()
            if u < 0 or v_coord < 0:
                continue
            grid_u, grid_v = _get_boustrophedon_grid_pos(v, u, v_coord, H, W, nrow, padding)
            cv2.circle(vis_img, (grid_u, grid_v), 5, pred_color, -1)

        for v in range(num_views):
            if gt_tracks[i, v, 2].item() <= 0:
                continue
            u, v_coord = gt_tracks[i, v, :2].cpu().numpy()
            grid_u, grid_v = _get_boustrophedon_grid_pos(v, u, v_coord, H, W, nrow, padding)
            cv2.circle(vis_img, (grid_u, grid_v), 6, gt_color, 2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis_img)


class MvConsistencyBenchmark:
    """Multi-view feature-correspondence consistency, evaluated on NAVI or
    ScanNet clusters -- both give (image, depth, intrinsics, Rt, mask) per
    view in the same shapes, so the rest of this class doesn't care which."""

    @dataclass(frozen=True)
    class Cfg:
        dataset: Literal["navi", "scannet", "megadepth"] = "navi"
        # None -> "data/navi_v1", "data/scannet_test_1500", or "data/megadepth"
        # depending on `dataset`.
        data_root: str | None = None
        num_views: int = 8
        num_corr: int = 100
        # Caps how many dataset sequences get evaluated. The default is
        # deliberately small: this also runs after every training-time eval
        # call, where the full split is far more than needed. `None` evaluates
        # the entire split, which is what a reported number should use.
        max_sequences: int | None = 20
        seed: int = 42
        visualize: bool = False
        # See `extract_dense_features`: bypass the multi-view decoder and score
        # the encoder's single-view features alone. Off by default (training
        # wants the full model's multi-view features); on for baseline checks.
        # Only applies to `correspondence_method="feature"` -- "attention" has
        # no single-view analogue, since it reads the decoder's own cross-view
        # attention.
        single_view: bool = False
        # "feature": 1-NN cosine match in the final decoder layer's dense
        # features. "attention": read the model's own softmax cross-view
        # attention at the last "global" inter-frame block -- does its actual
        # attention agree with correspondence, rather than its output feature
        # similarity? "attention_multilayer" averages that over every
        # cross-view block instead of just the last one.
        correspondence_method: Literal["feature", "attention", "attention_multilayer"] = "feature"
        # Only used by "attention". `None` probes the model's *last* cross-view
        # block, i.e. what it actually uses for its own output. Set an index to
        # probe a specific block instead; see each backbone's
        # `attention_probe` for which indices are valid. No effect for
        # "attention_multilayer", which always reads every block.
        attention_block_index: int | None = None
        # Only used by "attention_multilayer": soft-argmax temperature. Well
        # below 1 sharpens the match toward a hard argmax.
        attention_temperature: float = 0.0001
        # Only used by "feature". `None` reads dense features off the model's
        # final forward-pass output. Set an index to instead read raw
        # residual-stream tokens straight off a block, bypassing everything
        # after it -- the feature-method analogue of `attention_block_index`,
        # over the same index space. `single_view` has no effect when set.
        feature_block_index: int | None = None
        # Alternative to the two indices above: a fraction in `(0, 1]` of the
        # model's depth, resolved to the nearest *valid* block. Lets one
        # relative "how far into the network" axis compare backbones of very
        # different depths. `1.0` matches the `None` default. Mutually
        # exclusive with the corresponding absolute index.
        block_fraction: float | None = None
        # Also fit an essential matrix + RANSAC per view pair and report
        # rotation/translation error. Off by default: noisy on top of already
        # noisy correspondences, and the 2D/3D tracking metrics already capture
        # correspondence quality directly.
        estimate_relative_pose: bool = False

    def __init__(self, cfg: Cfg) -> None:
        assert not (cfg.correspondence_method != "feature" and cfg.single_view), (
            f"single_view has no effect with correspondence_method={cfg.correspondence_method!r} -- "
            "it bypasses the decoder attention is read from"
        )
        assert not (cfg.correspondence_method != "feature" and cfg.feature_block_index is not None), (
            f"feature_block_index has no effect with correspondence_method={cfg.correspondence_method!r} -- "
            "use attention_block_index instead"
        )
        assert not (cfg.single_view and cfg.feature_block_index is not None), (
            "single_view has no effect with feature_block_index set -- it already bypasses "
            "the multi-view decoder, reading a specific block does the same via a different path"
        )
        assert not (cfg.correspondence_method == "attention_multilayer" and cfg.attention_block_index is not None), (
            "attention_block_index has no effect with correspondence_method='attention_multilayer' -- "
            "it always reads every valid cross-view block"
        )
        assert not (cfg.correspondence_method == "attention_multilayer" and cfg.block_fraction is not None), (
            "block_fraction has no effect with correspondence_method='attention_multilayer' -- "
            "it always reads every valid cross-view block"
        )
        assert not (cfg.block_fraction is not None and cfg.attention_block_index is not None), (
            "set either block_fraction or attention_block_index, not both"
        )
        assert not (cfg.block_fraction is not None and cfg.feature_block_index is not None), (
            "set either block_fraction or feature_block_index, not both"
        )
        assert not (cfg.single_view and cfg.block_fraction is not None), (
            "single_view has no effect with block_fraction set -- it already bypasses "
            "the multi-view decoder, reading a specific block does the same via a different path"
        )
        assert cfg.block_fraction is None or 0 < cfg.block_fraction <= 1, (
            f"block_fraction={cfg.block_fraction} must be in (0, 1]"
        )
        self.cfg = cfg
        if cfg.dataset == "navi":
            # image_mean="None": Poincar3.forward already normalizes internally.
            self.dataset = NAVI(
                cfg.data_root or "data/navi_v1", split="valid", image_mean="None", num_views=cfg.num_views
            )
        elif cfg.dataset == "scannet":
            self.dataset = ScanNet(cfg.data_root or "data/scannet_test_1500", num_views=cfg.num_views)
        elif cfg.dataset == "megadepth":
            self.dataset = MegaDepth(cfg.data_root or "data/megadepth", num_views=cfg.num_views)
        else:
            raise ValueError(f"Unknown mv_consistency dataset: {cfg.dataset!r}")

    @torch.no_grad()
    def benchmark(self, model: nn.Module, step: int | None = None) -> dict[str, float]:
        model.eval()
        device = next(model.parameters()).device
        use_attention = self.cfg.correspondence_method == "attention"
        use_attention_multilayer = self.cfg.correspondence_method == "attention_multilayer"
        if use_attention_multilayer:
            probes = get_all_attention_probes(model)
            if not probes:
                raise RuntimeError(
                    "correspondence_method='attention_multilayer' needs `get_all_attention_probes(model)` to "
                    "return at least one probe -- this backbone has no genuine cross-view attention to read at "
                    "all (e.g. a single-view-only baseline like DINOv3)."
                )
        elif self.cfg.block_fraction is not None:
            block_index = resolve_block_fraction(model, self.cfg.block_fraction, use_attention)
        else:
            block_index = self.cfg.attention_block_index if use_attention else self.cfg.feature_block_index
        use_feature_block = (not use_attention) and (not use_attention_multilayer) and (block_index is not None)
        if use_attention:
            probe = get_attention_probe(model, block_index)
        elif use_feature_block:
            probe = get_feature_probe(model, block_index)
        else:
            probe = None
        if (use_attention or use_feature_block) and probe is None:
            probe_fn = "model.attention_probe()" if use_attention else "model.attention_probe()/model.feature_probe()"
            cfg_name = "attention_block_index" if use_attention else "feature_block_index"
            raise RuntimeError(
                f"correspondence_method={self.cfg.correspondence_method!r} needs `{probe_fn}` to return a "
                "probe target -- this backbone has no attention/features to read at the targeted block "
                "(e.g. the targeted block isn't a 'global'/cross-view block -- check "
                f"register_attention_block_indices, or mv_consistency.{cfg_name})."
            )

        all_2d_errors: list[torch.Tensor] = []
        all_3d_errors: list[torch.Tensor] = []
        all_rot_errors: list[float] = []
        all_trans_errors: list[float] = []
        num_sequences = len(self.dataset) if self.cfg.max_sequences is None else min(len(self.dataset), self.cfg.max_sequences)

        for idx in tqdm(range(num_sequences), desc=f"mv_consistency/{self.cfg.dataset}/{self.cfg.correspondence_method}"):
            batch = self.dataset[idx]
            images = batch["image"].to(device)
            depths = batch["depth"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            Rts = batch["Rt"].to(device)
            masks = batch["mask"].to(device)

            masked_coords = masks[0].squeeze().nonzero(as_tuple=False)
            if len(masked_coords) == 0:
                continue
            num_tracks = min(self.cfg.num_corr, len(masked_coords))

            set_all_seeds(self.cfg.seed + idx)
            sel = torch.randperm(len(masked_coords))[:num_tracks]
            start_points = masked_coords[sel][:, [1, 0]].float()

            H, W = images.shape[-2:]
            V = images.shape[0]

            if use_attention:
                q_all, k_all, scale = extract_global_attention_qk(probe, images)
                Hf, Wf = H // model.patch_size, W // model.patch_size
                patch_token_start = probe.patch_token_start
                num_tokens = patch_token_start + Hf * Wf
                q_source = q_all[:, patch_token_start : patch_token_start + Hf * Wf, :]
                q_source = q_source.reshape(-1, Hf, Wf, q_source.shape[-1])
            elif use_attention_multilayer:
                qks = extract_global_attention_qk_multilayer(probes, images)
                Hf, Wf = H // model.patch_size, W // model.patch_size
                patch_token_start = probes[0].patch_token_start
                num_tokens = patch_token_start + Hf * Wf
                q_alls = [q_all for q_all, _, _ in qks]
                k_alls = [k_all for _, k_all, _ in qks]
                scales = [scale for _, _, scale in qks]
            elif use_feature_block:
                feats = extract_dense_features_at_block(probe, images, model.patch_size)
                Hf, Wf = feats.shape[-2:]
                feat_s = feats[0:1]
            else:
                feats = extract_dense_features(model, images, single_view=self.cfg.single_view)
                Hf, Wf = feats.shape[-2:]
                feat_s = feats[0:1]
            uv_s_feat = start_points * torch.tensor([Wf / W, Hf / H], device=device)

            pred_tracks = torch.zeros((num_tracks, V, 2), device=device)
            pred_tracks[:, 0] = start_points
            for v in range(1, V):
                if use_attention:
                    target_start = v * num_tokens + patch_token_start
                    uv_t_feat = find_correspondences_attention_based(
                        uv_s_feat, q_source, k_all, scale, target_start, Hf, Wf
                    )
                    pred_tracks[:, v] = uv_t_feat * torch.tensor([W / Wf, H / Hf], device=device)
                elif use_attention_multilayer:
                    # already pixel-space, unlike the other two methods' patch-grid-space matches
                    target_start = v * num_tokens + patch_token_start
                    pred_tracks[:, v] = find_correspondences_attention_multilayer_based(
                        start_points, q_alls, k_alls, scales, patch_token_start, target_start, Hf, Wf, H, W,
                        self.cfg.attention_temperature,
                    )
                else:
                    uv_t_feat = find_correspondences_feature_based(uv_s_feat, feat_s, feats[v : v + 1])
                    pred_tracks[:, v] = uv_t_feat * torch.tensor([W / Wf, H / Hf], device=device)

                if self.cfg.estimate_relative_pose:
                    pose_error = estimate_relative_pose_error(
                        start_points, pred_tracks[:, v], intrinsics[0], intrinsics[v], Rts[0], Rts[v]
                    )
                    if pose_error is not None:
                        rot_err, trans_err = pose_error
                        all_rot_errors.append(rot_err)
                        if not np.isnan(trans_err):
                            all_trans_errors.append(trans_err)

            gt_tracks = calculate_gt_tracks(
                start_points, depths, intrinsics, Rts, source_inverse=True, target_inverse=False
            )

            if self.cfg.visualize:
                visualize_comparison_tracks(
                    list(images), pred_tracks, gt_tracks, f"viz/mv_consistency/sample_{idx}.png"
                )

            errors_2d, errors_3d = compute_errors_for_sample(pred_tracks, gt_tracks, depths, intrinsics)
            if errors_2d.numel() > 0:
                all_2d_errors.append(errors_2d.cpu())
                all_3d_errors.append(errors_3d.cpu())

        summary = _summarize_errors(all_2d_errors, all_3d_errors, metric_scale=self.cfg.dataset != "megadepth")
        if all_rot_errors:
            summary["rot_err_deg"] = float(np.mean(all_rot_errors))
        if all_trans_errors:
            summary["trans_ang_err_deg"] = float(np.mean(all_trans_errors))
        if step is not None and summary and is_main_process():
            try:
                import wandb

                # Full `summary` has ~12+ keys (acc_{1,2,5,10,25,50}px, ate_2d_px,
                # acc_{1,2,5,10}cm, ate_3d_cm, ...) per dataset/correspondence_method --
                # logging all of it clutters wandb with dozens of near-redundant
                # threshold curves. Keep only the headline metrics here; the full
                # summary is still returned (and e.g. `eval.py`'s `_report` writes
                # it in full to JSON/wandb for standalone eval runs).
                wandb_keys = ("ate_2d_px", "acc_10px", "acc_25px", "acc_50px")
                wandb.log(
                    {
                        f"mv_consistency/{self.cfg.dataset}/{self.cfg.correspondence_method}/{k}": summary[k]
                        for k in wandb_keys
                        if k in summary
                    },
                    step=step,
                )
            except Exception:
                pass
        return summary
