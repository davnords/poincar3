from __future__ import annotations

import torch
import torch.nn.functional as F

from .pose_enc import extri_intri_to_pose_encoding


def _filter_by_quantile(
    loss: torch.Tensor, valid_range: float, *, min_elements: int = 1000, hard_max: float = 100.0
) -> torch.Tensor:
    """Clamp to `hard_max`, then drop the worst `1 - valid_range` fraction by
    quantile (a single bad-annotation/sensor-noise pixel can otherwise
    dominate a `.mean()` over hundreds of thousands of valid pixels). Ported
    from VGGT's `filter_by_quantile`
    (`VGGT's `training/loss.py``) -- its own default depth recipe
    (`depth.valid_range: 0.98`, `VGGT's `training/config/
    default.yaml`) always applies this on top of the per-element clamp,
    dropped in this file's original port.

    Args:
        loss: flat per-element loss values (already indexed to valid pixels).
        valid_range: quantile threshold in `(0, 1]`, e.g. `0.98` keeps the
            best 98% of elements. Values `<= 0` disable filtering.
    """
    if valid_range <= 0 or loss.numel() <= min_elements:
        return loss.clamp(max=hard_max)

    loss = loss.clamp(max=hard_max)
    quantile_thresh = min(torch.quantile(loss.detach(), valid_range).item(), hard_max)
    quantile_mask = loss < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss[quantile_mask]
    return loss


def compute_camera_loss(
    pred_pose_encoding: torch.Tensor,
    gt_poses: torch.Tensor,
    gt_K: torch.Tensor,
    image_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Flat L1 over the whole 9D `absT_quaR_FoV` pose encoding, which is more
    stable here than VGGT's Huber with separately-weighted
    translation/rotation/FOV components and per-component outlier clamping.
    `CameraHead` is single-shot (see its docstring), so there is only ever one
    refinement stage to weight.

    Args:
        pred_pose_encoding: `CameraHead.forward`'s single-element list,
            unwrapped by the caller (`pred_pose_enc_list[-1]`).
        gt_poses: `(B, N, 4, 4)` cam-from-world extrinsics.
        gt_K: `(B, N, 3, 3)` intrinsics.
        image_hw: `(height, width)` `pred_pose_encoding` was computed at.
    """
    gt_pose_encoding = extri_intri_to_pose_encoding(gt_poses[..., :3, :], gt_K, image_hw)
    loss = F.l1_loss(pred_pose_encoding, gt_pose_encoding)
    return {"loss_camera": loss}


def _gradient_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss between predicted and GT spatial gradients (adjacent-pixel
    differences of the `pred - gt` residual, x and y), masked to pixels valid
    on both sides of each finite difference.

    Ported from VGGT's `gradient_loss` (`VGGT's `training/loss.py``),
    dropping its optional confidence-weighting branch: VGGT's own default
    recipe (`gradient_loss_fn="grad"`, no "conf" in the name) never exercises
    it, so plain unweighted masked L1 is what "matching VGGT" actually means
    here.

    Args:
        pred, gt: `(B, H, W, C)`.
        mask: `(B, H, W)` bool, True = valid.
    """
    mask = mask[..., None].expand(-1, -1, -1, pred.shape[-1])
    diff = (pred - gt) * mask

    grad_x = (diff[:, :, 1:] - diff[:, :, :-1]).abs() * (mask[:, :, 1:] & mask[:, :, :-1])
    grad_y = (diff[:, 1:, :] - diff[:, :-1, :]).abs() * (mask[:, 1:, :] & mask[:, :-1, :])
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    divisor = mask.sum()
    if divisor == 0:
        return (0.0 * pred).mean()
    return (grad_x.sum() + grad_y.sum()) / divisor


def _gradient_loss_multi_scale(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, *, scales: int = 4
) -> torch.Tensor:
    """Average `_gradient_loss` over `scales` progressively 2x-subsampled
    resolutions, so both fine and coarse depth-edge structure get a
    gradient-consistency signal. VGGT's `gradient_loss_multi_scale_wrapper`,
    `scales=4` is its own default for the plain (non-"normal") gradient loss.
    """
    total = pred.new_zeros(())
    for scale in range(scales):
        step = 2**scale
        total = total + _gradient_loss(pred[:, ::step, ::step], gt[:, ::step, ::step], mask[:, ::step, ::step])
    return total / scales


def compute_depth_loss(
    pred_depth: torch.Tensor,
    pred_depth_conf: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_depth_mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    alpha: float = 0.2,
    gradient_scales: int = 4,
    valid_range: float = 0.98,
) -> dict[str, torch.Tensor]:
    """Confidence-weighted L1 depth loss over valid GT-depth pixels
    (`gamma * |pred - gt| * conf - alpha * log(conf)`, encourages low
    confidence on hard/uncertain regions) plus the plain (unweighted)
    regression term and a multi-scale spatial-gradient term on the depth map
    (`_gradient_loss_multi_scale`) -- VGGT's own default depth-loss recipe,
    all three summed (`loss_depth = loss_conf.mean() + loss_reg.mean() +
    loss_grad`, matching how `MultitaskLoss` sums `loss_conf_depth +
    loss_reg_depth + loss_grad_depth` in `third_party/vggt/training/
    loss.py`). `loss_conf`/`loss_reg` are quantile-filtered
    (`_filter_by_quantile`) before averaging, same as VGGT's
    `regression_loss` -- without it, a single bad-annotation/sensor-noise
    pixel can otherwise dominate the mean over hundreds of thousands of
    valid pixels.

    Args:
        pred_depth, pred_depth_conf: `(B, N, H, W)`.
        gt_depth: `(B, N, H, W)`.
        gt_depth_mask: `(B, N, H, W)` bool, True = valid.
    """
    if gt_depth_mask.sum() == 0:
        dummy = (0.0 * pred_depth).mean()
        return {"loss_depth": dummy, "loss_depth_reg": dummy, "loss_depth_grad": dummy}

    loss_reg = (pred_depth[gt_depth_mask] - gt_depth[gt_depth_mask]).abs()
    conf = pred_depth_conf[gt_depth_mask]
    loss_conf = gamma * loss_reg * conf - alpha * torch.log(conf)

    loss_conf = _filter_by_quantile(loss_conf, valid_range)
    loss_reg = _filter_by_quantile(loss_reg, valid_range)

    batch_size, num_views, height, width = pred_depth.shape
    loss_grad = _gradient_loss_multi_scale(
        pred_depth.reshape(batch_size * num_views, height, width, 1),
        gt_depth.reshape(batch_size * num_views, height, width, 1),
        gt_depth_mask.reshape(batch_size * num_views, height, width),
        scales=gradient_scales,
    )

    return {
        "loss_depth": loss_conf.mean() + loss_reg.mean() + loss_grad,
        "loss_depth_reg": loss_reg.mean(),
        "loss_depth_grad": loss_grad,
    }
