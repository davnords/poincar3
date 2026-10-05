from __future__ import annotations

import torch


def normalize_batch_scale(
    poses: torch.Tensor,
    depths: torch.Tensor,
    depth_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recenter each sequence's extrinsics on frame 0, then scale translation
    and depth by the mean of valid GT depth pixels across the sequence.

    Args:
        poses: `(B, N, 4, 4)` cam-from-world extrinsics.
        depths: `(B, N, H, W)` z-depth (meters), 0 where invalid.
        depth_masks: `(B, N, H, W)` bool, True = valid depth.

    Returns:
        `(poses_norm (B, N, 4, 4), depths_norm (B, N, H, W), scale (B,))`.
    """
    batch_size = poses.shape[0]

    # Re-express every camera relative to frame 0 (frame 0 becomes the new
    # "world" origin): new_cam_i_from_world = cam_i_from_world @ world_from_cam0.
    world_from_cam0 = torch.linalg.inv(poses[:, 0])
    poses_norm = poses @ world_from_cam0.unsqueeze(1)

    depth_sum = (depths * depth_masks).sum(dim=(1, 2, 3))
    valid_count = depth_masks.sum(dim=(1, 2, 3)).float()
    scale = (depth_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)

    poses_norm[..., :3, 3] = poses_norm[..., :3, 3] / scale.view(batch_size, 1, 1)
    depths_norm = depths / scale.view(batch_size, 1, 1, 1)

    return poses_norm, depths_norm, scale


def normalize_pose_scale(poses: torch.Tensor) -> torch.Tensor:
    """Recenter each sequence's extrinsics on frame 0, then scale translation
    by the mean camera-center distance from frame 0 -- the camera-loss GT
    target's own normalization, *not* `normalize_batch_scale`'s depth-derived
    `scale` above.

    Why a separate function: this training mixture spans wildly different
    depth regimes batch to batch (aerial/landmark/indoor/object-scan scenes
    mixed by `DynamicBatchSampler`) -- `normalize_batch_scale`'s mean-GT-depth
    `scale` swung ~300x across batches of a real run (3.75 to 1139.8, see
    `experiments/ffrecon/train_poincar3.py`'s wandb `scale` history), which is
    fine for `depths_norm` (each batch's own depth is self-normalized to mean
    ~1 regardless of domain) but corrupts `compute_camera_loss`'s flat-L1
    translation target: the target's effective units become a function of
    ambient scene depth, decoupled from actual camera motion, so a fixed-
    magnitude translation error costs 300x more loss in a close-up batch than
    a far-away one. That shows up as `loss_camera`/AUC@30 plateauing early
    while RRA (rotation, untouched by this scale factor) keeps climbing
    cleanly and RTA (translation) stalls/degrades -- confirmed against a real
    run's `relpose_megadepth/RRA@15` vs `RTA@15` curves.

    Mean camera-center distance is bounded ~O(1) by construction (it is
    exactly what gets divided out) and depends only on camera motion, not on
    scene content, so it cannot blow up the way a depth-derived scale can
    across a heterogeneous dataset mixture.

    Deliberately does *not* touch `depths_norm`/`normalize_batch_scale`'s own
    `poses_norm`: those two stay on the shared depth-derived scale, since
    anything composing depth with pose translation needs them on one scale.
    Only `compute_camera_loss`'s GT argument uses this function's output.

    Args:
        poses: `(B, N, 4, 4)` cam-from-world extrinsics.

    Returns:
        `(B, N, 4, 4)` recentered + rescaled extrinsics.
    """
    batch_size = poses.shape[0]
    world_from_cam0 = torch.linalg.inv(poses[:, 0])
    poses_norm = poses @ world_from_cam0.unsqueeze(1)

    R = poses_norm[..., :3, :3]
    t = poses_norm[..., :3, 3]
    # Camera centers in the recentered frame: c_i = -R_i^T t_i (c_0 == 0).
    centers = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)  # (B, N, 3)
    scale = centers.norm(dim=-1).mean(dim=1).clamp_min(1e-6)  # (B,)

    poses_norm = poses_norm.clone()
    poses_norm[..., :3, 3] = t / scale.view(batch_size, 1, 1)
    return poses_norm
