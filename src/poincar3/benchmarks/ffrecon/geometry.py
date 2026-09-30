from __future__ import annotations

import numpy as np
import torch


def closed_form_inverse_se3(se3: np.ndarray) -> np.ndarray:
    """Inverse of a batch of SE(3) matrices, `(N, 4, 4)` or `(N, 3, 4)` -> `(N, 4, 4)`."""
    if se3.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError(f"se3 must be of shape (N,4,4) or (N,3,4), got {se3.shape}.")
    R = se3[:, :3, :3]
    T = se3[:, :3, 3:]
    R_transposed = np.transpose(R, (0, 2, 1))
    top_right = -np.matmul(R_transposed, T)
    inverted = np.tile(np.eye(4), (len(R), 1, 1))
    inverted[:, :3, :3] = R_transposed
    inverted[:, :3, 3:] = top_right
    return inverted


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """depth_map: `(H, W)`. intrinsic: `(3, 3)`, zero skew. Returns `(H, W, 3)` camera-space points."""
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def depth_to_world_coords_points(
    depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """depth_map: `(H, W)`. extrinsic: `(3, 4)` cam-from-world (w2c), OpenCV
    convention. intrinsic: `(3, 3)`. Returns `(world_points (H,W,3),
    cam_points (H,W,3), valid_mask (H,W))`."""
    if depth_map is None:
        return None, None, None

    point_mask = depth_map > eps
    cam_coords = depth_to_cam_coords_points(depth_map, intrinsic)

    cam_to_world = closed_form_inverse_se3(extrinsic[None])[0]
    R_cam_to_world = cam_to_world[:3, :3]
    t_cam_to_world = cam_to_world[:3, 3]

    world_coords = np.dot(cam_coords, R_cam_to_world.T) + t_cam_to_world
    return world_coords, cam_coords, point_mask


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray | torch.Tensor,
    extrinsics_cam: np.ndarray | torch.Tensor,
    intrinsics_cam: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Batched version of `depth_to_world_coords_points`.

    Args:
        depth_map: `(S, H, W, 1)` or `(S, H, W)`.
        extrinsics_cam: `(S, 3, 4)`, cam-from-world.
        intrinsics_cam: `(S, 3, 3)`.

    Returns:
        `(S, H, W, 3)` world points.
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points = []
    for frame_idx in range(depth_map.shape[0]):
        d = depth_map[frame_idx]
        d = d.squeeze(-1) if d.ndim == 3 else d
        cur_world_points, _, _ = depth_to_world_coords_points(d, extrinsics_cam[frame_idx], intrinsics_cam[frame_idx])
        world_points.append(cur_world_points)
    return np.stack(world_points, axis=0)
