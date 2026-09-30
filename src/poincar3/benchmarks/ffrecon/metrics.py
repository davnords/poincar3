from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree as KDTree


def umeyama(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimates the Sim(3) transform `c, R, t` such that `c * R @ X + t ~ Y`.

    Args:
        X, Y: `(m, n)` -- m is point dimension, n is number of points, index-aligned.

    Returns:
        `(c (scale), R (m, m), t (m, 1))`.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t


def accuracy(
    gt_points: np.ndarray, rec_points: np.ndarray, gt_normals: np.ndarray | None = None, rec_normals: np.ndarray | None = None
) -> tuple[float, float] | tuple[float, float, float, float]:
    """Mean/median nearest-GT-point distance for each reconstructed point
    (+ mean/median normal-consistency if normals are given)."""
    gt_tree = KDTree(gt_points)
    distances, idx = gt_tree.query(rec_points, workers=-1)
    acc, acc_median = np.mean(distances), np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.abs(np.sum(gt_normals[idx] * rec_normals, axis=-1))
        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)
    return acc, acc_median


def completion(
    gt_points: np.ndarray, rec_points: np.ndarray, gt_normals: np.ndarray | None = None, rec_normals: np.ndarray | None = None
) -> tuple[float, float] | tuple[float, float, float, float]:
    """Mean/median nearest-reconstructed-point distance for each GT point
    (+ mean/median normal-consistency if normals are given)."""
    rec_tree = KDTree(rec_points)
    distances, idx = rec_tree.query(gt_points, workers=-1)
    comp, comp_median = np.mean(distances), np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.abs(np.sum(gt_normals * rec_normals[idx], axis=-1))
        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)
    return comp, comp_median


def estimate_normals_pca(points: np.ndarray, k: int = 30) -> np.ndarray:
    """Per-point surface normal via local-neighborhood PCA (open3d's
    `estimate_normals` default algorithm): the eigenvector of the smallest
    eigenvalue of each point's `k`-nearest-neighbor covariance matrix. Sign
    is arbitrary/inconsistent across points -- fine here, `accuracy`/
    `completion`'s normal-consistency metric takes `abs(dot(...))`.
    """
    if len(points) < 3:
        return np.zeros_like(points)
    tree = KDTree(points)
    k = min(k, len(points))
    _, idx = tree.query(points, k=k, workers=-1)
    neighbors = points[idx]  # (N, k, 3)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order
    normals = eigvecs[:, :, 0]  # eigenvector of the smallest eigenvalue
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(norms, 1e-12, None)


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    max_correspondence_distance: float,
    max_iterations: int = 30,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Point-to-point ICP (open3d's `registration_icp(..., threshold,
    trans_init, TransformationEstimationPointToPoint())`, reimplemented):
    alternates nearest-neighbor correspondence (discarding pairs farther
    apart than `max_correspondence_distance`, matching open3d's `threshold`)
    with a closed-form rigid (rotation + translation, no scale -- `umeyama`
    already handled scale) alignment via SVD (Kabsch algorithm), until the
    mean correspondence distance stops improving by more than `tolerance` or
    `max_iterations` is reached.

    Returns the accumulated `(4, 4)` transform to left-multiply onto
    homogeneous `source` points (equivalently: `new_source = (R @ source.T).T
    + t` using the returned matrix's `R`/`t` blocks).
    """
    target_tree = KDTree(target)
    transform = np.eye(4)
    src = source.copy()
    prev_error: float | None = None

    for _ in range(max_iterations):
        distances, indices = target_tree.query(src, workers=-1)
        valid = distances < max_correspondence_distance
        if valid.sum() < 3:
            break
        src_valid = src[valid]
        tgt_valid = target[indices[valid]]

        src_mean = src_valid.mean(axis=0)
        tgt_mean = tgt_valid.mean(axis=0)
        H = (src_valid - src_mean).T @ (tgt_valid - tgt_mean)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        S = np.diag([1.0, 1.0, d])
        R = Vt.T @ S @ U.T
        t = tgt_mean - R @ src_mean

        step = np.eye(4)
        step[:3, :3] = R
        step[:3, 3] = t
        transform = step @ transform
        src = (R @ src.T).T + t

        mean_error = float(distances[valid].mean())
        if prev_error is not None and abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """`(4, 4)` homogeneous transform applied to `(N, 3)` points."""
    R, t = transform[:3, :3], transform[:3, 3]
    return (R @ points.T).T + t
