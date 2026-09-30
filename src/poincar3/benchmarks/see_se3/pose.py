from __future__ import annotations

import numpy as np
from scipy.linalg import logm


def pose9d(poses_c2w: np.ndarray) -> np.ndarray:
    """`poses_c2w`: (N,4,4) camera-to-world matrices. Returns (N,9): 3D
    translation concatenated with the first two columns of the rotation
    matrix (the continuous 6D rotation representation of Zhou et al. [40],
    exactly as Appendix A.1 specifies for Metric M1's mutual k-nn pose
    space)."""
    t = poses_c2w[:, :3, 3]
    r6 = poses_c2w[:, :3, :2].reshape(-1, 6)
    return np.concatenate([t, r6], axis=-1)


def se3_log(T_rel: np.ndarray) -> np.ndarray:
    """`T_rel`: (N,4,4) relative rigid transforms. Returns (N,6): the se(3)
    twist `(Log(T_rel))^v = [v (translational velocity), omega (rotational
    velocity, axis-angle)]`, exactly Eq. (1) of the paper -- the literal
    matrix logarithm of the 4x4 homogeneous transform, read off via the
    standard vee map (the skew-symmetric part of the rotation block gives
    `omega`, the last column gives `v`).

    Uses `scipy.linalg.logm` directly on the 4x4 matrix rather than a
    hand-derived closed form for the translational Jacobian inverse -- exact
    (up to floating point) for any rotation angle strictly below pi, and the
    isolated failure mode (a rotation of exactly pi) is vanishingly unlikely
    for the frame-to-frame deltas this probe evaluates."""
    n = T_rel.shape[0]
    out = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        xi_hat = logm(T_rel[i]).real
        omega = np.array([xi_hat[2, 1], xi_hat[0, 2], xi_hat[1, 0]])
        v = xi_hat[:3, 3]
        out[i] = np.concatenate([v, omega])
    return out


def relative_pose_targets(poses_c2w: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """`poses_c2w`: (N,4,4). `pairs`: (M,2) int array of `(i, j)` index pairs.
    Returns (M,6): `se3_log(inv(P_i) @ P_j)` for each pair -- the Lie-algebra
    relative-pose target `Delta P` Metric M4 regresses (Eq. 1, `P_rel =
    P_t^-1 P_{t+s}`)."""
    P_i = poses_c2w[pairs[:, 0]]
    P_j = poses_c2w[pairs[:, 1]]
    T_rel = np.linalg.inv(P_i) @ P_j
    return se3_log(T_rel)
