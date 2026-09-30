from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoincareAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, geo_dim: int = 20):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, geo_dim),
        )
        self.readout = nn.Linear(geo_dim, 6, bias=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.readout(self.phi(z2) - self.phi(z1))


@dataclass(frozen=True)
class PoincareAdapterResult:
    r2: float
    r2_trans: float
    r2_rot: float
    mse: float
    # The trained adapter itself, so a caller can read its per-frame
    # `W . phi(z_t)` output (the quantity the paper's Fig. 1 unrolls) using
    # exactly the recipe the reported numbers came from, rather than
    # reimplementing the training loop in the plotting code.
    adapter: "PoincareAdapter | None" = None


def _r_squared(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Per-component R^2 (scored against the *test* targets' own mean, the
    standard `sklearn.metrics.r2_score` convention -- what lets a
    badly-miscalibrated baseline score arbitrarily negative, e.g. Table 1's
    R^2 ~= -5.29 for raw pixels), uniformly averaged over all 6 components --
    "we report the test R^2 (uniform average over the 6 components)"
    (Appendix A.2)."""
    ss_res = ((pred - target) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-8)
    return (1 - ss_res / ss_tot).mean().item()


def fit_poincare_adapter(
    z1_train: torch.Tensor,
    z2_train: torch.Tensor,
    y_train: torch.Tensor,
    z1_test: torch.Tensor,
    z2_test: torch.Tensor,
    y_test: torch.Tensor,
    hidden_dim: int = 64,
    geo_dim: int = 20,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    seed: int | None = None,
) -> PoincareAdapterResult:
    """Trains one Poincare Adapter (Appendix A.2's exact recipe: AdamW,
    lr=1e-3, weight_decay=1e-2, 10 epochs, batch_size=512, grad-clip norm
    1.0) on `(z1_train, z2_train) -> y_train`, and reports test R^2/MSE.

    `y_train`/`y_test`: (N,6) raw (un-normalized) se(3) targets (see
    `pose.relative_pose_targets`). Regression targets are z-scored (zero
    mean, unit variance) per component on the training set before training;
    predictions are un-normalized again before computing the test R^2/MSE, so
    both are reported in the targets' native units (meters / radians).

    `seed`: seeds the adapter's initialization and batch shuffling. M4 is
    genuinely seed-sensitive (the paper's own Appendix reports per-config
    seed sigma up to 1.03 on the translation component, and averages over 15
    seeds for its ablation numbers), so a single unseeded fit is not a
    reproducible number -- callers should fit several seeds and report the
    spread (see `SeeSE3Benchmark.Cfg.m4_num_seeds`)."""
    if seed is not None:
        torch.manual_seed(seed)
    device = z1_train.device
    in_dim = z1_train.shape[-1]
    adapter = PoincareAdapter(in_dim, hidden_dim, geo_dim).to(device)

    mean = y_train.mean(dim=0, keepdim=True)
    std = y_train.std(dim=0, keepdim=True).clamp_min(1e-8)
    y_train_z = (y_train - mean) / std

    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    n = z1_train.shape[0]
    adapter.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = adapter(z1_train[idx], z2_train[idx])
            loss = F.mse_loss(pred, y_train_z[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_clip)
            opt.step()

    adapter.eval()
    with torch.no_grad():
        pred_test = adapter(z1_test, z2_test) * std + mean

    return PoincareAdapterResult(
        r2=_r_squared(pred_test, y_test),
        r2_trans=_r_squared(pred_test[:, :3], y_test[:, :3]),
        r2_rot=_r_squared(pred_test[:, 3:], y_test[:, 3:]),
        mse=F.mse_loss(pred_test, y_test).item(),
        adapter=adapter,
    )
