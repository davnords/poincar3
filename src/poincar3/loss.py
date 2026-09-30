from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regularizers import KoLeoLoss, sinkhorn_knopp_teacher
from .types import Batch


class Poincar3Loss(nn.Module):
    """Latent-prediction loss between an `SSLModel`'s student and teacher.

    Three terms, all read off the decoder's final layer:

    * **Patch** -- iBOT-style cross-entropy between the student's softmax over
      the shared prototype head and a sharpened, Sinkhorn-Knopp normalized
      teacher target, over the *masked* patches only. Restricted to the frames
      both networks saw (the leading `batch.mask.shape[1]` of the teacher's
      sequence).
    * **Global** -- the same mechanism applied to the per-frame camera token
      through a separate prototype head, comparing student frame `i` against
      teacher frame `i`. Both are differently augmented views of the same
      underlying frame, so this is not a self-comparison.
    * **KoLeo** -- DINOv2's entropic regularizer on the raw, pre-head camera
      token, spreading scene embeddings apart.

    The teacher temperature is warmed up linearly (see `Cfg.teacher_temp`): a
    freshly initialized prototype head should not be hit with a fully sharp
    target from step 0.
    """

    @dataclass(frozen=True)
    class Cfg:
        global_loss_weight: float = 0.5
        koleo_loss_weight: float = 0.1
        student_temp: float = 0.1
        # Linearly warmed up from `warmup_teacher_temp` over
        # `teacher_temp_warmup_steps`, then held constant.
        warmup_teacher_temp: float = 0.04
        teacher_temp: float = 0.07
        teacher_temp_warmup_steps: int = 30_000
        sinkhorn_iterations: int = 3
        # The effective-rank SVD costs ~100ms/step, so it is only logged
        # periodically; the other collapse diagnostics run every step.
        rank_interval: int = 50

    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.koleo_loss = KoLeoLoss()

    def forward(self, *, batch: Batch, model: nn.Module, step: int) -> tuple[torch.Tensor, dict[str, float]]:
        student_out, teacher_out = model(batch)
        student_masked, _, student_global, student_global_raw = student_out
        teacher_masked, teacher_features, teacher_global, _ = teacher_out
        num_student_frames = batch.mask.shape[1]
        teacher_temp = self._teacher_temp(step)

        # Both sides were already gathered to the masked positions inside
        # `Poincar3.forward` (see its `head_mask` argument) -- flat [masked, D].
        teacher_masked = teacher_masked.detach()
        if teacher_masked.shape[0] == 0:
            raise ValueError("No masked patches in batch -- check batch.mask")
        teacher_probs = sinkhorn_knopp_teacher(teacher_masked, teacher_temp, self.cfg.sinkhorn_iterations)
        student_log_probs = F.log_softmax(student_masked.float() / self.cfg.student_temp, dim=-1)
        patch_loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()

        # Flatten [B, frames, D] to one row per (scene, frame), restricting the
        # teacher to the frames the student saw.
        teacher_global = teacher_global.detach()[:, :num_student_frames].reshape(-1, teacher_global.shape[-1])
        student_global = student_global.reshape(-1, student_global.shape[-1])
        teacher_global_probs = sinkhorn_knopp_teacher(teacher_global, teacher_temp, self.cfg.sinkhorn_iterations)
        student_global_log_probs = F.log_softmax(student_global.float() / self.cfg.student_temp, dim=-1)
        global_loss = -(teacher_global_probs * student_global_log_probs).sum(dim=-1).mean()

        koleo_loss = self.koleo_loss(student_global_raw.reshape(-1, student_global_raw.shape[-1]))

        loss = patch_loss + self.cfg.global_loss_weight * global_loss + self.cfg.koleo_loss_weight * koleo_loss
        stats = {
            "loss": loss.item(),
            "patch_loss": patch_loss.item(),
            "global_loss": global_loss.item(),
            "koleo_loss": koleo_loss.item(),
            "teacher_temp": teacher_temp,
        }

        # Collapse diagnostics on the teacher's raw, pre-head patch features.
        # Watch for std -> 0, effective_rank -> 1 or cross_sample_cosine -> 1;
        # all three move well before `patch_loss` visibly bottoms out.
        teacher_features = teacher_features[:, :num_student_frames].detach()
        stats["teacher_feature_std"] = _feature_std(teacher_features)
        stats["teacher_cross_sample_cosine"] = _cross_sample_cosine(teacher_features)
        if step % self.cfg.rank_interval == 0:
            stats["teacher_effective_rank"] = _effective_rank(teacher_features)
        return loss, stats

    def _teacher_temp(self, step: int) -> float:
        if step >= self.cfg.teacher_temp_warmup_steps:
            return self.cfg.teacher_temp
        frac = step / self.cfg.teacher_temp_warmup_steps
        return self.cfg.warmup_teacher_temp + frac * (self.cfg.teacher_temp - self.cfg.warmup_teacher_temp)


def _subsample(feats: torch.Tensor, max_tokens: int) -> torch.Tensor:
    flat = feats.reshape(-1, feats.shape[-1]).float()
    if flat.shape[0] > max_tokens:
        flat = flat[torch.randperm(flat.shape[0], device=flat.device)[:max_tokens]]
    return flat


def _feature_std(feats: torch.Tensor, max_tokens: int = 512) -> float:
    """Per-channel std of L2-normalized tokens: ~1/sqrt(C) if healthy, -> 0 if
    collapsed to a single point."""
    return F.normalize(_subsample(feats, max_tokens), dim=-1).std(dim=0).mean().item()


def _effective_rank(feats: torch.Tensor, max_tokens: int = 512) -> float:
    """exp(entropy of the singular-value spectrum) (RankMe, Garrido et al. 2023).
    Low even when the std looks fine means tokens collapse onto a low-dimensional
    subspace."""
    s = torch.linalg.svdvals(_subsample(feats, max_tokens))
    p = s / s.sum()
    return (-(p * p.clamp_min(1e-12).log()).sum()).exp().item()


def _cross_sample_cosine(feats: torch.Tensor) -> float:
    """Mean cosine similarity between different batch samples' pooled features.
    Near 1 means the model can no longer tell different inputs apart."""
    batch_size = feats.shape[0]
    if batch_size < 2:
        return float("nan")
    pooled = F.normalize(feats.float().mean(dim=(1, 2)), dim=-1)
    sim = pooled @ pooled.T
    return ((sim.sum() - batch_size) / (batch_size * (batch_size - 1))).item()
