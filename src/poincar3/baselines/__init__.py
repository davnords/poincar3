from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch.nn as nn

from poincar3.baselines.common import EncoderOnlyBaseline
from poincar3.baselines.dinov3 import load_frozen_dinov3_vitl
from poincar3.baselines.mum.adapter import load_mum_v1_baseline
from poincar3.device import device
from poincar3.model import Poincar3, SSLModel
from poincar3.run import load_run

Backbone = Literal[
    "poincar3",
    "poincar3_encoder",
    "mum_v1_decoder",
    "mum_v1_encoder",
    "dinov3_vitl",
]


def load_backbone(backbone: Backbone, run_path: str | None = None) -> tuple[nn.Module, int]:
    """Returns `(model, step)`. `step` is 0 for anything without a training run
    of its own, including the released checkpoint."""
    if backbone in ("poincar3", "poincar3_encoder"):
        model, step = _load_poincar3(run_path)
        if backbone == "poincar3_encoder":
            encoder = model.encoder
            model = EncoderOnlyBaseline(
                encoder, patch_size=encoder.patch_size, embed_dim=encoder.embed_dim
            ).to(device)
        return model, step
    if backbone in ("mum_v1_encoder", "mum_v1_decoder"):
        return load_mum_v1_baseline(device, mode=backbone.removeprefix("mum_v1_")), 0
    if backbone == "dinov3_vitl":
        return load_frozen_dinov3_vitl(device), 0
    raise ValueError(f"Unknown backbone: {backbone!r}")


def _load_poincar3(run_path: str | None) -> tuple[Poincar3, int]:
    if run_path is None:
        # No training run given: use the released checkpoint, which
        # `Poincar3()` auto-downloads.
        return Poincar3().to(device), 0
    run_cfg, step, weights, _, _ = load_run(Path(run_path))
    ssl_model = SSLModel(run_cfg.model).to(device)
    ssl_model.load_state_dict(weights)
    # The EMA teacher is the checkpoint worth evaluating; the student only
    # exists as the thing gradient descent updates.
    return ssl_model.teacher, step
