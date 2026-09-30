from __future__ import annotations

import torch

from poincar3.model import _build_encoder

from .common import EncoderOnlyBaseline


def load_frozen_dinov3_vitl(device: torch.device) -> EncoderOnlyBaseline:
    backbone = _build_encoder(encoder_size="vitl", pretrained=True)
    model = EncoderOnlyBaseline(backbone, patch_size=backbone.patch_size, embed_dim=backbone.embed_dim).to(device)
    model.eval()
    model.requires_grad_(False)
    return model
