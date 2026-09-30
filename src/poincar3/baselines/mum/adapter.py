from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .model import MuMAutoEncoder, mum_vitl16_decoderb

# Same values as `Poincar3`'s `_RESNET_MEAN`/`_RESNET_STD` (`model.py:14-15`), but
# applied by the caller here rather than internally by `forward` -- MuM's
# encoder/decoder expect already-normalized input, unlike `Poincar3.forward`.
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def load_frozen_mum(device: torch.device) -> MuMAutoEncoder:
    model = mum_vitl16_decoderb(pretrained=True).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def tap_ids_for_mum(model: MuMAutoEncoder) -> list[str]:
    """`[dec_frame_00, dec_global_00, dec_frame_01, dec_global_01, ...]` --
    matches the order `decoder_taps` appends to its output list."""
    tap_ids = []
    for i in range(len(model.decoder_frame_blocks)):
        tap_ids.append(f"dec_frame_{i:02d}")
        tap_ids.append(f"dec_global_{i:02d}")
    return tap_ids


@torch.no_grad()
def decoder_taps(model: MuMAutoEncoder, imgs: torch.Tensor) -> list[torch.Tensor]:
    """imgs: `(B, S, 3, H, W)`, raw `[0,1]` (imagenet-normalized here,
    mirroring `Poincar3.forward`'s internal normalization). Returns one
    `(B, S, num_patches, C)` tensor per decoder block (cls token dropped),
    in `tap_ids_for_mum`'s order. Ported from `MuMAutoEncoder.forward_decoder`
    (`model.py:235-283`), `mask_ratio=0.0` so no patches are actually
    dropped -- the encoder's `random_masking` becomes a no-op permutation
    that `ids_restore` undoes before the decoder ever sees the tokens.
    """
    B, S, C_in, H, W = imgs.shape
    mean = imgs.new_tensor(_RESNET_MEAN).view(1, 1, 3, 1, 1)
    std = imgs.new_tensor(_RESNET_STD).view(1, 1, 3, 1, 1)
    imgs = ((imgs - mean) / std).view(B * S, C_in, H, W)

    latent, _, ids_restore = model.forward_encoder(imgs, mask_ratio=0.0)

    num_patches_h, num_patches_w = H // model.patch_size, W // model.patch_size
    x = model.decoder_embed(latent)

    sin, cos = model.rope_embed_decoder(H=num_patches_h, W=num_patches_w)
    rope_frame = (sin, cos)
    pos_special = torch.zeros(1 + model.n_storage_tokens, sin.shape[-1], device=sin.device, dtype=sin.dtype)
    rope_global = (torch.cat([pos_special, sin]).repeat(S, 1), torch.cat([pos_special, cos]).repeat(S, 1))

    mask_tokens = model.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
    x = torch.cat([x[:, :1, :], x_], dim=1)

    _, P, C = x.shape
    taps: list[torch.Tensor] = []
    for frame_block, global_block in zip(model.decoder_frame_blocks, model.decoder_global_blocks):
        if x.shape != (B * S, P, C):
            x = x.view(B, S, P, C).view(B * S, P, C)
        x = frame_block(x, rope_frame)
        taps.append(x.reshape(B, S, P, C)[:, :, 1:, :])  # drop cls token

        x = x.view(B, S, P, C).view(B, S * P, C)
        x = global_block(x, rope_global)
        taps.append(x.reshape(B, S, P, C)[:, :, 1:, :])

    return taps


class MuMBaselineModel(nn.Module):
    """Wraps MuM v1's `MuMAutoEncoder` to expose the same interface `Poincar3`
    does, so it drops into `experiments/eval.py`'s benchmarks as a baseline in
    place of a trained `SSLModel.teacher`.

    `mode="decoder"` (default) reads `patch_features`/`patch_logits` off the
    final decoder block's raw dense tokens (`decoder_taps(...)[-1]`, i.e. the
    last `dec_global` tap -- the same cross-view-attended representation
    Poincar3's own `final_layer` is) -- a fair v1-vs-v2 comparison, since both
    have a real cross-view decoder. `mode="encoder"` instead bypasses the
    decoder entirely and uses the monocular encoder's own last-block patch
    tokens, *un-normalized* -- matching the paper's own matching baseline
    (`custom_fwd_matching` in `mum.model_loader`, which calls
    `forward_encoder(..., return_all_blocks=True)`, a path that returns
    before `self.norm` is ever applied), rather than `forward_features`'s
    normalized `"x_norm_patchtokens"` -- comparable to the `dinov3_vitl`
    baseline (`poincar3.baselines.common.EncoderOnlyBaseline`), or to Poincar3
    benchmarked with `single_view=True`.

    Either way, `patch_features`/`patch_logits` end up identical -- MuM v1 has
    no separate DINOHead-style prototype head, so there's nothing else to put
    in the `patch_logits` slot, mirroring how `EncoderOnlyBaseline` handles
    the same asymmetry.
    """

    def __init__(self, model: MuMAutoEncoder, mode: Literal["encoder", "decoder"] = "decoder"):
        super().__init__()
        self.model = model
        self.mode = mode
        self.patch_size = model.patch_size
        # In "decoder" mode, features come out of the decoder stack, which
        # runs at `decoder_embed.out_features` (e.g. 768 for vit_large), not
        # the encoder's `embed_dim` (1024) -- matchbench's `BaseMatcher`
        # hardcodes the channel count from this (via `register_token`) before
        # reshaping, so getting it wrong here breaks matchbench specifically
        # (the NYU-probe/mv_consistency benchmarks reshape dynamically and
        # never noticed).
        feature_dim = model.decoder_embed.out_features if mode == "decoder" else model.embed_dim
        self.register_token = nn.Parameter(torch.empty(1, 1, feature_dim), requires_grad=False)
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def encoder(self, images: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        # `return_all_blocks=True` exits `forward_encoder` before `self.norm`
        # ever runs (see class docstring) -- `[-1]` is the last block's raw
        # output, `[:, 1:]` drops the cls token (no storage tokens: MuM v1's
        # `n_storage_tokens=0`).
        return self.model.forward_encoder(images, 0, return_all_blocks=True)[-1][:, 1:]

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        # mask/head_mask ignored, as in `EncoderOnlyBaseline` -- MuM v1's
        # decoder has its own MAE-style masking (`decoder_taps`'s
        # `mask_ratio=0.0`), unrelated to Poincar3's iBOT-style `mask`.
        if self.mode == "decoder":
            patch_tokens = decoder_taps(self.model, images)[-1]  # (B, S, num_patches, C)
        else:
            # Same normalize-then-flatten-frames recipe as
            # `EncoderOnlyBaseline.forward` -- `images` arrives raw [0,1]
            # here (unlike `encoder()` above, whose callers pre-normalize).
            batch_size, num_frames, num_channels, height, width = images.shape
            images_norm = (images - self._resnet_mean) / self._resnet_std
            images_norm = images_norm.view(batch_size * num_frames, num_channels, height, width)
            patch_tokens = self.encoder(images_norm)
            patch_tokens = patch_tokens.view(batch_size, num_frames, *patch_tokens.shape[1:])
        return patch_tokens, patch_tokens, None, None

    def attention_probe(self, block_index: int | None = None):
        # Only "decoder" mode has cross-view attention to read -- "encoder"
        # mode bypasses the decoder entirely (see class docstring), so has no
        # analogue of Poincar3's inter-frame attention. Every `decoder_global_blocks`
        # entry does full cross-view attention (no Poincar3-style register-bottleneck
        # alternation), so any index is valid; default is the last one.
        if self.mode != "decoder":
            return None
        from poincar3.benchmarks.dense_features import AttentionProbe

        blocks = self.model.decoder_global_blocks
        idx = len(blocks) - 1 if block_index is None else block_index
        if not (0 <= idx < len(blocks)):
            raise ValueError(f"block_index={idx} out of range [0, {len(blocks)})")
        return AttentionProbe(
            attn_module=blocks[idx].attn,
            block_module=blocks[idx],
            patch_token_start=1 + self.model.n_storage_tokens,  # cls token (+ storage tokens, if any)
            run_forward=lambda images: self.forward(images.unsqueeze(0), mask=None),
        )

    def feature_probe(self, block_index: int | None = None):
        # "encoder" mode: monocular, no cross-view attention (see
        # `attention_probe`, which rightly returns `None` for it) -- but its
        # own single-view blocks still have dense features worth reading,
        # same `attention_probe`-vs-`feature_probe` asymmetry
        # `EncoderOnlyBaseline`/DINOv3 handles (see `get_feature_probe`).
        # "decoder" mode has no separate `feature_probe` of its own --
        # `attention_probe`'s cross-view blocks already serve both purposes.
        if self.mode != "encoder":
            return None
        from poincar3.benchmarks.dense_features import AttentionProbe

        blocks = self.model.blocks
        idx = len(blocks) - 1 if block_index is None else block_index
        if not (0 <= idx < len(blocks)):
            raise ValueError(f"block_index={idx} out of range [0, {len(blocks)})")
        return AttentionProbe(
            attn_module=blocks[idx].attn,
            block_module=blocks[idx],
            patch_token_start=1,  # cls token only -- no storage tokens on MuM v1's monocular encoder.
            run_forward=lambda images: self.forward(images.unsqueeze(0), mask=None),
        )


def load_mum_v1_baseline(device: torch.device, mode: Literal["encoder", "decoder"] = "decoder") -> MuMBaselineModel:
    return MuMBaselineModel(load_frozen_mum(device), mode=mode).to(device)
