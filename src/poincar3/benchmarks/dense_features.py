from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from poincar3.model import Poincar3


@dataclass(frozen=True)
class AttentionProbe:
    """Where to read a model's own cross-view attention *and* raw dense
    features from, at a given block -- for `extract_global_attention_qk` and
    `extract_dense_features_at_block` below. Any model that has genuine
    cross-view attention to probe (Poincar3, and MuM v1 in decoder mode)
    implements a one-arg
    `attention_probe(block_index=None)` method returning one of these (or
    `None`, for single-view-only baselines like DINOv3 with no cross-view
    attention at all -- see `get_attention_probe`).

    `attn_module` is whatever submodule directly calls
    `F.scaled_dot_product_attention` for the targeted cross-view block --
    e.g. `Poincar3.inter_frame_blocks[i].attn`. `block_module` is that same
    block's *parent* module (e.g. `Poincar3.inter_frame_blocks[i]`) -- every
    backbone's cross-view blocks run on tokens already reshaped to `(1,
    V*num_tokens, C)` (one view's `num_tokens`-token slice per view,
    concatenated in view order), so `block_module`'s own output, with no
    extra reshaping needed, is a drop-in dense-feature source at that depth.
    `patch_token_start` is the offset of patch tokens within one view's token
    block (register/camera/cls tokens come first). `run_forward` triggers one
    `attn_module`/`block_module` call, given `images` (V,3,H,W).

    `post_norm` is the backbone's own final normalization layer, when its
    canonical dense-feature output applies one to intermediate-block tokens.
    Only `extract_dense_features_at_block` uses it (never the attention path),
    and only for backbones that set it -- see `EncoderOnlyBaseline.feature_probe`,
    where it's DINOv3's `self.norm`, exactly what DINOv3's own
    `get_intermediate_layers(..., norm=True)` (the reference RoMa v2/RoMa-Omega
    descriptor path, `RoMa-Omega/src/romaomega/features.py`) applies to every
    layer it returns. `None` (the default) means "read the raw residual
    stream", which is the right convention for backbones whose native feature
    output is un-normalized -- Poincar3's decoder has no final norm at all."""

    attn_module: nn.Module
    block_module: nn.Module
    patch_token_start: int
    run_forward: Callable[[torch.Tensor], None]
    post_norm: nn.Module | None = None


def get_attention_probe(model: nn.Module, block_index: int | None = None) -> AttentionProbe | None:
    """Returns `model.attention_probe(block_index)` if `model` implements the
    method, else `None` -- e.g. single-view-only baselines (DINOv3, MuM v1 in
    encoder mode) have no cross-view attention to read.

    `block_index`: `None` (default) probes each model's *last* genuine
    cross-view block -- the block it actually uses for its own output, so a
    fair "does the block this model relies on encode correspondence"
    comparison. Pass an explicit index to probe a different block instead --
    useful for backbones that weren't trained to keep correspondence legible
    at their *last* block the way Poincar3's cross-view SSL training does:
    their raw attention can go diffuse there, even though a middle block still
    encodes it clearly. See each backbone's `attention_probe` for which indices
    are valid."""
    attention_probe = getattr(model, "attention_probe", None)
    return attention_probe(block_index) if attention_probe is not None else None


def get_feature_probe(model: nn.Module, block_index: int | None = None) -> AttentionProbe | None:
    """Like `get_attention_probe`, but for `extract_dense_features_at_block`
    rather than `extract_global_attention_qk`. Most backbones can serve both
    off the exact same probe (see `AttentionProbe`'s dual-purpose docstring),
    so this defaults to `model.attention_probe(block_index)`. Pass a
    `model.feature_probe(block_index)` method instead when a backbone needs a
    *different* probe for raw-feature reads than for cross-view attention --
    e.g. `EncoderOnlyBaseline`/DINOv3: single-view, so `attention_probe`
    always (rightly) returns `None`, since no cross-view attention exists to
    answer the "does the model's own attention agree with correspondence"
    question -- but its own single-view blocks still have perfectly good
    dense features worth reading, a different question `feature_probe`
    answers instead."""
    feature_probe = getattr(model, "feature_probe", None)
    if feature_probe is not None:
        return feature_probe(block_index)
    return get_attention_probe(model, block_index)


# No backbone probed here has more than a few dozen blocks -- generous enough
# headroom that `resolve_block_fraction` below never truncates a real depth.
_MAX_BLOCKS_TO_SCAN = 128


def resolve_block_fraction(model: nn.Module, fraction: float, use_attention: bool) -> int:
    """Maps `fraction` (in `(0, 1]`) to the nearest *valid* block index for
    `get_attention_probe`/`get_feature_probe` (whichever `use_attention`
    selects) -- lets one relative "how far into the network" x-axis compare
    backbones of very different depths and block-validity rules, e.g. for a
    layer-depth-vs-accuracy sweep across models.

    Deliberately brute-forces the valid-index set (probing every index up to
    `_MAX_BLOCKS_TO_SCAN` -- cheap, since `attention_probe`/`feature_probe`
    calls only inspect module structure, no forward pass) rather than
    computing `round(fraction * depth)` directly: several backbones have
    holes in their valid range that a naive depth-based formula would land on
    silently -- Poincar3's own register-bottleneck blocks have no
    patch-to-patch cross-view attention to read, for instance.
    `fraction=1.0` resolves to the last valid block, the same one the model's
    own final output goes through."""
    assert 0 < fraction <= 1, f"fraction={fraction} must be in (0, 1]"
    valid_indices = _scan_valid_block_indices(model, use_attention)
    if not valid_indices:
        method = "attention" if use_attention else "feature"
        raise RuntimeError(
            f"resolve_block_fraction: {type(model).__name__} has no valid {method} block in "
            f"range(0, {_MAX_BLOCKS_TO_SCAN}) -- this backbone/mode has no genuine cross-view "
            "attention to read at all (e.g. a single-view-only baseline probed for 'attention')."
        )
    target = fraction * valid_indices[-1]
    return min(valid_indices, key=lambda i: abs(i - target))


def _scan_valid_block_indices(model: nn.Module, use_attention: bool) -> list[int]:
    """Brute-force valid-index scan used by `resolve_block_fraction` --
    probing every index up to `_MAX_BLOCKS_TO_SCAN` is cheap
    (`attention_probe`/`feature_probe` calls only inspect module structure, no
    forward pass)."""
    get_probe = get_attention_probe if use_attention else get_feature_probe
    valid_indices = []
    for i in range(_MAX_BLOCKS_TO_SCAN):
        try:
            probe = get_probe(model, i)
        except ValueError:
            continue
        if probe is not None:
            valid_indices.append(i)
    return valid_indices


def _capture_sdpa_qk(attn_module: nn.Module, run_forward: Callable[[], None]) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Runs `run_forward()` and returns the exact `(q, k, scale)` tensors
    `attn_module`'s own forward pass hands to `F.scaled_dot_product_attention`
    -- post-qkv-projection, post q/k-norm, post-RoPE, whatever `attn_module`
    does internally before its own softmax -- without needing to replicate
    any of that internal recipe here. `q`/`k` come back exactly as SDPA
    receives them, `(batch, num_heads, N, head_dim)`.

    Works by monkeypatching `scaled_dot_product_attention` for the duration
    of `run_forward()`, gated by a forward pre/post hook on `attn_module` so
    only the call made *from inside* `attn_module` is captured -- every other
    attention block elsewhere in the model calls the same function during the
    same forward pass and must be ignored. Patches both
    `torch.nn.functional.scaled_dot_product_attention` (covers callers using
    `import torch.nn.functional as F` or `torch.nn.functional.sdpa(...)`,
    since both look the name up on the same module object at call time) and,
    if present, a `scaled_dot_product_attention` name bound directly in
    `attn_module`'s own defining module (covers `from torch.nn.functional
    import scaled_dot_product_attention`, which captures its own reference at
    import time -- patching `torch.nn.functional` alone can't reach that,
    e.g. Pi3's attention module)."""
    captured: dict[str, torch.Tensor | float] = {}
    active = {"on": False}
    real_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(query, key, value, *args, **kwargs):
        if active["on"] and "q" not in captured:
            captured["q"] = query.detach()
            captured["k"] = key.detach()
            captured["scale"] = kwargs.get("scale") or query.shape[-1] ** -0.5
        return real_sdpa(query, key, value, *args, **kwargs)

    owning_module = sys.modules.get(type(attn_module).__module__)
    has_own_binding = owning_module is not None and getattr(owning_module, "scaled_dot_product_attention", None) is real_sdpa

    pre_handle = attn_module.register_forward_pre_hook(lambda _m, _i: active.__setitem__("on", True))
    post_handle = attn_module.register_forward_hook(lambda _m, _i, _o: active.__setitem__("on", False))
    torch.nn.functional.scaled_dot_product_attention = patched_sdpa
    if has_own_binding:
        owning_module.scaled_dot_product_attention = patched_sdpa
    try:
        run_forward()
    finally:
        pre_handle.remove()
        post_handle.remove()
        torch.nn.functional.scaled_dot_product_attention = real_sdpa
        if has_own_binding:
            owning_module.scaled_dot_product_attention = real_sdpa

    if "q" not in captured:
        raise RuntimeError(
            f"{type(attn_module).__name__}.forward never called scaled_dot_product_attention during run_forward() "
            "-- attention_probe() picked the wrong module, or this backbone's attention doesn't go through SDPA."
        )
    return captured["q"], captured["k"], captured["scale"]


def extract_global_attention_qk(probe: AttentionProbe, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """images: (V,3,H,W) in [0,1], one covisible multi-view cluster (same as
    `extract_dense_features`'s default path). Returns `(q, k, scale)`: `q`/`k`
    are `(num_heads, V*num_tokens, head_dim)` (`num_tokens` = tokens per view,
    in the same order the model concatenates them -- see
    `probe.patch_token_start`), `scale` is the attention temperature."""
    q, k, scale = _capture_sdpa_qk(probe.attn_module, lambda: probe.run_forward(images))
    return q[0], k[0], scale  # squeeze the batch dim -- run_forward always calls with batch_size=1


def extract_dense_features_at_block(
    probe: AttentionProbe,
    images: torch.Tensor,
    patch_size: int,
    run_forward: Callable[[torch.Tensor], None] | None = None,
    apply_post_norm: bool = True,
) -> torch.Tensor:
    """images: (V,3,H,W) in [0,1]. Returns (V,C,Hf,Wf) dense patch features
    read directly off `probe.block_module`'s own output tokens -- the
    feature-method analogue of `extract_global_attention_qk`: instead of the
    model's own final forward-pass output (`extract_dense_features`), this
    captures a specific block's raw residual-stream tokens (bypassing
    everything after it, e.g. Poincar3's DINOHead), via a forward hook rather
    than a full re-implementation of each backbone's block loop.

    Works for either of `block_module`'s two possible output layouts -- the
    joint `(1, V*num_tokens, C)` cross-view blocks (`attention_probe()`) use,
    or a single-view block's own natural `(V, num_tokens, C)` (one row per
    view already, since `run_forward` always flattens to batch_size=1 --
    `feature_probe()`, e.g. `EncoderOnlyBaseline`/DINOv3) -- both reshape to
    `(V, num_tokens, C)` identically, so `num_tokens` (and from it,
    `patch_token_start`) is derived from the captured tensor's own element
    count rather than trusted from `probe.patch_token_start`, which only
    describes the joint-layout case.

    `run_forward`: defaults to `probe.run_forward` (always `images.unsqueeze
    (0)`, i.e. every image in `images` cross-attends as one shared multi-view
    cluster -- correct for `extract_global_attention_qk`'s genuinely-covisible
    -frames use case, which every `attention_probe()` is written for). Pass an
    override when `images` are instead *independent* (e.g. `Poincar3VisionEncoder`'s
    matchbench features, where they're unrelated pair-halves/batch entries
    that must never cross-attend) and the probed backbone might have real
    cross-view attention (MuM v1 decoder mode)
    -- unlike single-view-only backbones (DINOv3), where `probe.run_forward`'s
    batching is harmless (see `matchbench/encoder.py`'s caller).

    `apply_post_norm` (default True): run `probe.post_norm` over the captured
    patch tokens, when the probe supplies one. For DINOv3 that's its final
    `self.norm`, making this path agree with the backbone's own
    `get_intermediate_layers(..., norm=True)` -- the reference RoMa v2/
    RoMa-Omega descriptor (`RoMa-Omega/src/romaomega/features.py`), and also
    with what `extract_patch_features_batched` already returns for the same
    backbone via `forward_features`'s `x_norm_patchtokens`. A no-op for any
    probe whose `post_norm` is `None` (Poincar3, and every multi-view baseline).
    Pass False to read the raw, un-normalized residual stream instead."""
    V, _, H, W = images.shape
    Hf, Wf = H // patch_size, W // patch_size

    captured: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple | list) -> None:
        # DINOv3's own `SelfAttentionBlock.forward` (`poincar3.layers.block`)
        # returns a `List[Tensor]` rather than a bare `Tensor` whenever it's
        # called with a list input -- which `EncoderOnlyBaseline`'s path
        # always does (`forward_features` wraps a single-image-batch `Tensor`
        # into a length-1 list before running the block loop, DINOv2-style
        # multi-crop support this repo's own encoder inherited). `[0]` is
        # this single batch's tensor either way.
        captured["out"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = probe.block_module.register_forward_hook(hook)
    try:
        (run_forward or probe.run_forward)(images)
    finally:
        handle.remove()

    out = captured["out"]
    channels = out.shape[-1]
    num_tokens = out.numel() // (V * channels)
    patch_token_start = num_tokens - Hf * Wf

    tokens = out.reshape(V, num_tokens, channels)[:, patch_token_start:, :]  # (V, Hf*Wf, C)
    if apply_post_norm and probe.post_norm is not None:
        # LayerNorm is per-token, so normalizing the already-sliced patch
        # tokens is identical to normalizing the full sequence and slicing
        # after -- what `get_intermediate_layers` does.
        tokens = probe.post_norm(tokens)
    return tokens.permute(0, 2, 1).reshape(V, -1, Hf, Wf).float()


def extract_dense_features(model: Poincar3, images: torch.Tensor, single_view: bool = False) -> torch.Tensor:
    """images: (V,3,H,W) in [0,1]. Returns (V,C,Hf,Wf) dense patch features.

    By default, treats all V views as one multi-view forward pass through the
    full model (no mask -- teacher-style), reading the backbone's final decoder
    layer. If `single_view=True`, instead runs each view through the encoder
    alone, independently -- bypassing the multi-view decoder entirely, like a
    plain single-image backbone. Useful as a baseline/sanity check (e.g. to
    confirm the eval pipeline itself is discriminative: a pretrained encoder's
    single-view features should score far better than a random-init one's).
    """
    V, _, H, W = images.shape
    Hf, Wf = H // model.patch_size, W // model.patch_size

    if single_view:
        images_norm = (images - model._resnet_mean.squeeze(0)) / model._resnet_std.squeeze(0)
        out = model.encoder(images_norm, masks=None)
        patch_tokens = out["x_norm_patchtokens"] if isinstance(out, dict) else out
        return patch_tokens.permute(0, 2, 1).reshape(V, -1, Hf, Wf).float()

    patch_logits, patch_features, _, _ = model(images.unsqueeze(0), mask=None)
    # `Poincar3.Cfg.use_ibot_head=False` (the `poincar3.baselines.masked_mse`
    # ablation) builds no prototype head, so there are no logits to read --
    # fall back to the raw final-layer patch features, which are then the
    # model's entire output representation anyway.
    patch_tokens = (patch_features if patch_logits is None else patch_logits)[0]  # (V, num_patches, C)
    # Under autocast, the DINOHead output comes back as bfloat16 (it's a plain
    # nn.Linear, autocast casts it down). Downstream benchmark code
    # (F.grid_sample etc.) requires a consistent, known dtype, hence the
    # explicit cast here.
    return patch_tokens.permute(0, 2, 1).reshape(V, -1, Hf, Wf).float()


def extract_patch_features_batched(model: Poincar3, images: torch.Tensor) -> torch.Tensor:
    """images: (B,3,H,W) *independent* images -- unlike `extract_dense_features`,
    no covisibility between them is assumed. Each image is run as its own
    singleton-frame cluster (batch dim B, num_frames 1), so the cross-frame
    attention blocks degenerate to plain self-attention: batching many
    unrelated images together never lets them attend to each other, which a
    naive `extract_dense_features(images, ...)` call would do (it treats the
    whole first dimension as one multi-view cluster).

    Returns (B,C,Hf,Wf) raw pre-head patch tokens (`patch_features`,
    `embed_dim` channels) rather than `extract_dense_features`'s post-DINOHead
    `patch_logits` (`ibot_head_out_dim` channels, e.g. 16384) -- the
    representation space linear probes are normally trained on, and far
    cheaper to keep around per-image.
    """
    B, _, H, W = images.shape
    Hf, Wf = H // model.patch_size, W // model.patch_size
    _, patch_features, _, _ = model(images.unsqueeze(1), mask=None)
    patch_features = patch_features[:, 0]  # (B, num_patches, C)
    return patch_features.permute(0, 2, 1).reshape(B, -1, Hf, Wf).float()


def extract_multilayer_patch_features_batched(
    model: Poincar3, images: torch.Tensor, layer_idx: list[int]
) -> torch.Tensor:
    """Like `extract_patch_features_batched`, but concatenates raw patch
    tokens from multiple depths instead of just the final layer -- real RoMa
    v2/v3's `Descriptor.Cfg.layer_idx`
    concatenates two intermediate DINOv3 layers (`[11, 17]`) as the coarse
    matcher's input rather than only the last one.

    Reuses `capture_block_outputs=True`'s existing per-block return
    (`[frame_0, inter_0, frame_1, inter_1, ...]`, `2 * model.depth` entries,
    already reshaped to `(B, num_frames, num_tokens, C)`) rather than
    `AttentionProbe`'s `run_forward` (which does `images.unsqueeze(0)`,
    treating every image in `images` as one multi-frame *cluster* that can
    cross-attend -- wrong here, since im_A/im_B and unrelated pairs in the
    batch must stay independent, same guarantee `extract_patch_features_batched`
    makes via `images.unsqueeze(1)`).

    `2 * i + 1` (the entry *after* layer `i`'s inter-frame-attention step) is
    read for each requested `i`: with `num_frames=1` (every image its own
    singleton cluster), inter-frame attention degenerates to plain
    self-attention, so this stays independence-safe -- and it's the point
    where layer `i`'s residual-stream update is complete (what feeds layer
    `i + 1`), the closest analogue of a plain ViT's (DINOv3's) block-`i`
    output.
    """
    B, _, H, W = images.shape
    Hf, Wf = H // model.patch_size, W // model.patch_size
    *_, block_outputs = model(images.unsqueeze(1), mask=None, capture_block_outputs=True)
    feats = [block_outputs[2 * i + 1][:, 0, model.patch_token_start :, :] for i in layer_idx]  # each (B, N, C)
    feats = torch.cat(feats, dim=-1)  # (B, N, C * len(layer_idx))
    return feats.permute(0, 2, 1).reshape(B, -1, Hf, Wf).float()
