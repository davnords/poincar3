from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from poincar3.benchmarks.dense_features import (
    extract_dense_features_at_block,
    extract_multilayer_patch_features_batched,
    extract_patch_features_batched,
    get_feature_probe,
    resolve_block_fraction,
)
from poincar3.model import Poincar3

# `layer_idx`'s literal entries (the `(11, 17)` default comes from RoMa v2's
# own `Descriptor.Cfg.layer_idx`) are absolute indices into a 24-block network,
# which is what DINOv3-L's and MuM v1's monocular encoders are. Backbones whose
# *probed* stack is shallower (Poincar3's cross-view decoder is 12 blocks, MuM
# v1 decoder mode's 6) need them reinterpreted as fractional depth and
# remapped, or `(11, 17)` is simply out of range -- see `_resolve_layer_idx`.
_REFERENCE_DEPTH = 24


class VisionEncoder(ABC):
    """API a vision encoder must implement to fit in the benchmark."""

    patch_size: int
    embed_dim: int

    @abstractmethod
    def forward(
        self,
        x: Tensor,  # 2*B, C, H, W
    ) -> Tensor:  # 2*B, N, C
        raise NotImplementedError


def layer_norm_per_stream(features: list[Tensor]) -> list[Tensor]:
    """Parameter-free LayerNorm over the channel dim of each entry, applied
    independently per entry (one entry = one `layer_idx` layer, or one of the
    two streams `Matcher.Cfg.poincar3_frame_and_inter` concatenates).

    `Poincar3VisionEncoder.feature_norm`'s implementation, shared by the generic
    matchbench path (`forward` below) and the coarse matcher's own paired
    read (`coarse_match/matcher.py`'s `_extract_poincar3_paired_features`).
    Every entry must be channel-last. Normalizing per entry rather than over
    the concatenated blob is the point: two decoder depths (or the frame- and
    cross-view streams) sit at different scales, and a joint norm would let
    whichever one happens to be larger keep dominating the cosine similarity."""
    return [F.layer_norm(f, (f.shape[-1],)) for f in features]


class Poincar3VisionEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        layer_idx: list[int] | tuple[int, ...] | None = None,
        final_norm: bool = True,
        feature_norm: bool = False,
    ):
        """`layer_idx`: `None` (default) reads only the backbone's final
        layer; `embed_dim` is the backbone's own dim. A
        list concatenates raw patch tokens from those depths instead (real
        RoMa v2/v3's `Descriptor.Cfg.layer_idx=[11, 17]`) -- `embed_dim`
        scales up by
        `len(layer_idx)` accordingly, and entries are resolved relative to
        `model`'s actual depth (see `_resolve_layer_idx`), so the same
        literal `layer_idx` works across backbones of different depth
        instead of raising/silently misbehaving on any that aren't 24
        blocks deep. `model` is `Poincar3` for the `"poincar3"` backbone, or an
        `EncoderOnlyBaseline` (DINOv3, MuM v1) for any other -- both are
        dispatched correctly (see `forward`).

        `final_norm`: only affects the multi-layer (`layer_idx is not None`)
        read, and only for backbones whose feature probe supplies a
        `post_norm` -- in practice `EncoderOnlyBaseline`/DINOv3, whose
        `post_norm` is its final `self.norm`. True (default) matches
        DINOv3's own `get_intermediate_layers(..., norm=True)`, i.e. the
        descriptor real RoMa v2/RoMa-Omega feed their matcher
        (`RoMa-Omega/src/romaomega/features.py`), *and* what the single-layer
        path here already returns for the same backbone
        (`extract_patch_features_batched` -> `forward_features`'s
        `x_norm_patchtokens`) -- so single- and multi-layer reads agree.
        Before this existed the multi-layer branch hooked raw block outputs
        and skipped the norm entirely, silently making the DINOv3 baseline a
        different (un-normalized, outlier-heavy) descriptor than the
        reference recipe's. Pass False for that old behaviour. No effect on
        Poincar3, whose decoder has no final norm to apply.

        `feature_norm`: the Poincar3-side counterpart to `final_norm`. Poincar3's
        cross-view decoder ends in no normalization at all (`model.py`'s
        `_apply_patch_module` reads `final_layer` straight off the residual
        stream), so its raw, outlier-heavy tokens go directly into the coarse
        matcher's cosine similarity and DPT head -- the same handicap the
        DINOv3 baseline carried until `final_norm` was added. True applies a
        *parameter-free* LayerNorm per layer instead (`layer_norm_per_stream`)
        -- Poincar3 has no learned norm to borrow, and adding a learned one would
        be a change to the matcher rather than to how its features are read.
        False (default) preserves every run before this existed."""
        super().__init__()
        self.model = model
        self.patch_size = model.patch_size
        self.final_norm = final_norm
        self.feature_norm = feature_norm
        self.layer_idx = self._resolve_layer_idx(model, list(layer_idx)) if layer_idx is not None else None
        base_embed_dim = model.register_token.shape[-1]
        self.embed_dim = base_embed_dim * (len(self.layer_idx) if self.layer_idx else 1)

    @staticmethod
    def _resolve_layer_idx(model: nn.Module, layer_idx: list[int]) -> list[int]:
        """Reinterprets each `layer_idx` entry `i` as a fractional depth,
        `min(1.0, (i + 1) / _REFERENCE_DEPTH)`, and resolves it to a valid
        block index for `model`'s actual (possibly shallower, possibly
        "holey" -- register-bottleneck blocks) stack.
        Exact round-trip for any genuinely 24-deep backbone (DINOv3_vitl,
        MuM v1 encoder mode): `(11, 17)` resolves right back to `(11, 17)`.
        For Poincar3, reimplements `resolve_block_fraction`'s fraction-to-index
        mapping directly against `model.depth` rather than going through
        `get_feature_probe` -- Poincar3's own multi-layer path
        (`extract_multilayer_patch_features_batched`) reads `2 * model.depth`
        `capture_block_outputs` entries directly, a different (hole-free)
        indexing scheme than `attention_probe`/`feature_probe`'s."""
        fractions = [min(1.0, (i + 1) / _REFERENCE_DEPTH) for i in layer_idx]
        if isinstance(model, Poincar3):
            depth = model.depth
            return [min(depth - 1, max(0, round(f * depth) - 1)) for f in fractions]
        return [resolve_block_fraction(model, f, use_attention=False) for f in fractions]

    def forward(self, x: Tensor) -> Tensor:
        mean = self.model._resnet_mean.squeeze(1).to(x.dtype)  # (1,3,1,1)
        std = self.model._resnet_std.squeeze(1).to(x.dtype)
        x = (x * std + mean).clamp(0, 1)
        if self.layer_idx is None:
            feats = extract_patch_features_batched(self.model, x)  # (2B,C,Hf,Wf)
        elif isinstance(self.model, Poincar3):
            feats = extract_multilayer_patch_features_batched(self.model, x, self.layer_idx)
        else:
            # Every other baseline (DINOv3, but also genuinely multi-view
            # ones -- MuM v1 decoder mode).
            # `get_feature_probe(...).run_forward` defaults to
            # `images.unsqueeze(0)` (one shared cross-attending cluster) --
            # harmless for single-view-only DINOv3, but wrong for the
            # multi-view ones: `x` here is unrelated pair-halves/batch
            # entries that must stay independent, same guarantee the
            # real-Poincar3 branch above makes via `images.unsqueeze(1)`. Pass
            # that same convention as an explicit `run_forward` override
            # instead (every baseline's `forward` accepts `(B, V, ...)` with
            # the same signature, so this works uniformly regardless of
            # which backbone has real cross-view attention).
            per_layer = [
                extract_dense_features_at_block(
                    get_feature_probe(self.model, block_index=i),
                    x,
                    self.patch_size,
                    run_forward=lambda images: self.model(images.unsqueeze(1), mask=None),
                    apply_post_norm=self.final_norm,
                )
                for i in self.layer_idx
            ]
            feats = torch.cat(per_layer, dim=1)  # (2B, C*len(layer_idx), Hf, Wf)
        feats = feats.flatten(2).permute(0, 2, 1)  # (2B,N,C)
        if self.feature_norm:
            num_layers = len(self.layer_idx) if self.layer_idx else 1
            feats = torch.cat(layer_norm_per_stream(list(feats.chunk(num_layers, dim=-1))), dim=-1)
        return feats
