from __future__ import annotations

import torch
import torch.nn as nn

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class EncoderOnlyBaseline(nn.Module):
    def __init__(self, backbone: nn.Module, patch_size: int, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        # Only read for `.shape[-1]` by callers (matchbench/depth/normal
        # probes) to recover `embed_dim` -- Poincar3's own register tokens have
        # no analogue here, so this is a shapeholder, not a real feature.
        self.register_token = nn.Parameter(torch.empty(1, 1, embed_dim), requires_grad=False)
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def encoder(self, images: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        return self.backbone.forward_features(images)["x_norm_patchtokens"]

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        # mask/head_mask are ignored: this backbone has no masking mechanism,
        # and every eval caller invokes it with mask=None anyway.
        batch_size, num_frames, num_channels, height, width = images.shape
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)
        patch_tokens = self.encoder(images)
        patch_tokens = patch_tokens.view(batch_size, num_frames, *patch_tokens.shape[1:])
        return patch_tokens, patch_tokens, None, None

    def attention_probe(self, block_index: int | None = None):
        # Single-view only -- every frame runs through `self.backbone`
        # independently, so there's no cross-view attention to read off (see
        # `poincar3.benchmarks.dense_features.get_attention_probe`).
        return None

    def feature_probe(self, block_index: int | None = None):
        # Unlike `attention_probe` above, dense features *are* available at
        # any of `self.backbone`'s own single-view blocks -- there's just no
        # cross-view attention among them to read (see
        # `poincar3.benchmarks.dense_features.get_feature_probe`). `attn_module`
        # is filled in for API uniformity but never used by the feature path
        # (`extract_dense_features_at_block` only reads `block_module`).
        from poincar3.benchmarks.dense_features import AttentionProbe

        blocks = self.backbone.blocks
        idx = len(blocks) - 1 if block_index is None else block_index
        if not (0 <= idx < len(blocks)):
            raise ValueError(f"block_index={idx} out of range [0, {len(blocks)})")
        # DINOv3's own ViT counts special (cls + storage) tokens as
        # `n_storage_tokens`; the vendored DINOv2 one as `num_register_tokens`
        # -- either way, plus the leading cls token itself.
        num_special = getattr(self.backbone, "n_storage_tokens", None)
        if num_special is None:
            num_special = getattr(self.backbone, "num_register_tokens", 0)
        return AttentionProbe(
            attn_module=blocks[idx].attn,
            block_module=blocks[idx],
            patch_token_start=1 + num_special,
            run_forward=lambda images: self.forward(images.unsqueeze(0), mask=None),
            # This backbone's canonical dense-feature output is normalized:
            # `self.encoder` above reads `forward_features`'s
            # `x_norm_patchtokens`, and DINOv3's own
            # `get_intermediate_layers(..., norm=True)` -- the descriptor path
            # real RoMa v2/RoMa-Omega use (`RoMa-Omega/src/romaomega/features.py`)
            # -- applies `self.norm` to every intermediate layer it returns.
            # Handing it to the probe keeps the multi-layer read
            # (`extract_dense_features_at_block`, hooking a raw block output)
            # consistent with both. `self.norm` is the *patch* norm even when
            # `untie_cls_and_patch_norms` splits it (see
            # `vision_transformer.py`'s `get_intermediate_layers`), and patch
            # tokens are all this path keeps.
            post_norm=getattr(self.backbone, "norm", None),
        )
