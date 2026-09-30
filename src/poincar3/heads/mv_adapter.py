from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from ..layers.utils import named_apply
from ..layers.vision_transformer import init_weights_vit


class MultiViewAdapter(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        # 2 alternating frame/inter-frame pairs by default -- enough to let
        # information cross views at least once or twice, cheap enough that
        # it doesn't turn this into "finetune a whole new decoder from
        # scratch" (that's the separate, heavier protocol in
        # `train_poincar3.py`).
        depth: int = 2
        embed_dim: int = 768
        num_heads: int = 12
        mlp_ratio: float = 4.0
        # Kept small relative to `Poincar3.Cfg.num_register_tokens` (16) -- this
        # adapter is meant to be a cheap, shallow probe, not a full decoder;
        # registers here only give inter-frame attention a few extra
        # dumping-ground tokens beyond the dedicated camera token.
        num_register_tokens: int = 4

    def __init__(self, cfg: Cfg, in_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(in_dim, cfg.embed_dim) if in_dim != cfg.embed_dim else nn.Identity()
        self.rope_embed = RopePositionEmbedding(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        def _make_blocks() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    SelfAttentionBlock(
                        dim=cfg.embed_dim,
                        num_heads=cfg.num_heads,
                        ffn_ratio=cfg.mlp_ratio,
                        qkv_bias=True,
                        proj_bias=True,
                        ffn_bias=True,
                        ffn_layer=Mlp,
                        init_values=1e-5,
                        use_qk_norm=True,
                        mask_k_bias=True,
                    )
                    for _ in range(cfg.depth)
                ]
            )

        # Per-frame self-attention (rope-positioned over the patch grid,
        # prefix tokens untouched -- see `SelfAttention.apply_rope`).
        self.frame_blocks = _make_blocks()
        # Cross-frame self-attention over every frame's tokens flattened
        # together -- unlike `Poincar3.inter_frame_blocks`, always "global"
        # (no register-bottleneck attention type): at `depth=2` there's no
        # compute pressure that trick exists to relieve.
        self.inter_frame_blocks = _make_blocks()

        self.camera_token = nn.Parameter(torch.empty(1, 1, cfg.embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, cfg.num_register_tokens, cfg.embed_dim))
        self.num_register_tokens = cfg.num_register_tokens
        self.patch_token_start = cfg.num_register_tokens + 1

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)
        # `mask_k_bias=True` blocks need `init_weights_vit` to populate their
        # `LinearKMaskedBias.bias_mask` buffer (starts NaN) -- see
        # `Poincar3.init_weights`'s identical comment.
        named_apply(init_weights_vit, self.frame_blocks)
        named_apply(init_weights_vit, self.inter_frame_blocks)

    def forward(self, patch_features: torch.Tensor, patch_h: int, patch_w: int) -> list[torch.Tensor]:
        """
        Args:
            patch_features: frozen backbone output, `[B, num_frames,
                num_patches, D_in]`, `num_patches == patch_h * patch_w` in
                row-major order.
            patch_h, patch_w: the patch grid `patch_features` covers.

        Returns:
            `block_outputs`: `2 * cfg.depth` tensors, `[frame_0, inter_0,
            frame_1, inter_1, ...]`, each `[B, num_frames,
            patch_token_start + num_patches, embed_dim]`.
        """
        batch_size, num_frames, num_patches, _ = patch_features.shape
        if num_patches != patch_h * patch_w:
            raise ValueError(f"patch_features has {num_patches} patches, expected {patch_h}x{patch_w}={patch_h * patch_w}")

        x = self.proj(patch_features.float()).reshape(batch_size * num_frames, num_patches, self.cfg.embed_dim)
        camera = self.camera_token.expand(batch_size * num_frames, -1, -1)
        register = self.register_token.expand(batch_size * num_frames, -1, -1)
        tokens = torch.cat([camera, register, x], dim=1)
        num_tokens = tokens.shape[1]
        tokens = tokens.view(batch_size, num_frames, num_tokens, self.cfg.embed_dim)

        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_h, W=patch_w)
        frame_rope = (rope_sin.to(device=tokens.device, dtype=torch.float32), rope_cos.to(device=tokens.device, dtype=torch.float32))

        block_outputs: list[torch.Tensor] = []
        for block_idx in range(self.cfg.depth):
            flat = tokens.view(batch_size * num_frames, num_tokens, self.cfg.embed_dim)
            flat = self.frame_blocks[block_idx](flat, frame_rope)
            tokens = flat.view(batch_size, num_frames, num_tokens, self.cfg.embed_dim)
            block_outputs.append(tokens)

            flat_global = tokens.view(batch_size, num_frames * num_tokens, self.cfg.embed_dim)
            flat_global = self.inter_frame_blocks[block_idx](flat_global, None)
            tokens = flat_global.view(batch_size, num_frames, num_tokens, self.cfg.embed_dim)
            block_outputs.append(tokens)

        return block_outputs


def dense_head_hook_indices(depth: int) -> list[int]:
    """4 evenly-spaced tap indices spanning `MultiViewAdapter`'s `2*depth`-long
    `block_outputs` list, always including the first and last tap. Unlike
    `experiments/ffrecon/train_poincar3.py`'s `_dense_head_hook_indices` (which
    prefers inter-frame taps specifically, since `Poincar3.forward`'s raw
    `capture_block_outputs` alternates frame-only and cross-view-attended
    taps), every tap here is already post-inter-frame-attention, so plain
    even spacing is enough.
    """
    total = 2 * depth
    if total < 4:
        raise ValueError(f"MultiViewAdapter needs depth>=2 for a 4-hook DenseHead, got depth={depth} ({total} taps)")
    return [round(i * (total - 1) / 3) for i in range(4)]
