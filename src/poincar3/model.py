import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .layers import DINOHead, Mlp, RopePositionEmbedding, SelfAttentionBlock
from .layers.utils import named_apply
from .layers.vision_transformer import DinoVisionTransformer, init_weights_vit
from .types import Batch, Model

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

CHECKPOINT_URL = "https://github.com/davnords/storage/releases/download/poincar3/poincar3.pth"

_DINOV3_CHECKPOINTS = {
    "vits": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "vitb": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "vitl": "dinov3_vitl16_pretrain_lvd1689m-08c60483.pth",
}
_ENCODER_SIZES = {"vits": (384, 12, 6), "vitb": (768, 12, 12), "vitl": (1024, 24, 16)}


class Poincar3(Model):
    """Emergent multi-view geometry through self-distillation.

    A per-frame ViT encoder (`self.encoder`) followed by a cross-view decoder
    that alternates per-frame self-attention (`frame_blocks`) with inter-frame
    attention over the whole sequence (`inter_frame_blocks`).

    The same class is used as both student and teacher (see `SSLModel`): the
    student sees N masked views, the teacher sees those N views unmasked plus T
    extra ones. `loss.py` aligns the student's masked-patch predictions with the
    teacher's over the views both networks saw.

    Constructing it with no config downloads and loads the released pretrained
    weights:

        model = Poincar3()
    """

    @dataclass(frozen=True)
    class Cfg:
        embed_dim: int = 1024
        depth: int = 12
        num_heads: int = 16
        mlp_ratio: float = 4.0
        num_register_tokens: int = 16
        # Decoder blocks whose inter-frame attention runs over the register
        # tokens only, instead of over every patch of every frame.
        register_attention_block_indices: list[int] = field(default_factory=lambda: [2, 6, 9])
        encoder_size: Literal["vits", "vitb", "vitl"] = "vitl"
        # Initialize the encoder from a local DINOv3 checkpoint instead of from
        # scratch. The released model is trained from scratch.
        encoder_pretrained: bool = False
        # iBOT-style prototype head on the masked patch tokens.
        ibot_head_out_dim: int = 65536
        ibot_head_hidden_dim: int = 2048
        ibot_head_bottleneck_dim: int = 256
        # DINO-style prototype head on the per-frame camera token.
        global_head_out_dim: int = 65536
        global_head_hidden_dim: int = 2048
        global_head_bottleneck_dim: int = 256
        gradient_checkpointing: bool = False
        drop_path_rate: float = 0.0
        encoder_drop_path_rate: float = 0.0

    def __init__(self, cfg: Cfg | None = None):
        super().__init__()
        load_pretrained = cfg is None
        cfg = cfg or Poincar3.Cfg()

        self.encoder = _build_encoder(cfg.encoder_size, cfg.encoder_pretrained, cfg.encoder_drop_path_rate)
        self.proj = (
            nn.Linear(self.encoder.embed_dim, cfg.embed_dim)
            if self.encoder.embed_dim != cfg.embed_dim
            else nn.Identity()
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        def _block() -> SelfAttentionBlock:
            return SelfAttentionBlock(
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
                drop_path=cfg.drop_path_rate,
            )

        self.frame_blocks = nn.ModuleList([_block() for _ in range(cfg.depth)])
        self.inter_frame_blocks = nn.ModuleList([_block() for _ in range(cfg.depth)])

        self.depth = cfg.depth
        self.patch_size = 16
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.num_register_tokens = cfg.num_register_tokens
        # Registers first, then the camera token, then the patch tokens.
        self.patch_token_start = cfg.num_register_tokens + 1

        self.register_token = nn.Parameter(torch.empty(1, cfg.num_register_tokens, cfg.embed_dim))
        self.camera_token = nn.Parameter(torch.empty(1, 1, cfg.embed_dim))

        self.head = DINOHead(
            in_features=cfg.embed_dim,
            out_features=cfg.ibot_head_out_dim,
            hidden_features=cfg.ibot_head_hidden_dim,
            bottleneck_features=cfg.ibot_head_bottleneck_dim,
        )
        self.global_head = DINOHead(
            in_features=cfg.embed_dim,
            out_features=cfg.global_head_out_dim,
            hidden_features=cfg.global_head_hidden_dim,
            bottleneck_features=cfg.global_head_bottleneck_dim,
        )

        self.inter_frame_attention_types = ["global"] * cfg.depth
        for idx in cfg.register_attention_block_indices:
            if not 0 <= idx < cfg.depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_mean", _IMAGENET_MEAN), ("_std", _IMAGENET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

        if load_pretrained:
            state = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location="cpu")
            self.load_state_dict(state["model"], strict=True)

    def init_weights(self) -> None:
        nn.init.normal_(self.register_token, std=1e-3)
        nn.init.normal_(self.camera_token, std=1e-3)
        # The decoder blocks use `mask_k_bias`, whose bias-mask buffer starts
        # filled with NaN and is only set correctly by `init_weights_vit`. The
        # encoder does this internally; these blocks are ours.
        named_apply(init_weights_vit, self.frame_blocks)
        named_apply(init_weights_vit, self.inter_frame_blocks)

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        return_camera_tokens: bool = False,
        capture_block_outputs: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:
            images: `[batch, frames, 3, height, width]`.
            mask: `[batch, frames, patches]` bool, True = masked. Masked patches
                are replaced by the encoder's learned mask token before any
                self-attention runs. `None` for the teacher, which sees
                everything.
            head_mask: `[batch, frames, patches]` bool selecting the positions
                `self.head` is evaluated at. Independent of `mask`: the teacher
                sees every patch but only needs head outputs where the loss
                reads them. `None` runs the head densely, which eval paths want
                but training cannot afford -- `ibot_head_out_dim` makes those
                activations the dominant memory cost of a step.
            return_camera_tokens: also return the final layer's per-frame camera
                token, which the reconstruction heads read.
            capture_block_outputs: also return the full token tensor after every
                block, `2 * depth` tensors ordered
                `[frame_0, inter_0, frame_1, ...]`. Used by the DPT-style dense
                head and by layer-wise probing.

        Returns:
            `(patch_logits, patch_features, global_logits, global_raw)`, plus
            `camera_tokens` and/or `block_outputs` when requested.
            `patch_logits` is `self.head`'s output over the final decoder layer,
            either dense `[batch, frames, patches, ibot_head_out_dim]` or, with a
            `head_mask`, flattened to `[selected, ibot_head_out_dim]`.
            `patch_features` is always the dense pre-head patch tokens.
            `global_raw`/`global_logits` are the per-frame camera token before
            and after `self.global_head`.
        """
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        amp_enabled = images.device.type == "cuda"
        amp_dtype = torch.bfloat16 if not amp_enabled or torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            images = (images - self._mean) / self._std
            images = images.view(batch_size * num_frames, num_channels, height, width)
            mask = mask.reshape(batch_size * num_frames, -1) if mask is not None else None

            patch_tokens = self.encoder(images, masks=mask)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            patch_tokens = self.proj(patch_tokens)

            tokens = torch.cat(
                [
                    self.register_token.expand(batch_size * num_frames, -1, -1),
                    self.camera_token.expand(batch_size * num_frames, -1, -1),
                    patch_tokens,
                ],
                dim=1,
            )
            _, num_tokens, embed_dim = tokens.shape

            with torch.no_grad():
                rope_sin, rope_cos = self.rope_embed(H=height // self.patch_size, W=width // self.patch_size)
                frame_rope = (
                    rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                    rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
                )

            block_outputs: list[torch.Tensor] | None = [] if capture_block_outputs else None
            use_checkpoint = self.gradient_checkpointing and self.training
            shape = (batch_size, num_frames, num_tokens, embed_dim)
            for block_idx in range(self.depth):
                args = (*shape, block_idx)
                if use_checkpoint:
                    tokens = checkpoint(self._run_frame_block, tokens, *args, frame_rope, use_reentrant=False)
                else:
                    tokens = self._run_frame_block(tokens, *args, frame_rope)
                if block_outputs is not None:
                    block_outputs.append(tokens)

                attention_type = self.inter_frame_attention_types[block_idx]
                if use_checkpoint:
                    tokens = checkpoint(self._run_inter_frame_block, tokens, *args, attention_type, use_reentrant=False)
                else:
                    tokens = self._run_inter_frame_block(tokens, *args, attention_type)
                if block_outputs is not None:
                    block_outputs.append(tokens)

            # Only the final decoder layer is supervised, as in DINOv2/iBOT.
            camera_tokens = tokens[:, :, self.num_register_tokens]
            global_logits = self.global_head(camera_tokens)

            patch_features = tokens[:, :, self.patch_token_start :]
            # Gather before the head so its (large) output is never materialized
            # over the discarded majority of patches.
            head_input = patch_features[head_mask] if head_mask is not None else patch_features
            patch_logits = self.head(head_input)

        result = (patch_logits, patch_features, global_logits, camera_tokens)
        if return_camera_tokens:
            result = result + (camera_tokens,)
        if capture_block_outputs:
            result = result + (block_outputs,)
        return result

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        return self.frame_blocks[block_idx](tokens, rope_sincos)

    def _run_inter_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens = self.inter_frame_blocks[block_idx](tokens, None)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)
        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        start = self.patch_token_start
        register_tokens = tokens[:, :, :start].reshape(batch_size, num_frames * start, embed_dim)
        register_tokens = self.inter_frame_blocks[block_idx](register_tokens, None)
        register_tokens = register_tokens.view(batch_size, num_frames, start, embed_dim)
        return torch.cat([register_tokens, tokens[:, :, start:]], dim=2)

    def attention_probe(self, block_index: int | None = None):
        """Handle onto one inter-frame block's attention, for reading cross-view
        correspondences straight off the model (see `benchmarks.dense_features`).
        Defaults to the last block. Returns `None` for a "register" block, which
        has no direct patch-to-patch cross-view attention to read.
        """
        from poincar3.benchmarks.dense_features import AttentionProbe

        idx = self.depth - 1 if block_index is None else block_index
        if not 0 <= idx < self.depth:
            raise ValueError(f"block_index={idx} out of range [0, {self.depth})")
        if self.inter_frame_attention_types[idx] != "global":
            return None
        return AttentionProbe(
            attn_module=self.inter_frame_blocks[idx].attn,
            block_module=self.inter_frame_blocks[idx],
            patch_token_start=self.patch_token_start,
            run_forward=lambda images: self(images.unsqueeze(0), mask=None),
        )


class SSLModel(nn.Module):
    """A trainable student `Poincar3` plus an EMA teacher copy of it.

    The teacher starts as a deep copy of the student and is only ever updated by
    `update_teacher` (never by gradient descent).
    """

    def __init__(self, cfg: Poincar3.Cfg):
        super().__init__()
        self.cfg = cfg
        self.student = Poincar3(cfg)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)

    def forward(self, batch: Batch) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        # `batch.imgs` and `batch.student_imgs` are independently augmented views
        # of the same leading frames; the teacher additionally sees extra frames
        # the student never does.
        #
        # The teacher's `head_mask` is `batch.mask` padded with all-False extra
        # frames: the loss never looks past the student-visible frames, so no
        # head work is spent on frames it would discard.
        student_out = self.student(batch.student_imgs, mask=batch.mask, head_mask=batch.mask)
        with torch.no_grad():
            teacher_head_mask = batch.mask.new_zeros(batch.imgs.shape[0], batch.imgs.shape[1], batch.mask.shape[-1])
            teacher_head_mask[:, : batch.mask.shape[1]] = batch.mask
            teacher_out = self.teacher(batch.imgs, mask=None, head_mask=teacher_head_mask)
        return student_out, teacher_out

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.mul_(momentum).add_(student_param, alpha=1 - momentum)
        for teacher_buffer, student_buffer in zip(self.teacher.buffers(), self.student.buffers()):
            teacher_buffer.copy_(student_buffer)


def _build_encoder(
    encoder_size: Literal["vits", "vitb", "vitl"], pretrained: bool = False, drop_path_rate: float = 0.0
) -> DinoVisionTransformer:
    if encoder_size not in _ENCODER_SIZES:
        raise ValueError(f"Unknown encoder_size: {encoder_size}")
    embed_dim, depth, num_heads = _ENCODER_SIZES[encoder_size]

    model = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    if not pretrained:
        model.init_weights()
        return model

    path = Path(os.environ.get("DINOV3_DIR", ".")) / _DINOV3_CHECKPOINTS[encoder_size]
    if not path.exists():
        raise FileNotFoundError(
            f"DINOv3 weights not found at {path}. Request them from "
            "https://github.com/facebookresearch/dinov3, then put the file in the working "
            "directory or point $DINOV3_DIR at wherever it lives."
        )
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return model
