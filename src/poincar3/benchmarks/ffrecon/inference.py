from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from poincar3.device import device
from poincar3.heads.camera_head import CameraHead
from poincar3.heads.dense_head import DenseHead
from poincar3.heads.mv_adapter import MultiViewAdapter
from poincar3.heads.mv_adapter import dense_head_hook_indices as _adapter_hook_indices
from poincar3.heads.pose_enc import pose_encoding_to_extri_intri
from poincar3.model import Poincar3


class ReconPrediction:
    """One sequence's worth of feed-forward-reconstruction output.

    extrinsics: `(N, 3, 4)`, cam-from-world, OpenCV convention.
    intrinsics: `(N, 3, 3)`.
    depth: `(N, H, W)`, camera-space z-depth.
    depth_conf: `(N, H, W)`.
    """

    def __init__(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor,
    ):
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.depth = depth
        self.depth_conf = depth_conf


class ReconModel(nn.Module):
    """Common interface both checkpoint kinds implement. Subclasses only need
    `_forward_taps`; `infer` handles the shared pose-encoding conversion and the
    depth-head call."""

    camera_head: CameraHead
    depth_head: DenseHead

    def _forward_taps(self, imgs_b: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], int]:
        """imgs_b: `(1, N, 3, H, W)`. Returns `(camera_tap, dense_taps,
        patch_token_start)` -- `camera_tap` is the single 4D tap `CameraHead`
        reads, `dense_taps` the list `DenseHead` hooks into."""
        raise NotImplementedError

    @torch.no_grad()
    def infer(self, imgs: torch.Tensor) -> ReconPrediction:
        """imgs: `(N, 3, H, W)`, raw `[0, 1]` -- one sequence, no batch dim."""
        imgs_b = imgs.unsqueeze(0)
        camera_tap, dense_taps, patch_token_start = self._forward_taps(imgs_b)
        # `CameraHead` returns one prediction per refinement iteration; the last
        # is the final, most-refined pose.
        pred_pose_encoding = self.camera_head([camera_tap], patch_token_start)[-1]
        pred_depth, pred_depth_conf = self.depth_head(dense_taps, imgs_b, patch_token_start)

        height, width = imgs.shape[-2:]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pred_pose_encoding, (height, width))
        return ReconPrediction(
            extrinsics=extrinsics[0],
            intrinsics=intrinsics[0],
            depth=pred_depth[0],
            depth_conf=pred_depth_conf[0],
        )


def _normalize_block_outputs(block_outputs: list[torch.Tensor], batch_size: int, num_frames: int) -> list[torch.Tensor]:
    return [t.reshape(batch_size, num_frames, -1, t.shape[-1]) for t in block_outputs]


def _camera_readout_tokens(tap: torch.Tensor, num_register_tokens: int) -> torch.Tensor:
    """`CameraHead` reads its camera token at prefix position 0, but Poincar3
    lays its prefix out as registers-then-camera -- reorder for the head."""
    camera = tap[:, :, num_register_tokens : num_register_tokens + 1]
    registers = tap[:, :, :num_register_tokens]
    rest = tap[:, :, num_register_tokens + 1 :]
    return torch.cat([camera, registers, rest], dim=2)


def dense_head_hook_indices_poincar3(depth: int) -> list[int]:
    """Four evenly spaced decoder taps. `2 * i + 1` picks the inter-frame block
    of each frame/inter-frame pair, since `capture_block_outputs` emits both."""
    quarter_block_indices = [round(depth * f) - 1 for f in (0.25, 0.5, 0.75, 1.0)]
    return [2 * i + 1 for i in quarter_block_indices]


class Poincar3ReconModel(ReconModel):
    def __init__(self, model: Poincar3, camera_head: CameraHead, depth_head: DenseHead):
        super().__init__()
        self.model = model
        self.camera_head = camera_head
        self.depth_head = depth_head

    def _forward_taps(self, imgs_b: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], int]:
        batch_size, num_frames = imgs_b.shape[:2]
        *_, block_outputs = self.model(imgs_b, mask=None, head_mask=None, capture_block_outputs=True)
        tokens_list = _normalize_block_outputs(block_outputs, batch_size, num_frames)
        camera_tap = _camera_readout_tokens(tokens_list[-1], self.model.num_register_tokens)
        return camera_tap, tokens_list, self.model.patch_token_start


class AdapterReconModel(ReconModel):
    def __init__(self, backbone: nn.Module, adapter: MultiViewAdapter, camera_head: CameraHead, depth_head: DenseHead):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.camera_head = camera_head
        self.depth_head = depth_head

    def _forward_taps(self, imgs_b: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], int]:
        height, width = imgs_b.shape[-2:]
        patch_h, patch_w = height // self.backbone.patch_size, width // self.backbone.patch_size
        with torch.no_grad():
            _, patch_features, _, _ = self.backbone(imgs_b, mask=None, head_mask=None)
        block_outputs = self.adapter(patch_features, patch_h, patch_w)
        return block_outputs[-1], block_outputs, self.adapter.patch_token_start


def load_recon_checkpoint(
    checkpoint_path: str | Path, run_path: str | None = None, use_ema: bool = True
) -> ReconModel:
    """Loads a `checkpoint.pth` from `experiments/ffrecon/train.py` or
    `train_adapter.py` and returns the matching `ReconModel`, dispatching on the
    checkpoint's own `"backbone"` tag.

    Args:
        run_path: only used for an adapter checkpoint trained with
            `--backbone poincar3`; overrides the frozen backbone's pretraining
            run path recorded in the checkpoint, e.g. if it has since moved.
        use_ema: load the `*_ema` weights (default), which is what the periodic
            in-training eval scores. Falls back to the raw key if absent.
    """
    # `weights_only=False`: these checkpoints deliberately carry the non-tensor
    # state this function reads back -- the `"backbone"` tag it dispatches on
    # and the config dicts it rebuilds the architecture from. Safe because they
    # are written by this repo's own `save_checkpoint`.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def sd(key: str) -> dict:
        ema_key = f"{key}_ema"
        return ckpt[ema_key] if use_ema and ema_key in ckpt else ckpt[key]

    if "adapter" in ckpt:
        from poincar3.baselines import load_backbone

        backbone, _ = load_backbone(ckpt["backbone"], run_path or ckpt.get("run_path"))
        backbone.eval()
        backbone.requires_grad_(False)

        adapter_cfg = MultiViewAdapter.Cfg(**ckpt["adapter_cfg"])
        adapter = MultiViewAdapter(adapter_cfg, in_dim=backbone.register_token.shape[-1]).to(device)
        adapter.load_state_dict(sd("adapter"), strict=True)
        camera_head = CameraHead(dim_in=adapter_cfg.embed_dim).to(device)
        camera_head.load_state_dict(sd("camera_head"), strict=True)
        depth_head = DenseHead(
            dim_in=adapter_cfg.embed_dim,
            patch_size=backbone.patch_size,
            intermediate_layer_idx=_adapter_hook_indices(adapter_cfg.depth),
        ).to(device)
        depth_head.load_state_dict(sd("depth_head"), strict=True)
        recon_model: ReconModel = AdapterReconModel(backbone, adapter, camera_head, depth_head)

    elif ckpt.get("backbone") == "poincar3" and "model" in ckpt:
        model_cfg = Poincar3.Cfg(**ckpt["model_cfg"])
        model = Poincar3(model_cfg).to(device)
        model.load_state_dict(sd("model"), strict=True)
        camera_head = CameraHead(dim_in=model_cfg.embed_dim).to(device)
        camera_head.load_state_dict(sd("camera_head"), strict=True)
        depth_head = DenseHead(
            dim_in=model_cfg.embed_dim,
            patch_size=model.patch_size,
            intermediate_layer_idx=dense_head_hook_indices_poincar3(model_cfg.depth),
        ).to(device)
        depth_head.load_state_dict(sd("depth_head"), strict=True)
        recon_model = Poincar3ReconModel(model, camera_head, depth_head)

    else:
        raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path} (keys: {list(ckpt.keys())})")

    recon_model.eval()
    recon_model.requires_grad_(False)
    return recon_model
