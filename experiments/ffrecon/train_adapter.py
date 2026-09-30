from __future__ import annotations

import dataclasses
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from poincar3.baselines import Backbone, load_backbone
from poincar3.benchmarks.ffrecon import RelposeBenchmark
from poincar3.benchmarks.ffrecon.inference import AdapterReconModel
from poincar3.data.dynamic import DynamicBatchSampler
from poincar3.data.ffreconstruction.normalization import normalize_batch_scale, normalize_pose_scale
from poincar3.data.ffreconstruction.sequence import ReconBatch
from poincar3.device import device
from poincar3.distrib import init_distributed, is_distributed, is_main_process
from poincar3.finetune import (
    FinetuneReconBaseCfg,
    build_dataloader,
    build_ema,
    build_eval_loader,
    build_relpose_benchmarks,
    cli_main,
    clip_grad_norm,
    evaluate_held_out_loss,
    maybe_ddp,
    unwrap_ddp,
)
from poincar3.finetune import run_relpose_benchmarks as _run_relpose_benchmarks
from poincar3.heads.camera_head import CameraHead
from poincar3.heads.dense_head import DenseHead
from poincar3.heads.mv_adapter import MultiViewAdapter, dense_head_hook_indices
from poincar3.heads.recon_loss import compute_camera_loss, compute_depth_loss
from poincar3.logging import logger
from poincar3.run import init_wandb, set_torch_misc


@dataclass(frozen=True)
class ProbeReconAdapterCfg(FinetuneReconBaseCfg):
    # Which frozen backbone to probe (see `poincar3.baselines`). For
    # `poincar3`, `run_path` selects a pretraining run; omit it for the
    # released checkpoint.
    backbone: Backbone = "poincar3"
    run_path: str | None = None

    adapter: MultiViewAdapter.Cfg = MultiViewAdapter.Cfg()

    # Adapter + heads are much cheaper per frame than a full backbone
    # forward/backward, so a wider frame budget stays affordable.
    sampling: DynamicBatchSampler.Cfg = DynamicBatchSampler.Cfg(
        seq_len=(2, 24), teacher_len=(0, 0), num_frames=96
    )

    lr: float = 2e-4
    warmup_steps: int = 4_000
    # Applied uniformly to every trainable parameter through a single AdamW
    # group -- the no-decay split matters more for a full pretrained backbone
    # than for these small, freshly initialized modules.
    weight_decay: float = 0.01
    eval_interval: int = 2000

    out_dir: str = "experiments/ffrecon/runs/adapter"


def cosine_warmup_lr(step: int, max_steps: int, warmup_steps: int, min_factor: float = 0.01) -> float:
    if warmup_steps == 0 or step <= warmup_steps:
        return (1 - min_factor) * (step / max(warmup_steps, 1)) + min_factor
    rel = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return (1 - min_factor) * math.cos(0.5 * rel * math.pi) + min_factor


def load_frozen_backbone(cfg: ProbeReconAdapterCfg) -> nn.Module:
    model, step = load_backbone(cfg.backbone, cfg.run_path)
    logger.info(f"Loaded frozen {cfg.backbone} backbone (step {step}).")
    model.eval()
    model.requires_grad_(False)
    return model


def compute_loss(
    backbone: nn.Module,
    adapter: MultiViewAdapter,
    camera_head: nn.Module,
    depth_head: nn.Module,
    batch: ReconBatch,
    cfg: ProbeReconAdapterCfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    poses_norm, depths_norm, scale = normalize_batch_scale(batch.poses, batch.depths, batch.depth_masks)
    height, width = batch.imgs.shape[-2:]
    patch_h, patch_w = height // backbone.patch_size, width // backbone.patch_size

    with torch.no_grad():
        _, patch_features, _, _ = backbone(batch.imgs, mask=None, head_mask=None)
    block_outputs = adapter(patch_features, patch_h, patch_w)
    patch_token_start = unwrap_ddp(adapter).patch_token_start

    pred_pose_encoding = camera_head([block_outputs[-1]], patch_token_start)
    pred_depth, pred_depth_conf = depth_head(block_outputs, batch.imgs, patch_token_start)

    camera_losses = compute_camera_loss(
        pred_pose_encoding[-1], normalize_pose_scale(batch.poses), batch.K, (height, width)
    )
    depth_losses = compute_depth_loss(
        pred_depth,
        pred_depth_conf,
        depths_norm,
        batch.depth_masks,
        gamma=cfg.depth_conf_gamma,
        alpha=cfg.depth_conf_alpha,
    )

    loss = cfg.weight_camera * camera_losses["loss_camera"] + cfg.weight_depth * depth_losses["loss_depth"]
    stats = {
        **camera_losses,
        **depth_losses,
        "scale": scale.mean(),
        "num_views": batch.imgs.shape[1],
        "loss": loss,
    }
    return loss, stats


def evaluate(
    backbone: nn.Module,
    adapter: MultiViewAdapter,
    camera_head: nn.Module,
    depth_head: nn.Module,
    eval_loader: DataLoader,
    cfg: ProbeReconAdapterCfg,
) -> dict[str, float]:
    # `backbone` is permanently frozen and in `.eval()`, so it stays out of the
    # train/eval toggle list.
    return evaluate_held_out_loss(
        lambda batch: compute_loss(backbone, adapter, camera_head, depth_head, batch, cfg),
        [adapter, camera_head, depth_head],
        eval_loader,
        cfg.eval_batches,
    )


def run_relpose_benchmarks(
    benchmarks: list[tuple[str, RelposeBenchmark]],
    backbone: nn.Module,
    adapter: MultiViewAdapter,
    camera_head: nn.Module,
    depth_head: nn.Module,
    step: int,
) -> None:
    recon_model = AdapterReconModel(backbone, adapter, camera_head, depth_head)
    _run_relpose_benchmarks(benchmarks, recon_model, [adapter, camera_head, depth_head], step)


def save_checkpoint(
    run_dir: Path,
    step: int,
    cfg: ProbeReconAdapterCfg,
    adapter: nn.Module,
    camera_head: nn.Module,
    depth_head: nn.Module,
    optimizer: AdamW,
    ema: AveragedModel,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ema_adapter, ema_camera_head, ema_depth_head = ema.module[0], ema.module[1], ema.module[2]
    torch.save(
        {
            # The frozen backbone's weights aren't saved -- eval reloads them
            # from their original source -- but which source needs recording so
            # `load_recon_checkpoint` can rebuild it.
            "backbone": cfg.backbone,
            "run_path": cfg.run_path,
            "adapter_cfg": dataclasses.asdict(cfg.adapter),
            "adapter": adapter.state_dict(),
            "camera_head": camera_head.state_dict(),
            "depth_head": depth_head.state_dict(),
            # EMA copies -- what eval actually scores. The frozen backbone has
            # no EMA counterpart.
            "adapter_ema": ema_adapter.state_dict(),
            "camera_head_ema": ema_camera_head.state_dict(),
            "depth_head_ema": ema_depth_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        run_dir / "checkpoint.pth",
    )
    logger.info(f"Saved checkpoint at step {step} to {run_dir / 'checkpoint.pth'}")


def main(cfg: ProbeReconAdapterCfg) -> None:
    init_distributed()
    set_torch_misc(cfg)

    backbone = load_frozen_backbone(cfg)
    adapter = MultiViewAdapter(cfg.adapter, in_dim=backbone.register_token.shape[-1]).to(device)
    camera_head = CameraHead(dim_in=cfg.adapter.embed_dim).to(device)
    depth_head = DenseHead(
        dim_in=cfg.adapter.embed_dim,
        patch_size=backbone.patch_size,
        intermediate_layer_idx=dense_head_hook_indices(cfg.adapter.depth),
    ).to(device)
    # No EMA counterpart for the frozen backbone: its weights never move.
    ema, ema_live = build_ema([adapter, camera_head, depth_head], cfg.ema_decay)
    d_adapter = maybe_ddp(adapter)
    d_camera_head = maybe_ddp(camera_head)
    d_depth_head = maybe_ddp(depth_head)

    batch_sampler, train_loader = build_dataloader(cfg)
    eval_loader = build_eval_loader(cfg)
    relpose_benchmarks = build_relpose_benchmarks(cfg) if is_main_process() else []

    trainable = [*adapter.parameters(), *camera_head.parameters(), *depth_head.parameters()]
    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    run_name = cfg.name or f"{cfg.backbone}-adapter"
    run_dir = Path(cfg.out_dir) / run_name
    init_wandb(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        config=asdict(cfg),
        name=run_name,
        enabled=cfg.wandb,
    )

    adapter.train()
    camera_head.train()
    depth_head.train()

    epoch = 0
    batch_sampler.set_epoch(epoch)
    data_iter = iter(train_loader)
    pbar = tqdm(range(cfg.num_steps))
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            batch_sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = batch.to(device)

        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * cosine_warmup_lr(step, cfg.num_steps, cfg.warmup_steps)

        loss, stats = compute_loss(backbone, d_adapter, d_camera_head, d_depth_head, batch, cfg)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm([adapter, camera_head, depth_head], cfg.grad_clip_norm)

        # A degenerate batch (near-zero valid depth) can make the loss or
        # gradient non-finite; skip the update rather than corrupting the
        # adapter/heads with a NaN step. `loss` is each rank's own local value
        # (DDP only all-reduces gradients), so ranks could otherwise disagree
        # and desync their optimizer state -- take the MIN, so any rank seeing
        # a non-finite value skips for all.
        step_finite = bool(torch.isfinite(loss) and torch.isfinite(grad_norm))
        if is_distributed():
            finite_flag = torch.tensor(1.0 if step_finite else 0.0, device=device)
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            step_finite = bool(finite_flag.item())
        if step_finite:
            optimizer.step()
            ema.update_parameters(ema_live)
        else:
            logger.warning(f"step {step}: non-finite loss/grad -- skipping optimizer step.")

        if step % cfg.log_interval == 0 and is_main_process():
            log = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in stats.items()}
            log["grad_norm"] = float(grad_norm)
            log["step_skipped"] = float(not step_finite)
            log["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(log, step=step)
            pbar.set_description(f"loss={log['loss']:.4f} N={log['num_views']}")

        do_ckpt = step > 0 and step % cfg.ckpt_interval == 0
        do_eval = step % cfg.eval_interval == 0
        needs_sync = (do_ckpt or do_eval) and is_distributed()
        if needs_sync:
            dist.barrier()

        if do_eval and is_main_process():
            ema_adapter, ema_camera_head, ema_depth_head = ema.module[0], ema.module[1], ema.module[2]
            eval_stats = evaluate(backbone, ema_adapter, ema_camera_head, ema_depth_head, eval_loader, cfg)
            if eval_stats:
                wandb.log(eval_stats, step=step)
                logger.info(f"step {step}: eval_loss={eval_stats['eval_loss']:.4f}")
            run_relpose_benchmarks(relpose_benchmarks, backbone, ema_adapter, ema_camera_head, ema_depth_head, step)

        if do_ckpt and is_main_process():
            save_checkpoint(run_dir, step, cfg, adapter, camera_head, depth_head, optimizer, ema)

        if needs_sync:
            dist.barrier()

    if is_main_process():
        save_checkpoint(run_dir, cfg.num_steps, cfg, adapter, camera_head, depth_head, optimizer, ema)
    wandb.finish()


if __name__ == "__main__":
    cli_main(ProbeReconAdapterCfg, main)
