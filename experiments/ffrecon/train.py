from __future__ import annotations

import dataclasses
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

from poincar3.benchmarks.ffrecon import RelposeBenchmark
from poincar3.benchmarks.ffrecon.inference import (
    Poincar3ReconModel,
    _camera_readout_tokens,
    _normalize_block_outputs,
    dense_head_hook_indices_poincar3,
)
from poincar3.data.ffreconstruction.normalization import normalize_batch_scale, normalize_pose_scale
from poincar3.data.ffreconstruction.sequence import ReconBatch
from poincar3.device import device
from poincar3.distrib import init_distributed, is_distributed, is_main_process
from poincar3.finetune import (
    FinetuneReconBaseCfg,
    build_dataloader,
    build_ema,
    build_eval_loader,
    build_optimizer_and_scheduler,
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
from poincar3.heads.recon_loss import compute_camera_loss, compute_depth_loss
from poincar3.logging import logger
from poincar3.model import Poincar3
from poincar3.run import create_grad_scaler, init_wandb, set_torch_misc
from poincar3.run import load_run as load_pretrained_run


@dataclass(frozen=True)
class FinetuneReconCfg(FinetuneReconBaseCfg):
    # A pretraining run directory, as written by `poincar3.run.dump_run`. If
    # `None`, `pretrained` below decides what is loaded instead.
    run_path: str | None = None
    # Load the run's EMA teacher weights, which is what eval scores; set False
    # to start from the raw student. Ignored when `run_path is None`.
    load_teacher: bool = True
    # Only consulted when `run_path is None`. True loads the released
    # checkpoint; False finetunes a freshly constructed `Poincar3` instead --
    # the "no cross-view pretraining" control.
    pretrained: bool = True
    # Architecture for the `pretrained=False` control; ignored otherwise. Note
    # `Poincar3.Cfg.encoder_pretrained` defaults to False, so `--no-pretrained`
    # alone gives a fully random init -- pass
    # `--model-cfg.encoder-pretrained` for the DINOv3-encoder-only arm.
    model_cfg: Poincar3.Cfg = Poincar3.Cfg()

    out_dir: str = "experiments/ffrecon/runs/finetune"


def load_pretrained_model(cfg: FinetuneReconCfg) -> tuple[Poincar3, Poincar3.Cfg]:
    if cfg.run_path is None:
        if cfg.pretrained:
            logger.info("No --run-path given: finetuning the released Poincar3 checkpoint.")
            return Poincar3().to(device), Poincar3.Cfg()
        logger.info(
            "--no-pretrained: finetuning a freshly constructed Poincar3 "
            f"(encoder_pretrained={cfg.model_cfg.encoder_pretrained})."
        )
        return Poincar3(cfg.model_cfg).to(device), cfg.model_cfg

    run_cfg, step, weights, _, _ = load_pretrained_run(Path(cfg.run_path))
    model = Poincar3(run_cfg.model).to(device)
    prefix = "teacher." if cfg.load_teacher else "student."
    src_weights = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
    if not src_weights:
        raise ValueError(f"No '{prefix}*' keys found in {cfg.run_path}/weights.pth")
    model.load_state_dict(src_weights, strict=True)
    logger.info(f"Loaded {prefix.rstrip('.')} weights from {cfg.run_path} (step {step}).")
    return model, run_cfg.model


def compute_loss(
    model: Poincar3,
    camera_head: nn.Module,
    depth_head: nn.Module,
    batch: ReconBatch,
    cfg: FinetuneReconCfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    poses_norm, depths_norm, scale = normalize_batch_scale(batch.poses, batch.depths, batch.depth_masks)
    batch_size, num_frames, _, height, width = batch.imgs.shape
    raw_model = unwrap_ddp(model)

    with torch.set_grad_enabled(not cfg.freeze_backbone):
        *_, block_outputs = model(batch.imgs, mask=None, head_mask=None, capture_block_outputs=True)
    tokens_list = _normalize_block_outputs(block_outputs, batch_size, num_frames)
    camera_tap = _camera_readout_tokens(tokens_list[-1], raw_model.num_register_tokens)

    pred_pose_encoding = camera_head([camera_tap], raw_model.patch_token_start)
    pred_depth, pred_depth_conf = depth_head(tokens_list, batch.imgs, raw_model.patch_token_start)

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
        "num_views": num_frames,
        "loss": loss,
    }
    return loss, stats


def evaluate(
    model: Poincar3,
    camera_head: nn.Module,
    depth_head: nn.Module,
    eval_loader: DataLoader,
    cfg: FinetuneReconCfg,
) -> dict[str, float]:
    return evaluate_held_out_loss(
        lambda batch: compute_loss(model, camera_head, depth_head, batch, cfg),
        [model, camera_head, depth_head],
        eval_loader,
        cfg.eval_batches,
    )


def run_relpose_benchmarks(
    benchmarks: list[tuple[str, RelposeBenchmark]],
    model: Poincar3,
    camera_head: nn.Module,
    depth_head: nn.Module,
    step: int,
) -> None:
    recon_model = Poincar3ReconModel(model, camera_head, depth_head)
    _run_relpose_benchmarks(benchmarks, recon_model, [model, camera_head, depth_head], step)


def save_checkpoint(
    run_dir: Path,
    step: int,
    model: Poincar3,
    model_cfg: Poincar3.Cfg,
    camera_head: nn.Module,
    depth_head: nn.Module,
    optimizer: AdamW,
    ema: AveragedModel,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ema_model, ema_camera_head, ema_depth_head = ema.module[0], ema.module[1], ema.module[2]
    torch.save(
        {
            "backbone": "poincar3",
            # Lets `poincar3.benchmarks.ffrecon.inference.load_recon_checkpoint`
            # rebuild this exact architecture at eval time.
            "model_cfg": dataclasses.asdict(model_cfg),
            "model": model.state_dict(),
            "camera_head": camera_head.state_dict(),
            "depth_head": depth_head.state_dict(),
            # EMA copies -- what eval actually scores.
            "model_ema": ema_model.state_dict(),
            "camera_head_ema": ema_camera_head.state_dict(),
            "depth_head_ema": ema_depth_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        run_dir / "checkpoint.pth",
    )
    logger.info(f"Saved checkpoint at step {step} to {run_dir / 'checkpoint.pth'}")


def main(cfg: FinetuneReconCfg) -> None:
    # Must run before `build_dataloader`: `DynamicSampler` rank-shards at
    # construction time.
    init_distributed()
    set_torch_misc(cfg)

    model, model_cfg = load_pretrained_model(cfg)
    if cfg.freeze_backbone:
        model.requires_grad_(False)
        model.eval()
    camera_head = CameraHead(dim_in=model_cfg.embed_dim).to(device)
    depth_head = DenseHead(
        dim_in=model_cfg.embed_dim,
        patch_size=model.patch_size,
        intermediate_layer_idx=dense_head_hook_indices_poincar3(model_cfg.depth),
    ).to(device)
    ema, ema_live = build_ema([model, camera_head, depth_head], cfg.ema_decay)
    # A frozen backbone never receives gradients, so it needs no DDP wrapper.
    d_model = model if cfg.freeze_backbone else maybe_ddp(model)
    d_camera_head = maybe_ddp(camera_head)
    d_depth_head = maybe_ddp(depth_head)

    batch_sampler, loader = build_dataloader(cfg)
    eval_loader = build_eval_loader(cfg)
    # Only rank 0 evaluates, so no other rank needs the annotations in memory.
    relpose_benchmarks = build_relpose_benchmarks(cfg) if is_main_process() else []

    if cfg.freeze_backbone:
        backbone_modules, decoder_modules = [], []
    else:
        # Only `model.encoder` stays at the gentle `cfg.lr`; the cross-view
        # decoder trains at `head_lr` alongside the heads.
        decoder_tokens = nn.ParameterList([model.register_token, model.camera_token])
        decoder_modules = [model.proj, model.frame_blocks, model.inter_frame_blocks, decoder_tokens]
        backbone_modules = [model.encoder]
    optimizer, scheduler = build_optimizer_and_scheduler(
        cfg, backbone_modules, [camera_head, depth_head], decoder_modules
    )
    grad_scaler = create_grad_scaler()

    run_dir = Path(cfg.out_dir) / (cfg.name or "finetune")
    init_wandb(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        config=asdict(cfg),
        name=cfg.name,
        enabled=cfg.wandb,
    )

    if not cfg.freeze_backbone:
        model.train()
    camera_head.train()
    depth_head.train()

    epoch = 0
    batch_sampler.set_epoch(epoch)
    data_iter = iter(loader)
    pbar = tqdm(range(cfg.num_steps))
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            batch_sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = batch.to(device)

        loss, stats = compute_loss(d_model, d_camera_head, d_depth_head, batch, cfg)
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)  # must precede clipping, which needs real grad norms
        grad_norm = clip_grad_norm([model, camera_head, depth_head], cfg.grad_clip_norm)
        grad_scaler.step(optimizer)  # no-ops the update if any grad is inf/nan
        grad_scaler.update()
        scheduler.step()
        ema.update_parameters(ema_live)

        if step % cfg.log_interval == 0 and is_main_process():
            log = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in stats.items()}
            log["grad_norm"] = float(grad_norm)
            log["grad_scale"] = grad_scaler.get_scale()
            log["lr_backbone"] = 0.0 if cfg.freeze_backbone else optimizer.param_groups[0]["lr"]
            log["lr_head"] = optimizer.param_groups[-1]["lr"]
            log["batch_size"] = batch.imgs.shape[0]
            wandb.log(log, step=step)
            pbar.set_description(f"loss={log['loss']:.4f} N={log['num_views']} B={log['batch_size']}")

        # Checkpointing and eval run on rank 0 only; barrier on both sides so
        # the other ranks rendezvous here instead of idling in the next step's
        # all_reduce and eating into the NCCL watchdog timeout.
        do_ckpt = step > 0 and step % cfg.ckpt_interval == 0
        do_eval = step % cfg.eval_interval == 0
        needs_sync = (do_ckpt or do_eval) and is_distributed()
        if needs_sync:
            dist.barrier()

        if do_eval and is_main_process():
            ema_model, ema_camera_head, ema_depth_head = ema.module[0], ema.module[1], ema.module[2]
            eval_stats = evaluate(ema_model, ema_camera_head, ema_depth_head, eval_loader, cfg)
            if eval_stats:
                wandb.log(eval_stats, step=step)
                logger.info(f"step {step}: eval_loss={eval_stats['eval_loss']:.4f}")
            run_relpose_benchmarks(relpose_benchmarks, ema_model, ema_camera_head, ema_depth_head, step)

        if do_ckpt and is_main_process():
            save_checkpoint(run_dir, step, model, model_cfg, camera_head, depth_head, optimizer, ema)

        if needs_sync:
            dist.barrier()

    if is_main_process():
        save_checkpoint(run_dir, cfg.num_steps, model, model_cfg, camera_head, depth_head, optimizer, ema)
    wandb.finish()


if __name__ == "__main__":
    cli_main(FinetuneReconCfg, main)
