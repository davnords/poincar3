from __future__ import annotations

import json
import math
import os
import pickle
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Generic, TypeVar

import torch
import torch.amp.grad_scaler
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from poincar3.benchmarks.mv_consistency import MvConsistencyBenchmark
from poincar3.data.dynamic import DynamicBatchSampler, DynamicSampler
from poincar3.data.sequence_data import ComposedSequenceDataset, ScanNetPlusPlusSequence, VideoFrameSequence
from poincar3.device import device
from poincar3.distrib import is_distributed, is_main_process
from poincar3.logging import logger
from poincar3.loss import Poincar3Loss
from poincar3.model import Poincar3
from poincar3.types import Batch

EvalCallback = Callable[[nn.Module, int, Path], None]


@dataclass(frozen=True)
class TrainData:
    """The pretraining mixture. Each dataset carries its own `.weight`; set a
    weight to 0 to drop it from the mix."""

    re10k: VideoFrameSequence.Cfg = VideoFrameSequence.Cfg(data_root="data/re10k", split="train", weight=25_000)
    scannetpp: ScanNetPlusPlusSequence.Cfg = ScanNetPlusPlusSequence.Cfg(split="train", weight=25_000)


@dataclass(frozen=True)
class Poincar3Cfg:
    train_data: TrainData = TrainData()
    sampling: DynamicBatchSampler.Cfg = DynamicBatchSampler.Cfg(
        seq_len=(2, 24), teacher_len=(0, 12), num_frames=64
    )
    num_workers: int = 12
    prefetch_factor: int = 2

    model: Poincar3.Cfg = Poincar3.Cfg()
    loss: Poincar3Loss.Cfg = Poincar3Loss.Cfg()

    # Periodic multi-view correspondence evals during training.
    eval_scannet: MvConsistencyBenchmark.Cfg = MvConsistencyBenchmark.Cfg(
        dataset="scannet", correspondence_method="attention"
    )
    eval_megadepth: MvConsistencyBenchmark.Cfg = MvConsistencyBenchmark.Cfg(
        dataset="megadepth", correspondence_method="attention"
    )
    run_eval_scannet: bool = True
    run_eval_megadepth: bool = True

    # Run / logging
    name: str | None = None
    run_dir: str = "experiments/runs"
    resume_run: str | None = None
    wandb: bool = True
    wandb_entity: str | None = None
    wandb_project: str = "poincar3"

    # Optimization
    num_steps: int = 500_000
    lr: float = 2e-4
    warmup_steps: int = 10_000
    # Per-layer LR discount for the encoder (1.0 disables it).
    layerwise_decay: float = 1.0
    patch_embed_lr_mult: float = 1.0
    weight_decay: float = 0.04
    # Set to ramp weight decay up over the run (DINOv2 uses 0.04 -> 0.2);
    # `None` holds it flat.
    weight_decay_end: float | None = None
    grad_clip_norm: float = 1.0
    # EMA momentum: linear warmup from `ema_decay_warmup_start`, then flat.
    ema_decay: float = 0.999
    ema_decay_end: float | None = None
    ema_decay_warmup_start: float = 0.994
    ema_decay_warmup_steps: int = 30_000
    # Freeze the prototype heads' weight-normalized last layer for this many
    # steps, so early chaotic gradients don't drag it around while the teacher
    # temperature is still warming up (DINO's `freeze_last_layer_epochs`).
    freeze_last_layer_steps: int = 2000

    eval_interval: int = 10_000
    ckpt_interval: int = 2000
    # How often a checkpoint is archived into its own `step_<n>/` directory
    # instead of overwriting the run's latest.
    step_dir_interval: int = 25_000
    epoch_size: int = 2000

    omp_num_threads: int = 12
    dry_run: bool = False
    no_test: bool = False


CfgT = TypeVar("CfgT", bound=Poincar3Cfg)


@dataclass(frozen=True)
class CosineSchedule:
    """Optional linear warmup to `base_value`, then a cosine anneal to
    `final_value` over the rest of `total_steps`. `final_value=None` holds flat
    at `base_value` after warmup instead."""

    base_value: float
    total_steps: int
    final_value: float | None = None
    warmup_steps: int = 0
    start_warmup_value: float | None = None

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            start = self.base_value if self.start_warmup_value is None else self.start_warmup_value
            return start + (self.base_value - start) * step / self.warmup_steps
        final_value = self.base_value if self.final_value is None else self.final_value
        progress_steps = min(step - self.warmup_steps, self.total_steps - self.warmup_steps)
        progress = progress_steps / max(self.total_steps - self.warmup_steps, 1)
        return final_value + 0.5 * (self.base_value - final_value) * (1 + math.cos(math.pi * progress))


@dataclass
class RunState(Generic[CfgT]):
    cfg: CfgT
    step: int
    weights: dict[str, torch.Tensor] | None
    optimizer_state: dict | None
    scheduler_state: dict | None

    @property
    def is_resumed(self) -> bool:
        return self.weights is not None


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def set_torch_misc(cfg) -> None:
    torch.set_num_threads(cfg.omp_num_threads)
    os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision("highest")


def create_run_dir(base_dir: str, name: str | None) -> Path:
    return Path(base_dir) / str(name) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def init_wandb(
    *, entity: str | None, project: str, config: dict, name: str | None, enabled: bool = True, dry_run: bool = False
) -> tuple[wandb.sdk.wandb_run.Run | None, str | None]:
    mode = "online" if enabled and not dry_run and is_main_process() else "disabled"
    run = wandb.init(entity=entity, project=project, config=config, name=name, mode=mode)
    return run, (run.url if run and mode != "disabled" else None)


def _worker_init_fn(worker_id: int) -> None:
    # Each worker otherwise spreads tiny per-frame ops over every visible core,
    # where thread synchronization dwarfs the actual work. Parallelism comes
    # from `num_workers`.
    torch.set_num_threads(1)


def setup_sequence_data(cfg) -> tuple[DynamicBatchSampler, torch.utils.data.DataLoader]:
    composed = ComposedSequenceDataset(
        [
            (_dataset_class(dataset_cfg)(dataset_cfg), float(dataset_cfg.weight))
            for dataset_cfg in cfg.train_data.__dict__.values()
            if dataset_cfg.weight > 0
        ]
    )
    batch_sampler = DynamicBatchSampler(
        cfg.sampling, DynamicSampler(len(composed), shuffle=True, seed=cfg.sampling.seed)
    )
    loader = torch.utils.data.DataLoader[Batch](
        composed,
        batch_sampler=batch_sampler,
        num_workers=cfg.num_workers,
        collate_fn=Batch.collate,
        worker_init_fn=_worker_init_fn if cfg.num_workers > 0 else None,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        # Surface a stuck worker as an immediate, rank-attributable error
        # instead of stalling until the NCCL watchdog trips on an unrelated
        # collective ten minutes later.
        timeout=120,
    )
    return batch_sampler, loader


def _dataset_class(dataset_cfg):
    import sys

    return getattr(sys.modules[dataset_cfg.__module__], type(dataset_cfg).__qualname__.rsplit(".", 1)[0])


def build_eval_callback(cfg) -> EvalCallback | None:
    benchmarks = []
    if cfg.run_eval_scannet:
        benchmarks.append(MvConsistencyBenchmark(cfg.eval_scannet))
    if cfg.run_eval_megadepth:
        benchmarks.append(MvConsistencyBenchmark(cfg.eval_megadepth))
    if not benchmarks:
        return None

    def eval_callback(eval_model: nn.Module, step: int, run_dir: Path) -> None:
        for bench in benchmarks:
            try:
                bench.benchmark(eval_model, step=step)
            except Exception as e:
                logger.warning(f"{bench.cfg.dataset} eval failed at step {step}: {e}")

    return eval_callback


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def dump_run(
    *,
    run_dir: Path,
    cfg,
    step: int,
    weights: dict[str, torch.Tensor],
    optimizer_state: dict,
    make_step_dir: bool,
    scheduler_state: dict | None = None,
) -> None:
    logger.info(f"Dumping run at step {step}")
    if make_step_dir:
        run_dir = run_dir / f"step_{step}"
        run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(asdict(cfg), open(run_dir / "cfg.json", "w"), indent=2)
    pickle.dump(cfg, open(run_dir / "cfg.pkl", "wb"))
    (run_dir / "step.txt").write_text(str(step))
    torch.save(weights, run_dir / "weights.pth")
    torch.save(optimizer_state, run_dir / "optimizer_state.pth")
    if scheduler_state is not None:
        torch.save(scheduler_state, run_dir / "scheduler_state.pth")


def load_run(run_dir: Path):
    # `map_location="cpu"`: checkpoints are saved from rank 0's cuda:0, and this
    # runs before the process group is up, so an unmapped load would strand a
    # full copy of the weights on GPU 0 for every non-zero local rank.
    cfg = pickle.load(open(run_dir / "cfg.pkl", "rb"))
    step = int((run_dir / "step.txt").read_text())
    weights = torch.load(run_dir / "weights.pth", map_location="cpu")
    optimizer_state = torch.load(run_dir / "optimizer_state.pth", map_location="cpu")
    scheduler_path = run_dir / "scheduler_state.pth"
    scheduler_state = torch.load(scheduler_path, map_location="cpu") if scheduler_path.exists() else None
    logger.info(f"Loaded run at step {step}")
    return cfg, step, weights, optimizer_state, scheduler_state


def maybe_load_run(cfg: CfgT) -> RunState[CfgT]:
    if cfg.resume_run is None:
        if not cfg.dry_run:
            assert cfg.name is not None, "--name is required"
        return RunState[CfgT](cfg=cfg, step=0, weights=None, optimizer_state=None, scheduler_state=None)
    loaded_cfg, step, weights, optimizer_state, scheduler_state = load_run(Path(cfg.resume_run))
    return RunState[CfgT](
        cfg=loaded_cfg,
        step=step,
        weights=weights,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
    )


def setup_run(*, run_dir: Path, cfg, step: int, weights, optimizer_state, wandb_url: str | None = None) -> None:
    logger.info(f"Tracking run {cfg.name!r} at {run_dir}")
    dump_run(
        run_dir=run_dir, cfg=cfg, step=step, weights=weights, optimizer_state=optimizer_state, make_step_dir=False
    )
    if wandb_url:
        (run_dir / "wandb_url.txt").write_text(wandb_url)


def maybe_load_weights(model: nn.Module, run_state: RunState[CfgT]) -> None:
    if run_state.weights is not None:
        model.load_state_dict(run_state.weights)


def maybe_wrap_model_ddp(model: nn.Module) -> nn.Module:
    if is_distributed():
        return torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view=False)
    return model


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def create_grad_scaler(enabled: bool = True) -> torch.amp.grad_scaler.GradScaler:
    return torch.amp.grad_scaler.GradScaler(
        enabled=enabled,
        init_scale=65536,
        growth_interval=2000,
        backoff_factor=1 - 1e-8,
        growth_factor=1 + 1e-8,
    )


def _no_weight_decay(name: str) -> bool:
    """DINOv2's rule: biases, norm scales/biases and LayerScale gammas don't
    benefit from being pulled toward zero."""
    return name.endswith(".bias") or "norm" in name or "gamma" in name


def _vit_lr_decay_rate(name: str, layerwise_decay: float, num_layers: int) -> float:
    """DINOv2/v3-style layer-wise LR decay. Stem parameters get the heaviest
    discount, block `i` gets `layer_id = i + 1`, and anything past the last
    block (e.g. the final norm) is left undiscounted."""
    layer_id = num_layers + 1
    if any(t in name for t in (".patch_embed", ".cls_token", ".mask_token", ".storage_tokens")):
        layer_id = 0
    elif ".blocks." in name:
        layer_id = int(name.split(".blocks.")[1].split(".")[0]) + 1
    return layerwise_decay ** (num_layers + 1 - layer_id)


def create_optimizer_and_scheduler(
    model: nn.Module, run_state: RunState[CfgT]
) -> tuple[AdamW, torch.optim.lr_scheduler.LRScheduler]:
    """LR warms up linearly, then cosine-decays over the rest of `num_steps`.

    Encoder parameters get their own group per layer, scaled by
    `_vit_lr_decay_rate` (and by `patch_embed_lr_mult` for the stem); everything
    else runs at `lr`. Every group is split again by `_no_weight_decay` and
    tagged with a `wd_multiplier` that `train_loop` applies to the weight-decay
    schedule each step. Weight decay is deliberately *not* also scaled by
    `lr_multiplier`: AdamW's decoupled update is `param *= 1 - lr * wd`, so the
    discounted LR already discounts the effective decay.

    Expects the unwrapped student (not DDP, not `SSLModel`), so `model.encoder`
    is the backbone.
    """
    cfg = run_state.cfg
    num_layers = model.encoder.n_blocks
    param_groups: dict[tuple[str, float, bool], list[torch.nn.Parameter]] = defaultdict(list)
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            group = "encoder"
            lr_multiplier = _vit_lr_decay_rate(name, cfg.layerwise_decay, num_layers)
            if ".patch_embed" in name:
                lr_multiplier *= cfg.patch_embed_lr_mult
        else:
            group, lr_multiplier = "other", 1.0
        param_groups[(group, lr_multiplier, not _no_weight_decay(name))].append(param)

    optimizer = AdamW(
        [
            {
                "params": params,
                "lr": cfg.lr * lr_multiplier,
                "weight_decay": cfg.weight_decay * float(decay),
                "wd_multiplier": float(decay),
                "lr_multiplier": lr_multiplier,
                "group": group,
            }
            for (group, lr_multiplier, decay), params in param_groups.items()
            if params
        ],
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, total_iters=cfg.warmup_steps),
            CosineAnnealingLR(optimizer, T_max=cfg.num_steps - cfg.warmup_steps),
        ],
        milestones=[cfg.warmup_steps],
        last_epoch=run_state.step - 1 if run_state.is_resumed else -1,
    )
    if run_state.optimizer_state is not None:
        optimizer.load_state_dict(run_state.optimizer_state)
    if run_state.scheduler_state is not None:
        scheduler.load_state_dict(run_state.scheduler_state)
    return optimizer, scheduler


def _cancel_last_layer_gradients(model: nn.Module, step: int, freeze_steps: int) -> None:
    if step >= freeze_steps:
        return
    student = getattr(model, "module", model).student
    for head in (student.head, student.global_head):
        for p in head.last_layer.parameters():
            p.grad = None


def train_step(
    *,
    batch: Batch,
    model: nn.Module,
    step: int,
    loss: Poincar3Loss,
    grad_scaler: torch.amp.grad_scaler.GradScaler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg,
) -> float:
    t0 = time.perf_counter()
    loss_value, stats = loss(batch=batch, model=model, step=step)
    grad_scaler.scale(loss_value).backward()
    grad_scaler.unscale_(optimizer)
    _cancel_last_layer_gradients(model, step, cfg.freeze_last_layer_steps)

    grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    if grad_norm.isnan():
        logger.warning(f"NaN grad norm at step {step}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    scheduler.step()
    optimizer.zero_grad()

    if is_main_process():
        encoder_lrs = [g["lr"] for g in optimizer.param_groups if g["group"] == "encoder"]
        other_lr = next(g["lr"] for g in optimizer.param_groups if g["group"] == "other")
        wandb.log(
            {
                **stats,
                "total-grad-norm": grad_norm,
                "scale": grad_scaler.get_scale(),
                "encoder_lr_min": min(encoder_lrs),
                "encoder_lr_max": max(encoder_lrs),
                "lr": other_lr,
                "train-step-time": time.perf_counter() - t0,
            },
            step=step,
        )
    return loss_value.item()


def train_loop(
    *,
    cfg,
    model: nn.Module,
    d_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss: Poincar3Loss,
    grad_scaler: torch.amp.grad_scaler.GradScaler,
    sampler: DynamicBatchSampler | None,
    loader: torch.utils.data.DataLoader,
    run_dir: Path,
    step: int,
    epoch_size: int | None = None,
    eval_callback: EvalCallback | None = None,
) -> int:
    ema_schedule = CosineSchedule(
        cfg.ema_decay,
        cfg.num_steps,
        final_value=cfg.ema_decay_end,
        warmup_steps=cfg.ema_decay_warmup_steps,
        start_warmup_value=cfg.ema_decay_warmup_start,
    )
    wd_schedule = CosineSchedule(cfg.weight_decay, cfg.num_steps, final_value=cfg.weight_decay_end)

    epoch_size = epoch_size or len(loader)
    for epoch in range(step // epoch_size, cfg.num_steps // epoch_size + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in (pbar := tqdm(loader)):
            if step >= cfg.num_steps:
                break

            model.train(True)
            batch = batch.to(device)
            weight_decay = wd_schedule(step)
            for param_group in optimizer.param_groups:
                param_group["weight_decay"] = weight_decay * param_group["wd_multiplier"]

            loss_value = train_step(
                batch=batch,
                model=d_model,
                step=step,
                loss=loss,
                grad_scaler=grad_scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
            )
            ema_decay = ema_schedule(step)
            model.update_teacher(ema_decay)
            if is_main_process():
                wandb.log({"ema_decay": ema_decay, "weight_decay": weight_decay}, step=step)
            pbar.set_description(f"Loss: {loss_value:.4f}")

            # Checkpointing and evaluation run on rank 0 only. Barrier on both
            # sides so the other ranks rendezvous here explicitly instead of
            # racing into the next step's all_reduce and idling there, eating
            # into the NCCL watchdog's collective timeout.
            do_ckpt = step % cfg.ckpt_interval == 0 and not cfg.dry_run
            do_eval = step % cfg.eval_interval == 0 and not cfg.no_test
            needs_sync = (do_ckpt or do_eval) and is_distributed()

            if needs_sync:
                dist.barrier()
            if do_ckpt and is_main_process():
                dump_run(
                    run_dir=run_dir,
                    cfg=cfg,
                    step=step,
                    weights=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    make_step_dir=(step % cfg.step_dir_interval == 0),
                )
            if do_eval and is_main_process() and eval_callback is not None:
                eval_callback(model.teacher, step, run_dir)
            if needs_sync:
                dist.barrier()

            step += 1
            if step % epoch_size == 0:
                break

        if step >= cfg.num_steps:
            break

    if is_main_process() and not cfg.dry_run:
        dump_run(
            run_dir=run_dir,
            cfg=cfg,
            step=step,
            weights=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            make_step_dir=False,
        )
    if is_main_process():
        wandb.finish()
    return step
