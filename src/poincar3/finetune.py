from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
import tyro
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from poincar3.benchmarks.ffrecon import RelposeBenchmark
from poincar3.benchmarks.ffrecon.inference import ReconModel
from poincar3.data.dynamic import DynamicBatchSampler, DynamicSampler
from poincar3.data.ffreconstruction.sequence import (
    ComposedReconDataset,
    ReconBatch,
    ScanNetPlusPlusReconSequence,
)
from poincar3.data.photometric import PhotometricAug
from poincar3.device import device
from poincar3.distrib import is_distributed, setup_multinode_nccl_env
from poincar3.logging import logger
from poincar3.run import _no_weight_decay, _worker_init_fn


@dataclass(frozen=True)
class FinetuneReconBaseCfg:
    scannetpp: ScanNetPlusPlusReconSequence.Cfg = ScanNetPlusPlusReconSequence.Cfg(weight=20_000)
    # Per batch, sample one sequence length N and one aspect-ratio bucket
    # (long side `img_size`); B = floor(num_frames / N) sequences share it.
    # `teacher_len` is pinned at (0, 0): there is no teacher/student split
    # here, every sampled frame goes through the same forward pass.
    sampling: DynamicBatchSampler.Cfg = DynamicBatchSampler.Cfg(
        seq_len=(2, 24), teacher_len=(0, 0), num_frames=72
    )
    num_workers: int = 12
    omp_num_threads: int = 12

    num_steps: int = 200_000
    # Backbone LR. 2e-4 suits a freshly initialized decoder; drop to ~4e-5
    # when both encoder and decoder start from a cross-view-pretrained
    # checkpoint and only need gentle adaptation.
    lr: float = 2e-4
    # The camera/depth heads are always freshly initialized, so they train at
    # their own (larger) rate regardless of how `lr` is tuned.
    head_lr: float = 2e-4
    # Freeze the whole pretrained backbone and train only the heads on top of
    # its frozen features. The backbone then stays in `.eval()`, runs under
    # `torch.no_grad()`, and is excluded from DDP and the optimizer.
    freeze_backbone: bool = False

    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    # EMA of [model, camera_head, depth_head], which is what eval scores.
    ema_decay: float = 0.999

    weight_camera: float = 1.0
    weight_depth: float = 1.0
    depth_conf_gamma: float = 1.0
    depth_conf_alpha: float = 0.2

    # Held-out eval: the same loss on the "val" split (disjoint scenes), at a
    # fixed shape and with no augmentation, so numbers are comparable
    # step-to-step unlike the dynamic training batches.
    eval_interval: int = 2_000
    eval_batches: int = 20
    eval_batch_size: int = 4
    eval_num_views: int = 8
    eval_height: int = 256
    eval_width: int = 256

    # Relative-pose AUC, run at the same interval. `dataset` selects the whole
    # protocol, including whether images are square-resized or
    # aspect-preserving -- worth ~0.3 AUC@30 on MegaDepth -- so never mix the
    # two. Annotations are built by
    # `experiments/ffrecon/build_relpose_annotations.py`.
    run_relpose_megadepth_eval: bool = True
    relpose_megadepth: RelposeBenchmark.Cfg = RelposeBenchmark.Cfg(
        dataset="megadepth",
        data_dir="data/megadepth",
        anno_path="data/megadepth/annotations/test.jgz",
        num_frames=8,
        max_sequences=20,
        aspect_preserving=True,
    )
    run_relpose_re10k_eval: bool = True
    relpose_re10k: RelposeBenchmark.Cfg = RelposeBenchmark.Cfg(
        dataset="re10k",
        data_dir="data/re10k_test",
        anno_path="data/re10k_test/annotations/test.jgz",
        num_frames=8,
        max_sequences=20,
        aspect_preserving=False,
    )
    run_relpose_scannet_eval: bool = True
    relpose_scannet: RelposeBenchmark.Cfg = RelposeBenchmark.Cfg(
        dataset="scannet",
        data_dir="data/scannet_test_1500",
        anno_path="data/scannet_test_1500/annotations/test.jgz",
    )
    run_relpose_scannetpp_eval: bool = False
    relpose_scannetpp: RelposeBenchmark.Cfg = RelposeBenchmark.Cfg(dataset="scannetpp")

    log_interval: int = 20
    ckpt_interval: int = 10_000
    out_dir: str = "experiments/ffrecon/runs"
    name: str | None = None

    wandb: bool = True
    wandb_entity: str | None = None
    wandb_project: str = "poincar3-recon"


def build_dataloader(cfg: FinetuneReconBaseCfg) -> tuple[DynamicBatchSampler, DataLoader]:
    """Training loader, batched by `DynamicBatchSampler` (varying sequence
    length and aspect-ratio bucket per batch) rather than a fixed batch size."""
    composed = ComposedReconDataset([(ScanNetPlusPlusReconSequence(cfg.scannetpp), float(cfg.scannetpp.weight))])
    batch_sampler = DynamicBatchSampler(
        cfg.sampling, DynamicSampler(len(composed), shuffle=True, seed=cfg.sampling.seed)
    )
    loader = DataLoader(
        composed,
        batch_sampler=batch_sampler,
        num_workers=cfg.num_workers,
        collate_fn=ReconBatch.collate,
        worker_init_fn=_worker_init_fn if cfg.num_workers > 0 else None,
    )
    return batch_sampler, loader


def build_eval_loader(cfg: FinetuneReconBaseCfg) -> DataLoader:
    """Held-out val loader: fixed shape (no `DynamicBatchSampler`, so eval
    numbers are directly comparable across calls), disjoint scenes, and no
    augmentation."""
    val_cfg = dataclasses.replace(
        cfg.scannetpp,
        split="val",
        num_views=cfg.eval_num_views,
        height=cfg.eval_height,
        width=cfg.eval_width,
        rescale_aug=False,
        aug=PhotometricAug.Cfg(enabled=False),
    )
    composed = ComposedReconDataset([(ScanNetPlusPlusReconSequence(val_cfg), float(val_cfg.weight))])
    return DataLoader(
        composed, batch_size=cfg.eval_batch_size, num_workers=cfg.num_workers, collate_fn=ReconBatch.collate
    )


def build_relpose_benchmarks(cfg: FinetuneReconBaseCfg) -> list[tuple[str, RelposeBenchmark]]:
    """Built once up front: `RelposeBenchmark.__init__` loads and filters a
    whole annotation file, too slow to redo every `eval_interval`."""
    benchmarks: list[tuple[str, RelposeBenchmark]] = []
    if cfg.run_relpose_megadepth_eval:
        benchmarks.append(("megadepth", RelposeBenchmark(cfg.relpose_megadepth)))
    if cfg.run_relpose_re10k_eval:
        benchmarks.append(("re10k", RelposeBenchmark(cfg.relpose_re10k)))
    if cfg.run_relpose_scannet_eval:
        benchmarks.append(("scannet", RelposeBenchmark(cfg.relpose_scannet)))
    if cfg.run_relpose_scannetpp_eval:
        benchmarks.append(("scannetpp", RelposeBenchmark(cfg.relpose_scannetpp)))
    return benchmarks


def run_relpose_benchmarks(
    benchmarks: list[tuple[str, RelposeBenchmark]],
    recon_model: ReconModel,
    modules: Sequence[nn.Module],
    step: int,
) -> None:
    """Runs every benchmark against `recon_model`, toggling `modules` (the
    trainable ones -- a permanently frozen backbone is left out) to `.eval()`
    and back to `.train()` around the calls."""
    if not benchmarks:
        return
    for module in modules:
        module.eval()
    for name, bench in benchmarks:
        try:
            summary = bench.benchmark(recon_model, step=step, wandb_prefix=f"relpose_{name}")
            if summary:
                logger.info(f"step {step}: relpose/{name} AUC@30={summary['AUC@30']:.4f}")
        except Exception as e:  # noqa: BLE001 -- a failing benchmark must not crash training
            logger.warning(f"relpose/{name} eval failed at step {step}: {e}")
    for module in modules:
        module.train()


@torch.no_grad()
def evaluate_held_out_loss(
    compute_loss_fn: Callable[[ReconBatch], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    modules: Sequence[nn.Module],
    eval_loader: DataLoader,
    eval_batches: int,
) -> dict[str, float]:
    """Averages `compute_loss_fn`'s stats over up to `eval_batches` batches,
    toggling `modules` to `.eval()` and back to `.train()` around the loop."""
    for module in modules:
        module.eval()

    totals: dict[str, float] = {}
    count = 0
    eval_iter = iter(eval_loader)
    for _ in range(eval_batches):
        try:
            batch = next(eval_iter)
        except StopIteration:
            break
        batch = batch.to(device)
        _, stats = compute_loss_fn(batch)
        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                totals[k] = totals.get(k, 0.0) + v.item()
        count += 1

    for module in modules:
        module.train()
    if count == 0:
        return {}
    return {f"eval_{k}": v / count for k, v in totals.items()}


def maybe_ddp(module: nn.Module) -> nn.Module:
    """`find_unused_parameters=True`: a reconstruction forward pass never
    touches the SSL-only prototype heads, and DDP otherwise assumes every
    parameter participates in every step."""
    if not is_distributed():
        return module
    return DistributedDataParallel(module, find_unused_parameters=True, gradient_as_bucket_view=False)


def unwrap_ddp(module: nn.Module) -> nn.Module:
    """A DDP-wrapped module only forwards `_parameters`/`_buffers`/`_modules`
    through `__getattr__`; plain attributes like `patch_token_start` raise
    `AttributeError`. `compute_loss` is called with both raw and wrapped
    modules, so unwrap here."""
    return module.module if isinstance(module, DistributedDataParallel) else module


def clip_grad_norm(modules: Sequence[nn.Module], max_norm: float) -> torch.Tensor:
    """One joint gradient-norm clip over every trainable parameter, rather
    than a separate clip per module -- a per-module clip lets one module's
    exploding early gradient (e.g. a freshly initialized head) pass through
    undamped relative to the others.

    Returns the pre-clip total norm, for logging and non-finite-loss guards.
    """
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def build_ema(modules: Sequence[nn.Module], ema_decay: float) -> tuple[AveragedModel, nn.ModuleList]:
    """EMA over `modules` (e.g. `[model, camera_head, depth_head]`), which is
    what every held-out-loss / relpose-AUC eval scores.

    Returns `(ema, live)`: `live` wraps `modules` by reference, not by copy --
    pass that same object to `ema.update_parameters(live)` every step, on
    every rank (post-step parameters are already identical across ranks, so
    the EMAs stay in sync without a broadcast). Use `ema.module[i]`, in the
    same order as `modules`, at eval time.
    """
    live = nn.ModuleList(modules)
    ema = AveragedModel(live, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
    return ema, live


def build_optimizer_and_scheduler(
    cfg: FinetuneReconBaseCfg,
    backbone_modules: Sequence[nn.Module],
    head_modules: Sequence[nn.Module],
    decoder_modules: Sequence[nn.Module] = (),
) -> tuple[AdamW, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW with up to three LR groups: `backbone_modules` (a pretrained
    checkpoint needing only gentle adaptation) at `cfg.lr`, and
    `decoder_modules` plus `head_modules` -- both freshly initialized in the
    from-scratch controls -- at the larger `cfg.head_lr`. Each group is split
    again by weight-decay eligibility. Plain cosine decay from step 0, no
    warmup.

    Not used by `train_adapter.py`, whose trainable modules are much smaller
    and need their own warmup schedule.
    """
    def _param_groups(modules: Sequence[nn.Module], lr: float) -> list[dict]:
        by_decay: dict[bool, list[nn.Parameter]] = {True: [], False: []}
        for module in modules:
            for name, param in module.named_parameters():
                by_decay[not _no_weight_decay(name)].append(param)
        return [
            {"params": params, "lr": lr, "weight_decay": cfg.weight_decay * float(decay)}
            for decay, params in by_decay.items()
            if params
        ]

    optimizer = AdamW(
        _param_groups(backbone_modules, cfg.lr)
        + _param_groups(decoder_modules, cfg.head_lr)
        + _param_groups(head_modules, cfg.head_lr)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_steps)
    return optimizer, scheduler


CfgT = TypeVar("CfgT")


def cli_main(cfg_cls: type[CfgT], main_fn: Callable[[CfgT], None]) -> None:
    """`tyro.cli(cfg_cls)` plus clean process-group teardown on
    `KeyboardInterrupt`. Runs `setup_multinode_nccl_env()` first, before
    `main_fn` calls `init_distributed()` and creates the NCCL communicator.
    """
    setup_multinode_nccl_env()
    cfg = tyro.cli(cfg_cls)
    try:
        main_fn(cfg)
    except KeyboardInterrupt as e:
        if is_distributed():
            dist.destroy_process_group()
        raise e
