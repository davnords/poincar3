from __future__ import annotations

from dataclasses import asdict

import torch.distributed as dist
import tyro

from poincar3.device import device
from poincar3.distrib import init_distributed, is_distributed, is_main_process, setup_multinode_nccl_env
from poincar3.logging import logger
from poincar3.loss import Poincar3Loss
from poincar3.model import SSLModel
from poincar3.run import (
    Poincar3Cfg,
    build_eval_callback,
    create_grad_scaler,
    create_optimizer_and_scheduler,
    create_run_dir,
    init_wandb,
    maybe_load_run,
    maybe_load_weights,
    maybe_wrap_model_ddp,
    set_torch_misc,
    setup_run,
    setup_sequence_data,
    train_loop,
)


def main(cli_cfg: Poincar3Cfg) -> None:
    run_state = maybe_load_run(cli_cfg)
    cfg, step = run_state.cfg, run_state.step

    init_distributed()
    set_torch_misc(cfg)

    run_dir = create_run_dir(cfg.run_dir, cfg.name)
    if is_main_process() and not cfg.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    model = SSLModel(cfg.model).to(device)
    num_params = sum(p.numel() for p in model.student.parameters())
    logger.info(f"Model parameters (student): {num_params:,}")
    maybe_load_weights(model, run_state)
    d_model = maybe_wrap_model_ddp(model)

    # Only the student is optimized; the teacher moves via EMA in `train_loop`.
    optimizer, scheduler = create_optimizer_and_scheduler(model.student, run_state)
    loss_fn = Poincar3Loss(cfg.loss)
    grad_scaler = create_grad_scaler(enabled=(device.type == "cuda"))
    eval_callback = build_eval_callback(cfg)
    sampler, loader = setup_sequence_data(cfg)

    wandb_run, wandb_url = init_wandb(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        config=asdict(cfg),
        name=cfg.name,
        enabled=cfg.wandb,
        dry_run=cfg.dry_run,
    )
    if wandb_run is not None:
        wandb_run.summary["num_params"] = num_params
    if is_main_process() and not cfg.dry_run:
        setup_run(
            run_dir=run_dir,
            cfg=cfg,
            step=step,
            weights=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            wandb_url=wandb_url,
        )
        if run_state.is_resumed:
            (run_dir / "resumed_from.txt").write_text(str(cli_cfg.resume_run))

    train_loop(
        cfg=cfg,
        model=model,
        d_model=d_model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss_fn,
        grad_scaler=grad_scaler,
        sampler=sampler,
        loader=loader,
        run_dir=run_dir,
        step=step,
        epoch_size=cfg.epoch_size,
        eval_callback=eval_callback,
    )


if __name__ == "__main__":
    setup_multinode_nccl_env()
    try:
        main(tyro.cli(Poincar3Cfg))
    except KeyboardInterrupt:
        if is_distributed():
            dist.destroy_process_group()
        raise
