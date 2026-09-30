from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro
import wandb

from poincar3.benchmarks.ffrecon import PointcloudBenchmark, RelposeBenchmark, load_recon_checkpoint


@dataclass(frozen=True)
class EvalCfg:
    # A `checkpoint.pth` written by train.py or train_adapter.py.
    checkpoint: str
    # Only used for an adapter checkpoint trained with `--backbone poincar3`:
    # overrides the frozen backbone's pretraining run path recorded in the
    # checkpoint, e.g. if that directory has since moved.
    run_path: str | None = None
    # Score the checkpoint's EMA weights, matching what the periodic
    # in-training eval scores.
    use_ema: bool = True

    evaluation: Literal["relpose", "pointcloud"] = "relpose"
    relpose: RelposeBenchmark.Cfg = RelposeBenchmark.Cfg()
    pointcloud: PointcloudBenchmark.Cfg = PointcloudBenchmark.Cfg()

    out_dir: str | None = None  # None -> "<checkpoint's directory>/eval"

    wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str = "poincar3-ffrecon-eval"
    name: str | None = None


def _report(out_dir: Path, tag: str, summary: dict[str, float]) -> None:
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"evaluation": tag, "timestamp": datetime.now().isoformat(), "metrics": summary}
    out_path = out_dir / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"Saved results to {out_path}")

    # Also keep one `{tag: latest record}` file per checkpoint, so everything
    # it has ever scored is visible at a glance.
    summary_path = out_dir / "summary.json"
    all_results: dict[str, dict] = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    all_results[tag] = record
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"Updated {summary_path}")


def main(cfg: EvalCfg) -> None:
    recon_model = load_recon_checkpoint(cfg.checkpoint, run_path=cfg.run_path, use_ema=cfg.use_ema)

    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        config=asdict(cfg),
        name=cfg.name,
        mode="online" if cfg.wandb else "disabled",
    )

    if cfg.evaluation == "relpose":
        summary = RelposeBenchmark(cfg.relpose).benchmark(recon_model)
        tag = f"relpose_{cfg.relpose.dataset}"
    elif cfg.evaluation == "pointcloud":
        summary = PointcloudBenchmark(cfg.pointcloud).benchmark(recon_model)
        tag = f"pointcloud_{cfg.pointcloud.dataset}"
    else:
        raise ValueError(f"Unknown evaluation: {cfg.evaluation}")

    if not summary:
        print(f"No sequences scored for evaluation={cfg.evaluation!r} -- check dataset/annotation paths.")
    else:
        out_dir = Path(cfg.out_dir) if cfg.out_dir else Path(cfg.checkpoint).parent / "eval"
        _report(out_dir, tag, summary)
        wandb.log({f"eval/{tag}/{k}": v for k, v in summary.items()})

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(EvalCfg))
