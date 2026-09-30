from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro
import wandb

from poincar3.baselines import Backbone, load_backbone
from poincar3.benchmarks.matchbench import Poincar3VisionEncoder
from poincar3.benchmarks.matchbench import benchmark as run_matchbench
from poincar3.benchmarks.mv_consistency import MvConsistencyBenchmark
from poincar3.benchmarks.see_se3 import SeeSE3Benchmark
from poincar3.device import device


@dataclass(frozen=True)
class MatchBenchCfg:
    """Trains a grid of linear probes on frozen dense features to predict dense
    correspondences, scored by end-point error on covisible pairs."""

    dataset: Literal["megadepth", "scannet"] = "megadepth"
    matcher: Literal["linear", "knn"] = "linear"
    dataset_path: str | None = None
    batch_size: int = 8
    resolution: Literal["low", "medium", "high"] = "low"
    steps: int = 2500
    evals: int = 10
    amp: bool = True
    num_workers: int = 8


@dataclass(frozen=True)
class EvalCfg:
    evaluation: Literal["mvcorr", "matchbench", "see_se3"] = "mvcorr"
    backbone: Backbone = "poincar3"
    # A training run directory (or one of its `step_<n>/` subdirectories). Only
    # meaningful for the `poincar3`/`poincar3_encoder` backbones; omit it to
    # evaluate the released checkpoint.
    run_path: str | None = None

    mvcorr: MvConsistencyBenchmark.Cfg = MvConsistencyBenchmark.Cfg(
        dataset="scannet", correspondence_method="attention", max_sequences=None
    )
    matchbench: MatchBenchCfg = MatchBenchCfg()
    see_se3: SeeSE3Benchmark.Cfg = SeeSE3Benchmark.Cfg()

    output_dir: str = "experiments/eval"
    wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str = "poincar3"
    name: str | None = None


def _report(*, output_dir: Path, name: str, step: int, summary: dict[str, float]) -> None:
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_step{step}.json"
    out_path.write_text(
        json.dumps(
            {"evaluation": name, "step": step, "timestamp": datetime.now().isoformat(), "metrics": summary},
            indent=2,
        )
    )
    print(f"Saved results to {out_path}")
    wandb.log({f"eval/{name}/{k}": v for k, v in summary.items()})


def main(cfg: EvalCfg) -> None:
    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        config=asdict(cfg),
        name=cfg.name,
        mode="online" if cfg.wandb else "disabled",
    )
    output_dir = Path(cfg.output_dir) / cfg.backbone
    model, step = load_backbone(cfg.backbone, cfg.run_path)

    if cfg.evaluation == "mvcorr":
        summary = MvConsistencyBenchmark(cfg.mvcorr).benchmark(model, step=step)
        name = f"mvcorr_{cfg.mvcorr.dataset}_{cfg.mvcorr.correspondence_method}"
    elif cfg.evaluation == "matchbench":
        encoder = Poincar3VisionEncoder(model).to(device).eval()
        summary = run_matchbench(
            encoder,
            output_dir=str(output_dir / "matchbench"),
            batch_size=cfg.matchbench.batch_size,
            resolution=cfg.matchbench.resolution,
            dataset=cfg.matchbench.dataset,
            matcher=cfg.matchbench.matcher,
            dataset_path=cfg.matchbench.dataset_path,
            steps=cfg.matchbench.steps,
            evals=cfg.matchbench.evals,
            amp=cfg.matchbench.amp,
            num_workers=cfg.matchbench.num_workers,
            experiment_name=cfg.name,
        )
        name = f"matchbench_{cfg.matchbench.dataset}_{cfg.matchbench.matcher}"
    elif cfg.evaluation == "see_se3":
        summary = SeeSE3Benchmark(cfg.see_se3).benchmark(model, step=step)
        name = f"see_se3_{cfg.see_se3.dataset}_{cfg.see_se3.metric}"
    else:
        raise ValueError(f"Unknown evaluation: {cfg.evaluation}")

    _report(output_dir=output_dir, name=name, step=step, summary=summary)
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(EvalCfg))
