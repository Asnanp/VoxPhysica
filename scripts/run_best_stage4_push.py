#!/usr/bin/env python
"""Launch the best-shot Stage 4 height-first GPU run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Optional, Sequence

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.run_omega_ladder import _set_nested, build_stage_config  # noqa: E402


GPU_PYTHON = r"C:\Users\USER\anaconda3\python.exe"
STAGE_NAME = "stage4_proper_v2_height_first"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the best-shot Stage 4 height-first GPU experiment.")
    parser.add_argument("--config", default="configs/pibnn_base.yaml")
    parser.add_argument("--output-root", default="outputs/stage4_best_balanced_push")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--python", default=GPU_PYTHON)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-train", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _apply_best_push_overrides(config: dict[str, Any], *, epochs: int, device: str) -> dict[str, Any]:
    overrides = {
        "physics.enabled": False,
        "training.device": device,
        "training.allow_tf32": True,
        "training.batch_size": 8,
        "training.epochs": int(epochs),
        "training.optimizer.lr": 3.0e-05,
        "training.optimizer.weight_decay": 0.03,
        "training.scheduler.type": "cosine_annealing",
        "training.scheduler.T_max": int(epochs),
        "training.scheduler.min_lr": 1.0e-06,
        "training.scheduler.eta_min": 1.0e-06,
        "training.ema.decay": 0.9995,
        "training.early_stopping.monitor": "height_balanced_frontier_speaker_mae",
        "logging.checkpoint.monitor": "height_balanced_frontier_speaker_mae",
        "training.loss.task_weights.height": 4.0,
        "training.loss.task_weights.weight": 0.0,
        "training.loss.task_weights.age": 0.0,
        "training.loss.task_weights.gender": 0.0,
        "training.loss.task_weights.vtsl": 0.0,
        "training.loss.task_weights.physics_penalty": 0.0,
        "training.loss.physics_penalty_weight": 0.0,
        "training.loss.focal_after_epoch": 12,
        "training.loss.focal_ema_decay": 0.92,
        "training.speaker_batching.enabled": True,
        "training.speaker_batching.mode": "height_balanced",
        "training.speaker_batching.speakers_per_batch": 8,
        "training.speaker_batching.clips_per_speaker": 2,
        "training.speaker_batching.balance_gender": True,
        "training.speaker_batching.gender_balance_strength": 0.15,
        "training.speaker_batching.height_bin_weights.short": 1.7,
        "training.speaker_batching.height_bin_weights.medium": 1.0,
        "training.speaker_batching.height_bin_weights.tall": 1.7,
        "training.speaker_alignment.enable_pooled_height": True,
        "training.speaker_alignment.enable_consistency": True,
        "training.speaker_alignment.enable_ranking": False,
        "training.speaker_alignment.pooling_method": "mean",
        "training.speaker_alignment.consistency_mode": "variance",
        "training.speaker_alignment.warmup_start_epoch": 3,
        "training.speaker_alignment.warmup_end_epoch": 6,
        "training.speaker_alignment.pooled_height_weight_max": 0.10,
        "training.speaker_alignment.consistency_weight_max": 0.03,
        "training.speaker_alignment.ranking_weight_max": 0.0,
        "training.speaker_alignment.height_bin_loss_start_epoch": 1,
        "training.speaker_alignment.height_bin_loss_weight_short": 1.40,
        "training.speaker_alignment.height_bin_loss_weight_medium": 1.0,
        "training.speaker_alignment.height_bin_loss_weight_tall": 1.40,
        "training.speaker_alignment.short_height_threshold_cm": 160.0,
        "training.speaker_alignment.short_height_loss_weight": 1.75,
        "training.speaker_alignment.tall_height_threshold_cm": 175.0,
        "training.speaker_alignment.tall_height_loss_weight": 1.75,
        "training.feature_smoothing_std": 0.006,
        "training.lr_warmup_epochs": 2,
        "training.lr_warmup_start_factor": 0.20,
        "training.early_stopping.patience": 20,
        "model.v2.loss_weights.height": 4.0,
        "model.v2.loss_weights.weight": 0.0,
        "model.v2.loss_weights.age": 0.0,
        "model.v2.loss_weights.gender": 0.0,
        "model.v2.loss_weights.vtsl": 0.0,
        "model.v2.loss_weights.physics_penalty": 0.0,
        "model.v2.loss_weights.height_bin_aux": 0.20,
    }
    for dotted_key, value in overrides.items():
        _set_nested(config, dotted_key, value)
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = _resolve(args.config)
    output_root = _resolve(args.output_root)
    run_dir = os.path.join(output_root, f"seed_{int(args.seed)}")
    os.makedirs(run_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    config = build_stage_config(
        base_config,
        stage_name=STAGE_NAME,
        run_dir=run_dir,
        seed=int(args.seed),
        epochs_override=int(args.epochs),
    )
    config = _apply_best_push_overrides(config, epochs=int(args.epochs), device=str(args.device))

    config_out = os.path.join(run_dir, "config.yaml")
    metrics_out = os.path.join(run_dir, "metrics.json")
    manifest_out = os.path.join(run_dir, "launch_manifest.json")

    with open(config_out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    manifest = {
        "stage": STAGE_NAME,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "device": str(args.device),
        "config_path": config_out,
        "metrics_path": metrics_out,
        "python": _resolve(args.python),
    }
    with open(manifest_out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[BestStage4] Config -> {config_out}")
    print(f"[BestStage4] Metrics -> {metrics_out}")
    print(f"[BestStage4] Device -> {args.device}")

    if args.no_train:
        print("[BestStage4] Config generated only. Training skipped.")
        return 0

    cmd = [
        _resolve(args.python),
        os.path.join(ROOT, "scripts", "train.py"),
        "--config",
        config_out,
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--metrics-out",
        metrics_out,
    ]
    print("[BestStage4] " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"[BestStage4] Finished. Artifacts -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
