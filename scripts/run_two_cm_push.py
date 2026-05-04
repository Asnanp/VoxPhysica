#!/usr/bin/env python
"""Generate and optionally run the best-shot height-first experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PUSH_OVERRIDES: Dict[str, Any] = {
    "model.type": "two_cm_push_realworld_height_first",
    "model.v2.aggregation.method": "legacy_inverse_variance",
    "model.v2.reliability.mode": "handcrafted",
    "model.v2.ecapa_channels": 192,
    "model.v2.ecapa_scale": 4,
    "model.v2.conformer_d_model": 96,
    "model.v2.conformer_heads": 4,
    "model.v2.conformer_blocks": 2,
    "model.v2.dropout": 0.22,
    "model.v2.branch_dropout": 0.12,
    "model.v2.drop_path_rate": 0.04,
    "model.v2.height_context_hidden_dim": 224,
    "model.v2.height_context_blocks": 2,
    "model.v2.height_context_scale": 0.40,
    "model.v2.height_bin_hidden_dim": 96,
    "model.v2.height_bin_classes": 3,
    "model.v2.toggles.use_physics_branch": False,
    "model.v2.toggles.use_cross_attention": False,
    "model.v2.toggles.use_reliability_gate": False,
    "model.v2.toggles.use_height_prior": False,
    "model.v2.toggles.use_height_adapter": True,
    "model.v2.toggles.use_domain_adv": False,
    "model.v2.toggles.use_diversity_loss": False,
    "model.v2.toggles.use_feature_mixup": False,
    "model.v2.toggles.use_acoustic_physics_consistency": False,
    "model.v2.toggles.use_ranking_loss": False,
    "model.v2.toggles.use_speaker_consistency": False,
    "model.v2.toggles.use_uncertainty_calibration": False,
    "model.v2.toggles.use_shoulder_head": False,
    "model.v2.toggles.use_waist_head": False,
    "model.v2.toggles.use_kendall_weights": False,
    "model.v2.toggles.use_height_context_refiner": True,
    "model.v2.toggles.use_height_bin_aux": True,
    "model.v2.loss_weights.height": 4.0,
    "model.v2.loss_weights.weight": 0.0,
    "model.v2.loss_weights.age": 0.0,
    "model.v2.loss_weights.gender": 0.0,
    "model.v2.loss_weights.vtsl": 0.0,
    "model.v2.loss_weights.physics_penalty": 0.0,
    "model.v2.loss_weights.domain_adv": 0.0,
    "model.v2.loss_weights.ranking": 0.0,
    "model.v2.loss_weights.diversity": 0.0,
    "model.v2.loss_weights.speaker_consistency": 0.0,
    "model.v2.loss_weights.uncertainty_calibration": 0.0,
    "model.v2.loss_weights.height_bin_aux": 0.20,
    "physics.enabled": False,
    "physics.vtl_height_constraint.enabled": False,
    "physics.formant_vtl_constraint.enabled": False,
    "physics.f0_gender_constraint.enabled": False,
    "training.device": "cuda",
    "training.amp": True,
    "training.mixed_precision": True,
    "training.allow_tf32": True,
    "training.num_workers": 2,
    "training.persistent_workers": True,
    "training.epochs": 80,
    "training.optimizer.lr": 0.00005,
    "training.optimizer.weight_decay": 0.03,
    "training.scheduler.type": "cosine_annealing",
    "training.scheduler.T_max": 80,
    "training.scheduler.min_lr": 0.000001,
    "training.scheduler.eta_min": 0.000001,
    "training.loss.type": "vtsl_v2",
    "training.loss.focal_after_epoch": 12,
    "training.loss.focal_ema_decay": 0.95,
    "training.loss.task_weights.height": 4.0,
    "training.loss.task_weights.weight": 0.0,
    "training.loss.task_weights.age": 0.0,
    "training.loss.task_weights.gender": 0.0,
    "training.loss.task_weights.vtsl": 0.0,
    "training.loss.task_weights.physics_penalty": 0.0,
    "training.loss.task_weights.domain_adv": 0.0,
    "training.loss.physics_penalty_weight": 0.0,
    "training.augmentation.enabled": True,
    "training.augmentation.noise_p": 0.25,
    "training.augmentation.noise_std": 0.01,
    "training.augmentation.time_mask_p": 0.15,
    "training.augmentation.feat_mask_p": 0.10,
    "training.augmentation.scale_p": 0.15,
    "training.augmentation.scale_std": 0.03,
    "training.augmentation.temporal_jitter_p": 0.10,
    "training.ema.enabled": True,
    "training.ema.decay": 0.9995,
    "training.speaker_batching.enabled": False,
    "training.speaker_alignment.enable_pooled_height": False,
    "training.speaker_alignment.enable_consistency": False,
    "training.speaker_alignment.enable_ranking": False,
    "training.speaker_alignment.height_bin_loss_start_epoch": 1,
    "training.speaker_alignment.height_bin_loss_weight_short": 1.40,
    "training.speaker_alignment.height_bin_loss_weight_medium": 1.0,
    "training.speaker_alignment.height_bin_loss_weight_tall": 1.25,
    "training.speaker_alignment.short_height_threshold_cm": 160.0,
    "training.speaker_alignment.short_height_loss_weight": 0.80,
    "training.speaker_alignment.tall_height_threshold_cm": 175.0,
    "training.speaker_alignment.tall_height_loss_weight": 0.65,
    "training.feature_smoothing_std": 0.008,
    "training.lr_warmup_epochs": 1,
    "training.lr_warmup_start_factor": 0.25,
    "training.progress_log_interval_steps": 100,
    "training.early_stopping.enabled": True,
    "training.early_stopping.patience": 18,
    "training.early_stopping.monitor": "height_mae_speaker",
    "training.early_stopping.mode": "min",
    "logging.checkpoint.monitor": "height_mae_speaker",
    "logging.checkpoint.mode": "min",
}


def _set_nested(mapping: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def build_config(
    base_config: Mapping[str, Any],
    *,
    run_dir: str,
    seed: int,
    epochs: int | None,
    device: str,
) -> Dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    for dotted_key, value in PUSH_OVERRIDES.items():
        _set_nested(config, dotted_key, value)
    _set_nested(config, "training.seed", int(seed))
    _set_nested(config, "training.device", device)
    if epochs is not None:
        _set_nested(config, "training.epochs", int(epochs))
        _set_nested(config, "training.scheduler.T_max", int(epochs))
    _set_nested(config, "logging.tensorboard.log_dir", os.path.join(run_dir, "logs"))
    _set_nested(config, "logging.checkpoint.dir", os.path.join(run_dir, "ckpts"))
    _set_nested(config, "inference.output_dir", os.path.join(run_dir, "predictions"))
    _set_nested(config, "inference.checkpoint", os.path.join(run_dir, "ckpts", "best.ckpt"))
    return config


def summarize_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    records = list(records)
    metric_keys = (
        ("final_val", "height_mae_speaker"),
        ("final_val", "height_heightbin_short_speaker_mae"),
        ("final_val", "height_heightbin_tall_speaker_mae"),
        ("final_val", "height_heightbin_extreme_speaker_mae"),
        ("final_val", "height_quality_medium_speaker_mae"),
        ("final_test", "height_mae_speaker"),
        ("final_test", "height_heightbin_short_speaker_mae"),
        ("final_test", "height_heightbin_tall_speaker_mae"),
        ("final_test", "height_heightbin_extreme_speaker_mae"),
    )
    summary: Dict[str, float] = {"n_runs": float(len(records))}
    for section, metric_key in metric_keys:
        values: List[float] = []
        for record in records:
            payload = record.get(section, {})
            if isinstance(payload, Mapping):
                value = payload.get(metric_key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
        if values:
            summary[f"{section}_{metric_key}_mean"] = round(
                float(statistics.fmean(values)), 12
            )
            summary[f"{section}_{metric_key}_std"] = round(
                float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
                12,
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or run the height-first two-cm push experiment."
    )
    parser.add_argument("--config", default="configs/pibnn_base.yaml")
    parser.add_argument("--output-dir", default="outputs/two_cm_push_realworld")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run", action="store_true", help="Run training after writing configs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 17, 23])
    return parser.parse_args()


def main() -> int:
    import yaml

    args = parse_args()
    config_path = (
        args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    )
    output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(ROOT, args.output_dir)
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    completed_records: List[Dict[str, Any]] = []
    for seed in args.seeds:
        run_dir = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)
        run_config = build_config(
            base_config,
            run_dir=run_dir,
            seed=seed,
            epochs=args.epochs,
            device=args.device,
        )
        config_out = os.path.join(run_dir, "config.yaml")
        metrics_out = os.path.join(run_dir, "metrics.json")
        with open(config_out, "w", encoding="utf-8") as handle:
            yaml.safe_dump(run_config, handle, sort_keys=False)

        cmd = [
            args.python,
            os.path.join(ROOT, "scripts", "train.py"),
            "--config",
            config_out,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--metrics-out",
            metrics_out,
        ]
        print(f"[TwoCmPush] seed={seed}")
        print("  " + " ".join(cmd))
        if args.run:
            subprocess.run(cmd, check=True, cwd=ROOT)

        if os.path.exists(metrics_out):
            with open(metrics_out, "r", encoding="utf-8") as handle:
                record = json.load(handle)
            record["seed"] = seed
            completed_records.append(record)

    summary = summarize_metrics(completed_records)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[TwoCmPush] Wrote summary to {summary_path}")

    if summary:
        print(
            "[TwoCmPush] "
            f"val_h_spk={summary.get('final_val_height_mae_speaker_mean', float('nan')):.3f} "
            f"| val_short={summary.get('final_val_height_heightbin_short_speaker_mae_mean', float('nan')):.3f} "
            f"| val_tall={summary.get('final_val_height_heightbin_tall_speaker_mae_mean', float('nan')):.3f} "
            f"| val_edge={summary.get('final_val_height_heightbin_extreme_speaker_mae_mean', float('nan')):.3f} "
            f"| val_qmed={summary.get('final_val_height_quality_medium_speaker_mae_mean', float('nan')):.3f} "
            f"| test_h_spk={summary.get('final_test_height_mae_speaker_mean', float('nan')):.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
