#!/usr/bin/env python
"""Generate and optionally run VocalMorph V2 ablation and multi-seed experiments."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ABLATIONS: Dict[str, Dict[str, object]] = {
    "v2_small_physics": {},
    "v2_small_no_physics": {
        "model.v2.toggles.use_physics_branch": False,
        "model.v2.toggles.use_cross_attention": False,
        "model.v2.toggles.use_height_prior": False,
        "model.v2.toggles.use_reliability_gate": False,
        "model.v2.toggles.use_height_adapter": False,
        "model.v2.toggles.use_acoustic_physics_consistency": False,
    },
}


def _set_nested(mapping: Dict[str, object], dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _get_nested(mapping: Mapping[str, object], dotted_key: str, default: object = None) -> object:
    cursor: object = mapping
    for part in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def build_ablation_config(
    base_config: Mapping[str, object],
    overrides: Mapping[str, object],
    *,
    run_dir: str,
    seed: int,
) -> Dict[str, object]:
    config = copy.deepcopy(dict(base_config))
    for dotted_key, value in overrides.items():
        _set_nested(config, dotted_key, value)

    _set_nested(config, "training.seed", int(seed))
    _set_nested(config, "training.early_stopping.monitor", "height_mae_speaker")
    _set_nested(config, "training.early_stopping.mode", "min")
    _set_nested(config, "logging.checkpoint.monitor", "height_mae_speaker")
    _set_nested(config, "logging.checkpoint.mode", "min")
    _set_nested(config, "logging.tensorboard.log_dir", os.path.join(run_dir, "logs"))
    _set_nested(config, "logging.checkpoint.dir", os.path.join(run_dir, "ckpts"))
    return config


def summarize_metrics(records: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["ablation"]), []).append(record)

    summary: Dict[str, Dict[str, float]] = {}
    metric_keys = (
        ("final_val", "height_mae_speaker"),
        ("final_val", "height_rmse_speaker"),
        ("final_val", "height_median_ae_speaker"),
        ("final_test", "height_mae_speaker"),
        ("final_test", "height_rmse_speaker"),
        ("final_test", "height_median_ae_speaker"),
        ("final_val", "height_calibration_mae"),
    )
    for ablation_name, ablation_records in grouped.items():
        row: Dict[str, float] = {"n_runs": float(len(ablation_records))}
        for section, metric_key in metric_keys:
            values = []
            for record in ablation_records:
                section_data = record.get(section, {})
                if isinstance(section_data, Mapping):
                    value = section_data.get(metric_key)
                    if isinstance(value, (int, float)):
                        values.append(float(value))
            if values:
                row[f"{section}_{metric_key}_mean"] = round(float(statistics.fmean(values)), 12)
                row[f"{section}_{metric_key}_std"] = round(
                    float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
                    12,
                )
        summary[ablation_name] = row
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or generate VocalMorph V2 ablations.")
    parser.add_argument("--config", default="configs/pibnn_base.yaml")
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true", help="Actually run training commands")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to specific ablation names")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 17, 23])
    return parser.parse_args()


def main() -> int:
    import yaml

    args = parse_args()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    selected = set(args.only) if args.only else set(ABLATIONS.keys())
    unknown = sorted(selected - set(ABLATIONS.keys()))
    if unknown:
        raise ValueError(f"Unknown ablations requested: {unknown}")

    completed_records: List[Dict[str, object]] = []
    for ablation_name, overrides in ABLATIONS.items():
        if ablation_name not in selected:
            continue
        for seed in args.seeds:
            run_dir = os.path.join(output_dir, ablation_name, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            run_config = build_ablation_config(base_config, overrides, run_dir=run_dir, seed=seed)
            config_out = os.path.join(run_dir, "config.yaml")
            metrics_out = os.path.join(run_dir, "metrics.json")
            with open(config_out, "w") as f:
                yaml.safe_dump(run_config, f, sort_keys=False)

            cmd = [
                args.python,
                os.path.join(ROOT, "scripts", "train.py"),
                "--config",
                config_out,
                "--seed",
                str(seed),
                "--metrics-out",
                metrics_out,
            ]
            print(f"[Ablation] {ablation_name} seed={seed}")
            print("  " + " ".join(cmd))
            if args.run:
                subprocess.run(cmd, check=True, cwd=ROOT)

            if os.path.exists(metrics_out):
                with open(metrics_out, "r") as f:
                    record = json.load(f)
                record["ablation"] = ablation_name
                record["seed"] = seed
                completed_records.append(record)

    summary = summarize_metrics(completed_records)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Ablation] Wrote summary to {summary_path}")

    if summary:
        print("\nAblation Summary")
        for ablation_name, row in summary.items():
            val_mae = row.get("final_val_height_mae_speaker_mean", float("nan"))
            val_std = row.get("final_val_height_mae_speaker_std", float("nan"))
            cal_mae = row.get("final_val_height_calibration_mae_mean", float("nan"))
            print(
                f"  {ablation_name:26s} "
                f"val_h_spk={val_mae:.3f} +/- {val_std:.3f} "
                f"cal_mae={cal_mae:.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
