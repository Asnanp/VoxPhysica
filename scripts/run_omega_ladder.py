#!/usr/bin/env python
"""Generate and optionally run the VocalMorph Omega experiment ladder."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPLAY_ROOTS = [
    os.path.join(ROOT, "outputs", "ablations_canonical_resume_safe", "v2_small_no_physics"),
    os.path.join(ROOT, "outputs", "ablations", "v2_small_no_physics"),
]
REPLAY_CHECKPOINT_NAMES = ("best.ckpt", "last_good.ckpt", "last.ckpt", "best_model.pt")


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


def _deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


OMEGA_STAGES: Dict[str, Dict[str, Any]] = {
    "stage0_baseline_truth": {
        "mode": "replay",
        "description": "Replay the current strict frontier with legacy aggregation only.",
        "variant": "baseline_truth",
        "monitor": "height_mae_speaker",
        "overrides": {
            "model.type": "omega_stage0_no_physics_replay",
            "model.v2.aggregation.method": "legacy_inverse_variance",
            "model.v2.reliability.mode": "handcrafted",
            "model.v2.toggles.use_physics_branch": False,
            "model.v2.toggles.use_cross_attention": False,
            "model.v2.toggles.use_reliability_gate": False,
            "model.v2.toggles.use_height_prior": False,
            "model.v2.toggles.use_height_adapter": False,
            "model.v2.toggles.use_acoustic_physics_consistency": False,
            "training.speaker_batching.enabled": False,
            "training.speaker_alignment.enable_pooled_height": False,
            "training.speaker_alignment.enable_consistency": False,
            "training.speaker_alignment.enable_ranking": False,
        },
    },
    "stage1_aggregation_only": {
        "mode": "replay",
        "description": "Enable omega pooling with handcrafted reliability and no retraining.",
        "variant": "aggregation_only",
        "monitor": "height_mae_speaker",
        "overrides": {
            "model.type": "omega_stage1_no_physics_replay",
            "model.v2.aggregation.method": "omega_robust_reliability_pool",
            "model.v2.reliability.mode": "handcrafted",
            "model.v2.toggles.use_physics_branch": False,
            "model.v2.toggles.use_cross_attention": False,
            "model.v2.toggles.use_reliability_gate": False,
            "model.v2.toggles.use_height_prior": False,
            "model.v2.toggles.use_height_adapter": False,
            "model.v2.toggles.use_acoustic_physics_consistency": False,
            "training.speaker_batching.enabled": False,
            "training.speaker_alignment.enable_pooled_height": False,
            "training.speaker_alignment.enable_consistency": False,
            "training.speaker_alignment.enable_ranking": False,
        },
    },
    "stage2_speaker_structured_no_physics": {
        "mode": "train",
        "description": "Grouped-speaker batching with no-physics V2.",
        "variant": "speaker_structured_no_physics",
        "monitor": "height_mae_speaker",
        "overrides": {
            "model.type": "omega_stage2_no_physics",
            "model.v2.aggregation.method": "omega_robust_reliability_pool",
            "model.v2.reliability.mode": "handcrafted",
            "model.v2.toggles.use_physics_branch": False,
            "model.v2.toggles.use_cross_attention": False,
            "model.v2.toggles.use_reliability_gate": False,
            "model.v2.toggles.use_height_prior": False,
            "model.v2.toggles.use_height_adapter": False,
            "model.v2.toggles.use_acoustic_physics_consistency": False,
            "training.speaker_batching.enabled": True,
            "training.speaker_batching.speakers_per_batch": 8,
            "training.speaker_batching.clips_per_speaker": 2,
            "training.speaker_alignment.enable_pooled_height": False,
            "training.speaker_alignment.enable_consistency": False,
            "training.speaker_alignment.enable_ranking": False,
        },
    },
    "stage2b_hybrid_speaker_structured": {
        "mode": "train",
        "description": "Hybrid speaker batching with higher batch diversity and legacy aggregation.",
        "variant": "hybrid_speaker_structured_no_physics",
        "monitor": "height_mae_speaker",
        "overrides": {
            "model.type": "omega_stage2b_no_physics",
            "model.v2.aggregation.method": "legacy_inverse_variance",
            "model.v2.reliability.mode": "handcrafted",
            "model.v2.toggles.use_physics_branch": False,
            "model.v2.toggles.use_cross_attention": False,
            "model.v2.toggles.use_reliability_gate": False,
            "model.v2.toggles.use_height_prior": False,
            "model.v2.toggles.use_height_adapter": False,
            "model.v2.toggles.use_acoustic_physics_consistency": False,
            "training.speaker_batching.enabled": True,
            "training.speaker_batching.mode": "hybrid",
            "training.speaker_batching.clips_per_speaker": 2,
            "training.speaker_batching.paired_speakers_per_batch": 4,
            "training.speaker_batching.singleton_speakers_per_batch": 8,
            "training.speaker_alignment.enable_pooled_height": False,
            "training.speaker_alignment.enable_consistency": False,
            "training.speaker_alignment.enable_ranking": False,
        },
    },
    "stage3_speaker_alignment": {
        "mode": "train",
        "description": "Add pooled speaker loss, consistency, and ranking on the no-physics line.",
        "variant": "speaker_alignment_no_physics",
        "monitor": "height_mae_speaker",
        "overrides": {
            "model.type": "omega_stage3_no_physics",
            "model.v2.aggregation.method": "omega_robust_reliability_pool",
            "model.v2.reliability.mode": "handcrafted",
            "model.v2.toggles.use_physics_branch": False,
            "model.v2.toggles.use_cross_attention": False,
            "model.v2.toggles.use_reliability_gate": False,
            "model.v2.toggles.use_height_prior": False,
            "model.v2.toggles.use_height_adapter": False,
            "model.v2.toggles.use_acoustic_physics_consistency": False,
            "model.v2.loss_weights.weight": 0.25,
            "model.v2.loss_weights.age": 0.25,
            "model.v2.loss_weights.gender": 0.20,
            "training.speaker_batching.enabled": True,
            "training.speaker_batching.speakers_per_batch": 8,
            "training.speaker_batching.clips_per_speaker": 2,
            "training.speaker_alignment.enable_pooled_height": True,
            "training.speaker_alignment.enable_consistency": True,
            "training.speaker_alignment.enable_ranking": True,
            "training.speaker_alignment.pooled_height_weight_max": 0.35,
            "training.speaker_alignment.consistency_weight_max": 0.15,
            "training.speaker_alignment.ranking_weight_max": 0.10,
        },
    },
    "stage4_learned_reliability": {
        "mode": "train",
        "description": "Switch from handcrafted reliability to the learned metadata tower.",
        "variant": "learned_reliability_no_physics",
        "monitor": "height_mae_speaker",
        "overrides": {
            "model.type": "omega_stage4_no_physics",
            "model.v2.aggregation.method": "omega_robust_reliability_pool",
            "model.v2.reliability.mode": "learned",
            "model.v2.toggles.use_physics_branch": False,
            "model.v2.toggles.use_cross_attention": False,
            "model.v2.toggles.use_reliability_gate": False,
            "model.v2.toggles.use_height_prior": False,
            "model.v2.toggles.use_height_adapter": False,
            "model.v2.toggles.use_acoustic_physics_consistency": False,
            "training.speaker_batching.enabled": True,
            "training.speaker_batching.speakers_per_batch": 8,
            "training.speaker_batching.clips_per_speaker": 2,
            "training.speaker_alignment.enable_pooled_height": True,
            "training.speaker_alignment.enable_consistency": True,
            "training.speaker_alignment.enable_ranking": True,
        },
    },
    "stage5_physics_smart": {
        "mode": "train",
        "description": "Reintroduce the approved physics components only after the no-physics frontier improves.",
        "variant": "physics_smart",
        "monitor": "height_mae_speaker",
        "overrides": {
            "model.type": "omega_stage5_physics",
            "model.v2.aggregation.method": "omega_robust_reliability_pool",
            "model.v2.reliability.mode": "handcrafted",
            "model.v2.toggles.use_physics_branch": True,
            "model.v2.toggles.use_cross_attention": True,
            "model.v2.toggles.use_reliability_gate": True,
            "model.v2.toggles.use_height_prior": True,
            "model.v2.toggles.use_height_adapter": True,
            "model.v2.toggles.use_acoustic_physics_consistency": True,
            "model.v2.toggles.use_domain_adv": False,
            "model.v2.toggles.use_diversity_loss": False,
            "model.v2.toggles.use_ranking_loss": False,
            "model.v2.toggles.use_uncertainty_calibration": False,
            "training.speaker_batching.enabled": True,
            "training.speaker_batching.speakers_per_batch": 8,
            "training.speaker_batching.clips_per_speaker": 2,
            "training.speaker_alignment.enable_pooled_height": True,
            "training.speaker_alignment.enable_consistency": True,
            "training.speaker_alignment.enable_ranking": True,
        },
    },
    "stage6_flagship": {
        "mode": "train",
        "description": "Combined flagship candidate using only individually promoted Omega components.",
        "variant": "flagship_candidate",
        "monitor": "height_mae_speaker",
        "overrides": {
            "model.type": "omega_stage6_flagship",
            "model.v2.aggregation.method": "omega_robust_reliability_pool",
            "model.v2.reliability.mode": "handcrafted",
            "training.speaker_batching.enabled": True,
            "training.speaker_batching.speakers_per_batch": 8,
            "training.speaker_batching.clips_per_speaker": 2,
            "training.speaker_alignment.enable_pooled_height": True,
            "training.speaker_alignment.enable_consistency": True,
            "training.speaker_alignment.enable_ranking": True,
        },
    },
}


def environment_metadata() -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_enabled": bool(getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.enabled),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if torch.cuda.is_available():
        data["cuda_version"] = torch.version.cuda
        data["cudnn_version"] = torch.backends.cudnn.version()
        data["gpu_name"] = torch.cuda.get_device_name(0)
    return data


def build_stage_config(
    base_config: Mapping[str, Any],
    *,
    stage_name: str,
    run_dir: str,
    seed: int,
    epochs_override: Optional[int] = None,
) -> Dict[str, Any]:
    if stage_name not in OMEGA_STAGES:
        raise KeyError(f"Unknown Omega stage: {stage_name}")
    stage = OMEGA_STAGES[stage_name]
    config = copy.deepcopy(dict(base_config))
    for dotted_key, value in stage["overrides"].items():
        _set_nested(config, dotted_key, value)

    _set_nested(config, "training.seed", int(seed))
    if epochs_override is not None:
        _set_nested(config, "training.epochs", int(epochs_override))
    _set_nested(config, "training.allow_tf32", False)
    _set_nested(config, "training.device", config.get("training", {}).get("device", "auto"))
    _set_nested(config, "training.early_stopping.monitor", stage["monitor"])
    _set_nested(config, "training.early_stopping.mode", "min")
    _set_nested(config, "logging.checkpoint.monitor", stage["monitor"])
    _set_nested(config, "logging.checkpoint.mode", "min")
    _set_nested(config, "logging.tensorboard.log_dir", os.path.join(run_dir, "logs"))
    _set_nested(config, "logging.checkpoint.dir", os.path.join(run_dir, "ckpts"))
    return config


def append_markdown(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def write_markdown(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_experiment_matrix(path: str) -> None:
    rows = ["# Experiment Matrix\n", "\n", "| stage | mode | variant | description |\n", "| --- | --- | --- | --- |\n"]
    for stage_name, stage in OMEGA_STAGES.items():
        rows.append(
            f"| {stage_name} | {stage['mode']} | {stage['variant']} | {stage['description']} |\n"
        )
    write_markdown(path, "".join(rows))


def initialize_registry(path: str) -> None:
    if os.path.exists(path):
        return
    write_markdown(
        path,
        (
            "# Experiment Registry\n\n"
            "| stage | variant | seed | mode | legacy val | legacy test | omega val | omega test | "
            "legacy gap | omega gap | critical slices | calibration | decision | notes |\n"
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |\n"
        ),
    )


def initialize_decision_log(path: str) -> None:
    if os.path.exists(path):
        return
    write_markdown(path, "# Run Decision Log\n\n")


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if result != result:
        return float("nan")
    return result


def _fmt_metric(value: Any) -> str:
    number = _safe_float(value)
    return f"{number:.3f}" if number == number else "nan"


def _group_delta(metrics: Mapping[str, Any], keys: Sequence[str]) -> float:
    values = [_safe_float(metrics.get(key)) for key in keys]
    finite = [value for value in values if value == value]
    if len(finite) < 2:
        return float("nan")
    return max(finite) - min(finite)


def _quality_delta(metrics: Mapping[str, Any], suffix: str) -> float:
    prefix = "height_quality_"
    values = []
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        if not key.endswith(suffix):
            continue
        numeric = _safe_float(value)
        if numeric == numeric:
            values.append(numeric)
    if len(values) < 2:
        return float("nan")
    return max(values) - min(values)


def _critical_slice_summary(metrics: Mapping[str, Any], omega: bool = False) -> str:
    suffix = "_speaker_mae_omega" if omega else "_speaker_mae"
    source_delta = _group_delta(
        metrics,
        [f"height_source_nisp{suffix}", f"height_source_timit{suffix}"],
    )
    gender_delta = _group_delta(
        metrics,
        [f"height_gender_female{suffix}", f"height_gender_male{suffix}"],
    )
    height_delta = _group_delta(
        metrics,
        [
            f"height_heightbin_short{suffix}",
            f"height_heightbin_medium{suffix}",
            f"height_heightbin_tall{suffix}",
        ],
    )
    quality_delta = _quality_delta(metrics, suffix)
    label = "omega" if omega else "legacy"
    return (
        f"{label}[src={_fmt_metric(source_delta)},gender={_fmt_metric(gender_delta)},"
        f"height={_fmt_metric(height_delta)},quality={_fmt_metric(quality_delta)}]"
    )


def _critical_slice_max_degradation(metrics: Mapping[str, Any]) -> float:
    max_degradation = float("-inf")
    found = False
    for key, value in metrics.items():
        if not key.endswith("_speaker_mae"):
            continue
        if "_source_" not in key and "_gender_" not in key and "_heightbin_" not in key and "_quality_" not in key:
            continue
        omega_key = f"{key}_omega"
        if omega_key not in metrics:
            continue
        legacy = _safe_float(value)
        omega = _safe_float(metrics.get(omega_key))
        if legacy != legacy or omega != omega:
            continue
        max_degradation = max(max_degradation, omega - legacy)
        found = True
    return max_degradation if found else float("nan")


def _calibration_summary(metrics: Mapping[str, Any]) -> str:
    parts = []
    for key, label in (
        ("height_calibration_mae", "cal"),
        ("height_uncertainty_error_corr", "corr"),
        ("height_interval_68", "p68"),
        ("height_interval_95", "p95"),
    ):
        value = _safe_float(metrics.get(key))
        if value == value:
            parts.append(f"{label}={value:.3f}")
    return ", ".join(parts) if parts else "n/a"


def _extract_metrics_record(payload: Mapping[str, Any]) -> Dict[str, Any]:
    final_val = dict(payload.get("final_val", {}))
    final_test = dict(payload.get("final_test", {}))
    gaps = dict(payload.get("overfit_gaps", {}))
    return {
        "legacy_val": _safe_float(final_val.get("height_mae_speaker")),
        "legacy_test": _safe_float(final_test.get("height_mae_speaker")),
        "omega_val": _safe_float(final_val.get("height_mae_speaker_omega")),
        "omega_test": _safe_float(final_test.get("height_mae_speaker_omega")),
        "legacy_gap": _safe_float(gaps.get("height_mae_speaker_gap_val_minus_train")),
        "omega_gap": _safe_float(gaps.get("height_mae_speaker_omega_gap_val_minus_train")),
        "critical_slices": (
            f"{_critical_slice_summary(final_val, omega=False)}; "
            f"{_critical_slice_summary(final_val, omega=True)}; "
            f"max_deg={_fmt_metric(_critical_slice_max_degradation(final_val))}"
        ),
        "calibration": _calibration_summary(final_test),
        "max_critical_degradation": _critical_slice_max_degradation(final_val),
    }


def _decide_stage_outcome(stage_name: str, metrics: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    if not metrics:
        return "hold", "config generated"
    legacy_val = _safe_float(metrics.get("legacy_val"))
    legacy_test = _safe_float(metrics.get("legacy_test"))
    omega_val = _safe_float(metrics.get("omega_val"))
    omega_test = _safe_float(metrics.get("omega_test"))
    max_deg = _safe_float(metrics.get("max_critical_degradation"))
    if stage_name in {"stage0_baseline_truth", "stage1_aggregation_only"}:
        if legacy_val == legacy_val and legacy_test == legacy_test and omega_val == omega_val and omega_test == omega_test:
            val_gain = legacy_val - omega_val
            test_gain = legacy_test - omega_test
            if val_gain >= 0.10 and test_gain >= 0.10 and (max_deg != max_deg or max_deg <= 0.30):
                return (
                    "promote",
                    f"omega replay improved val by {val_gain:.3f} cm and test by {test_gain:.3f} cm",
                )
            return (
                "kill",
                f"omega replay did not clear thresholds (val_gain={val_gain:.3f}, test_gain={test_gain:.3f}, max_deg={_fmt_metric(max_deg)})",
            )
    return "hold", "metrics captured; promotion requires ladder-stage review"


def _resolve_replay_checkpoint(seed: int, replay_roots: Sequence[str]) -> Optional[str]:
    for root in replay_roots:
        if not root:
            continue
        resolved_root = root if os.path.isabs(root) else os.path.join(ROOT, root)
        seed_root = os.path.join(resolved_root, f"seed_{seed}")
        candidate_dirs = [os.path.join(seed_root, "ckpts"), seed_root]
        for directory in candidate_dirs:
            for name in REPLAY_CHECKPOINT_NAMES:
                candidate = os.path.join(directory, name)
                if os.path.exists(candidate):
                    return candidate
    return None


def append_registry_entry(path: str, record: Mapping[str, Any]) -> None:
    append_markdown(
        path,
        (
            f"| {record['stage']} | {record['variant']} | {record['seed']} | {record['mode']} | "
            f"{_fmt_metric(record.get('legacy_val'))} | {_fmt_metric(record.get('legacy_test'))} | "
            f"{_fmt_metric(record.get('omega_val'))} | {_fmt_metric(record.get('omega_test'))} | "
            f"{_fmt_metric(record.get('legacy_gap'))} | {_fmt_metric(record.get('omega_gap'))} | "
            f"{record.get('critical_slices', 'n/a')} | {record.get('calibration', 'n/a')} | "
            f"{record['decision']} | {record['notes']} |\n"
        ),
    )


def append_decision_log(path: str, record: Mapping[str, Any]) -> None:
    append_markdown(
        path,
        (
            f"## {record['stage']} / seed {record['seed']}\n"
            f"- Variant: `{record['variant']}`\n"
            f"- Mode: `{record['mode']}`\n"
            f"- Decision: `{record['decision']}`\n"
            f"- Notes: {record['notes']}\n"
            f"- Config diff: `{record['config_diff_path']}`\n\n"
            f"- Legacy val/test speaker MAE: `{_fmt_metric(record.get('legacy_val'))}` / `{_fmt_metric(record.get('legacy_test'))}`\n"
            f"- Omega val/test speaker MAE: `{_fmt_metric(record.get('omega_val'))}` / `{_fmt_metric(record.get('omega_test'))}`\n"
            f"- Train->val gaps: legacy `{_fmt_metric(record.get('legacy_gap'))}`, omega `{_fmt_metric(record.get('omega_gap'))}`\n"
            f"- Critical slices: {record.get('critical_slices', 'n/a')}\n"
            f"- Calibration: {record.get('calibration', 'n/a')}\n\n"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or run the VocalMorph Omega ladder.")
    parser.add_argument("--config", default="configs/pibnn_base.yaml")
    parser.add_argument("--output-dir", default="outputs/omega")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true", help="Launch train.py for train stages")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to stage names")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 17, 23])
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument(
        "--replay-root",
        nargs="+",
        default=DEFAULT_REPLAY_ROOTS,
        help="Checkpoint roots to search for replay stages",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    selected = list(OMEGA_STAGES.keys()) if not args.only else list(args.only)
    unknown = sorted(set(selected) - set(OMEGA_STAGES.keys()))
    if unknown:
        raise ValueError(f"Unknown Omega stages requested: {unknown}")

    experiment_matrix_path = os.path.join(ROOT, "experiments", "experiment_matrix.md")
    registry_path = os.path.join(ROOT, "experiments", "experiment_registry.md")
    decision_log_path = os.path.join(ROOT, "experiments", "run_decision_log.md")
    write_experiment_matrix(experiment_matrix_path)
    initialize_registry(registry_path)
    initialize_decision_log(decision_log_path)

    env_meta = environment_metadata()
    for stage_name in selected:
        stage = OMEGA_STAGES[stage_name]
        for seed in args.seeds:
            run_dir = os.path.join(output_dir, stage_name, stage["variant"], f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            config = build_stage_config(
                base_config,
                stage_name=stage_name,
                run_dir=run_dir,
                seed=seed,
                epochs_override=args.epochs,
            )
            config_out = os.path.join(run_dir, "config.yaml")
            config_diff_out = os.path.join(run_dir, "config_diff.json")
            metrics_out = os.path.join(run_dir, "metrics.json")
            with open(config_out, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            with open(config_diff_out, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "stage": stage_name,
                        "variant": stage["variant"],
                        "seed": seed,
                        "description": stage["description"],
                        "overrides": stage["overrides"],
                        "environment": env_meta,
                    },
                    handle,
                    indent=2,
                )

            decision = "hold"
            notes = "config generated"
            if args.run and stage["mode"] == "train":
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
                print("[Omega] " + " ".join(cmd))
                subprocess.run(cmd, check=True, cwd=ROOT)
                notes = "training completed"
            elif args.run and stage["mode"] == "replay":
                replay_ckpt = _resolve_replay_checkpoint(seed, args.replay_root)
                if replay_ckpt:
                    cmd = [
                        args.python,
                        os.path.join(ROOT, "scripts", "train.py"),
                        "--config",
                        config_out,
                        "--seed",
                        str(seed),
                        "--resume",
                        replay_ckpt,
                        "--eval-only",
                        "--metrics-out",
                        metrics_out,
                    ]
                    print("[Omega] " + " ".join(cmd))
                    subprocess.run(cmd, check=True, cwd=ROOT)
                    notes = f"replayed checkpoint {replay_ckpt}"
                else:
                    notes = "missing replay checkpoint for this seed"
            elif stage["mode"] == "replay":
                notes = "replay stage prepared; run with --run to score existing checkpoints"

            metrics_record: Optional[Dict[str, Any]] = None
            if os.path.exists(metrics_out):
                with open(metrics_out, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                metrics_record = _extract_metrics_record(payload)
                decision, auto_note = _decide_stage_outcome(stage_name, metrics_record)
                if auto_note:
                    notes = auto_note if notes == "config generated" else f"{notes}; {auto_note}"

            record = {
                "stage": stage_name,
                "variant": stage["variant"],
                "seed": seed,
                "mode": stage["mode"],
                "legacy_val": metrics_record.get("legacy_val") if metrics_record else float("nan"),
                "legacy_test": metrics_record.get("legacy_test") if metrics_record else float("nan"),
                "omega_val": metrics_record.get("omega_val") if metrics_record else float("nan"),
                "omega_test": metrics_record.get("omega_test") if metrics_record else float("nan"),
                "legacy_gap": metrics_record.get("legacy_gap") if metrics_record else float("nan"),
                "omega_gap": metrics_record.get("omega_gap") if metrics_record else float("nan"),
                "critical_slices": metrics_record.get("critical_slices", "n/a") if metrics_record else "n/a",
                "calibration": metrics_record.get("calibration", "n/a") if metrics_record else "n/a",
                "decision": decision,
                "notes": notes,
                "config_diff_path": config_diff_out,
            }
            append_registry_entry(registry_path, record)
            append_decision_log(decision_log_path, record)
    print(f"[Omega] Wrote ladder configs and registry under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
