#!/usr/bin/env python
"""Evaluate validation-trained ensembles across saved height checkpoints."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.diagnose_height_bias import (  # noqa: E402
    _feature_columns,
    _matrix,
    _regression_metrics,
    _resolve,
    _target,
    _write_records_csv,
    build_eval_stack,
    collect_speaker_records,
    configure_cuda_math,
    seed_everything,
    _prepare_config,
)
from scripts.train import _torch_load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stack VocalMorph checkpoint predictions.")
    parser.add_argument("--config", default="configs/pibnn_rtx3060_3cm_power.yaml")
    parser.add_argument(
        "--checkpoint-glob",
        default="outputs/checkpoints_rtx3060_3cm_extreme12gb/epoch_*.ckpt",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostics/checkpoint_ensemble_extreme12gb",
    )
    return parser.parse_args()


def _checkpoint_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load_for_replay(trainer, checkpoint_path: str) -> None:
    payload = _torch_load_checkpoint(checkpoint_path)
    trainer._load_model_checkpoint_state(payload["model_state_dict"])
    if payload.get("ema_state_dict") is not None:
        trainer.load_ema_state_dict(payload["ema_state_dict"])


def _records_by_speaker(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(row["speaker_id"]): row for row in records}


def _aligned_matrix(
    speakers: Sequence[str],
    by_checkpoint: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    method: str,
) -> np.ndarray:
    columns = []
    for label in by_checkpoint:
        values = []
        key = f"pred_{method}"
        for speaker_id in speakers:
            values.append(float(by_checkpoint[label][speaker_id][key]))
        columns.append(values)
    return np.asarray(columns, dtype=np.float32).T


def _speaker_truth(
    speakers: Sequence[str],
    records_by_speaker: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    return np.asarray(
        [float(records_by_speaker[speaker_id]["height_true"]) for speaker_id in speakers],
        dtype=np.float32,
    )


def _common_speakers(
    by_checkpoint: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> List[str]:
    common = None
    for records in by_checkpoint.values():
        keys = set(records.keys())
        common = keys if common is None else common & keys
    return sorted(common or [])


def _metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    metrics = _regression_metrics(y_true, y_pred)
    return {
        "name": name,
        "count": metrics["count"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "median_ae": metrics["median_ae"],
    }


def _meta_matrix(records: Sequence[Mapping[str, Any]], method: str) -> np.ndarray:
    return _matrix(records, _feature_columns(method))


def run_stackers(
    *,
    checkpoint_labels: Sequence[str],
    val_records_by_ckpt: Mapping[str, Mapping[str, Mapping[str, Any]]],
    test_records_by_ckpt: Mapping[str, Mapping[str, Mapping[str, Any]]],
    base_val_records: Sequence[Mapping[str, Any]],
    base_test_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    val_speakers = _common_speakers(val_records_by_ckpt)
    test_speakers = _common_speakers(test_records_by_ckpt)
    first_label = checkpoint_labels[0]
    y_val = _speaker_truth(val_speakers, val_records_by_ckpt[first_label])
    y_test = _speaker_truth(test_speakers, test_records_by_ckpt[first_label])
    results: List[Dict[str, Any]] = []

    for method in ("legacy", "omega", "mean"):
        x_val = _aligned_matrix(val_speakers, val_records_by_ckpt, method=method)
        x_test = _aligned_matrix(test_speakers, test_records_by_ckpt, method=method)
        for idx, label in enumerate(checkpoint_labels):
            results.append(_metric_row(f"{label}_{method}", y_test, x_test[:, idx]))

        results.append(_metric_row(f"avg_{method}", y_test, x_test.mean(axis=1)))
        results.append(_metric_row(f"median_{method}", y_test, np.median(x_test, axis=1)))

        affine = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("linear", LinearRegression()),
            ]
        )
        affine.fit(x_val, y_val)
        results.append(
            _metric_row(f"val_linear_stack_{method}", y_test, affine.predict(x_test))
        )

        ridge = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
            ]
        )
        ridge.fit(x_val, y_val)
        results.append(
            _metric_row(f"val_ridge_stack_{method}", y_test, ridge.predict(x_test))
        )

        try:
            huber = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("huber", HuberRegressor(alpha=0.03, epsilon=1.35, max_iter=1000)),
                ]
            )
            huber.fit(x_val, y_val)
            results.append(
                _metric_row(f"val_huber_stack_{method}", y_test, huber.predict(x_test))
            )
        except Exception as exc:
            results.append(
                {
                    "name": f"val_huber_stack_{method}",
                    "count": float(len(test_speakers)),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "median_ae": float("nan"),
                    "error": str(exc),
                }
            )

        base_val_by_id = _records_by_speaker(base_val_records)
        base_test_by_id = _records_by_speaker(base_test_records)
        val_meta_records = [base_val_by_id[speaker_id] for speaker_id in val_speakers]
        test_meta_records = [base_test_by_id[speaker_id] for speaker_id in test_speakers]
        x_val_meta = np.concatenate([x_val, _meta_matrix(val_meta_records, method)], axis=1)
        x_test_meta = np.concatenate(
            [x_test, _meta_matrix(test_meta_records, method)], axis=1
        )
        meta_ridge = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
            ]
        )
        meta_ridge.fit(x_val_meta, y_val)
        results.append(
            _metric_row(
                f"val_ridge_stack_meta_{method}",
                y_test,
                meta_ridge.predict(x_test_meta),
            )
        )

    results.sort(
        key=lambda row: row["mae"] if np.isfinite(float(row["mae"])) else float("inf")
    )
    return results


def main() -> int:
    args = parse_args()
    checkpoint_paths = sorted(glob.glob(_resolve(args.checkpoint_glob)))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matched: {args.checkpoint_glob}")

    prep_args = argparse.Namespace(
        config=args.config,
        checkpoint=None,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        include_train=False,
    )
    config = _prepare_config(prep_args)
    seed_everything(int(config["training"]["seed"]))
    configure_cuda_math(config["training"])
    trainer, _train_loader, val_loader, test_loader = build_eval_stack(config)

    val_records_by_ckpt: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    test_records_by_ckpt: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    checkpoint_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_records: List[Dict[str, Any]] = []
    labels: List[str] = []

    for path in checkpoint_paths:
        label = _checkpoint_label(path)
        labels.append(label)
        print(f"[Checkpoint Ensemble] Replaying {label}: {path}")
        _load_for_replay(trainer, path)
        val_records, val_summary = collect_speaker_records(trainer, val_loader, "val")
        test_records, test_summary = collect_speaker_records(
            trainer, test_loader, "test"
        )
        for row in val_records + test_records:
            copied = dict(row)
            copied["checkpoint"] = label
            all_records.append(copied)
        val_records_by_ckpt[label] = _records_by_speaker(val_records)
        test_records_by_ckpt[label] = _records_by_speaker(test_records)
        checkpoint_metrics[label] = {"val": val_summary, "test": test_summary}
        print(
            f"  val={val_summary.get('legacy_mae', float('nan')):.3f}cm "
            f"test={test_summary.get('legacy_mae', float('nan')):.3f}cm"
        )

    base_val = [
        dict(row, checkpoint=labels[0]) for row in val_records_by_ckpt[labels[0]].values()
    ]
    base_test = [
        dict(row, checkpoint=labels[0])
        for row in test_records_by_ckpt[labels[0]].values()
    ]
    ensemble_results = run_stackers(
        checkpoint_labels=labels,
        val_records_by_ckpt=val_records_by_ckpt,
        test_records_by_ckpt=test_records_by_ckpt,
        base_val_records=base_val,
        base_test_records=base_test,
    )

    output_dir = _resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    records_path = os.path.join(output_dir, "checkpoint_speaker_predictions.csv")
    summary_path = os.path.join(output_dir, "summary.json")
    _write_records_csv(records_path, all_records)
    summary = {
        "config": _resolve(args.config),
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_metrics": checkpoint_metrics,
        "ensemble_results": ensemble_results,
        "best": ensemble_results[0] if ensemble_results else {},
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)

    print("[Checkpoint Ensemble] Best validation-trained test result:")
    if ensemble_results:
        best = ensemble_results[0]
        print(
            f"  {best['name']} mae={best['mae']:.3f}cm "
            f"rmse={best['rmse']:.3f}cm median={best['median_ae']:.3f}cm"
        )
    print(f"[Checkpoint Ensemble] Wrote {records_path}")
    print(f"[Checkpoint Ensemble] Wrote {summary_path}")
    trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
