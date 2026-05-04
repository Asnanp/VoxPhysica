#!/usr/bin/env python
"""Export speaker predictions and test validation-trained height calibration.

This script is intentionally diagnostic: it never fits on test labels.  It replays
the checkpoint on validation/test, writes per-speaker records, and evaluates a
small set of calibration heads trained only on validation speakers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.train import (  # noqa: E402
    _coerce_bool,
    _coerce_int,
    _torch_load_checkpoint,
    configure_cuda_math,
    seed_everything,
)
from src.models.pibnn import build_model  # noqa: E402
from src.preprocessing.dataset import (  # noqa: E402
    VocalMorphDataset,
    build_dataloaders_from_dirs,
    build_worker_init_fn,
    collate_fn,
)
from src.training.trainer import VocalMorphTrainer  # noqa: E402
from src.utils.audit_utils import duration_bin, height_bin, quality_bucket  # noqa: E402


SPEAKER_META_KEYS = (
    "source_id",
    "gender",
    "duration_s",
    "speech_ratio",
    "snr_db_estimate",
    "capture_quality_score",
    "voiced_ratio",
    "clipped_ratio",
    "distance_cm_estimate",
    "f0_mean",
    "formant_spacing_mean",
    "vtl_mean",
    "valid_frames",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose VocalMorph height bias.")
    parser.add_argument("--config", default="configs/pibnn_rtx3060_3cm_power.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostics/height_bias_fast12gb",
        help="Directory for speaker CSV and summary JSON.",
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also export train speaker predictions for diagnosis. Not used for default calibration.",
    )
    return parser.parse_args()


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _resolve_checkpoint(config: Mapping[str, Any], explicit: Optional[str]) -> str:
    if explicit:
        path = _resolve(explicit)
    else:
        ckpt_dir = (
            config.get("logging", {})
            .get("checkpoint", {})
            .get("dir", "outputs/checkpoints")
        )
        path = os.path.join(_resolve(ckpt_dir), "best.ckpt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _finite_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(arr.std())


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"count": 0.0, "mae": float("nan"), "rmse": float("nan"), "median_ae": float("nan")}
    err = y_true[valid] - y_pred[valid]
    abs_err = np.abs(err)
    return {
        "count": float(valid.sum()),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "median_ae": float(np.median(abs_err)),
    }


def _slice_mae(
    records: Sequence[Mapping[str, Any]], pred_key: str, label_key: str
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label in sorted({str(row.get(label_key, "unknown")) for row in records}):
        rows = [row for row in records if str(row.get(label_key, "unknown")) == label]
        if not rows:
            continue
        y_true = np.asarray([float(row["height_true"]) for row in rows], dtype=np.float32)
        y_pred = np.asarray([float(row[pred_key]) for row in rows], dtype=np.float32)
        out[label] = _regression_metrics(y_true, y_pred)["mae"]
    return out


def _batch_tensor_to_numpy(batch: Mapping[str, Any], key: str) -> Optional[np.ndarray]:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().cpu().float().numpy()


def _denorm(trainer: VocalMorphTrainer, tensor: torch.Tensor, key: str) -> np.ndarray:
    return trainer._denorm_numpy(tensor.detach().cpu().float().numpy(), key)


@torch.no_grad()
def collect_speaker_records(
    trainer: VocalMorphTrainer,
    loader: DataLoader,
    split_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    trainer.model.eval()
    weight_backup = trainer._swap_in_ema_weights()
    try:
        pred_means = {"height": [], "weight": [], "age": [], "gender_pred": []}
        pred_vars = {"height": [], "weight": [], "age": []}
        gender_probs: List[torch.Tensor] = []
        quality_scores: List[torch.Tensor] = []
        targets: Dict[str, List[torch.Tensor]] = {
            "height_raw": [],
            "weight_raw": [],
            "age_raw": [],
            "gender": [],
            "source_id": [],
            "duration_s": [],
            "speech_ratio": [],
            "snr_db_estimate": [],
            "capture_quality_score": [],
            "voiced_ratio": [],
            "clipped_ratio": [],
            "distance_cm_estimate": [],
            "f0_mean": [],
            "formant_spacing_mean": [],
            "vtl_mean": [],
            "valid_frames": [],
        }
        speaker_ids: List[str] = []
        clip_height_preds: List[np.ndarray] = []
        clip_height_vars: List[np.ndarray] = []
        clip_meta_values: Dict[str, List[np.ndarray]] = defaultdict(list)

        n_steps = len(loader)
        for step_idx, batch in enumerate(loader):
            batch = trainer._to_device(batch)
            batch_targets = trainer._build_targets(batch, epoch=1)
            with torch.cuda.amp.autocast(enabled=trainer.use_amp):
                preds = trainer._forward_model(
                    batch=batch,
                    epoch=1,
                    step_idx=step_idx,
                    n_steps=n_steps,
                    train_mode=False,
                    targets=batch_targets if trainer.use_native_v2_loss else None,
                )

            ensemble = trainer._ensemble_predictions(batch)
            metric_height = preds["height"].detach()
            metric_weight = preds["weight"].detach()
            metric_age = preds["age"].detach()
            metric_height_var = preds.get(
                "height_var", torch.ones_like(preds["height"])
            ).detach()
            metric_weight_var = preds.get(
                "weight_var", torch.ones_like(preds["weight"])
            ).detach()
            metric_age_var = preds.get("age_var", torch.ones_like(preds["age"])).detach()
            metric_gender_pred = preds["gender_logits"].argmax(-1).detach()
            metric_gender_probs = torch.softmax(preds["gender_logits"], dim=-1).detach()
            metric_quality = preds.get(
                "quality_score", torch.ones_like(preds["height"])
            ).detach()

            if ensemble is not None:
                metric_height = ensemble["height"]
                metric_weight = ensemble["weight"]
                metric_age = ensemble["age"]
                metric_height_var = ensemble["height_var"]
                metric_weight_var = ensemble["weight_var"]
                metric_age_var = ensemble["age_var"]
                metric_gender_pred = ensemble["gender_pred"]
                metric_gender_probs = ensemble["gender_probs"]
                metric_quality = ensemble["quality_score"]

            pred_means["height"].append(metric_height.cpu())
            pred_means["weight"].append(metric_weight.cpu())
            pred_means["age"].append(metric_age.cpu())
            pred_vars["height"].append(metric_height_var.cpu())
            pred_vars["weight"].append(metric_weight_var.cpu())
            pred_vars["age"].append(metric_age_var.cpu())
            pred_means["gender_pred"].append(metric_gender_pred.cpu())
            gender_probs.append(metric_gender_probs.cpu())
            quality_scores.append(metric_quality.cpu())

            for key in (
                "height_raw",
                "weight_raw",
                "age_raw",
                "gender",
                "source_id",
                "duration_s",
                "speech_ratio",
                "snr_db_estimate",
                "capture_quality_score",
                "voiced_ratio",
                "clipped_ratio",
                "distance_cm_estimate",
                "f0_mean",
                "formant_spacing_mean",
                "vtl_mean",
            ):
                value = batch.get(key)
                if isinstance(value, torch.Tensor):
                    targets[key].append(value.detach().cpu())
                    np_value = _batch_tensor_to_numpy(batch, key)
                    if np_value is not None:
                        clip_meta_values[key].append(np_value)

            padding_mask = batch.get("padding_mask")
            if isinstance(padding_mask, torch.Tensor):
                valid_frames = (~padding_mask).sum(dim=1).to(dtype=torch.float32)
                targets["valid_frames"].append(valid_frames.detach().cpu())
                clip_meta_values["valid_frames"].append(valid_frames.detach().cpu().numpy())

            speaker_ids.extend([str(sid) for sid in batch.get("speaker_id", [])])
            clip_height_preds.append(_denorm(trainer, metric_height, "height"))
            clip_height_vars.append(metric_height_var.detach().cpu().float().numpy())

        if not speaker_ids:
            return [], {}

        preds_for_agg = {
            "height": torch.cat(pred_means["height"]).float(),
            "weight": torch.cat(pred_means["weight"]).float(),
            "age": torch.cat(pred_means["age"]).float(),
            "gender_probs": torch.cat(gender_probs).float(),
        }
        vars_for_agg = {
            "height": torch.cat(pred_vars["height"]).float(),
            "weight": torch.cat(pred_vars["weight"]).float(),
            "age": torch.cat(pred_vars["age"]).float(),
            "gender_probs": None,
        }
        metadata = {
            key: torch.cat(targets[key]).float()
            for key in (
                "duration_s",
                "speech_ratio",
                "snr_db_estimate",
                "capture_quality_score",
                "voiced_ratio",
                "clipped_ratio",
                "distance_cm_estimate",
                "valid_frames",
            )
            if targets.get(key)
        }
        quality = torch.cat(quality_scores).float()
        aggregated = {
            "legacy": trainer.model.aggregate_by_speaker(
                speaker_ids=speaker_ids,
                preds=preds_for_agg,
                variances=vars_for_agg,
                quality=quality,
                metadata=metadata,
                method="legacy_inverse_variance",
            ),
            "omega": trainer.model.aggregate_by_speaker(
                speaker_ids=speaker_ids,
                preds=preds_for_agg,
                variances=vars_for_agg,
                quality=quality,
                metadata=metadata,
                method="omega_robust_reliability_pool",
            ),
            "mean": trainer.model.aggregate_by_speaker(
                speaker_ids=speaker_ids,
                preds=preds_for_agg,
                variances=vars_for_agg,
                quality=quality,
                metadata=metadata,
                method="mean",
            ),
        }

        height_scale = 1.0
        if trainer.target_stats:
            height_scale = float(trainer.target_stats.get("height", {}).get("std", 1.0))

        height_true = np.concatenate(
            [tensor.float().numpy() for tensor in targets["height_raw"]], axis=0
        )
        speaker_truth: Dict[str, Dict[str, Any]] = {}
        meta_arrays = {
            key: np.concatenate(values, axis=0) if values else None
            for key, values in clip_meta_values.items()
        }
        clip_pred_arr = np.concatenate(clip_height_preds, axis=0)
        clip_var_arr = np.concatenate(clip_height_vars, axis=0)

        for idx, speaker_id in enumerate(speaker_ids):
            entry = speaker_truth.setdefault(
                speaker_id,
                {
                    "height_values": [],
                    "clip_pred_values": [],
                    "clip_var_values": [],
                    "meta": defaultdict(list),
                },
            )
            entry["height_values"].append(float(height_true[idx]))
            entry["clip_pred_values"].append(float(clip_pred_arr[idx]))
            entry["clip_var_values"].append(float(clip_var_arr[idx]))
            for key, values in meta_arrays.items():
                if values is not None and idx < len(values):
                    entry["meta"][key].append(float(values[idx]))

        records: List[Dict[str, Any]] = []
        for speaker_id, truth in sorted(speaker_truth.items()):
            if speaker_id not in aggregated["legacy"]["speaker"]:
                continue
            true_height = _finite_mean(truth["height_values"])
            meta = truth["meta"]
            row: Dict[str, Any] = {
                "split": split_name,
                "speaker_id": speaker_id,
                "height_true": true_height,
                "height_bin_true": height_bin(true_height),
                "clip_count": len(truth["clip_pred_values"]),
                "clip_pred_mean": _finite_mean(truth["clip_pred_values"]),
                "clip_pred_std": _finite_std(truth["clip_pred_values"]),
                "clip_pred_var_mean": _finite_mean(truth["clip_var_values"]),
            }
            for key in SPEAKER_META_KEYS:
                row[key] = _finite_mean(meta.get(key, []))
            source_lookup = {0: "TIMIT", 1: "NISP", 2: "EXTERNAL"}
            row["source"] = source_lookup.get(int(round(row.get("source_id", 0) or 0)), "UNKNOWN")
            row["gender_label"] = "male" if int(round(row.get("gender", 0) or 0)) == 1 else "female"
            row["duration_bin"] = duration_bin(float(row.get("duration_s", float("nan"))))
            row["quality_bucket"] = quality_bucket(
                float(row.get("capture_quality_score", float("nan")))
            )

            for method_name, payload in aggregated.items():
                speaker_payload = payload["speaker"][speaker_id]
                pred_norm = speaker_payload["height"].view(1).detach().cpu().numpy()
                row[f"pred_{method_name}"] = float(
                    trainer._denorm_numpy(pred_norm, "height")[0]
                )
                row[f"err_{method_name}"] = float(row[f"pred_{method_name}"] - true_height)
                row[f"abs_err_{method_name}"] = abs(row[f"err_{method_name}"])
                row[f"pred_std_{method_name}"] = float(
                    speaker_payload.get("height_std", torch.zeros(())).view(1).cpu().numpy()[0]
                ) * height_scale
                row[f"pooled_quality_{method_name}"] = float(
                    speaker_payload.get("quality", torch.tensor(float("nan"))).view(1).cpu().numpy()[0]
                )
                row[f"pooled_reliability_{method_name}"] = float(
                    speaker_payload.get("clip_reliability", torch.tensor(float("nan"))).view(1).cpu().numpy()[0]
                )
            records.append(row)

        summary = summarize_records(records)
        return records, summary
    finally:
        trainer._restore_weights(weight_backup)


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for method in ("legacy", "omega", "mean"):
        y_true = np.asarray([float(row["height_true"]) for row in records], dtype=np.float32)
        y_pred = np.asarray([float(row[f"pred_{method}"]) for row in records], dtype=np.float32)
        metrics = _regression_metrics(y_true, y_pred)
        for key, value in metrics.items():
            summary[f"{method}_{key}"] = value
    return summary


def _feature_columns(method: str) -> List[str]:
    return [
        f"pred_{method}",
        f"pred_std_{method}",
        f"pooled_quality_{method}",
        f"pooled_reliability_{method}",
        "clip_count",
        "clip_pred_mean",
        "clip_pred_std",
        "clip_pred_var_mean",
        "source_id",
        "gender",
        "duration_s",
        "speech_ratio",
        "snr_db_estimate",
        "capture_quality_score",
        "voiced_ratio",
        "clipped_ratio",
        "distance_cm_estimate",
        "f0_mean",
        "formant_spacing_mean",
        "vtl_mean",
        "valid_frames",
    ]


def _matrix(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> np.ndarray:
    rows = []
    for row in records:
        values = []
        for column in columns:
            value = row.get(column, float("nan"))
            try:
                values.append(float(value))
            except Exception:
                values.append(float("nan"))
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def _target(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["height_true"]) for row in records], dtype=np.float32)


def _evaluate_calibration(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    metrics = _regression_metrics(y_true, y_pred)
    return {
        "name": name,
        "count": metrics["count"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "median_ae": metrics["median_ae"],
    }


def _fit_pred_bin_residual(
    train_records: Sequence[Mapping[str, Any]], method: str
) -> Tuple[float, Dict[str, float]]:
    residuals_by_bin: Dict[str, List[float]] = defaultdict(list)
    global_residuals: List[float] = []
    for row in train_records:
        pred = float(row[f"pred_{method}"])
        residual = float(row["height_true"]) - pred
        if not np.isfinite(pred) or not np.isfinite(residual):
            continue
        residuals_by_bin[height_bin(pred)].append(residual)
        global_residuals.append(residual)
    global_shift = float(np.median(global_residuals)) if global_residuals else 0.0
    shifts = {
        label: float(np.median(values))
        for label, values in residuals_by_bin.items()
        if values
    }
    return global_shift, shifts


def _apply_pred_bin_residual(
    records: Sequence[Mapping[str, Any]], method: str, global_shift: float, shifts: Mapping[str, float]
) -> np.ndarray:
    preds = []
    for row in records:
        pred = float(row[f"pred_{method}"])
        preds.append(pred + float(shifts.get(height_bin(pred), global_shift)))
    return np.asarray(preds, dtype=np.float32)


def run_calibrators(
    val_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    y_val = _target(val_records)
    y_test = _target(test_records)
    for method in ("legacy", "omega", "mean"):
        pred_test = _matrix(test_records, [f"pred_{method}"])[:, 0]
        results.append(_evaluate_calibration(f"identity_{method}", y_test, pred_test))

        pred_val = _matrix(val_records, [f"pred_{method}"])[:, 0]
        affine = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("linear", LinearRegression()),
            ]
        )
        affine.fit(pred_val.reshape(-1, 1), y_val)
        results.append(
            _evaluate_calibration(
                f"val_affine_{method}",
                y_test,
                affine.predict(pred_test.reshape(-1, 1)).astype(np.float32),
            )
        )

        columns = _feature_columns(method)
        x_val = _matrix(val_records, columns)
        x_test = _matrix(test_records, columns)
        ridge = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
            ]
        )
        ridge.fit(x_val, y_val)
        results.append(
            _evaluate_calibration(
                f"val_ridge_meta_{method}",
                y_test,
                ridge.predict(x_test).astype(np.float32),
            )
        )

        residual = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
            ]
        )
        residual.fit(x_val, y_val - pred_val)
        residual_pred = pred_test + residual.predict(x_test).astype(np.float32)
        results.append(
            _evaluate_calibration(
                f"val_residual_ridge_{method}",
                y_test,
                residual_pred,
            )
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
                _evaluate_calibration(
                    f"val_huber_meta_{method}",
                    y_test,
                    huber.predict(x_test).astype(np.float32),
                )
            )
        except Exception as exc:
            results.append(
                {
                    "name": f"val_huber_meta_{method}",
                    "count": float(len(test_records)),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "median_ae": float("nan"),
                    "error": str(exc),
                }
            )

        global_shift, shifts = _fit_pred_bin_residual(val_records, method)
        results.append(
            _evaluate_calibration(
                f"val_predbin_median_shift_{method}",
                y_test,
                _apply_pred_bin_residual(test_records, method, global_shift, shifts),
            )
        )

    results.sort(key=lambda row: row["mae"] if math.isfinite(float(row["mae"])) else float("inf"))
    return results


def _write_records_csv(path: str, records: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in records for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(dict(row))


def _best_slice_report(records: Sequence[Mapping[str, Any]], pred_key: str) -> Dict[str, Dict[str, float]]:
    return {
        "height_bin": _slice_mae(records, pred_key, "height_bin_true"),
        "source": _slice_mae(records, pred_key, "source"),
        "gender": _slice_mae(records, pred_key, "gender_label"),
        "quality": _slice_mae(records, pred_key, "quality_bucket"),
        "duration": _slice_mae(records, pred_key, "duration_bin"),
    }


def _prepare_config(args: argparse.Namespace) -> Dict[str, Any]:
    with open(_resolve(args.config), "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train_cfg = config.setdefault("training", {})
    train_cfg["seed"] = int(train_cfg.get("seed", 42))
    train_cfg["epochs"] = _coerce_int(train_cfg.get("epochs", 100), "training.epochs")
    train_cfg["batch_size"] = _coerce_int(
        args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32),
        "training.batch_size",
    )
    train_cfg["num_workers"] = _coerce_int(
        args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 4),
        "training.num_workers",
    )
    train_cfg["device"] = args.device or train_cfg.get("device", "auto")
    if train_cfg.get("max_feature_frames") is not None:
        train_cfg["max_feature_frames"] = _coerce_int(
            train_cfg["max_feature_frames"], "training.max_feature_frames"
        )
    if train_cfg.get("prefetch_factor") is not None:
        train_cfg["prefetch_factor"] = _coerce_int(
            train_cfg["prefetch_factor"], "training.prefetch_factor"
        )
    if train_cfg.get("persistent_workers") is not None:
        train_cfg["persistent_workers"] = _coerce_bool(
            train_cfg["persistent_workers"], "training.persistent_workers"
        )
    config.setdefault("logging", {}).setdefault("tensorboard", {})["enabled"] = False
    return config


def build_eval_stack(
    config: Dict[str, Any],
) -> Tuple[VocalMorphTrainer, DataLoader, DataLoader, DataLoader]:
    train_cfg = config["training"]
    data_cfg = config["data"]
    feature_root = _resolve(data_cfg["features_dir"])
    stats_path = os.path.join(feature_root, "target_stats.json")
    target_stats = None
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as handle:
            target_stats = json.load(handle)
    config["target_stats"] = target_stats

    max_feature_frames = train_cfg.get("max_feature_frames")
    train_loader, val_loader, test_loader = build_dataloaders_from_dirs(
        train_dir=os.path.join(feature_root, "train"),
        val_dir=os.path.join(feature_root, "val"),
        test_dir=os.path.join(feature_root, "test"),
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        target_stats=target_stats,
        max_len=max_feature_frames,
        train_crop_mode=str(train_cfg.get("train_crop_mode", "head")),
        eval_crop_mode=str(train_cfg.get("eval_crop_mode", "center")),
        persistent_workers=train_cfg.get("persistent_workers"),
        prefetch_factor=train_cfg.get("prefetch_factor"),
        train_augment=False,
        speaker_batching=train_cfg.get("speaker_batching", {}),
        base_seed=int(train_cfg["seed"]),
        pin_memory=train_cfg.get("pin_memory", True),
    )

    inferred_input_dim = train_loader.dataset.infer_input_dim()
    config.setdefault("model", {})["input_dim"] = int(inferred_input_dim)
    model = build_model(config)
    if hasattr(model, "set_target_stats"):
        model.set_target_stats(target_stats)
    trainer = VocalMorphTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        target_stats=target_stats,
        train_eval_loader=None,
    )
    return trainer, train_loader, val_loader, test_loader


def build_train_eval_loader(config: Mapping[str, Any]) -> DataLoader:
    train_cfg = config["training"]
    feature_root = _resolve(config["data"]["features_dir"])
    dataset = VocalMorphDataset(
        os.path.join(feature_root, "train"),
        max_len=train_cfg.get("max_feature_frames"),
        target_stats=config.get("target_stats"),
        crop_mode=str(train_cfg.get("eval_crop_mode", "center")),
        augment=False,
    )
    pin = bool(train_cfg.get("pin_memory", True)) and torch.cuda.is_available()
    loader_kwargs: Dict[str, Any] = {}
    if train_cfg["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = (
            bool(train_cfg["persistent_workers"])
            if train_cfg.get("persistent_workers") is not None
            else True
        )
        if train_cfg.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(train_cfg["prefetch_factor"])
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=pin,
        worker_init_fn=build_worker_init_fn(int(train_cfg["seed"]) + 997),
        **loader_kwargs,
    )


def main() -> int:
    args = parse_args()
    config = _prepare_config(args)
    seed_everything(int(config["training"]["seed"]))
    configure_cuda_math(config["training"])

    checkpoint_path = _resolve_checkpoint(config, args.checkpoint)
    print(f"[Height Bias] Config: {_resolve(args.config)}")
    print(f"[Height Bias] Checkpoint: {checkpoint_path}")

    trainer, _train_loader, val_loader, test_loader = build_eval_stack(config)
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    trainer._load_model_checkpoint_state(checkpoint["model_state_dict"])
    if checkpoint.get("ema_state_dict") is not None:
        trainer.load_ema_state_dict(checkpoint["ema_state_dict"])

    all_records: List[Dict[str, Any]] = []
    split_summaries: Dict[str, Dict[str, float]] = {}
    if args.include_train:
        train_eval_loader = build_train_eval_loader(config)
        train_records, train_summary = collect_speaker_records(
            trainer, train_eval_loader, "train"
        )
        all_records.extend(train_records)
        split_summaries["train"] = train_summary

    val_records, val_summary = collect_speaker_records(trainer, val_loader, "val")
    test_records, test_summary = collect_speaker_records(trainer, test_loader, "test")
    all_records.extend(val_records)
    all_records.extend(test_records)
    split_summaries["val"] = val_summary
    split_summaries["test"] = test_summary

    calibration_results = run_calibrators(val_records, test_records)
    best = calibration_results[0] if calibration_results else {}

    output_dir = _resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "speaker_predictions.csv")
    summary_path = os.path.join(output_dir, "summary.json")
    _write_records_csv(csv_path, all_records)

    summary = {
        "config": _resolve(args.config),
        "checkpoint": checkpoint_path,
        "splits": split_summaries,
        "calibration_results": calibration_results,
        "best_validation_trained_calibrator": best,
        "test_legacy_slices": _best_slice_report(test_records, "pred_legacy"),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)

    print("[Height Bias] Split summary:")
    for split, metrics in split_summaries.items():
        print(
            f"  {split:5s} legacy={metrics.get('legacy_mae', float('nan')):.3f}cm "
            f"omega={metrics.get('omega_mae', float('nan')):.3f}cm "
            f"mean={metrics.get('mean_mae', float('nan')):.3f}cm"
        )
    if best:
        print(
            "[Height Bias] Best val-trained test calibrator: "
            f"{best['name']} mae={best['mae']:.3f}cm rmse={best['rmse']:.3f}cm"
        )
    print("[Height Bias] Test legacy slices:")
    for group, values in summary["test_legacy_slices"].items():
        printable = ", ".join(f"{key}={value:.3f}" for key, value in values.items())
        print(f"  {group}: {printable}")
    print(f"[Height Bias] Wrote {csv_path}")
    print(f"[Height Bias] Wrote {summary_path}")
    trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
