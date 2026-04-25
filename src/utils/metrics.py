"""VocalMorph evaluation metrics for strict honest reporting."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score

from src.utils.audit_utils import duration_bin, height_bin, quality_bucket


def _cat_or_empty(lst: List[torch.Tensor], dtype=np.float32) -> np.ndarray:
    if not lst:
        return np.array([], dtype=dtype)
    return torch.cat(lst).detach().cpu().numpy().astype(dtype, copy=False)


def _denorm(arr: np.ndarray, key: str, target_stats: Optional[Mapping[str, Mapping[str, float]]]) -> np.ndarray:
    if target_stats is None:
        return arr
    stats = target_stats.get(key, {})
    return arr * float(stats.get("std", 1.0)) + float(stats.get("mean", 0.0))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _regression_metrics(
    prefix: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if valid_mask is None:
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    else:
        valid_mask = valid_mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid_mask):
        return {}

    true = y_true[valid_mask]
    pred = y_pred[valid_mask]
    abs_err = np.abs(true - pred)
    return {
        f"{prefix}_count": float(true.size),
        f"{prefix}_mae": float(np.mean(abs_err)),
        f"{prefix}_rmse": _rmse(true, pred),
        f"{prefix}_median_ae": float(np.median(abs_err)),
    }


def _add_height_subgroup_metrics(
    metrics: Dict[str, float],
    *,
    height_true: np.ndarray,
    height_pred: np.ndarray,
    source_id: np.ndarray,
    gender_true: np.ndarray,
    duration_s: np.ndarray,
    capture_quality_score: np.ndarray,
) -> None:
    base_valid = np.isfinite(height_true) & np.isfinite(height_pred)
    if not np.any(base_valid):
        return

    groups = {
        "source_nisp": source_id == 1,
        "source_timit": source_id == 0,
        "gender_female": gender_true == 0,
        "gender_male": gender_true == 1,
    }

    for label, mask in groups.items():
        metrics.update(
            _regression_metrics(
                f"height_{label}",
                height_true,
                height_pred,
                valid_mask=base_valid & mask,
            )
        )

    for bin_label in ("short", "medium", "tall"):
        mask = np.array([height_bin(value) == bin_label for value in height_true], dtype=bool)
        metrics.update(
            _regression_metrics(
                f"height_heightbin_{bin_label}",
                height_true,
                height_pred,
                valid_mask=base_valid & mask,
            )
        )

    for bin_label in ("short", "medium", "long"):
        mask = np.array([duration_bin(value) == bin_label for value in duration_s], dtype=bool)
        metrics.update(
            _regression_metrics(
                f"height_duration_{bin_label}",
                height_true,
                height_pred,
                valid_mask=base_valid & mask,
            )
        )

    for bucket in ("low", "medium", "high"):
        mask = np.array(
            [quality_bucket(value) == bucket for value in capture_quality_score],
            dtype=bool,
        )
        metrics.update(
            _regression_metrics(
                f"height_quality_{bucket}",
                height_true,
                height_pred,
                valid_mask=base_valid & mask,
            )
        )


def compute_metrics(
    all_preds: Dict[str, List[torch.Tensor]],
    all_targets: Dict[str, List[torch.Tensor]],
    target_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, float]:
    """Compute honest clip-level metrics with subgroup slices."""

    height_pred = _denorm(_cat_or_empty(all_preds["height"]), "height", target_stats)
    weight_pred = _denorm(_cat_or_empty(all_preds["weight"]), "weight", target_stats)
    age_pred = _denorm(_cat_or_empty(all_preds["age"]), "age", target_stats)

    height_true = _cat_or_empty(all_targets["height_raw"])
    weight_true = _cat_or_empty(all_targets["weight_raw"])
    age_true = _cat_or_empty(all_targets["age_raw"])

    gender_pred = _cat_or_empty(all_preds["gender_pred"], dtype=np.int64)
    gender_true = _cat_or_empty(all_targets["gender"], dtype=np.int64)
    source_id = _cat_or_empty(all_targets.get("source_id", []), dtype=np.int64)
    duration_s = _cat_or_empty(all_targets.get("duration_s", []))
    capture_quality_score = _cat_or_empty(all_targets.get("capture_quality_score", []))

    metrics: Dict[str, float] = {
        "n_eval_clips": float(height_true.size),
    }
    metrics.update(_regression_metrics("height", height_true, height_pred))
    metrics.update(
        _regression_metrics(
            "weight",
            weight_true,
            weight_pred,
            valid_mask=np.isfinite(weight_true),
        )
    )
    metrics.update(_regression_metrics("age", age_true, age_pred))

    if gender_true.size > 0:
        valid_gender = np.isfinite(gender_true)
        if np.any(valid_gender):
            metrics["gender_acc"] = float(np.mean(gender_pred[valid_gender] == gender_true[valid_gender]))

    _add_height_subgroup_metrics(
        metrics,
        height_true=height_true,
        height_pred=height_pred,
        source_id=source_id if source_id.size == height_true.size else np.zeros_like(height_true, dtype=np.int64),
        gender_true=gender_true if gender_true.size == height_true.size else np.zeros_like(height_true, dtype=np.int64),
        duration_s=duration_s if duration_s.size == height_true.size else np.full_like(height_true, np.nan, dtype=np.float32),
        capture_quality_score=(
            capture_quality_score
            if capture_quality_score.size == height_true.size
            else np.full_like(height_true, np.nan, dtype=np.float32)
        ),
    )
    return metrics


def compute_full_eval(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    target_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Full evaluation report: MAE/RMSE/medianAE/R2 + gender accuracy."""

    results: Dict[str, Dict[str, float]] = {}
    for key in ("height", "weight", "age"):
        pred = _denorm(np.asarray(y_pred[key], dtype=np.float32), key, target_stats)
        true = np.asarray(y_true[key], dtype=np.float32)
        valid = np.isfinite(true) & np.isfinite(pred)
        if not np.any(valid):
            results[key] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "median_ae": float("nan"),
                "r2": float("nan"),
            }
            continue
        true = true[valid]
        pred = pred[valid]
        abs_err = np.abs(true - pred)
        results[key] = {
            "mae": float(np.mean(abs_err)),
            "rmse": _rmse(true, pred),
            "median_ae": float(np.median(abs_err)),
            "r2": float(r2_score(true, pred)) if true.size > 1 else float("nan"),
        }

    g_pred = np.asarray(y_pred["gender"], dtype=np.int64)
    g_true = np.asarray(y_true["gender"], dtype=np.int64)
    if g_true.size == 0:
        results["gender"] = {"accuracy": float("nan"), "report": {}, "confusion_matrix": []}
    else:
        results["gender"] = {
            "accuracy": float(np.mean(g_pred == g_true)),
            "report": classification_report(g_true, g_pred, target_names=["female", "male"], output_dict=True),
            "confusion_matrix": confusion_matrix(g_true, g_pred).tolist(),
        }
    return results


def meets_targets(eval_results: Mapping[str, Mapping[str, float]]) -> Dict[str, bool]:
    return {
        "height_ok": float(eval_results.get("height", {}).get("mae", 999.0)) <= 3.0,
        "weight_ok": float(eval_results.get("weight", {}).get("mae", 999.0)) <= 5.0,
        "age_ok": float(eval_results.get("age", {}).get("mae", 999.0)) <= 5.0,
        "gender_ok": float(eval_results.get("gender", {}).get("accuracy", 0.0)) >= 0.95,
    }


def print_eval_report(eval_results: Mapping[str, Mapping[str, float]]) -> None:
    targets = meets_targets(eval_results)
    print("\n" + "=" * 50)
    print("  VocalMorph Evaluation Report")
    print("=" * 50)
    for key in ("height", "weight", "age"):
        result = eval_results.get(key, {})
        ok = "OK" if targets.get(f"{key}_ok") else "FAIL"
        unit = "cm" if key == "height" else ("kg" if key == "weight" else "yr")
        print(
            f"  [{ok}] {key.capitalize():8s} "
            f"MAE={float(result.get('mae', float('nan'))):.2f}{unit} "
            f"RMSE={float(result.get('rmse', float('nan'))):.2f} "
            f"MedianAE={float(result.get('median_ae', float('nan'))):.2f}"
        )
    gender = eval_results.get("gender", {})
    ok = "OK" if targets.get("gender_ok") else "FAIL"
    print(f"  [{ok}] Gender   Acc={float(gender.get('accuracy', float('nan'))) * 100:.1f}%")
    print("=" * 50 + "\n")


__all__ = [
    "compute_full_eval",
    "compute_metrics",
    "meets_targets",
    "print_eval_report",
]
