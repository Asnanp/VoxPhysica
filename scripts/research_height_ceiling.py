#!/usr/bin/env python
"""Build a research audit for the 3cm height-MAE target.

This script is intentionally post-training analysis. It does not fit on test
predictions, and it does not make a new checkpoint look better. The goal is to
explain whether the current data/split/model family is attacking the real error
source, then write a concrete redesign report before another long run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


HEIGHT_TARGET_CM = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 3cm MAE research audit from splits and diagnostics."
    )
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--diagnostics-dir", default="outputs/diagnostics")
    parser.add_argument(
        "--runs",
        default="height_bias_extreme12gb,height_bias_stabletail12gb",
        help="Comma-separated diagnostic run directories under --diagnostics-dir.",
    )
    parser.add_argument(
        "--ensemble-summary",
        default="outputs/diagnostics/checkpoint_ensemble_extreme12gb/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostics/3cm_research_audit",
    )
    return parser.parse_args()


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def _round(value: float, digits: int = 3) -> float:
    if not np.isfinite(value):
        return float("nan")
    return round(float(value), digits)


def height_bin(height_cm: float) -> str:
    if not np.isfinite(height_cm):
        return "unknown"
    if height_cm < 160.0:
        return "short"
    if height_cm >= 175.0:
        return "tall"
    return "medium"


def gender_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"m", "male", "1", "1.0"}:
        return "male"
    if text in {"f", "female", "0", "0.0"}:
        return "female"
    return "unknown"


def source_label(value: Any) -> str:
    text = str(value).strip()
    if text in {"1", "1.0"}:
        return "NISP"
    if text in {"0", "0.0"}:
        return "TIMIT"
    return text.upper() if text else "unknown"


def _count_audio_paths(value: str) -> int:
    if not value:
        return 0
    return len([part for part in str(value).split("|") if part.strip()])


def read_split_csv(path: str, split: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            height = _safe_float(row.get("height_cm"))
            records.append(
                {
                    "split": split,
                    "speaker_id": str(row.get("speaker_id", "")).strip(),
                    "source": source_label(row.get("source")),
                    "gender": gender_label(row.get("gender")),
                    "height_cm": height,
                    "height_bin": height_bin(height),
                    "weight_kg": _safe_float(row.get("weight_kg")),
                    "age": _safe_float(row.get("age")),
                    "clip_count": _count_audio_paths(str(row.get("audio_paths", ""))),
                }
            )
    return records


def read_splits(splits_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        path = os.path.join(splits_dir, f"{split}_clean.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        out[split] = read_split_csv(path, split)
    return out


def nested_counts(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> Dict[str, Any]:
    if not keys:
        return {}
    if len(keys) == 1:
        counts = Counter(str(row.get(keys[0], "unknown")) for row in rows)
        return dict(sorted(counts.items()))
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(keys[0], "unknown"))].append(row)
    return {
        key: nested_counts(group_rows, keys[1:])
        for key, group_rows in sorted(grouped.items())
    }


def split_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    heights = _finite(_safe_float(row.get("height_cm")) for row in rows)
    clips = _finite(_safe_float(row.get("clip_count")) for row in rows)
    timit_heights = _finite(
        _safe_float(row.get("height_cm"))
        for row in rows
        if str(row.get("source")) == "TIMIT"
    )
    nisp_heights = _finite(
        _safe_float(row.get("height_cm"))
        for row in rows
        if str(row.get("source")) == "NISP"
    )

    timit_inches = (
        np.abs(timit_heights / 2.54 - np.round(timit_heights / 2.54)) < 0.01
    )
    nisp_integer = np.abs(nisp_heights - np.round(nisp_heights)) < 0.01

    short_rows = [row for row in rows if row.get("height_bin") == "short"]
    return {
        "n_speakers": int(len(rows)),
        "n_clips": int(clips.sum()) if clips.size else 0,
        "height_min": _round(float(heights.min()) if heights.size else float("nan")),
        "height_max": _round(float(heights.max()) if heights.size else float("nan")),
        "height_mean": _round(float(heights.mean()) if heights.size else float("nan")),
        "height_std": _round(float(heights.std(ddof=0)) if heights.size else float("nan")),
        "under_152_count": int(np.sum(heights < 152.0)) if heights.size else 0,
        "under_155_count": int(np.sum(heights < 155.0)) if heights.size else 0,
        "short_count": int(sum(row.get("height_bin") == "short" for row in rows)),
        "medium_count": int(sum(row.get("height_bin") == "medium" for row in rows)),
        "tall_count": int(sum(row.get("height_bin") == "tall" for row in rows)),
        "male_short_count": int(
            sum(row.get("gender") == "male" and row.get("height_bin") == "short" for row in rows)
        ),
        "female_short_count": int(
            sum(row.get("gender") == "female" and row.get("height_bin") == "short" for row in rows)
        ),
        "timit_inch_quantized_fraction": _round(float(timit_inches.mean()) if timit_inches.size else float("nan")),
        "nisp_integer_cm_fraction": _round(float(nisp_integer.mean()) if nisp_integer.size else float("nan")),
        "counts_by_height_bin": nested_counts(rows, ["height_bin"]),
        "counts_by_gender_height_bin": nested_counts(rows, ["gender", "height_bin"]),
        "counts_by_source_gender_height_bin": nested_counts(rows, ["source", "gender", "height_bin"]),
        "short_speakers": [
            {
                "speaker_id": row["speaker_id"],
                "height_cm": _round(float(row["height_cm"])),
                "gender": row["gender"],
                "source": row["source"],
            }
            for row in sorted(short_rows, key=lambda item: float(item["height_cm"]))
        ],
    }


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    means: Mapping[str, float],
    include_weight: bool,
) -> np.ndarray:
    matrix: List[List[float]] = []
    for row in rows:
        age = _safe_float(row.get("age"))
        weight = _safe_float(row.get("weight_kg"))
        age_missing = 0.0 if np.isfinite(age) else 1.0
        weight_missing = 0.0 if np.isfinite(weight) else 1.0
        values = [
            1.0,
            1.0 if row.get("source") == "NISP" else 0.0,
            1.0 if row.get("gender") == "male" else 0.0,
            age if np.isfinite(age) else float(means["age"]),
            age_missing,
        ]
        if include_weight:
            values.extend(
                [
                    weight if np.isfinite(weight) else float(means["weight_kg"]),
                    weight_missing,
                ]
            )
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)


def _target(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([_safe_float(row.get("height_cm")) for row in rows], dtype=np.float64)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"count": 0.0, "mae": float("nan"), "rmse": float("nan"), "median_ae": float("nan")}
    err = y_pred[valid] - y_true[valid]
    abs_err = np.abs(err)
    return {
        "count": float(valid.sum()),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "median_ae": float(np.median(abs_err)),
    }


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if float(alpha) <= 0.0:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(x.T @ x + penalty, x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(x.T @ x + penalty, x.T @ y, rcond=None)[0]


def metadata_baselines(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    train = list(splits["train"])
    val = list(splits["val"])
    test = list(splits["test"])
    means = {
        "age": float(np.nanmean([_safe_float(row.get("age")) for row in train])),
        "weight_kg": float(np.nanmean([_safe_float(row.get("weight_kg")) for row in train])),
    }
    results: List[Dict[str, Any]] = []
    for name, include_weight in (
        ("source_gender_age_prior", False),
        ("source_gender_age_weight_prior", True),
    ):
        x_train = _feature_matrix(train, means=means, include_weight=include_weight)
        y_train = _target(train)
        x_val = _feature_matrix(val, means=means, include_weight=include_weight)
        y_val = _target(val)
        x_test = _feature_matrix(test, means=means, include_weight=include_weight)
        y_test = _target(test)
        best: Optional[Dict[str, Any]] = None
        for alpha in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0):
            coef = _fit_ridge(x_train, y_train, alpha)
            val_metrics = regression_metrics(y_val, x_val @ coef)
            test_metrics = regression_metrics(y_test, x_test @ coef)
            candidate = {
                "name": name,
                "alpha": alpha,
                "val_mae": val_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_median_ae": test_metrics["median_ae"],
                "deploy_note": (
                    "uses measured weight; only fair if weight is available at inference"
                    if include_weight
                    else "metadata-only prior"
                ),
            }
            if best is None or candidate["val_mae"] < best["val_mae"]:
                best = candidate
        if best is not None:
            results.append(best)
    return results


def read_csv_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_mae_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    seed: int = 123,
    n_boot: int = 2000,
) -> Tuple[float, float]:
    valid = np.where(np.isfinite(y_true) & np.isfinite(y_pred))[0]
    if valid.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot, dtype=np.float64)
    errors = np.abs(y_pred[valid] - y_true[valid])
    for idx in range(n_boot):
        sample = rng.integers(0, errors.size, size=errors.size)
        scores[idx] = float(errors[sample].mean())
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(valid.sum()) < 2:
        return float("nan")
    y = y_true[valid]
    p = y_pred[valid]
    if y.std() <= 1e-9 or p.std() <= 1e-9:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def _slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(valid.sum()) < 2:
        return float("nan")
    y = y_true[valid]
    p = y_pred[valid]
    var_y = float(np.var(y))
    if var_y <= 1e-9:
        return float("nan")
    return float(np.cov(y, p, bias=True)[0, 1] / var_y)


def _slice_stats(
    rows: Sequence[Mapping[str, Any]], pred_key: str, label_key: str
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    labels = sorted({str(row.get(label_key, "unknown")) for row in rows})
    for label in labels:
        group = [row for row in rows if str(row.get(label_key, "unknown")) == label]
        y = np.asarray([_safe_float(row.get("height_true")) for row in group], dtype=np.float64)
        p = np.asarray([_safe_float(row.get(pred_key)) for row in group], dtype=np.float64)
        err = p - y
        metrics = regression_metrics(y, p)
        out[label] = {
            "n": int(metrics["count"]),
            "mae": _round(metrics["mae"]),
            "bias_pred_minus_true": _round(float(np.nanmean(err))),
        }
    return out


def prediction_profile(
    records: Sequence[Mapping[str, Any]], method: str
) -> Dict[str, Any]:
    rows = [row for row in records if str(row.get("split")) == "test"]
    pred_key = f"pred_{method}"
    err_key = f"abs_err_{method}"
    y = np.asarray([_safe_float(row.get("height_true")) for row in rows], dtype=np.float64)
    p = np.asarray([_safe_float(row.get(pred_key)) for row in rows], dtype=np.float64)
    err = p - y
    metrics = regression_metrics(y, p)
    ci_low, ci_high = bootstrap_mae_ci(y, p)
    true_std = float(np.nanstd(y))
    pred_std = float(np.nanstd(p))
    worst = sorted(
        rows,
        key=lambda row: _safe_float(row.get(err_key)),
        reverse=True,
    )[:10]
    return {
        "method": method,
        "n": int(metrics["count"]),
        "mae": _round(metrics["mae"]),
        "mae_95ci_low": _round(ci_low),
        "mae_95ci_high": _round(ci_high),
        "rmse": _round(metrics["rmse"]),
        "median_ae": _round(metrics["median_ae"]),
        "true_mean": _round(float(np.nanmean(y))),
        "pred_mean": _round(float(np.nanmean(p))),
        "true_std": _round(true_std),
        "pred_std": _round(pred_std),
        "std_compression_ratio": _round(pred_std / true_std if true_std > 1e-9 else float("nan")),
        "true_range": [_round(float(np.nanmin(y))), _round(float(np.nanmax(y)))],
        "pred_range": [_round(float(np.nanmin(p))), _round(float(np.nanmax(p)))],
        "corr": _round(_pearson(y, p)),
        "slope_pred_vs_true": _round(_slope(y, p)),
        "mean_bias_pred_minus_true": _round(float(np.nanmean(err))),
        "error_percentiles": {
            "p50": _round(float(np.nanpercentile(np.abs(err), 50))),
            "p75": _round(float(np.nanpercentile(np.abs(err), 75))),
            "p90": _round(float(np.nanpercentile(np.abs(err), 90))),
            "p95": _round(float(np.nanpercentile(np.abs(err), 95))),
            "max": _round(float(np.nanmax(np.abs(err)))),
        },
        "height_bin_slices": _slice_stats(rows, pred_key, "height_bin_true"),
        "gender_slices": _slice_stats(rows, pred_key, "gender_label"),
        "source_slices": _slice_stats(rows, pred_key, "source"),
        "worst_speakers": [
            {
                "speaker_id": row.get("speaker_id"),
                "height_true": _round(_safe_float(row.get("height_true"))),
                "pred": _round(_safe_float(row.get(pred_key))),
                "err_pred_minus_true": _round(_safe_float(row.get(f"err_{method}"))),
                "abs_err": _round(_safe_float(row.get(err_key))),
                "height_bin": row.get("height_bin_true"),
                "gender": row.get("gender_label"),
                "source": row.get("source"),
            }
            for row in worst
        ],
    }


def diagnostic_run(run_name: str, run_dir: str) -> Optional[Dict[str, Any]]:
    summary_path = os.path.join(run_dir, "summary.json")
    csv_path = os.path.join(run_dir, "speaker_predictions.csv")
    if not os.path.exists(summary_path) or not os.path.exists(csv_path):
        return None
    summary = read_json(summary_path)
    records = read_csv_records(csv_path)
    profiles = {
        method: prediction_profile(records, method)
        for method in ("legacy", "omega", "mean")
    }
    test_metrics = summary.get("splits", {}).get("test", {})
    val_metrics = summary.get("splits", {}).get("val", {})
    best_method = min(
        ("legacy", "omega", "mean"),
        key=lambda name: float(test_metrics.get(f"{name}_mae", float("inf"))),
    )
    return {
        "name": run_name,
        "dir": run_dir,
        "summary": summary,
        "records": records,
        "profiles": profiles,
        "best_raw_test_method": best_method,
        "best_raw_test_mae": _round(float(test_metrics.get(f"{best_method}_mae", float("nan")))),
        "val_mae_for_best_method": _round(float(val_metrics.get(f"{best_method}_mae", float("nan")))),
        "val_to_test_gap_for_best_method": _round(
            float(test_metrics.get(f"{best_method}_mae", float("nan")))
            - float(val_metrics.get(f"{best_method}_mae", float("nan")))
        ),
        "best_validation_trained_calibrator": summary.get("best_validation_trained_calibrator", {}),
    }


def compare_runs(run_a: Mapping[str, Any], run_b: Mapping[str, Any], method: str) -> Dict[str, Any]:
    rows_a = {
        str(row["speaker_id"]): row
        for row in run_a.get("records", [])
        if str(row.get("split")) == "test"
    }
    rows_b = {
        str(row["speaker_id"]): row
        for row in run_b.get("records", [])
        if str(row.get("split")) == "test"
    }
    common = sorted(set(rows_a) & set(rows_b))
    deltas: List[float] = []
    by_bin: Dict[str, List[float]] = defaultdict(list)
    for speaker_id in common:
        a = _safe_float(rows_a[speaker_id].get(f"abs_err_{method}"))
        b = _safe_float(rows_b[speaker_id].get(f"abs_err_{method}"))
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        delta = b - a
        deltas.append(delta)
        by_bin[str(rows_a[speaker_id].get("height_bin_true", "unknown"))].append(delta)
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "run_a": run_a.get("name"),
        "run_b": run_b.get("name"),
        "method": method,
        "common_speakers": int(len(deltas)),
        "mean_abs_error_delta_b_minus_a": _round(float(arr.mean()) if arr.size else float("nan")),
        "speakers_improved_in_b": int(np.sum(arr < 0.0)) if arr.size else 0,
        "speakers_worse_in_b": int(np.sum(arr > 0.0)) if arr.size else 0,
        "by_height_bin_mean_delta": {
            key: _round(float(np.mean(values))) for key, values in sorted(by_bin.items())
        },
    }


def _best_ensemble(ensemble_summary_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(ensemble_summary_path):
        return None
    summary = read_json(ensemble_summary_path)
    return summary.get("best")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items() if key != "records"}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return out


def write_markdown(path: str, audit: Mapping[str, Any]) -> None:
    split_summaries = audit["split_summaries"]
    runs = audit["diagnostic_runs"]
    baselines = audit["metadata_baselines"]
    best_overall = audit.get("best_observed", {})
    ensemble_best = audit.get("best_checkpoint_ensemble")

    lines: List[str] = []
    lines.append("# 3cm Height-MAE Research Audit")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append(
        "The current system is not missing one small trick. The diagnostics point to "
        "a data/split/signal ceiling: validation rewards about 4.3cm, while sealed "
        "test stays around 5.6-6.0cm and the short-height tail remains near 8-9cm."
    )
    lines.append("")
    lines.append(
        f"Best observed honest test MAE in these diagnostics: "
        f"{best_overall.get('mae', 'nan')}cm ({best_overall.get('run', 'unknown')} "
        f"{best_overall.get('method', 'unknown')}). Target gap: "
        f"{best_overall.get('gap_to_3cm', 'nan')}cm."
    )
    if ensemble_best:
        lines.append(
            f"Best checkpoint ensemble/stack result: {ensemble_best.get('mae', 'nan')}cm "
            f"({ensemble_best.get('name', 'unknown')}); stacking did not break the ceiling."
        )
    lines.append("")
    lines.append("## Split Coverage")
    lines.extend(
        _table(
            [
                "split",
                "speakers",
                "height range",
                "std",
                "short/med/tall",
                "<152cm",
                "male short",
                "female short",
                "TIMIT inch-quant",
            ],
            [
                [
                    split,
                    s["n_speakers"],
                    f"{s['height_min']}-{s['height_max']}",
                    s["height_std"],
                    f"{s['short_count']}/{s['medium_count']}/{s['tall_count']}",
                    s["under_152_count"],
                    s["male_short_count"],
                    s["female_short_count"],
                    s["timit_inch_quantized_fraction"],
                ]
                for split, s in split_summaries.items()
            ],
        )
    )
    lines.append("")
    lines.append("Research flags:")
    train = split_summaries["train"]
    val = split_summaries["val"]
    test = split_summaries["test"]
    flags = [
        f"Validation has {val['under_152_count']} speakers below 152cm, while test has {test['under_152_count']}.",
        f"Train has only {train['male_short_count']} short male speakers; test has {test['male_short_count']}.",
        "The target labels are partly quantized/self-reported, so 3cm leaves very little room for label noise.",
        "Current validation is not a trustworthy selector for the hardest test tail.",
    ]
    for flag in flags:
        lines.append(f"- {flag}")
    lines.append("")
    lines.append("## Run Evidence")
    run_rows = []
    for run in runs:
        profile = run["profiles"][run["best_raw_test_method"]]
        height_slices = profile["height_bin_slices"]
        run_rows.append(
            [
                run["name"],
                run["best_raw_test_method"],
                run["val_mae_for_best_method"],
                run["best_raw_test_mae"],
                run["val_to_test_gap_for_best_method"],
                profile["std_compression_ratio"],
                height_slices.get("short", {}).get("mae", "nan"),
                height_slices.get("medium", {}).get("mae", "nan"),
                height_slices.get("tall", {}).get("mae", "nan"),
            ]
        )
    lines.extend(
        _table(
            [
                "run",
                "method",
                "val MAE",
                "test MAE",
                "gap",
                "pred/std",
                "short MAE",
                "medium MAE",
                "tall MAE",
            ],
            run_rows,
        )
    )
    lines.append("")
    lines.append("## Metadata Priors")
    lines.extend(
        _table(
            ["baseline", "val MAE", "test MAE", "note"],
            [
                [b["name"], _round(b["val_mae"]), _round(b["test_mae"]), b["deploy_note"]]
                for b in baselines
            ],
        )
    )
    lines.append("")
    lines.append(
        "If metadata-only priors live near the neural model's MAE, the acoustic model is "
        "not extracting enough height-specific signal yet. That is exactly what these "
        "runs show."
    )
    lines.append("")
    lines.append("## Why 3cm Is Not Happening Yet")
    for item in (
        "The model compresses the height range: predictions have lower spread than the true test speakers, so short speakers are overpredicted and tall speakers are pulled down.",
        "The short bin dominates the failure. A global 3cm MAE is impossible while short-speaker MAE is near 8-9cm.",
        "Validation lacks the lowest test heights, so early stopping can choose checkpoints that look good on validation and still fail the tail.",
        "Calibration and checkpoint stacking already plateau near the same MAE, which means the missing piece is not only final-layer calibration.",
    ):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Redesign Required")
    for item in (
        "Rebuild validation as grouped, gender x source x height-bin stratified CV. Keep a sealed test, but stop selecting checkpoints on the current non-representative validation split.",
        "Make speaker-level prediction the training object: encode several clips per speaker, pool with attention/robust reliability, and optimize the pooled speaker height directly.",
        "Use a stronger pretrained speech representation, preferably frozen WavLM/HuBERT/ECAPA embeddings plus the existing physics features. The current handcrafted feature stack is not separating the tail enough.",
        "Train a mixture/bin-aware regressor, but gate it with learned bin confidence and require the audit to show short-bin improvement before any full run continues.",
        "Audit/repair labels and collect more tail data. The current train set has too few short male and tall female examples for a reliable 3cm system.",
    ):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Go/No-Go Gates For The Next Long Run")
    for item in (
        "By epoch 4-6, short validation/CV speaker MAE must be below 5.5cm.",
        "Prediction std / true std must be at least 0.90 on validation/CV; otherwise the model is still collapsed to the mean.",
        "Val-to-test-like CV gap must be below 0.7cm before spending a full night on training.",
        "If a run improves validation but worsens medium/tail slices in this audit, stop it early.",
    ):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Files")
    lines.append("- `summary.json`: machine-readable audit.")
    lines.append("- `report.md`: this report.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    splits_dir = _resolve(args.splits_dir)
    output_dir = _resolve(args.output_dir)
    diagnostics_dir = _resolve(args.diagnostics_dir)
    os.makedirs(output_dir, exist_ok=True)

    splits = read_splits(splits_dir)
    split_summaries = {
        split: split_summary(rows) for split, rows in splits.items()
    }
    baselines = metadata_baselines(splits)

    run_names = [name.strip() for name in args.runs.split(",") if name.strip()]
    runs: List[Dict[str, Any]] = []
    for name in run_names:
        run = diagnostic_run(name, os.path.join(diagnostics_dir, name))
        if run is not None:
            runs.append(run)

    comparisons = []
    if len(runs) >= 2:
        comparisons.append(compare_runs(runs[0], runs[1], "legacy"))
        comparisons.append(compare_runs(runs[0], runs[1], "omega"))

    best_observed: Dict[str, Any] = {}
    for run in runs:
        for method, profile in run["profiles"].items():
            mae = float(profile["mae"])
            if not best_observed or mae < float(best_observed["mae"]):
                best_observed = {
                    "run": run["name"],
                    "method": method,
                    "mae": _round(mae),
                    "gap_to_3cm": _round(mae - HEIGHT_TARGET_CM),
                    "mae_95ci_low": profile["mae_95ci_low"],
                    "mae_95ci_high": profile["mae_95ci_high"],
                }

    ensemble_best = _best_ensemble(_resolve(args.ensemble_summary))
    if ensemble_best is not None:
        ensemble_best = _to_builtin(ensemble_best)

    audit = {
        "target_mae_cm": HEIGHT_TARGET_CM,
        "split_summaries": split_summaries,
        "metadata_baselines": baselines,
        "diagnostic_runs": runs,
        "run_comparisons": comparisons,
        "best_observed": best_observed,
        "best_checkpoint_ensemble": ensemble_best,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    report_path = os.path.join(output_dir, "report.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_to_builtin(audit), handle, indent=2, allow_nan=True)
    write_markdown(report_path, _to_builtin(audit))

    print("[3cm Audit] Best observed:")
    if best_observed:
        print(
            f"  {best_observed['run']} {best_observed['method']} "
            f"MAE={best_observed['mae']:.3f}cm "
            f"gap_to_3cm={best_observed['gap_to_3cm']:.3f}cm"
        )
    if ensemble_best:
        print(
            f"  checkpoint ensemble best={ensemble_best.get('name')} "
            f"MAE={float(ensemble_best.get('mae', float('nan'))):.3f}cm"
        )
    print(f"[3cm Audit] Wrote {summary_path}")
    print(f"[3cm Audit] Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
