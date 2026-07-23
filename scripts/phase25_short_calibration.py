#!/usr/bin/env python
"""Phase 25: Height-Dependent Short Speaker Calibration.

Instead of a complex meta-model, this uses a simple height-dependent
source switching strategy:

1. For each test speaker, determine which prediction sources perform best
   in their height segment (short/medium/tall) based on VALIDATION data
2. Use predicted height (from any source) as a proxy to assign the right
   source for each speaker
3. Apply a smooth blend transition at height boundaries

Key insight from Phase22 blocker analysis:
- Short speakers (<165cm) are systematically over-predicted by 15-22cm
- Phase21_ultra_short achieves 5.72cm short MAE (vs Phase12 6.99cm)
- Different sources excel in different height ranges
- A height-dependent per-speaker source selector could capture this

Usage:
    python scripts/phase25_short_calibration.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 25: Height-dependent short speaker calibration"
    )
    parser.add_argument("--output-dir", default="outputs/phase25_short_calibration")
    parser.add_argument("--phase3-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ─── Candidate loading (same as phase22/24) ──────────────────────────────


def read_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"speaker_id", "height_cm"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing required columns: {sorted(missing)}")
    keep = ["speaker_id", "height_cm"]
    for col in ("source", "gender", "age"):
        if col in df.columns:
            keep.append(col)
    return df[keep].copy()


def is_prediction_column(df: pd.DataFrame, col: str) -> bool:
    lower = col.lower()
    if col in {"height_cm", "gender", "age", "speaker_id", "source", "source_id"}:
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    if any(token in lower for token in ("error", "mae", "rmse", "count", "probability", "std", "uncert", "abs_")):
        return False
    return lower.endswith("_cm") or "pred" in lower


def align_prediction(base: pd.DataFrame, pred_df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
    if "speaker_id" not in pred_df.columns or col not in pred_df.columns:
        return None
    pred_small = pred_df[["speaker_id", col]].copy()
    if pred_small["speaker_id"].duplicated().any():
        pred_small = pred_small.groupby("speaker_id", as_index=False)[col].mean(numeric_only=True)
    merged = base[["speaker_id"]].merge(pred_small, on="speaker_id", how="left")
    if merged[col].isna().any():
        return None
    values = merged[col].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        return None
    if float(np.max(np.abs(values))) > 300.0:
        return None
    return values


def candidate_val_paths(test_path: Path) -> List[Path]:
    names = []
    name = test_path.name
    replacements = [
        ("_predictions_test.csv", "_predictions_val.csv"),
        ("_predictions_test.csv", "_predictions_oof_dev.csv"),
        ("_test.csv", "_val.csv"),
        ("_test.csv", "_oof_dev.csv"),
        ("test_speaker_predictions.csv", "val_speaker_predictions.csv"),
        ("test_clip_predictions.csv", "val_clip_predictions.csv"),
        ("predictions_test.csv", "predictions_val.csv"),
        ("predictions_test.csv", "predictions_oof_dev.csv"),
    ]
    for old, new in replacements:
        if old in name:
            names.append(test_path.with_name(name.replace(old, new)))
    return list(dict.fromkeys(names))


def iter_prediction_csvs(outputs_root: Path, output_dir: Path) -> List[Path]:
    skip_parts = {".git", "__pycache__", "pytest-temp", output_dir.name.lower()}
    csvs: List[Path] = []
    for dirpath_str, dirnames, filenames in os.walk(str(outputs_root)):
        dirpath = Path(dirpath_str)
        dirnames[:] = [d for d in dirnames if d.lower() not in skip_parts]
        for filename in filenames:
            lower = filename.lower()
            if not lower.endswith(".csv"):
                continue
            if "prediction" not in lower or "test" not in lower:
                continue
            if "oracle" in lower:
                continue
            csvs.append(dirpath / filename)
    return csvs


def collect_predictions(
    outputs_root: Path, output_dir: Path,
    val_base: pd.DataFrame, test_base: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Collect all prediction candidates."""
    test_sources: Dict[str, np.ndarray] = {}
    val_sources: Dict[str, np.ndarray] = {}
    seen: set[bytes] = set()

    for test_path in iter_prediction_csvs(outputs_root, output_dir):
        try:
            test_df = pd.read_csv(test_path)
        except Exception:
            continue
        if "speaker_id" not in test_df.columns or "height_cm" not in test_df.columns:
            continue

        val_df: Optional[pd.DataFrame] = None
        for vp in candidate_val_paths(test_path):
            if vp.exists():
                try:
                    val_df = pd.read_csv(vp)
                    break
                except Exception:
                    val_df = None

        for col in [c for c in test_df.columns if is_prediction_column(test_df, c)]:
            test_pred = align_prediction(test_base, test_df, col)
            if test_pred is None:
                continue
            sig = np.round(test_pred, 4).tobytes()
            if sig in seen:
                continue
            seen.add(sig)
            val_pred = None
            if val_df is not None and col in val_df.columns:
                val_pred = align_prediction(val_base, val_df, col)
            if val_pred is None:
                continue

            name = f"{test_path.relative_to(outputs_root).as_posix()}:{col}"
            test_sources[name] = test_pred
            val_sources[name] = val_pred

    # Build DataFrames
    test_df = test_base[["speaker_id", "height_cm"]].copy()
    val_df = val_base[["speaker_id", "height_cm"]].copy()
    for name in sorted(test_sources.keys()):
        test_df[name] = test_sources[name]
        val_df[name] = val_sources[name]

    feature_cols = sorted(set(test_sources.keys()) & set(val_sources.keys()))
    return val_df, test_df, feature_cols


# ─── Metrics ───────────────────────────────────────────────────────────────


def metrics(y: np.ndarray, pred: np.ndarray, meta: pd.DataFrame) -> Dict[str, float]:
    err = np.asarray(pred, dtype=np.float32) - np.asarray(y, dtype=np.float32)
    ae = np.abs(err)
    out: Dict[str, float] = {
        "count": float(len(y)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "bias": float(np.mean(err)),
        "within_3cm": float(np.mean(ae <= 3.0)),
        "within_5cm": float(np.mean(ae <= 5.0)),
    }

    y_arr = np.asarray(y, dtype=np.float32)
    masks = {
        "short": y_arr < 165.0,
        "medium": (y_arr >= 165.0) & (y_arr < 178.0),
        "tall": y_arr >= 178.0,
    }
    if "source" in meta.columns:
        src = meta["source"].astype(str).str.upper().to_numpy()
        masks["source_nisp"] = src == "NISP"
        masks["source_timit"] = src == "TIMIT"
    if "gender" in meta.columns:
        g = meta["gender"].to_numpy()
        masks["female"] = g == 0
        masks["male"] = g == 1

    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        out[f"{name}_n"] = float(mask.sum())
        out[f"{name}_mae"] = float(np.mean(ae[mask]))
    return out


# ─── Height-Dependent Source Switching ─────────────────────────────────────


def compute_height_segment_errors(
    val_df: pd.DataFrame, y_val: np.ndarray, source_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-source MAE in each height segment on validation."""
    segments = {
        "short": y_val < 165.0,
        "medium": (y_val >= 165.0) & (y_val < 178.0),
        "tall": y_val >= 178.0,
    }
    result: Dict[str, Dict[str, float]] = {}
    for seg_name, mask in segments.items():
        n = int(mask.sum())
        if n == 0:
            continue
        result[seg_name] = {}
        for col in source_cols:
            pred = val_df[col].to_numpy(dtype=np.float32)
            err = np.abs(pred[mask] - y_val[mask])
            result[seg_name][col] = float(np.mean(err))
        result[seg_name]["_n"] = n
    return result


def height_dependent_switch(
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    source_cols: List[str],
    y_val: np.ndarray,
    y_test: np.ndarray,
    meta_val: pd.DataFrame,
    meta_test: pd.DataFrame,
    short_threshold: float = 168.0,
    tall_threshold: float = 175.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Height-dependent source switching.

    For each test speaker:
    1. Use predicted height as a proxy for true height
    2. If predicted-short, use the source with best validation short MAE
    3. If predicted-tall, use the source with best validation tall MAE
    4. Otherwise use the best overall source

    Uses a smooth sigmoid transition at height boundaries.
    """
    seg_errors = compute_height_segment_errors(val_df, y_val, source_cols)

    # Find best source per segment
    best_short = min(
        [(seg_errors["short"][c], c) for c in source_cols],
        key=lambda x: x[0]
    )[1]
    best_medium = min(
        [(seg_errors["medium"][c], c) for c in source_cols],
        key=lambda x: x[0]
    )[1]
    best_tall = min(
        [(seg_errors["tall"][c], c) for c in source_cols],
        key=lambda x: x[0]
    )[1]

    # Best overall source
    val_errs = np.zeros((len(y_val), len(source_cols)), dtype=np.float32)
    for j, col in enumerate(source_cols):
        val_errs[:, j] = np.abs(val_df[col].to_numpy(dtype=np.float32) - y_val)
    best_overall_idx = int(np.argmin(np.mean(val_errs, axis=0)))
    best_overall = source_cols[best_overall_idx]

    # Also find a "short-specialized" blend source (average of top-3 short sources)
    short_sorted = sorted(
        [(seg_errors["short"][c], c) for c in source_cols],
        key=lambda x: x[0]
    )
    top_short_cols = [c for _, c in short_sorted[:5]]
    top_short_test_pred = np.mean(
        np.column_stack([test_df[c].to_numpy(dtype=np.float32) for c in top_short_cols]),
        axis=1
    )
    top_short_val_pred = np.mean(
        np.column_stack([val_df[c].to_numpy(dtype=np.float32) for c in top_short_cols]),
        axis=1
    )

    print(f"\n[HeightSwitch] Best sources per segment:")
    print(f"  Short (<165cm val): {best_short} (MAE={seg_errors['short'][best_short]:.3f}cm)")
    print(f"  Medium (165-178cm): {best_medium} (MAE={seg_errors['medium'][best_medium]:.3f}cm)")
    print(f"  Tall (>178cm):      {best_tall} (MAE={seg_errors['tall'][best_tall]:.3f}cm)")
    print(f"  Overall best:       {best_overall} (MAE={float(np.mean(val_errs[:, best_overall_idx])):.3f}cm)")
    print(f"  Top-5 short avg: short MAE={seg_errors['short'].get('top_5_short_avg', 0):.3f}cm")

    # Use the top-5 short avg as the short-specialized source
    # Use best overall for medium/tall

    # Get base predictions
    base_test = test_df[best_overall].to_numpy(dtype=np.float32).copy()
    short_specialized_test = top_short_test_pred

    # Estimate height proxy: median of top-3 overall predictions
    top3_cols = sorted(
        [(float(np.mean(val_errs[:, j])), source_cols[j]) for j in range(len(source_cols))],
        key=lambda x: x[0]
    )[:3]
    top3_names = [c for _, c in top3_cols]
    height_proxy_test = np.median(
        np.column_stack([test_df[c].to_numpy(dtype=np.float32) for c in top3_names]),
        axis=1
    )
    height_proxy_val = np.median(
        np.column_stack([val_df[c].to_numpy(dtype=np.float32) for c in top3_names]),
        axis=1
    )

    print(f"\n[HeightSwitch] Using height proxy (median of top-3 sources)")

    # Smooth sigmoid blending: at short_threshold, blend from short to medium
    # Use height proxy to determine blend weight
    def sigmoid(x, center, width=3.0):
        return 1.0 / (1.0 + np.exp(-(x - center) / width))

    # Weight toward short-specialized for lower heights
    w_short_test = 1.0 - sigmoid(height_proxy_test, short_threshold, width=4.0)
    w_short_val = 1.0 - sigmoid(height_proxy_val, short_threshold, width=4.0)

    # Blend: short_specialized when proxy is low, best_overall when proxy is high
    test_pred = w_short_test * short_specialized_test + (1.0 - w_short_test) * base_test
    val_pred = w_short_val * top_short_val_pred + (1.0 - w_short_val) * val_df[best_overall].to_numpy(dtype=np.float32)

    info = {
        "method": "height_dependent_switch",
        "best_short_source": best_short,
        "best_medium_source": best_medium,
        "best_tall_source": best_tall,
        "best_overall_source": best_overall,
        "short_threshold_cm": short_threshold,
        "tall_threshold_cm": tall_threshold,
        "top_5_short_sources": top_short_cols,
        "n_switched_test": int(np.sum(w_short_test > 0.25)),
        "n_mostly_short_test": int(np.sum(w_short_test > 0.75)),
    }

    return test_pred, val_pred, info


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_root = resolve(args.outputs_root)

    val_base = read_base(resolve(args.phase3_val))
    test_base = read_base(resolve(args.phase3_test))
    y_val = val_base["height_cm"].to_numpy(dtype=np.float32)
    y_test = test_base["height_cm"].to_numpy(dtype=np.float32)

    print("=" * 60)
    print("  PHASE 25: HEIGHT-DEPENDENT SHORT CALIBRATION")
    print("=" * 60)

    # Step 1: Collect predictions
    print(f"\n[Step 1] Collecting prediction candidates...")
    val_df, test_df, source_cols = collect_predictions(
        outputs_root, output_dir, val_base, test_base
    )
    print(f"  Found {len(source_cols)} validation-paired prediction sources")

    # Step 2: Height-dependent switching
    print(f"\n[Step 2] Running height-dependent source switching...")
    test_pred, val_pred, info = height_dependent_switch(
        test_df, val_df, source_cols,
        y_val, y_test, val_base, test_base,
        short_threshold=168.0,
        tall_threshold=178.0,
    )

    # Step 3: Evaluate
    print(f"\n[Step 3] Evaluation")

    # Baselines
    # Best single source
    val_errs = np.zeros((len(y_val), len(source_cols)), dtype=np.float32)
    for j, col in enumerate(source_cols):
        val_errs[:, j] = np.abs(val_df[col].to_numpy(dtype=np.float32) - y_val)
    best_idx = int(np.argmin(np.mean(val_errs, axis=0)))
    best_source = source_cols[best_idx]
    best_test_pred = test_df[best_source].to_numpy(dtype=np.float32)
    best_test_metrics = metrics(y_test, best_test_pred, test_base)

    # Phase12 specifically
    phase12_cols = [c for c in source_cols if "phase12" in c.lower() and "pred_cm" in c]
    if phase12_cols:
        phase12_pred = test_df[phase12_cols[0]].to_numpy(dtype=np.float32)
        phase12_metrics = metrics(y_test, phase12_pred, test_base)
    else:
        phase12_metrics = {"mae": float("nan"), "short_mae": float("nan"), "within_3cm": 0}

    # Our model
    val_metrics = metrics(y_val, val_pred, val_base)
    test_metrics = metrics(y_test, test_pred, test_base)

    print(f"\n  {'Source':<35} {'Val MAE':>10} {'Test MAE':>10} {'Short MAE':>10} {'Within 3cm':>12}")
    print(f"  {'-'*80}")
    print(f"  {'Best single source':<35} {float(np.mean(val_errs[:, best_idx])):>8.3f}cm {best_test_metrics['mae']:>8.3f}cm {best_test_metrics.get('short_mae',0):>8.3f}cm {100*best_test_metrics.get('within_3cm',0):>8.1f}%")
    if phase12_metrics.get("mae") and not np.isnan(phase12_metrics["mae"]):
        print(f"  {'Phase12':<35} {'':>10} {phase12_metrics['mae']:>8.3f}cm {phase12_metrics.get('short_mae',0):>8.3f}cm {100*phase12_metrics.get('within_3cm',0):>8.1f}%")
    print(f"  {'>> Height switch':<35} {val_metrics['mae']:>8.3f}cm {test_metrics['mae']:>8.3f}cm {test_metrics.get('short_mae',0):>8.3f}cm {100*test_metrics.get('within_3cm',0):>8.1f}%")
    print(f"  {'-'*80}")

    # Blocker analysis
    err = np.abs(test_pred - y_test)
    order = np.argsort(err)[::-1]
    total_ae = float(np.sum(err))
    target_total = 3.0 * len(y_test)
    needed = max(0.0, total_ae - target_total)
    worst = 0
    cum = 0.0
    for e in sorted(err)[::-1]:
        if cum >= needed:
            break
        cum += float(e)
        worst += 1

    short_mask = y_test < 165.0
    mae_if_short_perfect = float(
        (np.sum(err[~short_mask]) if short_mask.any() else np.sum(err)) / len(y_test)
    )

    print(f"\n  Blocker analysis:")
    print(f"  {'':<35} {'Value':>15}")
    print(f"  {'-'*50}")
    print(f"  {'Test MAE':<35} {test_metrics['mae']:>10.3f}cm")
    print(f"  {'Short MAE':<35} {test_metrics.get('short_mae', 0):>10.3f}cm")
    print(f"  {'Total abs error':<35} {total_ae:>10.1f}cm")
    print(f"  {'Needed reduction to 3cm':<35} {needed:>10.1f}cm")
    print(f"  {'Worst speakers to fix':<35} {worst:>10d}")
    print(f"  {'MAE if short perfect':<35} {mae_if_short_perfect:>10.3f}cm")
    print(f"  {'Within 3cm':<35} {100*test_metrics.get('within_3cm',0):>10.1f}%")

    # Top blockers
    print(f"\n  Top 10 blockers:")
    for i, idx in enumerate(order[:10]):
        sid = test_base.iloc[idx]["speaker_id"]
        print(f"    {i+1:2d}. {sid:<20s} true={y_test[idx]:5.1f}cm pred={test_pred[idx]:5.1f}cm err={err[idx]:5.1f}cm")

    # Step 4: Write output
    print(f"\n[Step 4] Writing output CSVs")

    test_out = test_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    test_out["phase25_pred_cm"] = test_pred
    test_out["phase25_abs_error_cm"] = err
    test_out_path = output_dir / "predictions_test.csv"
    test_out.to_csv(test_out_path, index=False)
    print(f"  Wrote {test_out_path}")

    val_out = val_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    val_out["phase25_pred_cm"] = val_pred
    val_out["phase25_abs_error_cm"] = np.abs(val_pred - y_val)
    val_out_path = output_dir / "predictions_val.csv"
    val_out.to_csv(val_out_path, index=False)
    print(f"  Wrote {val_out_path}")

    # Report
    report = {
        "method": "height_dependent_switch",
        "info": info,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_best_source": {
            "name": best_source,
            "test_mae": best_test_metrics["mae"],
            "test_short_mae": best_test_metrics.get("short_mae", 0),
        },
        "blocker_analysis": {
            "total_abs_error_cm": total_ae,
            "target_total_abs_error_cm": target_total,
            "needed_reduction_cm": needed,
            "worst_speakers_if_perfect": worst,
            "mae_if_short_perfect": mae_if_short_perfect,
            "short_mae": test_metrics.get("short_mae", 0),
        },
        "reached_3cm": test_metrics["mae"] <= 3.0,
        "reached_3cm_if_short_fixed": mae_if_short_perfect <= 3.0,
    }

    report_path = output_dir / "phase25_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote {report_path}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY: Test MAE={test_metrics['mae']:.3f}cm, Short MAE={test_metrics.get('short_mae', 0):.3f}cm")
    print(f"  3cm reachable? {'YES 🎯' if test_metrics['mae'] <= 3.0 else 'NO'}")
    print(f"  If short perfect: {mae_if_short_perfect:.3f}cm {'(reached 3cm!) 🎯' if mae_if_short_perfect <= 3.0 else ''}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
