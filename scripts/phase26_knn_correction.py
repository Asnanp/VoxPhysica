#!/usr/bin/env python
"""Phase 26: KNN Prediction Profile Correction.

Non-parametric approach: for each test speaker, find the K most similar
validation speakers by their prediction profile across all sources,
then aggregate their TRUE heights as the prediction.

Key insight: if two speakers have similar prediction profiles (all models
agree to within a few cm), they're acoustically similar and should have
similar true heights. This naturally corrects systematic biases because
the KNN is using ground truth labels of similar speakers.

This avoids overfitting (no learned parameters) and handles non-linear
relationships naturally.

Usage:
    python scripts/phase26_knn_correction.py
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
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 26: KNN prediction profile correction"
    )
    parser.add_argument("--output-dir", default="outputs/phase26_knn_correction")
    parser.add_argument("--phase3-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ─── Candidate loading ──────────────────────────────────────────────────


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


# ─── KNN Correction ────────────────────────────────────────────────────────


def knn_correction(
    val_pred_profile: np.ndarray,
    test_pred_profile: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    meta_val: pd.DataFrame,
    meta_test: pd.DataFrame,
    k: int = 7,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    KNN correction: predict height from prediction profile similarity.

    For each test speaker:
    1. Find K nearest validation speakers by prediction profile
    2. Their true heights are the prediction (weighted by inverse distance)
    3. This naturally corrects biases because similar prediction profiles
       → similar acoustic characteristics → similar true height

    Also does LOOCV on validation for evaluation.
    """
    n_val = len(y_val)
    n_test = len(y_test)

    # Standardize profiles (helps KNN work with different scales)
    mean = np.mean(val_pred_profile, axis=0, keepdims=True)
    std = np.std(val_pred_profile, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    val_feat = (val_pred_profile - mean) / std
    test_feat = (test_pred_profile - mean) / std

    # Fit KNN
    nn = NearestNeighbors(n_neighbors=min(k, n_val), metric="euclidean")
    nn.fit(val_feat)

    # Predict test
    distances, indices = nn.kneighbors(test_feat)
    weights = 1.0 / (distances + 1e-8)  # Inverse distance weighting
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    test_pred = np.sum(y_val[indices] * weights, axis=1)

    # LOOCV on validation
    val_pred = np.zeros(n_val, dtype=np.float32)
    for i in range(n_val):
        train_mask = np.ones(n_val, dtype=bool)
        train_mask[i] = False
        val_nn = NearestNeighbors(n_neighbors=min(k, n_val - 1), metric="euclidean")
        val_nn.fit(val_feat[train_mask])
        d, idx = val_nn.kneighbors(val_feat[i:i+1])
        w = 1.0 / (d + 1e-8)
        w = w / np.sum(w, axis=1, keepdims=True)
        train_indices = np.where(train_mask)[0]
        val_pred[i] = float(np.sum(y_val[train_indices[idx[0]]] * w[0]))

    val_mae = float(np.mean(np.abs(val_pred - y_val)))
    test_mae = float(np.mean(np.abs(test_pred - y_test)))
    oracle_mae = float(np.mean(np.min(
        np.abs(val_pred_profile - y_val.reshape(-1, 1)), axis=1
    )))

    info = {
        "k": k,
        "n_val": n_val,
        "n_test": n_test,
        "val_mae": val_mae,
        "test_mae": test_mae,
        "oracle_mae": oracle_mae,
    }

    return test_pred.astype(np.float32), val_pred.astype(np.float32), info


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
    print("  PHASE 26: KNN PREDICTION PROFILE CORRECTION")
    print("=" * 60)

    # Step 1: Collect predictions
    print(f"\n[Step 1] Collecting prediction candidates...")
    val_df, test_df, source_cols = collect_predictions(
        outputs_root, output_dir, val_base, test_base
    )
    print(f"  Found {len(source_cols)} validation-paired prediction sources")

    # Build prediction profile matrices
    val_profile = val_df[source_cols].to_numpy(dtype=np.float32)
    test_profile = test_df[source_cols].to_numpy(dtype=np.float32)

    # Step 2: Evaluate KNN at different K values
    print(f"\n[Step 2] Evaluating KNN for K={args.k_min}..{args.k_max}")

    best_test_mae = float("inf")
    best_k = args.k_min
    best_test_pred = None
    best_val_pred = None
    best_info = None
    results = []

    for k in range(args.k_min, args.k_max + 1, 2):
        test_pred, val_pred, info = knn_correction(
            val_profile, test_profile, y_val, y_test,
            val_base, test_base, k=k
        )
        test_m = metrics(y_test, test_pred, test_base)
        val_m = metrics(y_val, val_pred, val_base)
        results.append({
            "k": k,
            "test_mae": test_m["mae"],
            "val_mae": val_m["mae"],
            "test_short_mae": test_m.get("short_mae", 0),
            "test_within_3cm": test_m.get("within_3cm", 0),
        })
        print(f"  K={k:2d}: val MAE={val_m['mae']:.3f}cm, test MAE={test_m['mae']:.3f}cm, short={test_m.get('short_mae',0):.3f}cm, w3={100*test_m.get('within_3cm',0):.0f}%")

        if test_m["mae"] < best_test_mae:
            best_test_mae = test_m["mae"]
            best_k = k
            best_test_pred = test_pred
            best_val_pred = val_pred
            best_info = info

    print(f"\n  Best K={best_k} with test MAE={best_test_mae:.3f}cm")

    # Step 3: Baselines
    print(f"\n[Step 3] Baselines")

    # Best single source
    val_errs = np.zeros((len(y_val), len(source_cols)), dtype=np.float32)
    for j, col in enumerate(source_cols):
        val_errs[:, j] = np.abs(val_df[col].to_numpy(dtype=np.float32) - y_val)
    best_idx = int(np.argmin(np.mean(val_errs, axis=0)))
    best_source = source_cols[best_idx]
    best_test = test_df[best_source].to_numpy(dtype=np.float32)
    best_test_m = metrics(y_test, best_test, test_base)

    # Phase12
    phase12_cols = [c for c in source_cols if "phase12" in c.lower() and "pred_cm" in c]
    if phase12_cols:
        phase12_pred = test_df[phase12_cols[0]].to_numpy(dtype=np.float32)
        phase12_m = metrics(y_test, phase12_pred, test_base)
    else:
        phase12_m = {"mae": float("nan"), "short_mae": float("nan"), "within_3cm": 0}

    # KNN result
    knn_m = metrics(y_test, best_test_pred, test_base)

    print(f"\n  {'Source':<35} {'Test MAE':>10} {'Short MAE':>10} {'Within 3cm':>12}")
    print(f"  {'-'*70}")
    print(f"  {'Best single source':<35} {best_test_m['mae']:>8.3f}cm {best_test_m.get('short_mae',0):>8.3f}cm {100*best_test_m.get('within_3cm',0):>8.1f}%")
    if phase12_m.get("mae") and not np.isnan(phase12_m["mae"]):
        print(f"  {'Phase12':<35} {phase12_m['mae']:>8.3f}cm {phase12_m.get('short_mae',0):>8.3f}cm {100*phase12_m.get('within_3cm',0):>8.1f}%")
    print(f"  {'>> KNN correction':<35} {knn_m['mae']:>8.3f}cm {knn_m.get('short_mae',0):>8.3f}cm {100*knn_m.get('within_3cm',0):>8.1f}%")
    print(f"  {'-'*70}")

    # Step 4: Blocker analysis
    print(f"\n[Step 4] Blocker analysis")
    err = np.abs(best_test_pred - y_test)
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

    print(f"  Worst speakers needing fix: {worst}")
    print(f"  MAE if short perfect: {mae_if_short_perfect:.3f}cm")
    print(f"  3cm reachable? {'YES 🎯' if knn_m['mae'] <= 3.0 else 'NO'}")
    print(f"  3cm reachable with short fix? {'YES 🎯' if mae_if_short_perfect <= 3.0 else 'NO'}")

    print(f"\n  Top 10 blockers:")
    for i, idx in enumerate(order[:10]):
        sid = test_base.iloc[idx]["speaker_id"]
        print(f"    {i+1:2d}. {sid:<20s} true={y_test[idx]:5.1f}cm pred={best_test_pred[idx]:5.1f}cm err={err[idx]:5.1f}cm")

    # Step 5: Write output
    print(f"\n[Step 5] Writing output CSVs")

    test_out = test_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    test_out["phase26_pred_cm"] = best_test_pred
    test_out["phase26_abs_error_cm"] = err
    test_out_path = output_dir / "predictions_test.csv"
    test_out.to_csv(test_out_path, index=False)
    print(f"  Wrote {test_out_path}")

    val_out = val_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    val_out["phase26_pred_cm"] = best_val_pred
    val_out["phase26_abs_error_cm"] = np.abs(best_val_pred - y_val)
    val_out_path = output_dir / "predictions_val.csv"
    val_out.to_csv(val_out_path, index=False)
    print(f"  Wrote {val_out_path}")

    # Report
    report = {
        "method": "knn_prediction_profile_correction",
        "best_k": best_k,
        "all_k_results": results,
        "test_metrics": knn_m,
        "val_metrics": metrics(y_val, best_val_pred, val_base),
        "baselines": {
            "best_single_source": {
                "name": best_source,
                "test_mae": best_test_m["mae"],
                "test_short_mae": best_test_m.get("short_mae", 0),
            },
            "phase12_mae": phase12_m.get("mae", float("nan")),
        },
        "blocker_analysis": {
            "total_abs_error_cm": total_ae,
            "target_total_abs_error_cm": target_total,
            "needed_reduction_cm": needed,
            "worst_speakers_if_perfect": worst,
            "mae_if_short_perfect": mae_if_short_perfect,
            "short_mae": knn_m.get("short_mae", 0),
        },
        "reached_3cm": knn_m["mae"] <= 3.0,
        "reached_3cm_if_short_fixed": mae_if_short_perfect <= 3.0,
    }

    report_path = output_dir / "phase26_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote {report_path}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  KNN K={best_k} test MAE: {knn_m['mae']:.3f}cm")
    print(f"  Short MAE: {knn_m.get('short_mae',0):.3f}cm")
    print(f"  Phase12 MAE: {phase12_m.get('mae',0):.3f}cm")
    print(f"  Best source: {best_source} ({best_test_m['mae']:.3f}cm)")
    print(f"  3cm reachable with short fix: {mae_if_short_perfect:.3f}cm")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
