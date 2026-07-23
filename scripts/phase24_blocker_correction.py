#!/usr/bin/env python
"""Phase 24: Targeted Blocker Correction Model.

Analyzes the Phase22 blockers and trains a per-speaker correction model
that specifically targets the systematic short-speaker over-prediction bias.

Key insight from Phase22:
- 18 worst blockers are ALL short speakers (<165cm) over-predicted by 15-22cm
- If all short speakers were perfect: 3.12cm MAE (vs 5.47cm deployable)
- Oracle exists at 1.78cm — signal IS there, just not selectable

Strategy:
1. Use all prediction sources as features (ensemble stacking)
2. Train Ridge regression on VAL set with short-speaker sample weighting
3. Apply to TEST set to produce corrected predictions
4. Evaluate improvement on blockers and overall MAE

Usage:
    python scripts/phase24_blocker_correction.py \\
        --output-dir outputs/phase24_blocker_correction \\
        --phase3-val outputs/phase3_target_domain_rescue/phase3_predictions_val.csv \\
        --phase3-test outputs/phase3_target_domain_rescue/phase3_predictions_test.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 24: Targeted short-speaker blocker correction"
    )
    parser.add_argument("--output-dir", default="outputs/phase24_blocker_correction")
    parser.add_argument("--phase3-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--short-weight", type=float, default=5.0,
                        help="Multiplicative weight for short speakers (<165cm) in training")
    parser.add_argument("--cv", type=int, default=5, help="CV folds for RidgeCV alpha search")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ─── Candidate loading (adapted from phase22) ──────────────────────────────


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


def collect_prediction_features(
    outputs_root: Path, output_dir: Path,
    val_base: pd.DataFrame, test_base: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Collect all prediction candidates as feature columns."""
    test_sources: Dict[str, np.ndarray] = {}
    val_sources: Dict[str, np.ndarray] = {}
    seen: set[bytes] = set()
    feature_meta: List[Dict[str, str]] = []

    for test_path in iter_prediction_csvs(outputs_root, output_dir):
        try:
            test_df = pd.read_csv(test_path)
        except Exception:
            continue
        if "speaker_id" not in test_df.columns or "height_cm" not in test_df.columns:
            continue

        # Find val path
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
                continue  # Need val-paired for stacking

            name = f"{test_path.relative_to(outputs_root).as_posix()}:{col}"
            test_sources[name] = test_pred
            val_sources[name] = val_pred
            feature_meta.append({"name": name, "path": str(test_path), "column": col})

    # Build DataFrames
    test_feat = test_base[["speaker_id", "height_cm"]].copy()
    val_feat = val_base[["speaker_id", "height_cm"]].copy()

    for name in sorted(test_sources.keys()):
        test_feat[name] = test_sources[name]
    for name in sorted(val_sources.keys()):
        val_feat[name] = val_sources[name]

    feature_cols = sorted(set(test_sources.keys()) & set(val_sources.keys()))
    return val_feat, test_feat, feature_cols


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

    # Height segments
    masks = {}
    if "source" in meta.columns:
        src = meta["source"].astype(str).str.upper().to_numpy()
        masks["source_nisp"] = src == "NISP"
        masks["source_timit"] = src == "TIMIT"
    if "gender" in meta.columns:
        g = meta["gender"].to_numpy()
        masks["female"] = g == 0
        masks["male"] = g == 1

    y_arr = np.asarray(y, dtype=np.float32)
    masks["short"] = y_arr < 165.0
    masks["medium"] = (y_arr >= 165.0) & (y_arr < 178.0)
    masks["tall"] = y_arr >= 178.0

    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        out[f"{name}_n"] = float(mask.sum())
        out[f"{name}_mae"] = float(np.mean(ae[mask]))

    return out


# ─── Blocker Analysis ──────────────────────────────────────────────────────


def analyze_blockers(
    y: np.ndarray, pred: np.ndarray,
    speaker_ids: Sequence[str], meta: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze which speakers are blocking 3cm."""
    err = np.abs(np.asarray(pred, dtype=np.float32) - np.asarray(y, dtype=np.float32))
    order = np.argsort(err)[::-1]

    blockers = []
    for idx in order[:25]:
        blockers.append({
            "speaker_id": str(speaker_ids[idx]),
            "height_cm": float(y[idx]),
            "pred_cm": float(pred[idx]),
            "abs_error_cm": float(err[idx]),
            "source": str(meta.iloc[idx].get("source", "")),
            "gender": int(meta.iloc[idx].get("gender", -1)),
        })

    # 3cm budget analysis
    total_ae = float(np.sum(err))
    target_total = 3.0 * len(y)
    needed = max(0.0, total_ae - target_total)
    worst_needed = 0
    cum = 0.0
    for e in sorted(err)[::-1]:
        if cum >= needed:
            break
        cum += float(e)
        worst_needed += 1

    short_mask = np.asarray(y, dtype=np.float32) < 165.0
    mae_if_short_perfect = float(
        (np.sum(err[~short_mask]) if short_mask.any() else np.sum(err)) / len(y)
    )

    return {
        "total_abs_error_cm": total_ae,
        "target_total_abs_error_cm": target_total,
        "needed_reduction_cm": needed,
        "worst_speakers_if_perfect": worst_needed,
        "mae_if_short_perfect": mae_if_short_perfect,
        "short_n": int(short_mask.sum()),
        "short_mae": float(np.mean(err[short_mask])) if short_mask.any() else 0.0,
        "top_blockers": blockers,
    }


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs_root = resolve(args.outputs_root)
    val_base = read_base(resolve(args.phase3_val))
    test_base = read_base(resolve(args.phase3_test))

    # ─── Step 1: Collect prediction features ───────────────────────────
    print("=" * 60)
    print("  PHASE 24: TARGETED BLOCKER CORRECTION")
    print("=" * 60)
    print(f"\n[Step 1] Collecting prediction features from {outputs_root}")

    val_feat_df, test_feat_df, feature_cols = collect_prediction_features(
        outputs_root, output_dir, val_base, test_base
    )

    if len(feature_cols) < 3:
        print(f"[ERROR] Only {len(feature_cols)} validation-paired features found. Need at least 3.")
        return 1

    print(f"  Found {len(feature_cols)} validation-paired prediction sources")

    # ─── Step 2: Build feature matrices ────────────────────────────────
    print(f"\n[Step 2] Building feature matrices")

    y_val = val_base["height_cm"].to_numpy(dtype=np.float32)
    y_test = test_base["height_cm"].to_numpy(dtype=np.float32)

    X_val_raw = val_feat_df[feature_cols].to_numpy(dtype=np.float32)
    X_test_raw = test_feat_df[feature_cols].to_numpy(dtype=np.float32)

    # Add meta features if available
    meta_features_val = []
    meta_features_test = []
    meta_names = []

    if "source" in val_base.columns:
        src_val = (val_base["source"].astype(str).str.upper() == "TIMIT").to_numpy(dtype=np.float32)
        src_test = (test_base["source"].astype(str).str.upper() == "TIMIT").to_numpy(dtype=np.float32)
        meta_features_val.append(src_val.reshape(-1, 1))
        meta_features_test.append(src_test.reshape(-1, 1))
        meta_names.append("is_timit")

    if "gender" in val_base.columns:
        g_val = val_base["gender"].to_numpy(dtype=np.float32).reshape(-1, 1)
        g_test = test_base["gender"].to_numpy(dtype=np.float32).reshape(-1, 1)
        meta_features_val.append(g_val)
        meta_features_test.append(g_test)
        meta_names.append("gender")

    # Add derived features: mean prediction, std prediction
    mean_pred_val = np.mean(X_val_raw, axis=1, keepdims=True)
    std_pred_val = np.std(X_val_raw, axis=1, keepdims=True)
    max_pred_val = np.max(X_val_raw, axis=1, keepdims=True)
    min_pred_val = np.min(X_val_raw, axis=1, keepdims=True)
    spread_pred_val = max_pred_val - min_pred_val

    mean_pred_test = np.mean(X_test_raw, axis=1, keepdims=True)
    std_pred_test = np.std(X_test_raw, axis=1, keepdims=True)
    max_pred_test = np.max(X_test_raw, axis=1, keepdims=True)
    min_pred_test = np.min(X_test_raw, axis=1, keepdims=True)
    spread_pred_test = max_pred_test - min_pred_test

    if meta_features_val:
        X_val = np.concatenate(
            [X_val_raw, mean_pred_val, std_pred_val, max_pred_val, spread_pred_val] + meta_features_val,
            axis=1
        )
        X_test = np.concatenate(
            [X_test_raw, mean_pred_test, std_pred_test, max_pred_test, spread_pred_test] + meta_features_test,
            axis=1
        )
    else:
        X_val = np.concatenate(
            [X_val_raw, mean_pred_val, std_pred_val, max_pred_val, spread_pred_val],
            axis=1
        )
        X_test = np.concatenate(
            [X_test_raw, mean_pred_test, std_pred_test, max_pred_test, spread_pred_test],
            axis=1
        )

    all_feature_names = list(feature_cols) + ["mean_pred", "std_pred", "max_pred", "spread_pred"] + meta_names
    print(f"  Feature dimension: {X_val.shape[1]} ({len(feature_cols)} prediction sources + {X_val.shape[1] - len(feature_cols)} derived)")

    # ─── Step 3: Train Ridge with short-speaker weighting ─────────────
    print(f"\n[Step 3] Training Ridge regression (alpha={args.alpha}, short_weight={args.short_weight})")

    # Compute sample weights: short speakers get higher weight
    short_mask_val = y_val < 165.0
    sample_weights = np.ones(len(y_val), dtype=np.float32)
    sample_weights[short_mask_val] *= args.short_weight
    n_short_val = int(short_mask_val.sum())
    print(f"  Val short speakers: {n_short_val}/{len(y_val)} (weighted x{args.short_weight})")

    # Scale features
    scaler = StandardScaler()
    X_val_s = scaler.fit_transform(X_val)
    X_test_s = scaler.transform(X_test)

    # RidgeCV to find best alpha if alpha=0 (auto)
    if args.alpha <= 0:
        print("  Running RidgeCV to find optimal alpha...")
        alphas = np.logspace(-1, 3, 10)
        model = RidgeCV(alphas=alphas, cv=args.cv)
        model.fit(X_val_s, y_val, sample_weight=sample_weights)
        best_alpha = model.alpha_
        print(f"  Selected alpha={best_alpha:.4f}")
    else:
        model = Ridge(alpha=args.alpha)
        model.fit(X_val_s, y_val, sample_weight=sample_weights)
        best_alpha = args.alpha

    # ─── Step 4: Evaluate ──────────────────────────────────────────────
    print(f"\n[Step 4] Evaluating correction model")

    val_pred = model.predict(X_val_s).astype(np.float32)
    test_pred = model.predict(X_test_s).astype(np.float32)

    val_metrics = metrics(y_val, val_pred, val_base)
    test_metrics = metrics(y_test, test_pred, test_base)

    # Baseline: simple mean of all predictions
    base_val_pred = np.mean(X_val_raw, axis=1)
    base_test_pred = np.mean(X_test_raw, axis=1)
    base_val_metrics = metrics(y_val, base_val_pred, val_base)
    base_test_metrics = metrics(y_test, base_test_pred, test_base)

    # Best single source baseline
    source_errors = np.zeros((len(y_val), len(feature_cols)), dtype=np.float32)
    for j, col in enumerate(feature_cols):
        source_errors[:, j] = np.abs(X_val_raw[:, j] - y_val)
    best_source_idx = int(np.argmin(np.mean(source_errors, axis=0)))
    best_source_name = feature_cols[best_source_idx]
    best_source_test_pred = X_test_raw[:, best_source_idx]
    best_source_test_metrics = metrics(y_test, best_source_test_pred, test_base)

    # Phase12 (known best single source) as baseline
    phase12_cols = [c for c in feature_cols if "phase12" in c.lower() and "pred_cm" in c]
    if phase12_cols:
        phase12_idx = feature_cols.index(phase12_cols[0])
        phase12_test_pred = X_test_raw[:, phase12_idx]
        phase12_test_metrics = metrics(y_test, phase12_test_pred, test_base)
    else:
        phase12_test_metrics = {"mae": float("nan"), "short_mae": float("nan")}

    print(f"\n  {'='*45}")
    print(f"  {'Baseline':<30} {'Val MAE':>10} {'Test MAE':>10} {'Short MAE':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Best single source':<30} {np.min(np.mean(source_errors, axis=0)):>8.3f}cm {best_source_test_metrics['mae']:>8.3f}cm {best_source_test_metrics.get('short_mae', 0):>8.3f}cm")
    print(f"  {'Simple average':<30} {base_val_metrics['mae']:>8.3f}cm {base_test_metrics['mae']:>8.3f}cm {base_test_metrics.get('short_mae', 0):>8.3f}cm")
    if phase12_test_metrics.get("mae") and not np.isnan(phase12_test_metrics["mae"]):
        print(f"  {'Phase12 (best known)':<30} {'':>10} {phase12_test_metrics['mae']:>8.3f}cm {phase12_test_metrics.get('short_mae', 0):>8.3f}cm")
    print(f"  {'-'*45}")
    print(f"  {'>> CORRECTION MODEL':<30} {val_metrics['mae']:>8.3f}cm {test_metrics['mae']:>8.3f}cm {test_metrics.get('short_mae', 0):>8.3f}cm")
    print(f"  {'='*45}")

    # ─── Step 5: Blocker analysis ──────────────────────────────────────
    print(f"\n[Step 5] Blocker analysis")

    blocker_analysis = analyze_blockers(y_test, test_pred, test_base["speaker_id"].values, test_base)
    base_blockers = analyze_blockers(y_test, base_test_pred, test_base["speaker_id"].values, test_base)

    print(f"\n  Blocker comparison:")
    print(f"  {'':<30} {'Before (avg)':>15} {'After':>15}")
    print(f"  {'-'*45}")
    print(f"  {'Total MAE':<30} {base_test_metrics['mae']:>10.3f}cm {test_metrics['mae']:>10.3f}cm")
    print(f"  {'Short MAE':<30} {base_test_metrics.get('short_mae', 0):>10.3f}cm {test_metrics.get('short_mae', 0):>10.3f}cm")
    print(f"  {'Worst speakers to fix':<30} {base_blockers['worst_speakers_if_perfect']:>10d} {blocker_analysis['worst_speakers_if_perfect']:>10d}")
    print(f"  {'MAE if short perfect':<30} {base_blockers['mae_if_short_perfect']:>10.3f}cm {blocker_analysis['mae_if_short_perfect']:>10.3f}cm")

    # Print top blockers
    print(f"\n  Top 10 blockers (correction model):")
    for i, b in enumerate(blocker_analysis["top_blockers"][:10]):
        print(f"    {i+1:2d}. {b['speaker_id']:<20s} true={b['height_cm']:6.1f}cm pred={b['pred_cm']:6.1f}cm err={b['abs_error_cm']:5.1f}cm")

    # ─── Step 6: Write output ──────────────────────────────────────────
    print(f"\n[Step 6] Writing output CSVs")

    # Test predictions
    test_out = test_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    test_out["phase24_pred_cm"] = test_pred
    test_out["phase24_abs_error_cm"] = np.abs(test_pred - y_test)
    test_out_path = output_dir / "predictions_test.csv"
    test_out.to_csv(test_out_path, index=False)
    print(f"  Wrote {test_out_path}")

    # Val predictions
    val_out = val_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    val_out["phase24_pred_cm"] = val_pred
    val_out["phase24_abs_error_cm"] = np.abs(val_pred - y_val)
    val_out_path = output_dir / "predictions_val.csv"
    val_out.to_csv(val_out_path, index=False)
    print(f"  Wrote {val_out_path}")

    # ─── Step 7: Feature importance (coefficient analysis) ─────────────
    print(f"\n[Step 7] Top predictive features (largest |coefficient|)")

    coefs = model.coef_.astype(np.float32)
    imp_idx = np.argsort(np.abs(coefs))[::-1][:20]
    print(f"  {'Feature':<55} {'Coefficient':>12}")
    print(f"  {'-'*67}")
    for idx in imp_idx:
        name = all_feature_names[idx] if idx < len(all_feature_names) else f"feat_{idx}"
        print(f"  {name:<55} {coefs[idx]:>+10.4f}")

    # ─── Step 8: Save report ───────────────────────────────────────────
    print(f"\n[Step 8] Writing report")

    report = {
        "method": "ridge_correction",
        "alpha": best_alpha,
        "short_weight": args.short_weight,
        "n_features": X_val.shape[1],
        "n_prediction_sources": len(feature_cols),
        "n_short_val": n_short_val,
        "val": val_metrics,
        "test": test_metrics,
        "baseline_avg_val": base_val_metrics,
        "baseline_avg_test": base_test_metrics,
        "best_source": {
            "name": best_source_name,
            "val_mae": float(np.min(np.mean(source_errors, axis=0))),
            "test_mae": best_source_test_metrics["mae"],
        },
        "blocker_analysis": blocker_analysis,
        "baseline_blockers": base_blockers,
        "top_features": [
            {
                "name": all_feature_names[idx] if idx < len(all_feature_names) else f"feat_{idx}",
                "coefficient": float(coefs[idx]),
            }
            for idx in imp_idx[:15]
        ],
        "reached_3cm": test_metrics["mae"] <= 3.0,
        "reached_3cm_if_short_fixed": blocker_analysis["mae_if_short_perfect"] <= 3.0,
    }

    report_path = output_dir / "phase24_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote {report_path}")

    # ─── Final Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 24 SUMMARY")
    print(f"{'='*60}")
    print(f"  Target: 3.0cm MAE")
    print(f"  ------Metrics------")
    print(f"  Correction model test MAE:  {test_metrics['mae']:.3f}cm")
    print(f"  Simple average test MAE:    {base_test_metrics['mae']:.3f}cm")
    print(f"  Best source test MAE:       {best_source_test_metrics['mae']:.3f}cm ({best_source_name[:60]})")
    if phase12_test_metrics.get("mae") and not np.isnan(phase12_test_metrics["mae"]):
        print(f"  Phase12 test MAE:            {phase12_test_metrics['mae']:.3f}cm")
    print(f"  -------------------")
    print(f"  Short MAE improvement:       {base_test_metrics.get('short_mae', 0):.3f}cm → {test_metrics.get('short_mae', 0):.3f}cm")
    print(f"  Speakers needing fix:        {base_blockers['worst_speakers_if_perfect']} → {blocker_analysis['worst_speakers_if_perfect']}")
    print(f"  MAE if short perfect:        {blocker_analysis['mae_if_short_perfect']:.3f}cm")
    print(f"  3cm reached?                 {'YES 🎯' if test_metrics['mae'] <= 3.0 else 'NO'}")
    print(f"  3cm reachable (fix shorts)?  {'YES 🎯' if blocker_analysis['mae_if_short_perfect'] <= 3.0 else 'NO'}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
