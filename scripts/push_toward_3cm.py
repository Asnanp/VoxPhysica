#!/usr/bin/env python
"""
3cm Push: Per-Speaker Hybrid Ensemble.

Combines:
1. phase12_residual_guard (best single source: 4.95cm test MAE)
2. V5 clip-level variance-weighted predictions (using height_var_norm for confidence)
3. Low-dimensional feature-based KNN for per-speaker blend weights
4. Additional V5 omega predictions as fallback

Outputs a prediction CSV for the phase22 gauntlet.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_prediction(base: pd.DataFrame, csv_path: Path, col: str) -> np.ndarray:
    """Load and align a prediction column with the base speaker order."""
    df = pd.read_csv(csv_path)
    if "speaker_id" not in df.columns or col not in df.columns:
        raise ValueError(f"Missing speaker_id or {col} in {csv_path}")
    merged = base[["speaker_id"]].merge(df[["speaker_id", col]], on="speaker_id", how="left")
    if merged[col].isna().any():
        raise ValueError(f"Missing predictions for some speakers in {csv_path}:{col}")
    return merged[col].to_numpy(dtype=np.float32)


def compute_v5_variance_weighted(
    clip_csv: Path,
    base: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute V5 variance-weighted speaker predictions.
    
    Uses clip-level height_pred_cm and height_var_norm to weight
    each clip's prediction by inverse variance per speaker.
    
    Returns:
        variance_weighted_pred: per-speaker predictions (inverse-variance weighted)
        raw_avg_pred: per-speaker predictions (simple average)
    """
    df = pd.read_csv(clip_csv)
    
    if "speaker_id" not in df.columns or "height_pred_cm" not in df.columns:
        raise ValueError(f"Missing required columns in {clip_csv}")
    if "height_var_norm" not in df.columns:
        print(f"[WARNING] height_var_norm not found, using uniform weights")
        df["height_var_norm"] = 1.0
    
    # Per speaker: weighted average by inverse variance
    speaker_weighted = {}
    speaker_raw = {}
    
    for sid, group in df.groupby("speaker_id"):
        preds = group["height_pred_cm"].to_numpy(dtype=np.float32)
        vars_ = group["height_var_norm"].to_numpy(dtype=np.float32)
        
        # Clip variance: clamp to avoid division by zero
        vars_clamped = np.clip(vars_, 1e-6, None)
        weights = 1.0 / vars_clamped
        
        weighted_pred = float(np.average(preds, weights=weights))
        raw_pred = float(np.mean(preds))
        
        speaker_weighted[sid] = weighted_pred
        speaker_raw[sid] = raw_pred
    
    # Align with base
    weighted_out = np.array([speaker_weighted.get(sid, 160.0) for sid in base["speaker_id"]], dtype=np.float32)
    raw_out = np.array([speaker_raw.get(sid, 160.0) for sid in base["speaker_id"]], dtype=np.float32)
    
    return weighted_out, raw_out


def compute_v5_omega_weighted(
    speaker_csv: Path,
    base: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Load V5 omega predictions (speaker-level). Returns None if unavailable."""
    if not os.path.exists(speaker_csv):
        print(f"  [WARN] Speaker CSV not found: {speaker_csv}")
        return None
    df = pd.read_csv(speaker_csv)
    if "speaker_id" not in df.columns:
        print(f"  [WARN] Missing speaker_id in {speaker_csv}")
        return None
    
    omega_col = None
    for c in df.columns:
        if "omega" in c.lower() and "_cm" in c.lower():
            omega_col = c
            break
    
    if omega_col is None:
        print(f"  [WARN] No omega prediction column in {speaker_csv}, skipping")
        return None
    
    merged = base[["speaker_id"]].merge(df[["speaker_id", omega_col]], on="speaker_id", how="left")
    if merged[omega_col].isna().any():
        print(f"  [WARN] Missing omega predictions for some speakers, skipping")
        return None
    
    omega_pred = merged[omega_col].to_numpy(dtype=np.float32)
    return omega_pred


def extract_low_dim_features(
    features_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    """
    Extract ONLY the essential acoustic features per speaker (not the full 136-dim sequence).
    
    Returns DataFrame with ~12 features per speaker.
    """
    split_dir = features_dir / split_name
    npz_files = sorted(split_dir.glob("*.npz"))
    
    speaker_data: Dict[str, Dict[str, List]] = {}
    
    for npz_path in npz_files:
        try:
            data = dict(np.load(npz_path, allow_pickle=True))
        except Exception:
            continue
        
        sid = str(data.get("speaker_id", b"unknown"))
        if isinstance(sid, bytes):
            sid = sid.decode("utf-8")
        
        if sid not in speaker_data:
            speaker_data[sid] = {
                "f0": [], "vtl": [], "formant_spacing": [],
                "jitter": [], "shimmer": [], "hnr": [],
                "duration": [], "voiced_ratio": [], "speech_ratio": [],
                "snr": [], "quality_score": [], "clipped_ratio": [],
                "n_clips": 0,
            }
        
        sp = speaker_data[sid]
        for key, lst_key in [
            ("f0_mean", "f0"), ("vtl_mean", "vtl"),
            ("formant_spacing_mean", "formant_spacing"),
            ("jitter", "jitter"), ("shimmer", "shimmer"), ("hnr", "hnr"),
            ("duration_s", "duration"), ("voiced_ratio", "voiced_ratio"),
            ("speech_ratio", "speech_ratio"), ("snr_db_estimate", "snr"),
            ("capture_quality_score", "quality_score"),
            ("clipped_ratio", "clipped_ratio"),
        ]:
            val = data.get(key)
            if isinstance(val, np.ndarray) and val.ndim == 0:
                sp[lst_key].append(float(val))
        
        sp["n_clips"] += 1
    
    rows = []
    for sid, sp in sorted(speaker_data.items()):
        row = {"speaker_id": sid, "n_clips": sp["n_clips"]}
        for key, lst_key in [
            ("f0_mean", "f0"), ("vtl_mean", "vtl"),
            ("formant_spacing_mean", "formant_spacing"),
            ("jitter", "jitter"), ("shimmer", "shimmer"), ("hnr", "hnr"),
            ("duration_s", "duration"), ("voiced_ratio", "voiced_ratio"),
            ("speech_ratio", "speech_ratio"), ("snr_db", "snr"),
            ("quality_score", "quality_score"), ("clipped_ratio", "clipped_ratio"),
        ]:
            vals = sp[lst_key]
            row[key] = float(np.mean(vals)) if vals else 0.0
        
        # Add gender from sample data
        rows.append(row)
    
    result = pd.DataFrame(rows).fillna(0.0)
    
    # Add source and gender
    sid_to_source: Dict[str, str] = {}
    sid_to_gender: Dict[str, float] = {}
    
    for npz_path in npz_files:
        try:
            data = dict(np.load(npz_path, allow_pickle=True))
            sid = str(data.get("speaker_id", b"unknown"))
            if isinstance(sid, bytes):
                sid = sid.decode("utf-8")
            if sid not in sid_to_source:
                src = data.get("source")
                if isinstance(src, (bytes, str)):
                    sid_to_source[sid] = src.decode("utf-8") if isinstance(src, bytes) else str(src)
                g = data.get("gender")
                if isinstance(g, np.ndarray) and g.ndim == 0:
                    sid_to_gender[sid] = float(g)
        except Exception:
            continue
    
    result["source_encoded"] = result["speaker_id"].map(
        lambda sid: 0 if sid_to_source.get(sid, "").upper() == "NISP" else 1
    ).fillna(0)
    result["gender"] = result["speaker_id"].map(sid_to_gender).fillna(0.0)
    
    return result.set_index("speaker_id")


def per_speaker_blend(
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
    val_preds: Dict[str, np.ndarray],
    test_preds: Dict[str, np.ndarray],
    y_val: np.ndarray,
    y_test: np.ndarray,
    k: int = 7,
    blend_alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Per-speaker KNN blend of prediction sources.
    
    For each test speaker:
    1. Find K similar validation speakers (by acoustic features)
    2. For those neighbors, compute which blend weight between the two sources
       would have been optimal
    3. Apply the average optimal weight to the test speaker
    
    Uses only the phase12_pred and V5_variance_weighted predictions.
    """
    # Build feature matrix (low-dim)
    feature_cols = [c for c in val_features.columns if c not in ("n_clips",)]
    val_feat = val_features[feature_cols].to_numpy(dtype=np.float32)
    test_feat = test_features[feature_cols].to_numpy(dtype=np.float32)
    
    # Normalize
    feat_mean = np.nanmean(val_feat, axis=0)
    feat_std = np.nanstd(val_feat, axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    val_feat = (val_feat - feat_mean) / feat_std
    test_feat = (test_feat - feat_mean) / feat_std
    
    source_names = list(test_preds.keys())
    n_val = len(y_val)
    n_test = len(y_test)
    
    # For each validation speaker, compute optimal blend weight between primary and secondary
    primary_name = "phase12_pred_cm"
    secondary_name = "v5_var_weighted_cm"
    
    if primary_name not in val_preds or secondary_name not in val_preds:
        # Fallback: use all available sources
        primary_name = source_names[0]
        secondary_name = source_names[1] if len(source_names) > 1 else source_names[0]
    
    p_val = val_preds[primary_name]
    s_val = val_preds[secondary_name]
    p_test = test_preds[primary_name]
    s_test = test_preds[secondary_name]
    
    # For each validation speaker, find optimal blend weight alpha (0..1)
    # where prediction = alpha * primary + (1-alpha) * secondary
    val_optimal_alpha = np.zeros(n_val, dtype=np.float32)
    for i in range(n_val):
        err_primary = abs(float(p_val[i] - y_val[i]))
        err_secondary = abs(float(s_val[i] - y_val[i]))
        
        if err_primary + err_secondary < 1e-6:
            val_optimal_alpha[i] = 0.5
        elif err_primary < err_secondary:
            val_optimal_alpha[i] = 1.0  # Primary was better
        else:
            val_optimal_alpha[i] = 0.0  # Secondary was better
    
    # For each test speaker, KNN to get blend weight
    test_preds_out = np.zeros(n_test, dtype=np.float32)
    val_preds_loocv = np.zeros(n_val, dtype=np.float32)
    test_alphas = np.zeros(n_test, dtype=np.float32)
    
    for i in range(n_test):
        dists = np.sqrt(np.sum((val_feat - test_feat[i:i+1]) ** 2, axis=1))
        nearest = np.argsort(dists)[:k]
        neighbor_alphas = val_optimal_alpha[nearest]
        
        # Weight by inverse distance
        weights = 1.0 / (dists[nearest] + 1e-6)
        weights = weights / weights.sum()
        
        alpha = float(np.average(neighbor_alphas, weights=weights))
        alpha = np.clip(alpha, 0.0, 1.0)
        test_alphas[i] = alpha
        test_preds_out[i] = alpha * p_test[i] + (1.0 - alpha) * s_test[i]
    
    # LOOCV for validation
    for i in range(n_val):
        train_mask = np.ones(n_val, dtype=bool)
        train_mask[i] = False
        dists = np.sqrt(np.sum((val_feat[train_mask] - val_feat[i:i+1]) ** 2, axis=1))
        nearest_val = np.argsort(dists)[:k]
        val_indices = np.where(train_mask)[0]
        neighbor_indices = val_indices[nearest_val]
        neighbor_alphas = val_optimal_alpha[neighbor_indices]
        
        alpha = float(np.mean(neighbor_alphas))
        alpha = np.clip(alpha, 0.0, 1.0)
        val_preds_loocv[i] = alpha * p_val[i] + (1.0 - alpha) * s_val[i]
    
    val_mae = float(np.mean(np.abs(val_preds_loocv - y_val)))
    test_mae = float(np.mean(np.abs(test_preds_out - y_test)))
    
    # Static blends for comparison
    blend_05 = 0.5 * p_test + 0.5 * s_test
    blend_05_mae = float(np.mean(np.abs(blend_05 - y_test)))
    primary_only_mae = float(np.mean(np.abs(p_test - y_test)))
    secondary_only_mae = float(np.mean(np.abs(s_test - y_test)))
    
    info = {
        "primary_source": primary_name,
        "secondary_source": secondary_name,
        "k": k,
        "val_loocv_mae": round(val_mae, 4),
        "test_mae": round(test_mae, 4),
        "primary_only_test_mae": round(primary_only_mae, 4),
        "secondary_only_test_mae": round(secondary_only_mae, 4),
        "static_50_50_test_mae": round(blend_05_mae, 4),
        "mean_alpha": round(float(np.mean(test_alphas)), 4),
        "oracle_val_mae": round(float(np.mean(np.minimum(
            np.abs(p_val - y_val), np.abs(s_val - y_val)
        ))), 4),
    }
    
    print(f"[Blend] K={k}: val_loocv={val_mae:.4f} test={test_mae:.4f}")
    print(f"[Blend] Primary only: {primary_only_mae:.4f} | Secondary only: {secondary_only_mae:.4f}")
    print(f"[Blend] Static 50/50: {blend_05_mae:.4f} | Mean alpha: {float(np.mean(test_alphas)):.4f}")
    print(f"[Blend] Oracle (val min of two): {info['oracle_val_mae']:.4f}")
    
    return test_preds_out, val_preds_loocv, info


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="3cm Push: Hybrid Per-Speaker Ensemble")
    parser.add_argument("--output-dir", default="outputs/push_toward_3cm")
    parser.add_argument("--phase3-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--features-dir", default="data/features_v4")
    parser.add_argument("--k-neighbors", type=int, default=7)
    parser.add_argument("--v5-clip-test", default="outputs/v5_3cm_architecture_predictions/test_clip_predictions.csv")
    parser.add_argument("--v5-clip-val", default="outputs/v5_3cm_architecture_predictions/val_clip_predictions.csv")
    parser.add_argument("--v5-speaker-test", default="outputs/v5_3cm_architecture_predictions/test_speaker_predictions.csv")
    parser.add_argument("--v5-speaker-val", default="outputs/v5_3cm_architecture_predictions/val_speaker_predictions.csv")
    parser.add_argument("--phase12-test", default="outputs/phase12_residual_guard/phase12_predictions_test.csv")
    parser.add_argument("--phase12-val", default="outputs/phase12_residual_guard/phase12_predictions_val.csv")
    args = parser.parse_args()
    
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base data
    val_base = pd.read_csv(resolve(args.phase3_val))
    test_base = pd.read_csv(resolve(args.phase3_test))
    y_val = val_base["height_cm"].to_numpy(dtype=np.float32)
    y_test = test_base["height_cm"].to_numpy(dtype=np.float32)
    print(f"[3cmPush] Val: {len(val_base)} spk, Test: {len(test_base)} spk")
    
    # ─── Step 1: Load Phase12 predictions ────────────────────────────────
    print("\n=== Step 1: Phase12 Predictions ===")
    phase12_test_path = resolve(args.phase12_test)
    phase12_val_path = resolve(args.phase12_val)
    
    phase12_test_col = None
    phase12_val_col = None
    
    test_df_12 = pd.read_csv(phase12_test_path)
    for c in test_df_12.columns:
        if "pred" in c.lower() and "_cm" in c.lower():
            phase12_test_col = c
            break
    
    if os.path.exists(phase12_val_path):
        val_df_12 = pd.read_csv(phase12_val_path)
        for c in val_df_12.columns:
            if "pred" in c.lower() and "_cm" in c.lower():
                phase12_val_col = c
                break
    
    if phase12_test_col is None:
        print("[ERROR] No prediction column found in phase12 test CSV")
        return 1
    
    # Align
    merged_test_12 = test_base[["speaker_id"]].merge(
        test_df_12[["speaker_id", phase12_test_col]], on="speaker_id", how="left"
    )
    phase12_test_pred = merged_test_12[phase12_test_col].to_numpy(dtype=np.float32)
    
    phase12_val_pred = None
    if phase12_val_col is not None:
        merged_val_12 = val_base[["speaker_id"]].merge(
            val_df_12[["speaker_id", phase12_val_col]], on="speaker_id", how="left"
        )
        phase12_val_pred = merged_val_12[phase12_val_col].to_numpy(dtype=np.float32)
    
    phase12_test_mae = float(np.mean(np.abs(phase12_test_pred - y_test)))
    print(f"  Phase12 test MAE: {phase12_test_mae:.3f}cm")
    
    # ─── Step 2: V5 Variance-Weighted Predictions ────────────────────────
    print("\n=== Step 2: V5 Variance-Weighted Clip Predictions ===")
    
    v5_clip_test = resolve(args.v5_clip_test)
    v5_clip_val = resolve(args.v5_clip_val)
    
    # Test
    v5_var_test, v5_raw_test = compute_v5_variance_weighted(v5_clip_test, test_base)
    v5_var_test_mae = float(np.mean(np.abs(v5_var_test - y_test)))
    v5_raw_test_mae = float(np.mean(np.abs(v5_raw_test - y_test)))
    print(f"  V5 simple avg test MAE: {v5_raw_test_mae:.3f}cm")
    print(f"  V5 variance-weighted test MAE: {v5_var_test_mae:.3f}cm")
    
    # Val
    v5_var_val, v5_raw_val = compute_v5_variance_weighted(v5_clip_val, val_base)
    v5_var_val_mae = float(np.mean(np.abs(v5_var_val - y_val)))
    print(f"  V5 simple avg val MAE: {float(np.mean(np.abs(v5_raw_val - y_val))):.3f}cm")
    print(f"  V5 variance-weighted val MAE: {v5_var_val_mae:.3f}cm")
    
    # V5 omega (speaker-level alternative) — optional
    v5_speaker_test = resolve(args.v5_speaker_test)
    v5_omega_test = compute_v5_omega_weighted(v5_speaker_test, test_base)
    v5_omega_test_mae = None
    if v5_omega_test is not None:
        v5_omega_test_mae = float(np.mean(np.abs(v5_omega_test - y_test)))
        print(f"  V5 omega test MAE: {v5_omega_test_mae:.3f}cm")
    
    # ─── Step 3: Extract low-dim features ────────────────────────────────
    print("\n=== Step 3: Low-Dim Speaker Features ===")
    features_dir = resolve(args.features_dir)
    val_features = extract_low_dim_features(features_dir, "val")
    test_features = extract_low_dim_features(features_dir, "test")
    print(f"  Val features: {val_features.shape}")
    print(f"  Test features: {test_features.shape}")
    print(f"  Feature cols: {[c for c in val_features.columns if c not in ('n_clips',)][:5]}...")
    
    # ─── Step 4: Per-Speaker Blending ────────────────────────────────────
    print("\n=== Step 4: Per-Speaker KNN Blend ===")
    
    # Align features with base speaker order
    val_feat_aligned = val_features.reindex(val_base["speaker_id"].values).fillna(0.0)
    test_feat_aligned = test_features.reindex(test_base["speaker_id"].values).fillna(0.0)
    
    # Build prediction dicts
    val_pred_dict = {
        "phase12_pred_cm": phase12_val_pred,
        "v5_var_weighted_cm": v5_var_val,
        "v5_raw_avg_cm": v5_raw_val,
    }
    test_pred_dict = {
        "phase12_pred_cm": phase12_test_pred,
        "v5_var_weighted_cm": v5_var_test,
        "v5_raw_avg_cm": v5_raw_test,
    }
    if v5_omega_test is not None:
        test_pred_dict["v5_omega_cm"] = v5_omega_test
    
    # Blend phase12 + V5 variance-weighted
    for k in [3, 5, 7, 11, 15]:
        test_pred, val_pred, info = per_speaker_blend(
            val_feat_aligned, test_feat_aligned,
            val_pred_dict, test_pred_dict,
            y_val, y_test,
            k=k,
        )
        print()
    
    # Default K=7
    test_pred, val_pred, info = per_speaker_blend(
        val_feat_aligned, test_feat_aligned,
        val_pred_dict, test_pred_dict,
        y_val, y_test,
        k=7,
    )
    
    # ─── Step 5: Write output ────────────────────────────────────────────
    print("\n=== Step 5: Writing Predictions ===")
    
    # Per-speaker blend prediction
    result_df = test_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    result_df["knn_blend_pred_cm"] = test_pred
    result_df["knn_blend_abs_error"] = np.abs(test_pred - y_test)
    test_blend_path = output_dir / "knn_blend_predictions_test.csv"
    result_df.to_csv(test_blend_path, index=False)
    print(f"  Wrote {test_blend_path}")
    
    # Also write val predictions
    val_result = val_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    val_result["knn_blend_pred_cm"] = val_pred
    val_result["knn_blend_abs_error"] = np.abs(val_pred - y_val)
    val_blend_path = output_dir / "knn_blend_predictions_val.csv"
    val_result.to_csv(val_blend_path, index=False)
    print(f"  Wrote {val_blend_path}")
    
    # Also write V5 variance-weighted as a standalone prediction
    v5_test = test_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    v5_test["v5_var_weighted_cm"] = v5_var_test
    v5_test["v5_var_abs_error"] = np.abs(v5_var_test - y_test)
    v5_test_path = output_dir / "v5_var_weighted_predictions_test.csv"
    v5_test.to_csv(v5_test_path, index=False)
    print(f"  Wrote {v5_test_path}")
    
    v5_val = val_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    v5_val["v5_var_weighted_cm"] = v5_var_val
    v5_val["v5_var_abs_error"] = np.abs(v5_var_val - y_val)
    v5_val_path = output_dir / "v5_var_weighted_predictions_val.csv"
    v5_val.to_csv(v5_val_path, index=False)
    print(f"  Wrote {v5_val_path}")
    
    # Write phase12 as validation-paired prediction
    p12_test = test_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    p12_test["phase12_pred_cm"] = phase12_test_pred
    p12_test["phase12_abs_error"] = np.abs(phase12_test_pred - y_test)
    p12_test_path = output_dir / "phase12_pred_predictions_test.csv"
    p12_test.to_csv(p12_test_path, index=False)
    print(f"  Wrote {p12_test_path}")
    
    if phase12_val_pred is not None:
        p12_val = val_base[["speaker_id", "source", "gender", "height_cm"]].copy()
        p12_val["phase12_pred_cm"] = phase12_val_pred
        p12_val["phase12_abs_error"] = np.abs(phase12_val_pred - y_val)
        p12_val_path = output_dir / "phase12_pred_predictions_val.csv"
        p12_val.to_csv(p12_val_path, index=False)
        print(f"  Wrote {p12_val_path}")
    
    # Write report
    report = {
        "phase12_test_mae": round(phase12_test_mae, 4),
        "v5_var_test_mae": round(v5_var_test_mae, 4),
        "v5_raw_test_mae": round(v5_raw_test_mae, 4),
        "knn_blend_test_mae": round(info["test_mae"], 4),
        "knn_blend_val_loocv_mae": round(info["val_loocv_mae"], 4),
        "knn_k": info["k"],
        "mean_alpha": info["mean_alpha"],
        "oracle_val_min_of_two": info["oracle_val_mae"],
        "description": "KNN per-speaker blend of Phase12 + V5 variance-weighted",
    }
    
    report_path = output_dir / "push_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote {report_path}")
    
    print(f"\n{'='*60}")
    print(f"  3CM PUSH RESULTS")
    print(f"  Phase12:           {phase12_test_mae:.3f}cm")
    print(f"  V5 raw avg:        {v5_raw_test_mae:.3f}cm")
    print(f"  V5 variance-wtd:   {v5_var_test_mae:.3f}cm")
    if v5_omega_test_mae is not None:
        print(f"  V5 omega:          {v5_omega_test_mae:.3f}cm")
    print(f"  KNN blend test:    {info['test_mae']:.3f}cm")
    print(f"  Oracle (val):      {info['oracle_val_mae']:.3f}cm")
    print(f"  TARGET:            3.000cm")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
