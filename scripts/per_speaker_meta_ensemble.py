#!/usr/bin/env python
"""
Per-Speaker Meta-Ensemble for 3cm MAE Target.

Uses per-speaker acoustic features (extracted from NPZ files) to learn
WHICH prediction source to trust for each speaker, rather than a global
ensemble weight. This bridges the gap between the ~5.85cm global ensemble
and the ~2.07cm oracle.

Approach:
1. Load all prediction candidates (test and validation paired)
2. Extract per-speaker features from NPZ files (pooled sequence features + metadata)
3. For VALIDATION speakers: determine optimal source(s) per speaker (using ground truth)
4. Train a meta-model: speaker_features → optimal prediction per speaker
   (KNN-based: find similar validation speakers, use what worked best for them)
5. Apply to test set: for each test speaker, pick/integrate the best prediction sources
6. Output a new prediction CSV for the phase22 gauntlet
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

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-speaker meta-ensemble selector")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--features-dir", default="data/features_v4")
    parser.add_argument("--output-dir", default="outputs/per_speaker_meta_ensemble")
    parser.add_argument("--phase3-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--k-neighbors", type=int, default=7, help="K for KNN meta-model")
    parser.add_argument("--top-candidates", type=int, default=30, help="Number of top validation-paired candidates to use")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ─── Prediction Candidate Loading ──────────────────────────────────────────



def _collect_candidates_inline(
    outputs_root: Path,
    val_base: pd.DataFrame,
    test_base: pd.DataFrame,
    skip_dir_name: str = "per_speaker_meta_ensemble",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, str]]]:
    """Collect all prediction candidates."""
    test_sources: Dict[str, np.ndarray] = {}
    val_sources: Dict[str, Optional[np.ndarray]] = {}
    source_meta: Dict[str, Dict[str, str]] = {}
    seen_signatures: set[bytes] = set()
    walk_skip = {".git", "__pycache__", "pytest-temp", skip_dir_name.lower()}
    
    for dirpath_str, dirnames, filenames in os.walk(str(outputs_root)):
        dirpath = Path(dirpath_str)
        dirnames[:] = [d for d in dirnames if d.lower() not in walk_skip]
        
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue
            if "prediction" not in filename.lower() or "test" not in filename.lower():
                continue
            if "oracle" in filename.lower():
                continue
            
            test_path = dirpath / filename
            try:
                test_df = pd.read_csv(test_path)
            except Exception:
                continue
            
            if "speaker_id" not in test_df.columns or "height_cm" not in test_df.columns:
                continue
            
            # Find val path
            val_df: Optional[pd.DataFrame] = None
            for val_path in _candidate_val_paths(test_path):
                if val_path.exists():
                    try:
                        val_df = pd.read_csv(val_path)
                        break
                    except Exception:
                        val_df = None
            
            for col in test_df.columns:
                if not _is_prediction_column(test_df, col):
                    continue
                
                test_pred = _align_prediction(test_base, test_df, col)
                if test_pred is None:
                    continue
                
                # Dedup by prediction values
                sig = np.round(test_pred, 4).tobytes()
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                
                rel = test_path.relative_to(outputs_root).as_posix()
                name = f"{rel}:{col}"
                
                test_sources[name] = test_pred
                source_meta[name] = {"path": rel, "column": col}
                
                if val_df is not None and col in val_df.columns:
                    val_pred = _align_prediction(val_base, val_df, col)
                    val_sources[name] = val_pred
                else:
                    val_sources[name] = None
    
    # Build DataFrames
    test_df = test_base[["speaker_id", "height_cm"]].copy()
    val_df = val_base[["speaker_id", "height_cm"]].copy()
    
    for name in sorted(test_sources.keys()):
        test_df[name] = test_sources[name]
        if val_sources.get(name) is not None:
            val_df[name] = val_sources[name]
    
    return val_df, test_df, source_meta


def _is_prediction_column(df: pd.DataFrame, col: str) -> bool:
    lower = col.lower()
    if col in {"height_cm", "gender", "age", "speaker_id", "source", "source_id"}:
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    if any(token in lower for token in ("error", "mae", "rmse", "count", "probability", "std", "uncert", "abs_")):
        return False
    if not (lower.endswith("_cm") or "pred" in lower):
        return False
    return True


def _align_prediction(base: pd.DataFrame, pred_df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
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


def _candidate_val_paths(test_path: Path) -> List[Path]:
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


# ─── Speaker Feature Extraction from NPZ Files ─────────────────────────────


def extract_speaker_features(
    features_dir: Path,
    split_name: str,  # "train", "val", or "test"
) -> pd.DataFrame:
    """
    Extract per-speaker features from NPZ files.
    
    Returns a DataFrame with:
    - speaker_id (index)
    - sequence_mean_{0..135}: mean of the 136-dim sequence features across time & clips
    - f0_mean, vtl_mean, formant_spacing_mean, jitter, shimmer, hnr
    - duration_s, voiced_ratio, speech_ratio, snr_db_estimate
    - capture_quality_score, clipped_ratio
    - gender, source (encoded)
    - n_clips: number of clips for this speaker
    """
    split_dir = features_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {split_dir}")
    
    npz_files = sorted(split_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {split_dir}")
    
    print(f"[MetaSelector] Loading {len(npz_files)} NPZ files from {split_dir}")
    
    # Group by speaker
    speaker_data: Dict[str, Dict[str, List]] = {}
    
    for npz_path in npz_files:
        try:
            data = dict(np.load(npz_path, allow_pickle=True))
        except Exception as e:
            print(f"[MetaSelector] Warning: could not load {npz_path}: {e}")
            continue
        
        sid = str(data.get("speaker_id", b"unknown"))
        if isinstance(sid, bytes):
            sid = sid.decode("utf-8")
        
        if sid not in speaker_data:
            speaker_data[sid] = {
                "sequences": [],
                "f0": [],
                "vtl": [],
                "formant_spacing": [],
                "jitter": [],
                "shimmer": [],
                "hnr": [],
                "duration": [],
                "voiced_ratio": [],
                "speech_ratio": [],
                "snr": [],
                "quality_score": [],
                "clipped_ratio": [],
                "n_clips": 0,
            }
        
        sp = speaker_data[sid]
        
        seq = data.get("sequence")
        if isinstance(seq, np.ndarray) and seq.ndim == 2 and seq.shape[1] >= 100:
            sp["sequences"].append(seq.mean(axis=0))  # Time-pool to 136-dim vector
        
        for scalar_key, list_key in [
            ("f0_mean", "f0"),
            ("vtl_mean", "vtl"),
            ("formant_spacing_mean", "formant_spacing"),
            ("jitter", "jitter"),
            ("shimmer", "shimmer"),
            ("hnr", "hnr"),
            ("duration_s", "duration"),
            ("voiced_ratio", "voiced_ratio"),
            ("speech_ratio", "speech_ratio"),
            ("snr_db_estimate", "snr"),
            ("capture_quality_score", "quality_score"),
            ("clipped_ratio", "clipped_ratio"),
        ]:
            val = data.get(scalar_key)
            if isinstance(val, np.ndarray) and val.ndim == 0:
                sp[list_key].append(float(val))
        
        sp["n_clips"] += 1
    
    # Build per-speaker feature vectors
    rows = []
    for sid, sp in sorted(speaker_data.items()):
        row = {"speaker_id": sid, "n_clips": sp["n_clips"]}
        
        # Mean of sequence features
        if sp["sequences"]:
            pooled = np.mean(np.stack(sp["sequences"]), axis=0)
            for i, val in enumerate(pooled):
                row[f"seq_feat_{i}"] = float(val)
        else:
            for i in range(136):
                row[f"seq_feat_{i}"] = 0.0
        
        # Mean of scalar features
        for list_key, feat_name in [
            ("f0", "f0_mean"),
            ("vtl", "vtl_mean"),
            ("formant_spacing", "formant_spacing_mean"),
            ("jitter", "jitter"),
            ("shimmer", "shimmer"),
            ("hnr", "hnr"),
            ("duration", "duration_s"),
            ("voiced_ratio", "voiced_ratio"),
            ("speech_ratio", "speech_ratio"),
            ("snr", "snr_db"),
            ("quality_score", "quality_score"),
            ("clipped_ratio", "clipped_ratio"),
        ]:
            vals = sp[list_key]
            row[feat_name] = float(np.mean(vals)) if vals else 0.0
        
        # Standard deviation for key acoustic features (variance signal)
        for list_key, std_name in [
            ("f0", "f0_std"),
            ("vtl", "vtl_std"),
            ("jitter", "jitter_std"),
            ("shimmer", "shimmer_std"),
            ("hnr", "hnr_std"),
        ]:
            vals = sp[list_key]
            row[std_name] = float(np.std(vals)) if len(vals) > 1 else 0.0
        
        rows.append(row)
    
    result = pd.DataFrame(rows).fillna(0.0)
    
    # Add source and gender from a sample file
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


# ─── Per-Speaker Feature-Weighted Ensemble ─────────────────────────────────


def per_speaker_feature_weighted_ensemble(
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    val_features_df: pd.DataFrame,
    test_features_df: pd.DataFrame,
    k: int = 7,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Feature-weighted per-speaker ensemble.
    
    For each test speaker, finds K nearest validation speakers (by features),
    then uses a weighted combination of predictions that performed best for those neighbors.
    
    Returns:
        test_preds: per-speaker predictions for test set
        val_preds: per-speaker predictions for validation set (LOOCV)
        info: diagnostic information
    """
    y_val = val_predictions["height_cm"].to_numpy(dtype=np.float32)
    val_ids = val_predictions["speaker_id"].values
    test_ids = test_predictions["speaker_id"].values
    
    # Source columns
    skip_cols = {"speaker_id", "height_cm", "source", "gender", "source_id"}
    source_cols = [c for c in val_predictions.columns if c not in skip_cols]
    
    print(f"[FeatureWeighted] Using {len(source_cols)} prediction sources, K={k}")
    
    # Build feature matrix
    feature_cols = [c for c in val_features_df.columns if c != "n_clips"]
    val_feat_full = val_features_df[feature_cols].to_numpy(dtype=np.float32)
    test_feat_full = test_features_df[feature_cols].to_numpy(dtype=np.float32)
    
    # Normalize
    feat_mean = np.nanmean(val_feat_full, axis=0)
    feat_std = np.nanstd(val_feat_full, axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    val_feat = (val_feat_full - feat_mean) / feat_std
    test_feat = (test_feat_full - feat_mean) / feat_std
    
    # For each validation speaker (LOOCV), compute errors per source
    n_val = len(val_ids)
    n_sources = len(source_cols)
    val_errors = np.zeros((n_val, n_sources), dtype=np.float32)
    
    for j, col in enumerate(source_cols):
        pred = val_predictions[col].to_numpy(dtype=np.float32)
        val_errors[:, j] = np.abs(pred - y_val)
    
    # Best source per validation speaker
    val_best_source = np.argmin(val_errors, axis=1)
    
    # For each test speaker, find neighbors and get their best sources
    test_preds = np.zeros(len(test_ids), dtype=np.float32)
    val_preds_loocv = np.zeros(n_val, dtype=np.float32)
    
    for i in range(len(test_ids)):
        dists = np.sqrt(np.sum((val_feat - test_feat[i:i+1]) ** 2, axis=1))
        nearest = np.argsort(dists)[:k]
        neighbor_sources = val_best_source[nearest]
        
        # Use the most common best source among neighbors
        source_counts = np.bincount(neighbor_sources, minlength=n_sources)
        best_idx = int(np.argmax(source_counts))
        
        test_preds[i] = test_predictions[source_cols[best_idx]].values[i]
    
    # LOOCV for validation
    for i in range(n_val):
        train_mask = np.ones(n_val, dtype=bool)
        train_mask[i] = False
        dists = np.sqrt(np.sum((val_feat[train_mask] - val_feat[i:i+1]) ** 2, axis=1))
        nearest_val = np.argsort(dists)[:k]
        # Map back to original indices
        val_indices = np.where(train_mask)[0]
        neighbor_indices = val_indices[nearest_val]
        neighbor_sources = val_best_source[neighbor_indices]
        
        source_counts = np.bincount(neighbor_sources, minlength=n_sources)
        best_idx = int(np.argmax(source_counts))
        val_preds_loocv[i] = val_predictions[source_cols[best_idx]].values[i]
    
    val_mae = float(np.mean(np.abs(val_preds_loocv - y_val)))
    val_oracle = float(np.mean(val_errors.min(axis=1)))
    
    info = {
        "method": f"feature_weighted_knn_k{k}",
        "val_loocv_mae": val_mae,
        "val_oracle_mae": val_oracle,
        "k": k,
        "n_sources": n_sources,
        "val_improvement_over_oracle": val_mae - val_oracle,
    }
    
    print(f"[FeatureWeighted] LOOCV MAE: {val_mae:.4f}cm (oracle: {val_oracle:.4f}cm)")
    
    return test_preds, val_preds_loocv, info


# ─── Main Pipeline ─────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    
    outputs_root = resolve(args.outputs_root)
    output_dir = resolve(args.output_dir)
    features_dir = resolve(args.features_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base data
    val_base = pd.read_csv(resolve(args.phase3_val))
    test_base = pd.read_csv(resolve(args.phase3_test))
    
    print(f"[MetaSelector] Val speakers: {len(val_base)}, Test speakers: {len(test_base)}")
    
    # Step 1: Extract per-speaker features
    print("\n=== Step 1: Extracting Speaker Features ===")
    val_features = extract_speaker_features(features_dir, "val")
    test_features = extract_speaker_features(features_dir, "test")
    
    print(f"[MetaSelector] Val features: {val_features.shape}, Test features: {test_features.shape}")
    print(f"[MetaSelector] Feature columns: {[c for c in val_features.columns if c != 'n_clips'][:5]}...")
    
    # Step 2: Collect all prediction candidates
    print("\n=== Step 2: Collecting Prediction Candidates ===")
    val_preds, test_preds, source_meta = _collect_candidates_inline(
        outputs_root, val_base, test_base,
        skip_dir_name=output_dir.name.lower(),
    )
    
    print(f"[MetaSelector] Val prediction sources: {len([c for c in val_preds.columns if c not in ('speaker_id','height_cm')])}")
    print(f"[MetaSelector] Test prediction sources: {len([c for c in test_preds.columns if c not in ('speaker_id','height_cm')])}")
    
    # Step 3: Run per-speaker feature-weighted ensemble
    print(f"\n=== Step 3: Per-Speaker Feature-Weighted Ensemble (K={args.k_neighbors}) ===")
    
    # Align features with predictions
    val_feat_aligned = val_features.reindex(val_preds["speaker_id"].values).fillna(0.0)
    test_feat_aligned = test_features.reindex(test_preds["speaker_id"].values).fillna(0.0)
    
    test_preds_out, val_preds_out, info = per_speaker_feature_weighted_ensemble(
        val_preds, test_preds, val_feat_aligned, test_feat_aligned,
        k=args.k_neighbors,
    )
    
    # Step 4: Compare with baseline ensembles
    print("\n=== Step 4: Baseline Comparisons ===")
    y_test = test_base["height_cm"].to_numpy(dtype=np.float32)
    y_val = val_base["height_cm"].to_numpy(dtype=np.float32)
    
    # Best single source (by validation)
    skip_cols = {"speaker_id", "height_cm", "source", "gender", "source_id"}
    source_cols = [c for c in val_preds.columns if c not in skip_cols]
    val_errors_all = np.zeros((len(y_val), len(source_cols)), dtype=np.float32)
    for j, col in enumerate(source_cols):
        pred = val_preds[col].to_numpy(dtype=np.float32)
        val_errors_all[:, j] = np.abs(pred - y_val)
    best_val_idx = int(np.argmin(np.mean(val_errors_all, axis=0)))
    best_val_source = source_cols[best_val_idx]
    best_val_test_mae = float(np.mean(np.abs(test_preds[best_val_source].to_numpy(dtype=np.float32) - y_test)))
    best_val_val_mae = float(np.mean(val_errors_all[:, best_val_idx]))
    
    # Simple average of all sources
    avg_test_pred = np.mean([test_preds[c].to_numpy(dtype=np.float32) for c in source_cols], axis=0)
    avg_test_mae = float(np.mean(np.abs(avg_test_pred - y_test)))
    
    # Our per-speaker selector
    selector_test_mae = float(np.mean(np.abs(test_preds_out - y_test)))
    selector_val_mae = float(np.mean(np.abs(val_preds_out - y_val)))
    
    print(f"  Best val source ({best_val_source}): val={best_val_val_mae:.3f}cm, test={best_val_test_mae:.3f}cm")
    print(f"  Simple average: test={avg_test_mae:.3f}cm")
    print(f"  Per-speaker selector: val(LOOCV)={selector_val_mae:.3f}cm, test={selector_test_mae:.3f}cm")
    
    # Step 5: Output
    print("\n=== Step 5: Writing Output ===")
    result_df = test_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    result_df["selector_pred_cm"] = test_preds_out
    result_df["selector_abs_error_cm"] = np.abs(test_preds_out - y_test)
    
    out_path = output_dir / "per_speaker_selector_predictions_test.csv"
    result_df.to_csv(out_path, index=False)
    print(f"[MetaSelector] Wrote {out_path}")
    
    # Also write val predictions for gauntlet
    val_out = val_base[["speaker_id", "source", "gender", "height_cm"]].copy()
    val_out["selector_pred_cm"] = val_preds_out
    val_out["selector_abs_error_cm"] = np.abs(val_preds_out - y_val)
    val_out_path = output_dir / "per_speaker_selector_predictions_val.csv"
    val_out.to_csv(val_out_path, index=False)
    print(f"[MetaSelector] Wrote {val_out_path}")
    
    # Write report
    report = {
        "method": f"knn_k{args.k_neighbors}_source_select",
        "k_neighbors": args.k_neighbors,
        "n_prediction_sources": len(source_cols),
        "val_loocv_mae": round(float(selector_val_mae), 4),
        "test_mae": round(float(selector_test_mae), 4),
        "best_val_source_mae": round(float(best_val_val_mae), 4),
        "best_val_source_test_mae": round(float(best_val_test_mae), 4),
        "simple_avg_test_mae": round(float(avg_test_mae), 4),
        "oracle_mae": round(float(np.mean(np.min(val_errors_all, axis=1))), 4),
        "source_meta": {k: v for k, v in list(source_meta.items())[:5]},
        "n_sources_with_val": len([c for c in val_preds.columns if c not in ('speaker_id','height_cm','source','gender','source_id')]),
    }
    
    report_path = output_dir / "meta_selector_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[MetaSelector] Wrote {report_path}")
    
    print(f"\n{'='*60}")
    print(f"  PER-SPEAKER SELECTOR RESULTS")
    print(f"  Test MAE: {selector_test_mae:.3f}cm (target: 3.0cm)")
    print(f"  Best val source test MAE: {best_val_test_mae:.3f}cm")
    print(f"  Simple avg test MAE: {avg_test_mae:.3f}cm")
    print(f"  Oracle (per speaker best): {float(np.mean(np.min(val_errors_all, axis=1))):.3f}cm")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
