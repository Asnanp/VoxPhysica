#!/usr/bin/env python
"""
Multi-fold Cross-Validated XGBoost Height Predictor.

Splits the 97 validation speakers into K stratified folds (by height bin),
trains separate XGBoost models per fold, and generates out-of-fold (OOF)
predictions for a realistic (non-overfit) validation estimate.

Then trains a final model on ALL validation data and predicts the test set.

Outputs Phase22-compatible CSVs with OOF-based val predictions + final test predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed, install with: pip install xgboost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-fold CV clip-level XGBoost height predictor")
    parser.add_argument("--features-dir", default="data/features_v4")
    parser.add_argument("--output-dir", default="outputs/clip_xgboost_kfold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-strategy", choices=["mean", "max", "mean_max"], default="mean",
                        help="How to pool the sequence features over time")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of within-fold training clips held for early stopping")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_clip_data(features_dir: Path, split: str, pool_strategy: str,
                   ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, float], Dict[str, List[int]]]:
    """Load all clips from a split directory.

    Returns:
        X: (n_clips, n_features) float32 array
        y: (n_clips,) float32 height array
        speaker_ids: list of speaker_id strings (n_clips,)
        speaker_heights: dict mapping speaker_id -> mean height in cm
        speaker_map: mapping from speaker_id to list of clip indices
    """
    split_dir = features_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    npz_files = sorted(split_dir.glob("*.npz"))
    print(f"[Data] Loading {len(npz_files)} clips from {split_dir}")

    all_features: List[np.ndarray] = []
    all_heights: List[float] = []
    all_speakers: List[str] = []
    speaker_map: Dict[str, List[int]] = {}
    n_skipped = 0

    for npz_path in tqdm(npz_files, desc=f"Loading {split}"):
        try:
            data = dict(np.load(npz_path, allow_pickle=True))
        except Exception:
            n_skipped += 1
            continue

        seq = data.get("sequence")
        if seq is None or not isinstance(seq, np.ndarray) or seq.ndim != 2:
            n_skipped += 1
            continue

        # Pool the sequence: (time, 136) -> (136,)
        if pool_strategy == "mean":
            pooled = np.mean(seq, axis=0)
        elif pool_strategy == "max":
            pooled = np.max(seq, axis=0)
        else:  # mean_max
            pooled = np.concatenate([np.mean(seq, axis=0), np.max(seq, axis=0)])

        # Add scalar features
        scalar_feats = []
        scalar_keys = [
            "f0_mean", "vtl_mean", "formant_spacing_mean", "hnr", "jitter",
            "shimmer", "snr_db_estimate", "voiced_ratio", "speech_ratio",
            "duration_s", "clipped_ratio", "capture_quality_score",
        ]
        for key in scalar_keys:
            val = data.get(key)
            if isinstance(val, np.ndarray) and val.ndim == 0:
                scalar_feats.append(float(val))
            else:
                scalar_feats.append(0.0)

        # Gender as feature
        g = data.get("gender")
        if isinstance(g, np.ndarray) and g.ndim == 0:
            scalar_feats.append(float(g))
        else:
            scalar_feats.append(0.0)

        features = np.concatenate([pooled, np.array(scalar_feats, dtype=np.float32)])

        height = float(data.get("height_cm", np.nan))
        if np.isnan(height):
            n_skipped += 1
            continue

        sid = str(data.get("speaker_id", b"unknown"))
        if isinstance(sid, bytes):
            sid = sid.decode("utf-8")

        idx = len(all_heights)
        all_features.append(features)
        all_heights.append(height)
        all_speakers.append(sid)
        if sid not in speaker_map:
            speaker_map[sid] = []
        speaker_map[sid].append(idx)

    if n_skipped > 0:
        print(f"[Data] Skipped {n_skipped} files with missing/invalid data")

    X = np.stack(all_features, axis=0)
    y = np.array(all_heights, dtype=np.float32)
    speaker_ids = all_speakers

    print(f"[Data] Loaded {len(X)} clips, {len(speaker_map)} speakers, {X.shape[1]} features")
    print(f"[Data] Height range: {y.min():.1f} - {y.max():.1f}cm")

    # Build per-speaker height map
    speaker_heights: Dict[str, float] = {}
    for sid, clip_idxs in speaker_map.items():
        speaker_heights[sid] = float(np.mean(y[clip_idxs]))

    return X, y, speaker_ids, speaker_heights, speaker_map


def speaker_stratified_folds(
    speaker_ids: List[str],
    speaker_map: Dict[str, List[int]],
    speaker_heights: Dict[str, float],
    n_folds: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create stratified folds at the speaker level.

    Each fold is (train_clip_indices, val_clip_indices).

    Stratification ensures roughly equal height-bin distribution per fold.
    """
    rng = np.random.RandomState(seed)

    # Build speaker-level data
    speaker_list = sorted(speaker_map.keys())
    n_speakers = len(speaker_list)

    # Assign height bins
    bins = [0, 162, 178, 999]
    bin_labels = ["short", "medium", "tall"]

    speaker_heights_arr = []
    speaker_bins = []
    for sid in speaker_list:
        sp_h = speaker_heights[sid]
        speaker_heights_arr.append(sp_h)
        bin_idx = 0
        for b_idx in range(len(bins) - 1):
            if bins[b_idx] <= sp_h < bins[b_idx + 1]:
                bin_idx = b_idx
                break
        speaker_bins.append(bin_idx)

    speaker_heights_arr = np.array(speaker_heights_arr, dtype=np.float32)
    speaker_bins = np.array(speaker_bins, dtype=np.int32)

    print(f"[CV] Stratifying {n_speakers} speakers across {n_folds} folds")
    for b_idx, label in enumerate(bin_labels):
        count = int(np.sum(speaker_bins == b_idx))
        if count > 0:
            print(f"      {label}: {count} speakers")

    # Assign speakers to folds using stratified shuffle
    fold_assignments = np.full(n_speakers, -1, dtype=np.int32)

    # Process each bin separately
    for b_idx in range(len(bins) - 1):
        bin_speaker_indices = np.where(speaker_bins == b_idx)[0]
        rng.shuffle(bin_speaker_indices)
        for i, sp_idx in enumerate(bin_speaker_indices):
            fold_assignments[sp_idx] = int(i % n_folds)

    # Check all assigned
    assert np.all(fold_assignments >= 0), "Some speakers not assigned to a fold!"

    # Count per fold
    fold_counts = np.bincount(fold_assignments.astype(np.int64))
    for f in range(n_folds):
        print(f"  Fold {f}: {int(fold_counts[f])} speakers")

    # Build clip-level fold assignments
    speaker_to_fold: Dict[str, int] = {}
    for i, sid in enumerate(speaker_list):
        speaker_to_fold[sid] = int(fold_assignments[i])

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_folds):
        train_mask = np.array([
            speaker_to_fold[sid] != fold_idx for sid in speaker_ids
        ], dtype=bool)
        val_mask = ~train_mask
        folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))

    return folds


def train_xgboost_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val_early: np.ndarray, y_val_early: np.ndarray,
    params: Dict[str, Any],
) -> Any:
    """Train a single XGBoost model with early stopping."""
    early_stop = params.get("early_stopping_rounds", 50)
    model = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=1.0,
        min_child_weight=3.0,
        gamma=0.5,
        early_stopping_rounds=early_stop,
        random_state=params["seed"],
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val_early, y_val_early)],
        verbose=False,
    )
    return model


def compute_speaker_metrics(clip_preds: np.ndarray, clip_true: np.ndarray,
                            speaker_ids: List[str]) -> Dict[str, Any]:
    """Compute per-speaker metrics from clip-level predictions."""
    speaker_preds: Dict[str, List[float]] = {}
    speaker_true: Dict[str, List[float]] = {}
    for sid, pred, true_val in zip(speaker_ids, clip_preds, clip_true):
        if sid not in speaker_preds:
            speaker_preds[sid] = []
            speaker_true[sid] = []
        speaker_preds[sid].append(float(pred))
        speaker_true[sid].append(float(true_val))

    sp_ids = np.array(list(speaker_preds.keys()))
    sp_pred = np.array([float(np.mean(speaker_preds[sid])) for sid in sp_ids], dtype=np.float32)
    sp_true = np.array([float(np.mean(speaker_true[sid])) for sid in sp_ids], dtype=np.float32)

    sp_mae = float(np.mean(np.abs(sp_pred - sp_true)))
    sp_bias = float(np.mean(sp_pred - sp_true))

    # Per-height metrics
    height_metrics = {}
    for lo, hi, label in [(0, 162, "short"), (162, 178, "medium"), (178, 999, "tall")]:
        mask = (sp_true >= lo) & (sp_true < hi)
        if int(mask.sum()) > 0:
            height_metrics[label] = {
                "n": int(mask.sum()),
                "mae": float(np.mean(np.abs(sp_pred[mask] - sp_true[mask]))),
                "bias": float(np.mean(sp_pred[mask] - sp_true[mask])),
            }

    return {
        "n_speakers": len(sp_ids),
        "mae": sp_mae,
        "bias": sp_bias,
        "per_height": height_metrics,
        "speaker_ids": sp_ids,
        "speaker_preds": sp_pred,
        "speaker_true": sp_true,
    }


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_dir = resolve(args.features_dir)

    if not HAS_XGB:
        print("\n[ERROR] xgboost not installed. Run: pip install xgboost")
        return 1

    print("=" * 65)
    print("  K-FOLD CROSS-VALIDATED XGBOOST HEIGHT PREDICTOR")
    print("=" * 65)

    # ========== Load Data ==========
    print("\n=== Loading Validation Clips ===")
    X_val, y_val, speakers_val, speaker_heights_val, sp_map_val = load_clip_data(
        features_dir, "val", args.pool_strategy)
    n_val_clips = len(X_val)

    print("\n=== Loading Test Clips ===")
    X_test, y_test, speakers_test, speaker_heights_test, sp_map_test = load_clip_data(
        features_dir, "test", args.pool_strategy)
    n_test_clips = len(X_test)

    n_features = X_val.shape[1]
    n_scalar = 13  # 12 acoustic + 1 gender

    # ========== Create Stratified Folds ==========
    print(f"\n{'=' * 65}")
    print(f"  CREATING {args.n_folds} STRATIFIED FOLDS")
    print(f"  {n_val_clips} clips from {len(sp_map_val)} validation speakers")
    print(f"{'=' * 65}")

    folds = speaker_stratified_folds(
        speakers_val, sp_map_val, speaker_heights_val, args.n_folds, args.seed)

    # ========== K-Fold Training ==========
    print(f"\n{'=' * 65}")
    print(f"  K-FOLD TRAINING")
    print(f"{'=' * 65}")

    oof_preds = np.full(n_val_clips, np.nan, dtype=np.float32)
    fold_metrics = []

    for fold_idx, (train_clip_idxs, val_clip_idxs) in enumerate(folds):
        n_train = len(train_clip_idxs)
        n_val_fold = len(val_clip_idxs)

        # Further split train into train + early_stop (by clips, not speakers)
        # to avoid leaking speaker info into early stopping
        n_val_early = max(1, int(n_train * args.val_split))
        perm = rng.permutation(n_train)
        es_idx = perm[:n_val_early]
        tr_idx = perm[n_val_early:]

        X_tr = X_val[train_clip_idxs[tr_idx]]
        y_tr = y_val[train_clip_idxs[tr_idx]]
        X_es = X_val[train_clip_idxs[es_idx]]
        y_es = y_val[train_clip_idxs[es_idx]]
        X_val_fold = X_val[val_clip_idxs]
        y_val_fold = y_val[val_clip_idxs]

        print(f"\n  Fold {fold_idx + 1}/{args.n_folds}: "
              f"train={len(tr_idx)} clips, early_stop={len(es_idx)}, val={n_val_fold}")

        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "seed": args.seed + fold_idx,
            "early_stopping_rounds": args.early_stopping,
        }

        model = train_xgboost_fold(X_tr, y_tr, X_es, y_es, params)
        best_iter = model.best_iteration + 1 if model.best_iteration and model.best_iteration > 0 else args.n_estimators
        print(f"    Best iteration: {best_iter}")

        # Predict on held-out fold
        fold_preds = model.predict(X_val_fold).astype(np.float32)
        oof_preds[val_clip_idxs] = fold_preds

        # Save fold model
        model.save_model(str(output_dir / f"xgb_fold_{fold_idx}.ubj"))

        fold_fold_mae = float(np.mean(np.abs(fold_preds - y_val_fold)))
        fold_sp_metrics = compute_speaker_metrics(
            fold_preds, y_val_fold,
            [speakers_val[i] for i in val_clip_idxs])
        print(f"    Fold clip MAE: {fold_fold_mae:.4f}cm  "
              f"Fold speaker MAE: {fold_sp_metrics['mae']:.4f}cm")

        fold_metrics.append({
            "fold": fold_idx,
            "best_iteration": best_iter,
            "clip_mae": round(fold_fold_mae, 4),
            "speaker_mae": round(fold_sp_metrics["mae"], 4),
        })

    # ========== OOF Validation ==========
    print(f"\n{'=' * 65}")
    print(f"  OOF VALIDATION RESULTS (honest estimate)")
    print(f"{'=' * 65}")

    # Check for NaN (shouldn't happen)
    oof_valid = ~np.isnan(oof_preds)
    if not oof_valid.all():
        missing = int(np.sum(~oof_valid))
        print(f"  WARNING: {missing} clips missing OOF predictions!")

    oof_clip_mae = float(np.mean(np.abs(oof_preds[oof_valid] - y_val[oof_valid])))
    print(f"  OOF clip MAE: {oof_clip_mae:.4f}cm")

    oof_sp_metrics = compute_speaker_metrics(
        oof_preds[oof_valid], y_val[oof_valid],
        [speakers_val[i] for i in range(n_val_clips) if oof_valid[i]])
    print(f"  OOF speaker MAE: {oof_sp_metrics['mae']:.4f}cm")
    print(f"  OOF speaker bias: {oof_sp_metrics['bias']:+.4f}cm")
    for label, hm in oof_sp_metrics["per_height"].items():
        print(f"    {label} (n={hm['n']}): MAE={hm['mae']:.3f}cm bias={hm['bias']:+.3f}cm")

    # ========== Train Final Model on ALL Validation Data ==========
    print(f"\n{'=' * 65}")
    print(f"  TRAINING FINAL MODEL (all validation data)")
    print(f"{'=' * 65}")

    n_val_early = max(1, int(n_val_clips * args.val_split))
    perm = rng.permutation(n_val_clips)
    es_idx = perm[:n_val_early]
    tr_idx = perm[n_val_early:]

    X_tr_final = X_val[tr_idx]
    y_tr_final = y_val[tr_idx]
    X_es_final = X_val[es_idx]
    y_es_final = y_val[es_idx]

    print(f"  Train clips: {len(tr_idx)}, Early stop: {len(es_idx)}")

    final_model = train_xgboost_fold(X_tr_final, y_tr_final, X_es_final, y_es_final, {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "seed": args.seed + 999,
        "early_stopping_rounds": args.early_stopping,
    })
    final_best_iter = final_model.best_iteration + 1 if final_model.best_iteration and final_model.best_iteration > 0 else args.n_estimators
    print(f"  Best iteration: {final_best_iter}")

    # Full validation predictions (not OOF, for reference)
    full_val_preds = final_model.predict(X_val).astype(np.float32)
    full_val_mae = float(np.mean(np.abs(full_val_preds - y_val)))
    print(f"  Full val clip MAE: {full_val_mae:.4f}cm (overfit, not honest)")

    # ========== Test Predictions ==========
    print(f"\n{'=' * 65}")
    print(f"  TEST PREDICTIONS")
    print(f"{'=' * 65}")

    test_preds = final_model.predict(X_test).astype(np.float32)
    test_sp_metrics = compute_speaker_metrics(
        test_preds, y_test, speakers_test)
    print(f"  Test speaker MAE: {test_sp_metrics['mae']:.4f}cm")
    print(f"  Test speaker bias: {test_sp_metrics['bias']:+.4f}cm")
    for label, hm in test_sp_metrics["per_height"].items():
        print(f"    {label} (n={hm['n']}): MAE={hm['mae']:.3f}cm bias={hm['bias']:+.3f}cm")

    # ========== Write Predictions ==========
    print(f"\n{'=' * 65}")
    print(f"  WRITING PREDICTIONS")
    print(f"{'=' * 65}")

    # --- Clip-level CSVs ---
    # Val OOF clip predictions
    val_oof_clip = pd.DataFrame({
        "speaker_id": speakers_val,
        "height_cm": y_val,
        "clip_xgb_oof_pred_cm": oof_preds,  # NaN for any missing
    })
    val_oof_clip.to_csv(output_dir / "val_clip_predictions.csv", index=False)
    print(f"  Wrote: {output_dir / 'val_clip_predictions.csv'} (OOF clip preds)")

    # Test clip predictions
    test_clip_df = pd.DataFrame({
        "speaker_id": speakers_test,
        "height_cm": y_test,
        "clip_xgb_pred_cm": test_preds,
    })
    test_clip_df.to_csv(output_dir / "test_clip_predictions.csv", index=False)
    print(f"  Wrote: {output_dir / 'test_clip_predictions.csv'} (test clip preds)")

    # --- Speaker-level CSVs (for Phase22) ---
    # Val OOF speaker predictions
    val_sp_rows = []
    for sid in oof_sp_metrics["speaker_ids"]:
        # Find the matching OOF preds for this speaker
        sp_mask = np.array([s == sid for s in speakers_val])
        sp_oof = oof_preds[sp_mask]
        sp_oof_valid = ~np.isnan(sp_oof)
        if sp_oof_valid.any():
            pred_val = float(np.mean(sp_oof[sp_oof_valid]))
        else:
            # Fallback: use full model prediction
            pred_val = float(np.mean(full_val_preds[sp_mask]))
        true_val = float(np.mean(y_val[sp_mask]))
        val_sp_rows.append({
            "speaker_id": sid,
            "height_cm": true_val,
            "clip_xgb_pred_cm": pred_val,  # OOF-based
        })
    sp_val_df = pd.DataFrame(val_sp_rows)
    sp_val_df.to_csv(output_dir / "val_speaker_predictions.csv", index=False)
    print(f"  Wrote: {output_dir / 'val_speaker_predictions.csv'} (OOF speaker preds)")

    # Test speaker predictions
    sp_test_rows = []
    for sid in test_sp_metrics["speaker_ids"]:
        sid_idx = list(test_sp_metrics["speaker_ids"]).index(sid)
        sp_test_rows.append({
            "speaker_id": sid,
            "height_cm": float(test_sp_metrics["speaker_true"][sid_idx]),
            "clip_xgb_pred_cm": float(test_sp_metrics["speaker_preds"][sid_idx]),
        })
    sp_test_df = pd.DataFrame(sp_test_rows)
    sp_test_df.to_csv(output_dir / "test_speaker_predictions.csv", index=False)
    print(f"  Wrote: {output_dir / 'test_speaker_predictions.csv'} (test speaker preds)")

    # Save model
    final_model.save_model(str(output_dir / "xgb_final_model.ubj"))
    print(f"  Saved: {output_dir / 'xgb_final_model.ubj'}")
    print(f"  (Fold models saved during training loop)")

    # ========== Report ==========
    report = {
        "n_folds": args.n_folds,
        "n_val_speakers": len(sp_map_val),
        "n_test_speakers": len(sp_map_test),
        "n_val_clips": n_val_clips,
        "n_test_clips": n_test_clips,
        "n_features": n_features,
        "oof_clip_mae": round(float(oof_clip_mae), 4),
        "oof_speaker_mae": round(float(oof_sp_metrics["mae"]), 4),
        "oof_speaker_bias": round(float(oof_sp_metrics["bias"]), 4),
        "oof_per_height": {
            k: {"n": v["n"], "mae": round(v["mae"], 4), "bias": round(v["bias"], 4)}
            for k, v in oof_sp_metrics["per_height"].items()
        },
        "test_speaker_mae": round(float(test_sp_metrics["mae"]), 4),
        "test_speaker_bias": round(float(test_sp_metrics["bias"]), 4),
        "test_per_height": {
            k: {"n": v["n"], "mae": round(v["mae"], 4), "bias": round(v["bias"], 4)}
            for k, v in test_sp_metrics["per_height"].items()
        },
        "full_val_clip_mae": round(float(full_val_mae), 4),
        "final_model_best_iteration": final_best_iter,
        "fold_metrics": fold_metrics,
    }

    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote: {output_dir / 'report.json'}")

    # ========== Summary ==========
    print(f"\n{'=' * 65}")
    print(f"  K-FOLD XGBOOST RESULTS")
    print(f"  {'Fold-wise best iterations:':30s} {[m['best_iteration'] for m in fold_metrics]}")
    print(f"  {'OOF clip MAE:':30s} {oof_clip_mae:.4f}cm (honest)")
    print(f"  {'OOF speaker MAE:':30s} {oof_sp_metrics['mae']:.4f}cm (honest)")
    print(f"  {'Full val clip MAE:':30s} {full_val_mae:.4f}cm (overfit)")
    print(f"  {'Test speaker MAE:':30s} {test_sp_metrics['mae']:.4f}cm (real)")
    print(f"  {'Short test MAE:':30s} {test_sp_metrics['per_height'].get('short', {}).get('mae', 0):.4f}cm")
    print(f"{'=' * 65}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
