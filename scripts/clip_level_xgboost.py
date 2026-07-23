#!/usr/bin/env python
"""
Path A: Clip-Level XGBoost Height Predictor.

Trains XGBoost on 1158 validation clips (each with V5 sequence features 
and scalar acoustics) to predict height_cm directly. Runs inference on 
1155 test clips, averages per speaker, and writes Phase22-compatible CSVs.

Unlike the V5 neural network, XGBoost learns a fundamentally different 
function that may capture patterns the NN misses — providing diversity 
to the ensemble/prediction pool.
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
    parser = argparse.ArgumentParser(description="Clip-level XGBoost height predictor")
    parser.add_argument("--features-dir", default="data/features_v4")
    parser.add_argument("--output-dir", default="outputs/clip_xgboost_predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-strategy", choices=["mean", "max", "mean_max"], default="mean",
                        help="How to pool the sequence features over time")
    parser.add_argument("--use-scalar-features", action="store_true", default=True)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of val clips to hold out for early stopping")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_clips(features_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """Load all clips from a split directory.

    Returns:
        features: (n_clips, n_features) array
        heights: (n_clips,) array
        speaker_ids: list of speaker_id strings
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
        if args.pool_strategy == "mean":
            pooled = np.mean(seq, axis=0)
        elif args.pool_strategy == "max":
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

    print(f"[Data] Loaded {len(X)} clips, {len(speaker_map)} speakers, {X.shape[1]} features")
    print(f"[Data] Height range: {y.min():.1f} - {y.max():.1f}cm")

    return X, y, all_speakers, speaker_map


def main() -> int:
    global args
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_dir = resolve(args.features_dir)

    print("=" * 60)
    print("CLIP-LEVEL XGBOOST HEIGHT PREDICTOR")
    print("=" * 60)

    # Load data
    print("\n=== Loading Validation Clips ===")
    X_val, y_val, speakers_val, speaker_map_val = load_clips(features_dir, "val")

    print("\n=== Loading Test Clips ===")
    X_test, y_test, speakers_test, speaker_map_test = load_clips(features_dir, "test")

    # Hold out a split for early stopping
    n_val = len(X_val)
    n_train = int(n_val * (1.0 - args.val_split))
    indices = np.random.permutation(n_val)
    train_idx = indices[:n_train]
    early_stop_idx = indices[n_train:]

    X_train = X_val[train_idx]
    y_train = y_val[train_idx]
    X_early_stop = X_val[early_stop_idx]
    y_early_stop = y_val[early_stop_idx]

    print(f"\nTrain: {len(X_train)} clips, Early stop: {len(X_early_stop)} clips")
    print(f"Test: {len(X_test)} clips, {len(speaker_map_test)} speakers")

    # Train XGBoost
    if not HAS_XGB:
        print("\n[ERROR] xgboost not installed. Run: pip install xgboost")
        return 1

    print(f"\n=== Training XGBoost ===")
    print(f"  n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"  lr={args.learning_rate}, early_stopping={args.early_stopping}")

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=1.0,
        min_child_weight=3.0,
        gamma=0.5,
        random_state=args.seed,
        verbosity=1,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_early_stop, y_early_stop)],
        verbose=True,
    )

    # Evaluate on validation
    val_preds = model.predict(X_val).astype(np.float32)
    val_mae = float(np.mean(np.abs(val_preds - y_val)))
    print(f"\n  Val MAE (clip-level): {val_mae:.4f}cm")

    # Per-speaker validation metrics
    val_speaker_preds = {}
    val_speaker_true = {}
    for sid, clip_idxs in speaker_map_val.items():
        val_speaker_preds[sid] = float(np.mean(val_preds[clip_idxs]))
        val_speaker_true[sid] = float(np.mean(y_val[clip_idxs]))

    val_sp_ids = np.array(list(val_speaker_preds.keys()))
    val_sp_pred = np.array([val_speaker_preds[sid] for sid in val_sp_ids])
    val_sp_true = np.array([val_speaker_true[sid] for sid in val_sp_ids])
    val_sp_mae = float(np.mean(np.abs(val_sp_pred - val_sp_true)))
    print(f"  Val MAE (speaker-level): {val_sp_mae:.4f}cm")

    # Predict on test
    print("\n=== Predicting Test Clips ===")
    test_preds = model.predict(X_test).astype(np.float32)

    # Per-speaker test metrics
    test_speaker_preds = {}
    test_speaker_true = {}
    for sid, clip_idxs in speaker_map_test.items():
        test_speaker_preds[sid] = float(np.mean(test_preds[clip_idxs]))
        test_speaker_true[sid] = float(np.mean(y_test[clip_idxs]))

    test_sp_ids = np.array(list(test_speaker_preds.keys()))
    test_sp_preds = np.array([test_speaker_preds[sid] for sid in test_sp_ids])
    test_sp_true = np.array([test_speaker_true[sid] for sid in test_sp_ids])
    test_sp_mae = float(np.mean(np.abs(test_sp_preds - test_sp_true)))
    print(f"  Test MAE (speaker-level): {test_sp_mae:.4f}cm")

    # Per-height metrics
    for name, sp_ids, sp_preds, sp_true in [
        ("Val", val_sp_ids, val_sp_pred, val_sp_true),
        ("Test", test_sp_ids, test_sp_preds, test_sp_true),
    ]:
        for lo, hi, label in [(0, 162, "short"), (162, 178, "medium"), (178, 999, "tall")]:
            mask = (sp_true >= lo) & (sp_true < hi)
            if mask.any():
                mae = float(np.mean(np.abs(sp_preds[mask] - sp_true[mask])))
                bias = float(np.mean(sp_preds[mask] - sp_true[mask]))
                print(f"  {name} {label} (n={mask.sum()}): MAE={mae:.3f} bias={bias:+.3f}")

    # Write clip-level predictions
    print("\n=== Writing Predictions ===")
    clip_df = pd.DataFrame({
        "speaker_id": speakers_test,
        "height_cm": y_test,
        "clip_xgb_pred_cm": test_preds,
    })
    clip_df.to_csv(output_dir / "test_clip_predictions.csv", index=False)
    print(f"  Wrote {output_dir / 'test_clip_predictions.csv'}")

    val_clip_df = pd.DataFrame({
        "speaker_id": speakers_val,
        "height_cm": y_val,
        "clip_xgb_pred_cm": val_preds,
    })
    val_clip_df.to_csv(output_dir / "val_clip_predictions.csv", index=False)
    print(f"  Wrote {output_dir / 'val_clip_predictions.csv'}")

    # Write speaker-level predictions
    sp_rows = []
    for sid in test_sp_ids:
        sp_rows.append({
            "speaker_id": sid,
            "height_cm": test_speaker_true[sid],
            "clip_xgb_pred_cm": test_speaker_preds[sid],
        })
    sp_test_df = pd.DataFrame(sp_rows)
    sp_test_df.to_csv(output_dir / "test_speaker_predictions.csv", index=False)
    print(f"  Wrote {output_dir / 'test_speaker_predictions.csv'}")

    sp_val_rows = []
    for sid in val_sp_ids:
        sp_val_rows.append({
            "speaker_id": sid,
            "height_cm": val_speaker_true[sid],
            "clip_xgb_pred_cm": val_speaker_preds[sid],
        })
    sp_val_df = pd.DataFrame(sp_val_rows)
    sp_val_df.to_csv(output_dir / "val_speaker_predictions.csv", index=False)
    print(f"  Wrote {output_dir / 'val_speaker_predictions.csv'}")

    # Save model
    model.save_model(str(output_dir / "xgb_model.ubj"))
    print(f"  Saved model to {output_dir / 'xgb_model.ubj'}")

    # Feature importance
    importance = model.feature_importances_
    n_scalar = len([
        "f0_mean", "vtl_mean", "formant_spacing_mean", "hnr", "jitter",
        "shimmer", "snr_db_estimate", "voiced_ratio", "speech_ratio",
        "duration_s", "clipped_ratio", "capture_quality_score",
    ]) + 1  # +1 for gender
    n_seq = X_val.shape[1] - n_scalar
    top_seq_idx = np.argsort(importance[:n_seq])[::-1][:10]
    top_scalar_idx = np.argsort(importance[n_seq:])[::-1][:10]

    report = {
        "val_mae_clip": round(val_mae, 4),
        "val_mae_speaker": round(val_sp_mae, 4),
        "test_mae_speaker": round(test_sp_mae, 4),
        "n_train_clips": len(X_train),
        "n_val_clips": len(X_early_stop),
        "n_test_clips": len(X_test),
        "n_test_speakers": len(test_sp_ids),
        "n_features": X_val.shape[1],
    }

    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote {output_dir / 'report.json'}")

    print(f"\n{'=' * 60}")
    print(f"  CLIP-LEVEL XGBOOST RESULTS")
    print(f"  Val speaker MAE: {val_sp_mae:.3f}cm")
    print(f"  Test speaker MAE: {test_sp_mae:.3f}cm")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
