#!/usr/bin/env python
"""
Nuclear 3cm Push: Speaker-level ML stack from raw features.

Aggregates SSL embeddings + physics scalars per speaker, trains diverse
base learners with KFold OOF, stacks with Huber meta-learner, applies
gender-conditional calibration.

CPU-only. Runs in ~5 min.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import HuberRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed, skipping LGB models")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed, skipping XGB models")

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "features_v4_combo_full_ssl"
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = ROOT / "outputs" / "nuclear_3cm_push"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHYSICS_SCALARS = [
    "f0_mean", "formant_spacing_mean", "vtl_mean",
    "jitter", "shimmer", "hnr",
    "duration_s", "voiced_ratio",
    "snr_db_estimate", "speech_ratio",
]


def load_clip_features(split: str) -> pd.DataFrame:
    """Load all clips for a split, extract scalar + SSL features."""
    feat_path = FEAT_DIR / split
    files = sorted(os.listdir(feat_path))
    rows = []
    for fname in files:
        if not fname.endswith(".npz"):
            continue
        data = np.load(feat_path / fname, allow_pickle=True)
        row = {
            "speaker_id": str(data["speaker_id"]),
            "gender": int(data["gender"]),
            "height_cm": float(data["height_cm"]),
            "source": str(data["source"]),
        }
        # Physics scalars
        for k in PHYSICS_SCALARS:
            v = float(data[k])
            row[k] = v if np.isfinite(v) else np.nan
        # SSL embedding
        ssl = data["ssl_embedding"].astype(np.float32)
        for i, v in enumerate(ssl):
            row[f"ssl_{i}"] = float(v)
        # Sequence summary: mean across time of 264-dim feature
        seq = data["sequence"].astype(np.float32)  # (T, 264)
        seq_mean = np.nanmean(seq, axis=0)
        for i, v in enumerate(seq_mean):
            row[f"seqmean_{i}"] = float(v) if np.isfinite(v) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_speaker(clip_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clip-level features to speaker level."""
    ssl_cols = [c for c in clip_df.columns if c.startswith("ssl_")]
    seq_cols = [c for c in clip_df.columns if c.startswith("seqmean_")]

    groups = clip_df.groupby("speaker_id")
    records = []
    for sid, g in groups:
        rec = {
            "speaker_id": sid,
            "height_cm": g["height_cm"].iloc[0],
            "gender": g["gender"].iloc[0],
            "source": g["source"].iloc[0],
            "n_clips": len(g),
        }
        # Physics: mean, std, median, min, max, range
        for k in PHYSICS_SCALARS:
            vals = g[k].dropna()
            if len(vals) == 0:
                rec[f"{k}_mean"] = np.nan
                rec[f"{k}_std"] = 0.0
                rec[f"{k}_med"] = np.nan
                rec[f"{k}_min"] = np.nan
                rec[f"{k}_max"] = np.nan
                rec[f"{k}_range"] = 0.0
            else:
                rec[f"{k}_mean"] = vals.mean()
                rec[f"{k}_std"] = vals.std() if len(vals) > 1 else 0.0
                rec[f"{k}_med"] = vals.median()
                rec[f"{k}_min"] = vals.min()
                rec[f"{k}_max"] = vals.max()
                rec[f"{k}_range"] = vals.max() - vals.min()

        # SSL: mean + std
        for c in ssl_cols:
            vals = g[c].values
            rec[f"{c}_mean"] = np.mean(vals)
            rec[f"{c}_std"] = np.std(vals) if len(vals) > 1 else 0.0

        # Sequence: mean + std
        for c in seq_cols:
            vals = g[c].values
            rec[f"{c}_mean"] = np.mean(vals)
            rec[f"{c}_std"] = np.std(vals) if len(vals) > 1 else 0.0

        records.append(rec)
    return pd.DataFrame(records)


def build_feature_matrix(
    speaker_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """Build feature matrix from speaker-level aggregated data."""
    # Pick all numeric columns except target/id
    exclude = {"speaker_id", "height_cm", "source"}
    feat_cols = [
        c for c in speaker_df.columns
        if c not in exclude and speaker_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]
    X = speaker_df[feat_cols].values.astype(np.float64)
    # Fill NaN with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            med = np.nanmedian(col)
            X[mask, j] = med if np.isfinite(med) else 0.0
    return X, feat_cols


def get_base_models() -> Dict:
    """Return dict of (name -> model constructor)."""
    models = {}
    models["ridge_a1"] = lambda: Ridge(alpha=1.0)
    models["ridge_a10"] = lambda: Ridge(alpha=10.0)
    models["ridge_a100"] = lambda: Ridge(alpha=100.0)
    models["krr_rbf"] = lambda: KernelRidge(alpha=10.0, kernel="rbf", gamma=0.001)
    models["huber"] = lambda: HuberRegressor(epsilon=1.35, max_iter=500, alpha=1.0)
    models["et_200"] = lambda: ExtraTreesRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    models["et_500"] = lambda: ExtraTreesRegressor(
        n_estimators=500, max_depth=16, min_samples_leaf=3, random_state=17, n_jobs=-1
    )
    models["rf_300"] = lambda: RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=4, random_state=42, n_jobs=-1
    )
    models["gbr_200"] = lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    models["knn_10"] = lambda: KNeighborsRegressor(n_neighbors=10, weights="distance")
    models["knn_20"] = lambda: KNeighborsRegressor(n_neighbors=20, weights="distance")

    if HAS_XGB:
        models["xgb_200"] = lambda: xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        models["xgb_500"] = lambda: xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.6, reg_alpha=2.0, reg_lambda=10.0,
            random_state=17, n_jobs=-1, verbosity=0,
        )

    if HAS_LGB:
        models["lgb_200"] = lambda: lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        models["lgb_500"] = lambda: lgb.LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.6, reg_alpha=2.0, reg_lambda=10.0,
            random_state=17, n_jobs=-1, verbose=-1,
        )

    return models


def oof_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    models: Dict,
    n_folds: int = 5,
    n_repeats: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Train base models with repeated KFold, return OOF + val + test predictions."""
    n_train = len(y_train)
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    model_names = list(models.keys())
    n_models = len(model_names)

    oof_preds = np.zeros((n_train, n_models))
    val_preds = np.zeros((n_val, n_models))
    test_preds = np.zeros((n_test, n_models))
    oof_counts = np.zeros((n_train, n_models))

    kf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)

    for mi, name in enumerate(model_names):
        print(f"  Training {name}...", end="", flush=True)
        fold_val_preds = []
        fold_test_preds = []
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
            model = models[name]()
            model.fit(X_train[tr_idx], y_train[tr_idx])
            oof_preds[va_idx, mi] += model.predict(X_train[va_idx])
            oof_counts[va_idx, mi] += 1
            fold_val_preds.append(model.predict(X_val))
            fold_test_preds.append(model.predict(X_test))
        # Average OOF
        mask = oof_counts[:, mi] > 0
        oof_preds[mask, mi] /= oof_counts[mask, mi]
        val_preds[:, mi] = np.mean(fold_val_preds, axis=0)
        test_preds[:, mi] = np.mean(fold_test_preds, axis=0)
        oof_mae = mean_absolute_error(y_train[mask], oof_preds[mask, mi])
        val_mae = mean_absolute_error(y_val, val_preds[:, mi])
        test_mae = mean_absolute_error(y_test, test_preds[:, mi])
        print(f" OOF={oof_mae:.3f} val={val_mae:.3f} test={test_mae:.3f}")

    return oof_preds, val_preds, test_preds, model_names


def stack_blend(
    oof_preds: np.ndarray,
    y_train: np.ndarray,
    val_preds: np.ndarray,
    test_preds: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Level-2 stacking with Huber + Ridge ensemble."""
    scaler = StandardScaler()
    oof_s = scaler.fit_transform(oof_preds)
    val_s = scaler.transform(val_preds)
    test_s = scaler.transform(test_preds)

    meta_models = {
        "huber_stack": HuberRegressor(epsilon=1.35, max_iter=1000, alpha=0.1),
        "ridge_stack": RidgeCV(alphas=[0.1, 1, 10, 50, 100]),
    }
    meta_val = []
    meta_test = []
    for name, model in meta_models.items():
        model.fit(oof_s, y_train)
        vp = model.predict(val_s)
        tp = model.predict(test_s)
        print(f"  {name}: val={mean_absolute_error(y_val, vp):.3f} test={mean_absolute_error(y_test, tp):.3f}")
        meta_val.append(vp)
        meta_test.append(tp)

    # Average meta predictions
    val_stack = np.mean(meta_val, axis=0)
    test_stack = np.mean(meta_test, axis=0)
    oof_stack = np.mean(
        [m.predict(oof_s) for m in meta_models.values()], axis=0
    )
    return oof_stack, val_stack, test_stack


def gender_conditional_calibration(
    val_pred: np.ndarray,
    val_true: np.ndarray,
    val_gender: np.ndarray,
    test_pred: np.ndarray,
    test_true: np.ndarray,
    test_gender: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-gender shift+scale calibration fitted on val."""
    cal_val = val_pred.copy()
    cal_test = test_pred.copy()

    for g in [0, 1]:  # 0=female, 1=male
        vm = val_gender == g
        tm = test_gender == g
        if vm.sum() < 5:
            continue
        # Fit simple shift on val
        residuals = val_true[vm] - val_pred[vm]
        shift = np.median(residuals)
        cal_val[vm] += shift
        cal_test[tm] += shift

    return cal_val, cal_test


def short_speaker_rescue(
    pred: np.ndarray,
    true: np.ndarray,
    gender: np.ndarray,
    height_threshold: float = 160.0,
) -> np.ndarray:
    """If predicted height is very close to mean, push short predictions down."""
    # This is applied on val to find the optimal shrink factor
    rescued = pred.copy()
    # Identify likely-short speakers (predicted < threshold)
    short_mask = pred < height_threshold
    if short_mask.sum() > 0:
        # Mild additional downward push for short predictions
        rescued[short_mask] -= 0.5
    return rescued


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NUCLEAR 3CM PUSH - Speaker ML Stack")
    print("=" * 60)

    # 1) Load clips
    print("\n[1/6] Loading clip features...")
    train_clips = load_clip_features("train")
    val_clips = load_clip_features("val")
    test_clips = load_clip_features("test")
    print(f"  Clips: train={len(train_clips)} val={len(val_clips)} test={len(test_clips)}")

    # Filter out CELEB (no height labels in val/test)
    # Keep only speakers that appear in splits
    train_split = pd.read_csv(SPLIT_DIR / "train_clean.csv")
    val_split = pd.read_csv(SPLIT_DIR / "val_clean.csv")
    test_split = pd.read_csv(SPLIT_DIR / "test_clean.csv")

    train_clips = train_clips[train_clips["speaker_id"].isin(train_split["speaker_id"])]
    val_clips = val_clips[val_clips["speaker_id"].isin(val_split["speaker_id"])]
    test_clips = test_clips[test_clips["speaker_id"].isin(test_split["speaker_id"])]
    print(f"  After split filter: train={len(train_clips)} val={len(val_clips)} test={len(test_clips)}")

    # 2) Aggregate to speaker level
    print("\n[2/6] Aggregating to speaker level...")
    train_sp = aggregate_speaker(train_clips)
    val_sp = aggregate_speaker(val_clips)
    test_sp = aggregate_speaker(test_clips)
    print(f"  Speakers: train={len(train_sp)} val={len(val_sp)} test={len(test_sp)}")

    # Also add train_plus_external for more training data
    ext_split = SPLIT_DIR / "train_plus_external_fast4.csv"
    if ext_split.exists():
        ext_df = pd.read_csv(ext_split)
        ext_speakers = set(ext_df["speaker_id"]) - set(train_split["speaker_id"])
        if ext_speakers:
            ext_clips = train_clips[train_clips["speaker_id"].isin(ext_speakers)]
            if len(ext_clips) > 0:
                ext_sp = aggregate_speaker(ext_clips)
                print(f"  External speakers added: {len(ext_sp)}")
                train_sp = pd.concat([train_sp, ext_sp], ignore_index=True)

    # 3) Build feature matrix
    print("\n[3/6] Building feature matrix...")
    X_train, feat_cols = build_feature_matrix(train_sp)
    X_val, _ = build_feature_matrix(val_sp)
    X_test, _ = build_feature_matrix(test_sp)
    y_train = train_sp["height_cm"].values.astype(np.float64)
    y_val = val_sp["height_cm"].values.astype(np.float64)
    y_test = test_sp["height_cm"].values.astype(np.float64)
    gender_val = val_sp["gender"].values
    gender_test = test_sp["gender"].values
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Standardize for linear models (tree models don't care)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Optional PCA for KNN/Ridge (keep 95% variance)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_val_pca = pca.transform(X_val_s)
    X_test_pca = pca.transform(X_test_s)
    print(f"  PCA components (95% var): {X_train_pca.shape[1]}")

    # 4) Train base models
    print("\n[4/6] Training base models (5-fold x3 repeat)...")
    models = get_base_models()

    # Full features for tree models, PCA for linear/KNN
    pca_models = {"ridge_a1", "ridge_a10", "ridge_a100", "krr_rbf", "huber", "knn_10", "knn_20"}
    tree_models = {k: v for k, v in models.items() if k not in pca_models}
    linear_models = {k: v for k, v in models.items() if k in pca_models}

    print("  --- Tree/Ensemble models (full features) ---")
    oof_tree, val_tree, test_tree, names_tree = oof_train(
        X_train, y_train, X_val, X_test, tree_models
    )

    print("  --- Linear/KNN models (PCA features) ---")
    oof_lin, val_lin, test_lin, names_lin = oof_train(
        X_train_pca, y_train, X_val_pca, X_test_pca, linear_models
    )

    # Combine
    oof_all = np.hstack([oof_tree, oof_lin])
    val_all = np.hstack([val_tree, val_lin])
    test_all = np.hstack([test_tree, test_lin])
    all_names = names_tree + names_lin

    # 5) Level-2 stacking
    print("\n[5/6] Level-2 stacking...")
    oof_stack, val_stack, test_stack = stack_blend(
        oof_all, y_train, val_all, test_all, y_val, y_test
    )

    # 6) Gender calibration + short rescue
    print("\n[6/6] Gender calibration...")
    val_cal, test_cal = gender_conditional_calibration(
        val_stack, y_val, gender_val, test_stack, y_test, gender_test
    )

    # Simple average of stacked + calibrated (hedge)
    val_final = 0.5 * val_stack + 0.5 * val_cal
    test_final = 0.5 * test_stack + 0.5 * test_cal

    # Also try: just the best individual model vs stack vs calibrated
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Per-model results
    print(f"\n{'Model':30s}  {'Val MAE':>8s}  {'Test MAE':>8s}  {'Test <=3cm':>10s}")
    print("-" * 62)
    for i, name in enumerate(all_names):
        vm = mean_absolute_error(y_val, val_all[:, i])
        tm = mean_absolute_error(y_test, test_all[:, i])
        w3 = 100 * np.mean(np.abs(test_all[:, i] - y_test) <= 3.0)
        print(f"  {name:28s}  {vm:8.3f}  {tm:8.3f}  {w3:9.1f}%")

    print("-" * 62)
    vm = mean_absolute_error(y_val, val_stack)
    tm = mean_absolute_error(y_test, test_stack)
    w3 = 100 * np.mean(np.abs(test_stack - y_test) <= 3.0)
    print(f"  {'STACKED':28s}  {vm:8.3f}  {tm:8.3f}  {w3:9.1f}%")

    vm = mean_absolute_error(y_val, val_cal)
    tm = mean_absolute_error(y_test, test_cal)
    w3 = 100 * np.mean(np.abs(test_cal - y_test) <= 3.0)
    print(f"  {'STACKED+GENCAL':28s}  {vm:8.3f}  {tm:8.3f}  {w3:9.1f}%")

    vm = mean_absolute_error(y_val, val_final)
    tm = mean_absolute_error(y_test, test_final)
    w3 = 100 * np.mean(np.abs(test_final - y_test) <= 3.0)
    print(f"  {'FINAL (avg stack+cal)':28s}  {vm:8.3f}  {tm:8.3f}  {w3:9.1f}%")

    # Save predictions
    test_out = test_sp[["speaker_id", "height_cm", "gender"]].copy()
    test_out["stack_pred_cm"] = test_stack
    test_out["calibrated_pred_cm"] = test_cal
    test_out["final_pred_cm"] = test_final
    test_out["abs_error_cm"] = np.abs(test_final - y_test)
    test_out.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    val_out = val_sp[["speaker_id", "height_cm", "gender"]].copy()
    val_out["stack_pred_cm"] = val_stack
    val_out["calibrated_pred_cm"] = val_cal
    val_out["final_pred_cm"] = val_final
    val_out.to_csv(OUT_DIR / "val_predictions.csv", index=False)

    # Summary
    summary = {
        "n_base_models": len(all_names),
        "base_models": all_names,
        "n_features": X_train.shape[1],
        "n_pca": X_train_pca.shape[1],
        "train_n": int(len(y_train)),
        "val": {
            "n": int(len(y_val)),
            "stack_mae": float(mean_absolute_error(y_val, val_stack)),
            "calibrated_mae": float(mean_absolute_error(y_val, val_cal)),
            "final_mae": float(mean_absolute_error(y_val, val_final)),
            "final_within_3cm": float(np.mean(np.abs(val_final - y_val) <= 3.0)),
        },
        "test": {
            "n": int(len(y_test)),
            "stack_mae": float(mean_absolute_error(y_test, test_stack)),
            "calibrated_mae": float(mean_absolute_error(y_test, test_cal)),
            "final_mae": float(mean_absolute_error(y_test, test_final)),
            "final_within_3cm": float(np.mean(np.abs(test_final - y_test) <= 3.0)),
        },
        "per_model_test_mae": {
            name: float(mean_absolute_error(y_test, test_all[:, i]))
            for i, name in enumerate(all_names)
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {OUT_DIR}")
    print("Done.")
