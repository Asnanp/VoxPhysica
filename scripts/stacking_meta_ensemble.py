#!/usr/bin/env python
"""
Stacking Meta-Ensemble: trains an XGBoost regressor on the predictions from
all Phase22 candidates (v5 arch, v5 direct, K-fold XGBoost, Phase12, etc.)
plus metadata, to predict speaker height.

The meta-model learns which sources to trust for which speakers, going beyond
simple convex blends.

Usage:
    python scripts/stacking_meta_ensemble.py \
        --output-dir outputs/stacking_meta_ensemble \
        --device cuda
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
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phase22_3cm_reality_gauntlet import (  # type: ignore[import]
    Candidate,
    json_ready,
    load_candidates,
    metrics,
    read_base,
    oracle_result,
    error_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stacking Meta-Ensemble for height prediction")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/stacking_meta_ensemble")
    parser.add_argument("--phase3-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.6)
    parser.add_argument("--colsample-bytree", type=float, default=0.4)
    parser.add_argument("--reg-alpha", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of val held out for early stopping within each fold")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_feature_matrix(
    candidates: Sequence[Candidate],
    val_base: pd.DataFrame,
    test_base: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, List[str]]:
    """Build feature matrices from candidate predictions + metadata.
    
    Returns:
        X_val: (n_val, n_features) float32
        X_test: (n_test, n_features) float32
        val_meta: metadata DataFrame
        test_meta: metadata DataFrame
        feature_names: list of feature names
    """
    n_val = len(val_base)
    n_test = len(test_base)
    
    features_val: Dict[str, np.ndarray] = {}
    features_test: Dict[str, np.ndarray] = {}
    
    # 1. Base prediction features from all validation-paired candidates
    for cand in candidates:
        if cand.val_pred is None:
            continue
        # Sanitize name for column naming
        col_name = f"pred_{len(features_val)}"
        features_val[col_name] = cand.val_pred.astype(np.float32)
        features_test[col_name] = cand.test_pred.astype(np.float32)
    
    n_base = len(features_val)
    print(f"  Base prediction features: {n_base}")
    
    # 2. Derived statistical features
    if n_base >= 2:
        pred_val_mat = np.stack(list(features_val.values()), axis=1)  # (n_val, n_base)
        pred_test_mat = np.stack(list(features_test.values()), axis=1)  # (n_test, n_base)
        
        features_val["mean_pred"] = np.mean(pred_val_mat, axis=1).astype(np.float32)
        features_test["mean_pred"] = np.mean(pred_test_mat, axis=1).astype(np.float32)
        
        features_val["median_pred"] = np.median(pred_val_mat, axis=1).astype(np.float32)
        features_test["median_pred"] = np.median(pred_test_mat, axis=1).astype(np.float32)
        
        features_val["std_pred"] = np.std(pred_val_mat, axis=1).astype(np.float32)
        features_test["std_pred"] = np.std(pred_test_mat, axis=1).astype(np.float32)
        
        features_val["min_pred"] = np.min(pred_val_mat, axis=1).astype(np.float32)
        features_test["min_pred"] = np.min(pred_test_mat, axis=1).astype(np.float32)
        
        features_val["max_pred"] = np.max(pred_val_mat, axis=1).astype(np.float32)
        features_test["max_pred"] = np.max(pred_test_mat, axis=1).astype(np.float32)
        
        features_val["range_pred"] = (features_val["max_pred"] - features_val["min_pred"]).astype(np.float32)
        features_test["range_pred"] = (features_test["max_pred"] - features_test["min_pred"]).astype(np.float32)
        
        # Spread: how far the best individual deviates from the mean
        if False:  # disabled - need error not prediction
            pass
        
        # How many sources predict above/below mean
        above_mean_val = (pred_val_mat > features_val["mean_pred"][:, None]).sum(axis=1).astype(np.float32)
        above_mean_test = (pred_test_mat > features_test["mean_pred"][:, None]).sum(axis=1).astype(np.float32)
        features_val["n_above_mean"] = above_mean_val
        features_test["n_above_mean"] = above_mean_test
    
    # 3. Metadata features
    if "source" in val_base.columns:
        features_val["source_nisp"] = (val_base["source"].astype(str).str.upper() == "NISP").to_numpy(dtype=np.float32)
        features_test["source_nisp"] = (test_base["source"].astype(str).str.upper() == "NISP").to_numpy(dtype=np.float32)
    
    if "gender" in val_base.columns:
        # gender is 0=female, 1=male in this dataset
        features_val["gender_male"] = (val_base["gender"].to_numpy(dtype=np.float32) > 0.5).astype(np.float32)
        features_test["gender_male"] = (test_base["gender"].to_numpy(dtype=np.float32) > 0.5).astype(np.float32)
    
    # 4. Assemble matrices
    feature_names = sorted(features_val.keys())
    X_val = np.column_stack([features_val[name] for name in feature_names]).astype(np.float32)
    X_test = np.column_stack([features_test[name] for name in feature_names]).astype(np.float32)
    
    print(f"  Feature matrix: {X_val.shape[1]} features, {X_val.shape[0]} val, {X_test.shape[0]} test")
    
    return X_val, X_test, val_base.copy(), test_base.copy(), feature_names


def train_oof_xgboost(
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    feature_names: List[str],
) -> Tuple[np.ndarray, xgb.XGBRegressor, float]:
    """Train K-fold XGBoost with OOF predictions on validation set.
    
    Proper K-fold: train only on train_idx, use ALL of val_idx for OOF predictions.
    Within each fold, further split train_idx into train + early_stopping so
    the model doesn't overfit internally.
    
    Returns:
        oof_preds: OOF predictions (n_val,) — truly out-of-fold
        final_model: model trained on full val set
        oof_mae: OOF MAE
    """
    kf = KFold(n_splits=int(args.n_folds), shuffle=True, random_state=int(args.seed))
    oof_preds = np.zeros(len(y_val), dtype=np.float32)
    fold_maes = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_val)):
        # Split the TRAINING portion further into train + early_stopping
        n_train = len(train_idx)
        n_early = max(3, int(n_train * float(args.val_split)))
        rng = np.random.RandomState(int(args.seed) + fold * 7)
        es_idx_in_train = rng.choice(n_train, size=n_early, replace=False)
        tr_mask = np.ones(n_train, dtype=bool)
        tr_mask[es_idx_in_train] = False
        
        train_inner_idx = train_idx[tr_mask]
        es_idx = train_idx[es_idx_in_train]
        
        X_tr = X_val[train_inner_idx]
        y_tr = y_val[train_inner_idx]
        X_es = X_val[es_idx]
        y_es = y_val[es_idx]
        X_v = X_val[val_idx]
        y_v = y_val[val_idx]
        
        model = xgb.XGBRegressor(
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
            min_child_weight=float(args.min_child_weight),
            subsample=float(args.subsample),
            colsample_bytree=float(args.colsample_bytree),
            reg_alpha=float(args.reg_alpha),
            reg_lambda=float(args.reg_lambda),
            random_state=int(args.seed) + fold,
            verbosity=0,
            early_stopping_rounds=int(args.early_stopping),
            eval_metric="mae",
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_es, y_es)],
            verbose=False,
        )
        
        pred = model.predict(X_v, iteration_range=(0, model.best_iteration + 1))
        oof_preds[val_idx] = pred.astype(np.float32)
        
        fold_mae = float(mean_absolute_error(y_v, pred))
        fold_maes.append(fold_mae)
        
        best_iter = int(model.best_iteration + 1)
        print(f"  Fold {fold + 1}: best_iter={best_iter}, fold_mae={fold_mae:.4f}cm", flush=True)
    
    oof_mae = float(mean_absolute_error(y_val, oof_preds))
    print(f"  OOF MAE: {oof_mae:.4f}cm", flush=True)
    
    # Train final model on full validation set (with internal early stopping)
    print("  Training final model on full val set...", flush=True)
    final_model = xgb.XGBRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        min_child_weight=float(args.min_child_weight),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_alpha=float(args.reg_alpha),
        reg_lambda=float(args.reg_lambda),
        random_state=int(args.seed) + 999,
        verbosity=0,
        early_stopping_rounds=int(args.early_stopping),
        eval_metric="mae",
    )
    n_v = len(X_val)
    n_early = max(3, int(n_v * float(args.val_split)))
    rng = np.random.RandomState(int(args.seed) + 888)
    es_idx = rng.choice(n_v, size=n_early, replace=False)
    tr_mask = np.ones(n_v, dtype=bool)
    tr_mask[es_idx] = False
    
    final_model.fit(
        X_val[tr_mask], y_val[tr_mask],
        eval_set=[(X_val[es_idx], y_val[es_idx])],
        verbose=False,
    )
    
    return oof_preds, final_model, oof_mae


def main() -> int:
    args = parse_args()
    np.random.seed(int(args.seed))
    
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs_root = resolve(args.outputs_root)
    
    # Load base data
    val_base = read_base(resolve(args.phase3_val))
    test_base = read_base(resolve(args.phase3_test))
    val_y = val_base["height_cm"].to_numpy(dtype=np.float32)
    test_y = test_base["height_cm"].to_numpy(dtype=np.float32)
    
    print(f"[stacking] val: {len(val_y)} speakers, test: {len(test_y)} speakers")
    
    # Load candidates (use the actual output dir to avoid self-crawl issues)
    dummy_output_dir = resolve("outputs/phase22_kfold_selector")
    print("[stacking] loading candidates...", flush=True)
    candidates = load_candidates(outputs_root, dummy_output_dir, val_base, test_base)
    if not candidates:
        raise RuntimeError("No candidates found")
    val_paired = [c for c in candidates if c.val_pred is not None]
    print(f"[stacking] {len(candidates)} total, {len(val_paired)} val-paired")
    
    # Build feature matrix from all val-paired candidates
    print("[stacking] building feature matrix...", flush=True)
    X_val, X_test, val_meta, test_meta, feature_names = build_feature_matrix(
        val_paired, val_base, test_base,
    )
    
    n_features = X_val.shape[1]
    n_train = len(val_y)
    print(f"[stacking] features: {n_features}, training samples: {n_train}")
    print(f"[stacking] p/n ratio: {n_features / max(1, n_train):.3f}")
    
    if n_features >= n_train:
        print(f"[WARN] p >= n! Features ({n_features}) >= training samples ({n_train}).")
        print("  Heavy regularization required. Using max_depth=2, colsample=0.3, reg_alpha=3.")
        # Override with stronger regularization
        args.max_depth = min(int(args.max_depth), 2)
        args.colsample_bytree = min(float(args.colsample_bytree), 0.3)
        args.reg_alpha = max(float(args.reg_alpha), 3.0)
        args.reg_lambda = max(float(args.reg_lambda), 3.0)
        args.min_child_weight = max(float(args.min_child_weight), 8.0)
        args.subsample = min(float(args.subsample), 0.5)
    
    # Train OOF XGBoost
    print("[stacking] training K-fold XGBoost meta-model...", flush=True)
    oof_preds, final_model, oof_mae = train_oof_xgboost(X_val, val_y, args, feature_names)
    
    # Predict on test
    print("[stacking] predicting on test set...", flush=True)
    test_preds = final_model.predict(X_test, iteration_range=(0, final_model.best_iteration + 1))
    test_preds = test_preds.astype(np.float32)
    
    # Compute test metrics
    test_mae = float(mean_absolute_error(test_y, test_preds))
    test_ae = np.abs(test_preds - test_y)
    test_short_mae = float(test_ae[test_y < 162.0].mean()) if (test_y < 162.0).sum() > 0 else 0.0
    test_within_3cm = float(np.mean(test_ae <= 3.0))
    
    # Val metrics
    val_ae = np.abs(oof_preds - val_y)
    val_short_mae = float(val_ae[val_y < 162.0].mean()) if (val_y < 162.0).sum() > 0 else 0.0
    val_within_3cm = float(np.mean(val_ae <= 3.0))
    
    print(f"\n[stacking] OOF val MAE: {oof_mae:.4f}cm (short={val_short_mae:.4f}, within3={100*val_within_3cm:.1f}%)")
    print(f"[stacking] Test MAE: {test_mae:.4f}cm (short={test_short_mae:.4f}, within3={100*test_within_3cm:.1f}%)")
    
    # Feature importance
    importance = final_model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    print(f"\n  Top 15 features by gain:")
    for i, (feat, gain) in enumerate(sorted_imp[:15]):
        # Map feature index to name with bounds guard
        try:
            idx = int(feat.replace("f", ""))
            name = feature_names[idx] if 0 <= idx < len(feature_names) else feat
        except (ValueError, IndexError):
            name = feat
        print(f"  {i + 1:3d}. {name:40s} gain={gain:.1f}")
    
    # Save predictions
    import csv
    # Val predictions (OOF)
    val_out_path = output_dir / "val_speaker_predictions.csv"
    fieldnames = ["speaker_id", "height_cm", "stacking_oof_pred_cm", "stacking_oof_abs_error"]
    with open(val_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (_, row) in enumerate(val_base.iterrows()):
            writer.writerow({
                "speaker_id": row["speaker_id"],
                "height_cm": f"{float(row['height_cm']):.6f}",
                "stacking_oof_pred_cm": f"{float(oof_preds[i]):.6f}",
                "stacking_oof_abs_error": f"{float(val_ae[i]):.6f}",
            })
    print(f"\n  Wrote OOF val preds: {val_out_path}")
    
    # Test predictions
    test_out_path = output_dir / "test_speaker_predictions.csv"
    with open(test_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker_id", "height_cm", "stacking_pred_cm", "stacking_abs_error"])
        writer.writeheader()
        for i, (_, row) in enumerate(test_base.iterrows()):
            writer.writerow({
                "speaker_id": row["speaker_id"],
                "height_cm": f"{float(row['height_cm']):.6f}",
                "stacking_pred_cm": f"{float(test_preds[i]):.6f}",
                "stacking_abs_error": f"{float(test_ae[i]):.6f}",
            })
    print(f"  Wrote test preds: {test_out_path}")
    
    # Create clip-level equivalents (not available - skip)
    
    # Compute oracles for comparison
    print("\n[stacking] computing oracles...", flush=True)
    global_oracle = oracle_result(candidates, test_y, test_base, require_val=False)
    paired_oracle = oracle_result(candidates, test_y, test_base, require_val=True)
    
    # Compare with best sources
    print()
    print("=" * 65)
    print("  STACKING META-ENSEMBLE RESULTS")
    print("=" * 65)
    print("  %-40s %8s %8s %9s" % ("Method", "MAE", "Short", "Within3%"))
    print("  %-40s %8s %8s %9s" % ("-" * 40, "-" * 8, "-" * 8, "-" * 9))
    
    # Phase12 baseline
    phase12 = next((c for c in val_paired if "phase12" in c.name), None)
    if phase12:
        p12_mae = float(np.mean(np.abs(phase12.test_pred - test_y)))
        short_mask = test_y < 162.0
        p12_short = float(np.mean(np.abs(phase12.test_pred[short_mask] - test_y[short_mask]))) if short_mask.sum() > 0 else 0
        p12_within3 = float(np.mean(np.abs(phase12.test_pred - test_y) <= 3.0))
        print("  %-40s %8.4f %8.4f %8.1f%%" % ("Best Phase12", p12_mae, p12_short, 100 * p12_within3))
    
    # kNN blend baseline
    knn = next((c for c in val_paired if "knn" in c.name.lower() or "push_toward" in c.name), None)
    if knn:
        knn_mae = float(np.mean(np.abs(knn.test_pred - test_y)))
        short_mask = test_y < 162.0
        knn_short = float(np.mean(np.abs(knn.test_pred[short_mask] - test_y[short_mask]))) if short_mask.sum() > 0 else 0
        knn_within3 = float(np.mean(np.abs(knn.test_pred - test_y) <= 3.0))
        print("  %-40s %8.4f %8.4f %8.1f%%" % ("kNN blend (k=7)", knn_mae, knn_short, 100 * knn_within3))
    
    # XGBoost single-split (overfit reference)
    xgb_cand = next((c for c in val_paired if "clip_xgboost" in c.name), None)
    if xgb_cand:
        xgb_mae = float(np.mean(np.abs(xgb_cand.test_pred - test_y)))
        short_mask = test_y < 162.0
        xgb_short = float(np.mean(np.abs(xgb_cand.test_pred[short_mask] - test_y[short_mask]))) if short_mask.sum() > 0 else 0
        xgb_within3 = float(np.mean(np.abs(xgb_cand.test_pred - test_y) <= 3.0))
        print("  %-40s %8.4f %8.4f %8.1f%%" % ("XGBoost single-split", xgb_mae, xgb_short, 100 * xgb_within3))
    
    # Stacking
    print("  %-40s %8.4f %8.4f %8.1f%%" % ("Stacking meta-ensemble", test_mae, test_short_mae, 100 * test_within_3cm))
    
    print("  %-40s %8s %8s %9s" % ("-" * 40, "-" * 8, "-" * 8, "-" * 9))
    go = global_oracle["metrics"]
    print("  %-40s %8.4f %8.4f %8.1f%%" % ("Global oracle (all candidates)", go["mae"], go.get("short_mae", 0), 100 * go.get("within_3cm", 0)))
    po = paired_oracle["metrics"]
    print("  %-40s %8.4f %8.4f %8.1f%%" % ("Val-paired oracle", po["mae"], po.get("short_mae", 0), 100 * po.get("within_3cm", 0)))
    print("=" * 65)
    
    # Save report
    report = {
        "val": {
            "mae": float(oof_mae),
            "short_mae": float(val_short_mae),
            "within_3cm": float(val_within_3cm),
        },
        "test": {
            "mae": float(test_mae),
            "short_mae": float(test_short_mae),
            "within_3cm": float(test_within_3cm),
        },
        "model_params": {
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "min_child_weight": float(args.min_child_weight),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_alpha": float(args.reg_alpha),
            "reg_lambda": float(args.reg_lambda),
        },
        "n_features": int(n_features),
        "n_candidates": len(val_paired),
        "n_val": int(n_train),
        "n_test": len(test_y),
        "feature_importance": {feature_names[int(k.replace("f",""))]: float(v)
                              for k, v in sorted_imp[:30]},
        "global_oracle": {"mae": global_oracle["metrics"]["mae"],
                          "short_mae": global_oracle["metrics"].get("short_mae", 0)},
        "paired_oracle": {"mae": paired_oracle["metrics"]["mae"],
                          "short_mae": paired_oracle["metrics"].get("short_mae", 0)},
    }
    
    (output_dir / "stacking_report.json").write_text(
        json.dumps(json_ready(report), indent=2, allow_nan=True), encoding="utf-8")
    print(f"\nReport: {output_dir / 'stacking_report.json'}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
