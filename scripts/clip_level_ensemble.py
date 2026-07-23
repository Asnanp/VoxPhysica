#!/usr/bin/env python
"""
Clip-level convex ensemble.

Instead of blending at the speaker level (97 data points), this blends at the
CLIP level (1158 data points) using K-fold cross-validation. This gives the
convex optimizer far more training data to learn robust weights.

After blending at the clip level, predictions are averaged per speaker for the
final speaker-level predictions that Phase22 consumes.

Usage:
    python scripts/clip_level_ensemble.py --output-dir outputs/clip_ensemble
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phase22_improved_selector import resolve  # type: ignore[import]
from phase22_3cm_reality_gauntlet import metrics  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clip-level convex ensemble")
    parser.add_argument("--output-dir", default="outputs/clip_ensemble")
    parser.add_argument("--phase3-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--n-folds", type=int, default=5, help="K-fold CV folds over clips")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blend-probes", type=int, default=80000)
    parser.add_argument("--blend-batch", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--target-mae", type=float, default=3.0)
    return parser.parse_args()


def read_phase3(path: str | Path) -> pd.DataFrame:
    """Read Phase3 base with speaker_ids and height_cm."""
    df = pd.read_csv(resolve(path))
    return df


def load_clip_predictions(
    input_dir: str,
    split: str,
) -> Optional[pd.DataFrame]:
    """Try to load a clip-level prediction CSV from a directory."""
    dir_path = resolve(input_dir)
    # Different possible filenames
    candidates = [
        dir_path / f"{split}_clip_predictions.csv",
    ]
    for cp in candidates:
        if cp.exists():
            df = pd.read_csv(cp)
            # Find prediction column - prefer clip_xgb_pred_cm or height_pred_cm
            pred_cols = [c for c in df.columns if c not in {"speaker_id", "height_cm",
                         "source_id", "gender", "source", "height_abs_error_cm",
                         "weight_kg", "weight_pred_kg", "age", "age_pred",
                         "weight_kg_pred", "height_var_norm", "quality_score"}]
            if pred_cols:
                return df
    return None


def discover_clip_sources(
    outputs_root: str | Path,
) -> List[Dict[str, Any]]:
    """Discover all clip-level prediction sources in the outputs tree."""
    root = resolve(outputs_root)
    sources = []

    # Known directories with clip predictions
    known_dirs = [
        "clip_xgboost_predictions",
        "clip_xgboost_kfold",
        "v5_3cm_architecture_predictions",
        "v5_1_direct_3cm_predictions",
    ]

    for dir_name in known_dirs:
        dir_path = root / dir_name
        if not dir_path.is_dir():
            continue

        # Try to load val and test clip predictions
        for split in ["val", "test"]:
            csv_path = dir_path / f"{split}_clip_predictions.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            pred_col = None
            for c in df.columns:
                if c not in {"speaker_id", "height_cm", "source_id", "gender",
                             "source", "height_abs_error_cm", "weight_kg",
                             "weight_pred_kg", "age", "age_pred", "weight_kg_pred",
                             "height_var_norm", "quality_score", "height_abs_error"}:
                    if pred_col is None or "pred" in c.lower():
                        pred_col = c

            if pred_col is None:
                continue

            sources.append({
                "name": f"{dir_name}/{split}_clip_predictions.csv:{pred_col}",
                "dir": dir_name,
                "split": split,
                "path": str(csv_path),
                "column": pred_col,
                "df": df,
            })

    return sources


def build_clip_matrix(
    val_sources: List[Dict[str, Any]],
    test_sources: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build N_clips x N_sources matrices for val and test.
    
    Returns:
        X_val, X_test, y_val, y_test, val_speaker_ids, test_speaker_ids, source_names
    """
    # All sources process the same NPZ files in sorted order, so clip ordering
    # should match. Extract height targets from the first source's dataframe
    # (all sources should have identical height_cm values per clip).
    
    val_cols: List[np.ndarray] = []
    test_cols: List[np.ndarray] = []
    source_names: List[str] = []

    for vs, ts in zip(val_sources, test_sources):
        v_df = vs["df"]
        t_df = ts["df"]
        val_col = vs["column"]
        test_col = ts["column"]

        val_cols.append(v_df[val_col].values.astype(np.float32))
        test_cols.append(t_df[test_col].values.astype(np.float32))
        source_names.append(vs["name"])

    X_val = np.column_stack(val_cols)
    X_test = np.column_stack(test_cols)
    # Use clip-level height from the first source (all should match)
    y_val = val_sources[0]["df"]["height_cm"].to_numpy(dtype=np.float32).copy()
    y_test = test_sources[0]["df"]["height_cm"].to_numpy(dtype=np.float32).copy()
    val_speaker_ids = val_sources[0]["df"]["speaker_id"].to_numpy().copy()
    test_speaker_ids = test_sources[0]["df"]["speaker_id"].to_numpy().copy()

    return X_val, X_test, y_val, y_test, val_speaker_ids, test_speaker_ids, source_names


def clip_convex_blend(
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold: int,
    device: torch.device,
    probes: int,
    batch_size: int,
    train_idxs: np.ndarray,
    val_idxs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Learn convex blend weights on a subset of clips and apply to held-out clips.
    
    Returns:
        (oof_preds, best_weights_np, val_mae)
    """
    # Subset to training clips
    X_tr = torch.tensor(X_val[train_idxs], dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_val[train_idxs], dtype=torch.float32, device=device)
    X_held = torch.tensor(X_val[val_idxs], dtype=torch.float32, device=device)
    y_held = torch.tensor(y_val[val_idxs], dtype=torch.float32, device=device)

    n_sources = X_val.shape[1]
    best_score = float("inf")
    best_weights: Optional[torch.Tensor] = None

    generator = torch.Generator(device=device)
    generator.manual_seed(int(42 + fold))

    remaining = probes
    bs = min(batch_size, remaining)

    while remaining > 0:
        b = min(int(bs), remaining)
        remaining -= b
        raw = torch.rand((b, n_sources), device=device, generator=generator)
        weights = raw.pow(4.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        pred = weights @ X_tr.T
        err = pred - y_tr.view(1, -1)
        ae = torch.abs(err)
        mae = ae.mean(dim=1)
        p90 = torch.quantile(ae, 0.90, dim=1)
        bias = torch.abs(err.mean(dim=1))

        score = mae + 0.020 * p90 + 0.035 * bias
        value, arg = torch.min(score, dim=0)

        if float(value.item()) < best_score:
            best_score = float(value.item())
            best_weights = weights[int(arg.item())].detach().clone()

    if best_weights is None:
        # Fallback: uniform weights
        best_weights = torch.ones(n_sources, device=device) / n_sources

    # Apply to held-out
    held_pred = (best_weights @ X_held.T).detach().cpu().numpy().astype(np.float32)
    held_mae = float(np.mean(np.abs(held_pred - y_held.cpu().numpy())))

    return held_pred, best_weights.detach().cpu().numpy(), held_mae


def main() -> int:
    args = parse_args()

    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        print("[clip_ensemble] CUDA not available, using CPU (slower)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base
    val_base = read_phase3(args.phase3_val)
    test_base = read_phase3(args.phase3_test)
    val_y = val_base["height_cm"].to_numpy(dtype=np.float32)
    test_y = test_base["height_cm"].to_numpy(dtype=np.float32)

    # Load clip-level predictions from val and test directories
    outputs_root = ROOT / "outputs"
    all_sources = discover_clip_sources(outputs_root)
    val_sources = [s for s in all_sources if s["split"] == "val"]
    test_sources = [s for s in all_sources if s["split"] == "test"]

    # Pair val and test sources by directory name
    val_by_dir = {s["dir"]: s for s in val_sources}
    test_by_dir = {s["dir"]: s for s in test_sources}
    common_dirs = sorted(set(val_by_dir.keys()) & set(test_by_dir.keys()))

    paired_val = [val_by_dir[d] for d in common_dirs]
    paired_test = [test_by_dir[d] for d in common_dirs]

    if len(paired_val) < 2:
        print(f"[clip_ensemble] ERROR: only {len(paired_val)} clip-level sources found (need >= 2)")
        print("  Sources found:")
        for s in all_sources:
            print(f"    {s['dir']}/{s['split']}:{s['column']}")
        return 1

    print(f"[clip_ensemble] Found {len(paired_val)} paired clip-level sources:")
    for s in paired_val:
        print(f"  {s['name']}")

    # Build clip matrices (y targets from clip CSVs, not speaker base)
    X_val, X_test, y_val, y_test, val_sp_ids_clip, test_sp_ids_clip, source_names = build_clip_matrix(
        paired_val, paired_test,
    )
    n_clips = len(X_val)
    n_sources = X_val.shape[1]
    print(f"[clip_ensemble] Matrix: {n_clips} val clips x {n_sources} sources")
    print(f"[clip_ensemble] y_val: {len(y_val)} clips, y_test: {len(y_test)} clips")

    # NaN guard
    if np.any(np.isnan(X_val)) or np.any(np.isnan(X_test)):
        print("[clip_ensemble] ERROR: NaN found in clip prediction matrix!")
        print(f"  val NaN count: {np.isnan(X_val).sum()} / {X_val.size}")
        print(f"  test NaN count: {np.isnan(X_test).sum()} / {X_test.size}")
        return 1

    # ========== K-fold OOF blending over clips ==========
    n_folds = int(args.n_folds)
    indices = np.arange(n_clips)
    np.random.RandomState(int(args.seed)).shuffle(indices)

    fold_size = n_clips // n_folds
    oof_preds = np.zeros(n_clips, dtype=np.float32)
    fold_weights: List[Dict[str, Any]] = []

    print(f"[clip_ensemble] Running {n_folds}-fold CV over {n_clips} clips...")
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n_clips
        val_idxs = indices[start:end]
        train_idxs = np.setdiff1d(indices, val_idxs, assume_unique=True)

        held_pred, w, mae = clip_convex_blend(
            X_val, y_val, fold, device,
            int(args.blend_probes), int(args.blend_batch),
            train_idxs, val_idxs,
        )
        oof_preds[val_idxs] = held_pred
        fold_weights.append({
            "fold": fold,
            "n_train": int(len(train_idxs)),
            "n_val": int(len(val_idxs)),
            "val_mae": float(mae),
            "weights": {source_names[i]: float(w[i]) for i in range(n_sources) if abs(float(w[i])) > 1e-4},
        })
        print(f"  Fold {fold}: val_mae={mae:.4f}cm  weights={dict(zip(source_names, w.round(4)))}")

    oof_speaker_ids = val_sp_ids_clip

    # Speaker-level OOF metrics
    oof_df = pd.DataFrame({"speaker_id": oof_speaker_ids, "height_cm": y_val, "pred_cm": oof_preds})
    sp_grouped = oof_df.groupby("speaker_id").agg({"height_cm": "first", "pred_cm": "mean"})
    oof_sp_mae = float(np.mean(np.abs(sp_grouped["height_cm"] - sp_grouped["pred_cm"])))
    oof_clip_mae = float(np.mean(np.abs(oof_preds - y_val)))
    print(f"\n[clip_ensemble] OOF clip MAE: {oof_clip_mae:.4f}cm")
    print(f"[clip_ensemble] OOF speaker MAE: {oof_sp_mae:.4f}cm")

    # Save OOF clip predictions
    clip_oof_df = pd.DataFrame({
        "speaker_id": oof_speaker_ids,
        "height_cm": y_val,
        "pred_cm": oof_preds,
    })
    clip_oof_df.to_csv(output_dir / "oof_clip_predictions.csv", index=False)

    # ========== Train on ALL val clips for test ==========
    print(f"\n[clip_ensemble] Training full model on all {n_clips} val clips...")
    # Run convex search on ALL clips
    X_all = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_all = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    best_score = float("inf")
    best_weights: Optional[torch.Tensor] = None

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 999)

    remaining = int(args.blend_probes)
    bs = min(int(args.blend_batch), remaining)

    while remaining > 0:
        b = min(int(bs), remaining)
        remaining -= b
        raw = torch.rand((b, n_sources), device=device, generator=generator)
        weights = raw.pow(4.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        pred = weights @ X_all.T
        err = pred - y_all.view(1, -1)
        ae = torch.abs(err)
        mae = ae.mean(dim=1)
        p90 = torch.quantile(ae, 0.90, dim=1)
        bias = torch.abs(err.mean(dim=1))

        score = mae + 0.020 * p90 + 0.035 * bias
        value, arg = torch.min(score, dim=0)

        if float(value.item()) < best_score:
            best_score = float(value.item())
            best_weights = weights[int(arg.item())].detach().clone()

    if best_weights is None:
        best_weights = torch.ones(n_sources, device=device) / n_sources

    # Apply to test clips
    test_clip_preds = (best_weights @ X_test_t.T).detach().cpu().numpy().astype(np.float32)
    test_clip_mae = float(np.mean(np.abs(test_clip_preds - y_test)))
    print(f"[clip_ensemble] Full model test clip MAE: {test_clip_mae:.4f}cm")
    print(f"[clip_ensemble] Convex weights:")
    for i, name in enumerate(source_names):
        print(f"  {name}: {float(best_weights[i]):.4f}")

    # ========== Average to speaker level ==========
    # Val speakers
    val_sp_preds = oof_df.groupby("speaker_id")["pred_cm"].mean().to_dict()
    val_sp_ids = list(val_sp_preds.keys())
    val_sp_vals = [val_sp_preds[sid] for sid in val_sp_ids]
    # Map to base
    val_sp_map = val_base.set_index("speaker_id")["height_cm"].to_dict()
    val_sp_y = np.array([val_sp_map[sid] for sid in val_sp_ids], dtype=np.float32)
    val_sp_pred = np.array(val_sp_vals, dtype=np.float32)
    val_sp_mae = float(np.mean(np.abs(val_sp_pred - val_sp_y)))

    # Test speakers
    test_sp_ids = test_sp_ids_clip
    test_df = pd.DataFrame({"speaker_id": test_sp_ids, "height_cm": y_test, "pred_cm": test_clip_preds})
    test_sp_df = test_df.groupby("speaker_id")[["height_cm", "pred_cm"]].mean().reset_index()
    test_sp_mae = float(np.mean(np.abs(test_sp_df["height_cm"] - test_sp_df["pred_cm"])))

    print(f"\n{'=' * 60}")
    print(f"  SPEAKER-LEVEL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Val speaker MAE (OOF): {val_sp_mae:.4f}cm  (n={len(val_sp_ids)})")
    print(f"  Test speaker MAE:      {test_sp_mae:.4f}cm  (n={len(test_sp_df)})")
    print(f"{'=' * 60}")

    # ========== Save predictions for Phase22 ==========
    # Speaker-level predictions
    val_out = val_base[["speaker_id", "height_cm"]].copy()
    val_out["clip_ensemble_pred_cm"] = val_out["speaker_id"].map(val_sp_preds)

    test_out = test_base[["speaker_id", "height_cm"]].copy()
    test_pred_map = dict(zip(test_sp_df["speaker_id"], test_sp_df["pred_cm"]))
    missing_test = ~test_out["speaker_id"].isin(test_pred_map.keys())
    if missing_test.any():
        print(f"[clip_ensemble] WARNING: {missing_test.sum()} test speakers missing predictions!")
    test_out["clip_ensemble_pred_cm"] = test_out["speaker_id"].map(test_pred_map)

    val_out.to_csv(output_dir / "val_speaker_predictions.csv", index=False)
    test_out.to_csv(output_dir / "test_speaker_predictions.csv", index=False)
    print(f"[clip_ensemble] Saved speaker predictions to {output_dir}/")

    # Clip-level predictions
    clip_val_out = pd.DataFrame({
        "speaker_id": oof_speaker_ids,
        "height_cm": y_val,
        "clip_ensemble_pred_cm": oof_preds,
    })
    clip_test_out = pd.DataFrame({
        "speaker_id": test_sp_ids,
        "height_cm": y_test,
        "clip_ensemble_pred_cm": test_clip_preds,
    })
    clip_val_out.to_csv(output_dir / "val_clip_predictions.csv", index=False)
    clip_test_out.to_csv(output_dir / "test_clip_predictions.csv", index=False)
    print(f"[clip_ensemble] Saved clip predictions to {output_dir}/")

    # Metrics JSON
    report = {
        "val_clip_mae": float(oof_clip_mae),
        "val_speaker_mae": float(val_sp_mae),
        "test_clip_mae": float(test_clip_mae),
        "test_speaker_mae": float(test_sp_mae),
        "n_clips": int(n_clips),
        "n_sources": int(n_sources),
        "n_folds": int(n_folds),
        "fold_weights": fold_weights,
        "full_weights": {source_names[i]: float(best_weights[i]) for i in range(n_sources)},
        "source_names": source_names,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[clip_ensemble] Report saved to {output_dir}/report.json")

    # Quick per-height analysis
    val_sp_anal = oof_df.groupby("speaker_id").agg({"height_cm": "first", "pred_cm": "mean"}).reset_index()
    for sp_df, name in [(val_sp_anal, "val"), (test_sp_df, "test")]:
        err = sp_df["height_cm"] - sp_df["pred_cm"]
        short_mask = sp_df["height_cm"] < 162
        tall_mask = sp_df["height_cm"] >= 178
        med_mask = ~short_mask & ~tall_mask
        print(f"\n  {name.upper()} per-height:")
        print(f"    Short (<162): n={short_mask.sum():3d}  MAE={np.mean(np.abs(err[short_mask])):.4f}cm  bias={np.mean(err[short_mask]):+.4f}cm")
        print(f"    Medium:       n={med_mask.sum():3d}  MAE={np.mean(np.abs(err[med_mask])):.4f}cm")
        print(f"    Tall (>=178): n={tall_mask.sum():3d}  MAE={np.mean(np.abs(err[tall_mask])):.4f}cm  bias={np.mean(err[tall_mask]):+.4f}cm")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
