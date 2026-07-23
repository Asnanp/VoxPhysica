#!/usr/bin/env python
"""Post-hoc blend of existing speaker-level predictions toward 3 cm MAE.

Loads val and test speaker predictions from several finished runs, fits a
non-negative simplex blend on val (sum to 1, no leakage) plus optional
isotonic recalibration on val, and reports test speaker MAE.

This does NOT retrain any model. It only re-mixes cached predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# (label, val_csv, val_col, test_csv, test_col)
SOURCES: List[Tuple[str, Path, str, Path, str]] = [
    (
        "research_3cm_edge",
        OUT / "research_3cm_ensemble/predictions_val.csv",
        "calibrated_edge_pred_cm",
        OUT / "research_3cm_ensemble/predictions_test.csv",
        "calibrated_edge_pred_cm",
    ),
    (
        "research_direct_edge",
        OUT / "research_direct/predictions_val.csv",
        "calibrated_edge_pred_cm",
        OUT / "research_direct/predictions_test.csv",
        "calibrated_edge_pred_cm",
    ),
    (
        "stacking_meta",
        OUT / "stacking_meta_ensemble/val_speaker_predictions.csv",
        "stacking_oof_pred_cm",
        OUT / "stacking_meta_ensemble/test_speaker_predictions.csv",
        "stacking_pred_cm",
    ),
    (
        "v5_3cm_var_weighted",
        OUT / "v5_3cm_architecture_predictions/val_speaker_predictions.csv",
        "v5_var_weighted_cm",
        OUT / "v5_3cm_architecture_predictions/test_speaker_predictions.csv",
        "v5_var_weighted_cm",
    ),
    (
        "v5_1_direct",
        OUT / "v5_1_direct_3cm_predictions/val_speaker_predictions.csv",
        "height_pred_cm",
        OUT / "v5_1_direct_3cm_predictions/test_speaker_predictions.csv",
        "height_pred_cm",
    ),
    (
        "short_speaker_corrected",
        OUT / "short_speaker_correction/corrected_predictions_val.csv",
        "pred_height_cm",
        OUT / "short_speaker_correction/corrected_predictions_test.csv",
        "pred_height_cm",
    ),
    (
        "speaker_gpu_combo",
        OUT / "speaker_gpu_combo_full_ssl_cuda/predictions_val.csv",
        "pred_cm",
        OUT / "speaker_gpu_combo_full_ssl_cuda/predictions_test.csv",
        "pred_cm",
    ),
    (
        "speaker_gpu_target_ssl",
        OUT / "speaker_gpu_target_ssl_cuda/predictions_val.csv",
        "pred_cm",
        OUT / "speaker_gpu_target_ssl_cuda/predictions_test.csv",
        "pred_cm",
    ),
]


def _load(csv: Path, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    if "speaker_id" not in df.columns:
        raise ValueError(f"{csv} missing speaker_id")
    if pred_col not in df.columns:
        raise ValueError(f"{csv} missing {pred_col}; has {list(df.columns)}")
    keep = ["speaker_id", pred_col]
    if "height_cm" in df.columns:
        keep.append("height_cm")
    return df[keep].rename(columns={pred_col: "pred"})


def assemble(split: str) -> Tuple[pd.DataFrame, List[str]]:
    base: pd.DataFrame | None = None
    labels: List[str] = []
    for label, val_csv, val_col, test_csv, test_col in SOURCES:
        csv = val_csv if split == "val" else test_csv
        col = val_col if split == "val" else test_col
        if not csv.exists():
            print(f"[skip] {label}: missing {csv}")
            continue
        df = _load(csv, col).rename(columns={"pred": label})
        labels.append(label)
        if base is None:
            base = df
        else:
            cols = ["speaker_id", label]
            if "height_cm" not in base.columns and "height_cm" in df.columns:
                cols.append("height_cm")
            base = base.merge(df[cols], on="speaker_id", how="inner")
    if base is None:
        raise RuntimeError("no sources loaded")
    base = base.dropna(subset=labels + ["height_cm"]).reset_index(drop=True)
    return base, labels


def fit_simplex(P: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Simplex blend minimizing mean abs error plus L2 spread penalty.

    Small `ridge > 0` discourages corner solutions when val n is small.
    """
    n_src = P.shape[1]
    x0 = np.full(n_src, 1.0 / n_src)
    uniform = x0.copy()

    def loss(w: np.ndarray) -> float:
        pred = P @ w
        mae_loss = float(np.mean(np.abs(pred - y)))
        spread = float(np.sum((w - uniform) ** 2))
        return mae_loss + ridge * spread

    bounds = [(0.0, 1.0)] * n_src
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return w


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def within(pred: np.ndarray, y: np.ndarray, k: float) -> float:
    return float(np.mean(np.abs(pred - y) <= k))


def isotonic_fit_apply(p_val, y_val, p_test):
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return iso.transform(p_val), iso.transform(p_test)


LEAKY = {"stacking_meta"}  # val column is leaky / not honest OOF


def main() -> None:
    val_df, val_labels = assemble("val")
    test_df, test_labels = assemble("test")
    common = [l for l in val_labels if l in test_labels and l not in LEAKY]
    print(f"Sources used ({len(common)}):")
    for l in common:
        print(f"  - {l}")

    val_df = val_df.dropna(subset=common + ["height_cm"])
    test_df = test_df.dropna(subset=common + ["height_cm"])

    Pv = val_df[common].to_numpy(dtype=np.float64)
    yv = val_df["height_cm"].to_numpy(dtype=np.float64)
    Pt = test_df[common].to_numpy(dtype=np.float64)
    yt = test_df["height_cm"].to_numpy(dtype=np.float64)

    print(f"\nVal n={len(yv)}, Test n={len(yt)}")

    print("\nPer-source MAE:")
    print(f"  {'source':30s}  val      test")
    for i, l in enumerate(common):
        print(f"  {l:30s}  {mae(Pv[:, i], yv):6.3f}  {mae(Pt[:, i], yt):6.3f}")

    w = fit_simplex(Pv, yv, ridge=0.5)
    print("\nFitted simplex weights (val-MAE + L2 spread):")
    for l, wi in zip(common, w):
        print(f"  {l:30s}  {wi:.4f}")

    val_blend = Pv @ w
    test_blend = Pt @ w
    print(f"\nBlend val  MAE: {mae(val_blend, yv):.4f}")
    print(f"Blend test MAE: {mae(test_blend, yt):.4f}")
    print(f"Blend val  within 3cm: {100 * within(val_blend, yv, 3):.1f}%")
    print(f"Blend test within 3cm: {100 * within(test_blend, yt, 3):.1f}%")

    val_iso, test_iso = isotonic_fit_apply(val_blend, yv, test_blend)
    print(f"\nBlend+isotonic val  MAE: {mae(val_iso, yv):.4f}")
    print(f"Blend+isotonic test MAE: {mae(test_iso, yt):.4f}")
    print(f"Blend+isotonic val  within 3cm: {100 * within(val_iso, yv, 3):.1f}%")
    print(f"Blend+isotonic test within 3cm: {100 * within(test_iso, yt, 3):.1f}%")

    out_dir = OUT / "blend_3cm_posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_df = test_df.copy()
    test_df["blend_pred_cm"] = test_blend
    test_df["blend_iso_pred_cm"] = test_iso
    test_df["blend_abs_error_cm"] = np.abs(test_blend - yt)
    test_df["blend_iso_abs_error_cm"] = np.abs(test_iso - yt)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    val_df = val_df.copy()
    val_df["blend_pred_cm"] = val_blend
    val_df["blend_iso_pred_cm"] = val_iso
    val_df.to_csv(out_dir / "val_predictions.csv", index=False)

    summary = {
        "sources": common,
        "weights": {l: float(wi) for l, wi in zip(common, w)},
        "val": {
            "n": int(len(yv)),
            "blend_mae_cm": mae(val_blend, yv),
            "blend_iso_mae_cm": mae(val_iso, yv),
            "blend_within_3cm": within(val_blend, yv, 3),
            "blend_iso_within_3cm": within(val_iso, yv, 3),
        },
        "test": {
            "n": int(len(yt)),
            "blend_mae_cm": mae(test_blend, yt),
            "blend_iso_mae_cm": mae(test_iso, yt),
            "blend_within_3cm": within(test_blend, yt, 3),
            "blend_iso_within_3cm": within(test_iso, yt, 3),
        },
        "per_source_mae": {
            l: {"val": mae(Pv[:, i], yv), "test": mae(Pt[:, i], yt)}
            for i, l in enumerate(common)
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
