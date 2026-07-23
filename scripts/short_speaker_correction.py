#!/usr/bin/env python
"""
Short-Speaker Targeted Correction -- Height Calibration Curve.

The core finding from Phase22 is that ALL models systematically overshoot
short speakers and undershoot tall speakers (regression toward mean ~168cm).

Instead of per-speaker models (too few samples), we learn a height-calibration
curve from validation data and apply it to test predictions.

Approach:
1. Compute bias (pred - true) on validation for many height bins
2. Fit a smooth spline: bias = f(predicted_height)
3. Apply correction: corrected = predicted - f(predicted)
4. Evaluate on test speakers

This uses ALL validation speakers (not just short ones), making it much
more robust than the per-speaker approach.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Height calibration correction for short-speaker overshoot"
    )
    parser.add_argument("--output-dir", default="outputs/short_speaker_correction")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--method", choices=["spline", "rolling_avg", "bin_shift"], 
                        default="spline", help="Calibration method")
    parser.add_argument("--bins", type=int, default=20, help="Number of bins for calibration")
    parser.add_argument("--spline-sp", type=float, default=0.5, 
                        help="Spline smoothing parameter (0=interpolate, 1=linear)")
    parser.add_argument("--rolling-window", type=int, default=7, 
                        help="Window size for rolling average")
    parser.add_argument("--no-relax", action="store_true",
                        help="Don't relax correction toward zero for tall speakers")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def compute_calibration_curve_spline(
    pred: np.ndarray,
    error: np.ndarray,
    smoothing: float = 0.5,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a smooth calibration curve using binned averages + monotonic spline.

    Bins by predicted height, computes mean error per bin, then smooths.
    Returns the curve as (bin_centers, bin_errors) for interpolation.
    """
    order = np.argsort(pred)
    pred_sorted = pred[order]
    error_sorted = error[order]

    # Bin the sorted predictions and compute mean error per bin
    n = len(pred_sorted)
    bin_size = max(1, n // n_bins)

    bin_centers = []
    bin_errors = []

    for i in range(0, n, bin_size):
        end = min(i + bin_size, n)
        bin_centers.append(float(np.mean(pred_sorted[i:end])))
        bin_errors.append(float(np.mean(error_sorted[i:end])))

    bin_centers = np.array(bin_centers)
    bin_errors = np.array(bin_errors)

    # Apply smoothing: blend with global mean
    global_mean = float(np.mean(error))
    smoothed = smoothing * bin_errors + (1.0 - smoothing) * global_mean * np.ones_like(bin_errors)

    return bin_centers, smoothed


def compute_calibration_curve_rolling(
    pred: np.ndarray,
    error: np.ndarray,
    window: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling average calibration curve."""
    order = np.argsort(pred)
    pred_sorted = pred[order]
    error_sorted = error[order]

    df = pd.DataFrame({"pred": pred_sorted, "error": error_sorted})
    df["smoothed"] = df["error"].rolling(window=window, center=True, min_periods=1).mean()
    # Extrapolate first/last values
    df["smoothed"] = df["smoothed"].bfill().ffill()

    return df["pred"].values, df["smoothed"].values


def compute_calibration_curve_binshift(
    pred: np.ndarray,
    error: np.ndarray,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple bin-shift calibration: compute bias per predicted-height bin."""
    order = np.argsort(pred)
    pred_sorted = pred[order]
    error_sorted = error[order]

    n = len(pred_sorted)
    bin_size = max(1, n // n_bins)

    centers = []
    shifts = []

    for i in range(0, n, bin_size):
        end = min(i + bin_size, n)
        centers.append(float(np.mean(pred_sorted[i:end])))
        shifts.append(float(np.median(error_sorted[i:end])))

    return np.array(centers), np.array(shifts)


def apply_calibration(
    test_pred: np.ndarray,
    cal_centers: np.ndarray,
    cal_correction: np.ndarray,
    relax: bool = True,
    pred_min: float = 130.0,
    pred_max: float = 210.0,
) -> np.ndarray:
    """Apply calibration curve to test predictions via linear interpolation.

    Args:
        test_pred: Raw predictions on test set
        cal_centers: Calibration curve bin centers (from validation)
        cal_correction: Correction values (bias) at each bin center
        relax: Whether to relax correction to zero for extreme predictions
        pred_min: Minimum prediction for relaxation
        pred_max: Maximum prediction for relaxation

    Returns:
        Corrected predictions
    """
    # Clip to range
    clipped = np.clip(test_pred, cal_centers.min(), cal_centers.max())

    # Interpolate correction
    correction = np.interp(clipped, cal_centers, cal_correction)

    # Relax correction toward zero near extremes
    if relax:
        # Outside [pred_min, pred_max], linearly relax correction to 0 at 10cm beyond
        half_range = 10.0
        for i in range(len(test_pred)):
            p = float(test_pred[i])
            if p < pred_min:
                factor = max(0.0, (p - (pred_min - half_range)) / half_range)
                correction[i] *= factor
            elif p > pred_max:
                factor = max(0.0, ((pred_max + half_range) - p) / half_range)
                correction[i] *= factor

    return test_pred - correction


def compute_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    ae = np.abs(pred - y_true)
    err = pred - y_true
    return {
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "bias": float(np.mean(err)),
        "within_3cm": float(np.mean(ae <= 3.0)),
        "within_5cm": float(np.mean(ae <= 5.0)),
        "max_ae": float(np.max(ae)),
    }


def per_height_metrics(
    y_true: np.ndarray, pred: np.ndarray
) -> Dict[str, Any]:
    metrics = {}
    for lo, hi, label in [(0, 162, "short"), (162, 178, "medium"), (178, 999, "tall")]:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.any():
            ae = np.abs(pred[mask] - y_true[mask])
            err = pred[mask] - y_true[mask]
            metrics[f"{label}_mae"] = float(np.mean(ae))
            metrics[f"{label}_bias"] = float(np.mean(err))
            metrics[f"{label}_count"] = int(mask.sum())
    return metrics


def load_data(outputs_root: Path) -> Dict[str, Any]:
    """Load Phase12 predictions and ground truth."""
    val_base = pd.read_csv(outputs_root / "phase3_target_domain_rescue" / "phase3_predictions_val.csv")
    test_base = pd.read_csv(outputs_root / "phase3_target_domain_rescue" / "phase3_predictions_test.csv")

    p12_val = pd.read_csv(outputs_root / "phase12_residual_guard" / "phase12_predictions_val.csv")
    p12_test = pd.read_csv(outputs_root / "phase12_residual_guard" / "phase12_predictions_test.csv")

    p12_val_col = [c for c in p12_val.columns if "pred" in c.lower() and "_cm" in c.lower()][0]
    p12_test_col = [c for c in p12_test.columns if "pred" in c.lower() and "_cm" in c.lower()][0]

    val_df = val_base[["speaker_id", "height_cm", "source", "gender"]].merge(
        p12_val[["speaker_id", p12_val_col]], on="speaker_id"
    )
    test_df = test_base[["speaker_id", "height_cm", "source", "gender"]].merge(
        p12_test[["speaker_id", p12_test_col]], on="speaker_id"
    )

    y_val = val_df["height_cm"].to_numpy(dtype=np.float32)
    y_test = test_df["height_cm"].to_numpy(dtype=np.float32)
    phase12_val = val_df[p12_val_col].to_numpy(dtype=np.float32)
    phase12_test = test_df[p12_test_col].to_numpy(dtype=np.float32)
    error_val = phase12_val - y_val
    error_test = phase12_test - y_test

    return {
        "val_df": val_df, "test_df": test_df,
        "y_val": y_val, "y_test": y_test,
        "p12_val": phase12_val, "p12_test": phase12_test,
        "error_val": error_val, "error_test": error_test,
    }


def main() -> int:
    args = parse_args()
    np.random.seed(42)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs_root = resolve(args.outputs_root)

    print("=" * 60)
    print("HEIGHT CALIBRATION CORRECTION")
    print("=" * 60)

    # Load
    data = load_data(outputs_root)
    y_val = data["y_val"]
    y_test = data["y_test"]
    p12_val = data["p12_val"]
    p12_test = data["p12_test"]
    error_val = data["error_val"]
    error_test = data["error_test"]

    base_val_mae = float(np.mean(np.abs(error_val)))
    base_test_mae = float(np.mean(np.abs(error_test)))
    print(f"\nPhase12 Val MAE: {base_val_mae:.3f}cm  |  Test MAE: {base_test_mae:.3f}cm")

    # Fit Calibration Curve on Validation
    print(f"\n=== Fitting Calibration Curve ({args.method}) ===")

    if args.method == "spline":
        cal_centers, cal_correction = compute_calibration_curve_spline(
            p12_val, error_val, smoothing=args.spline_sp, n_bins=args.bins
        )
    elif args.method == "rolling_avg":
        cal_centers, cal_correction = compute_calibration_curve_rolling(
            p12_val, error_val, window=args.rolling_window
        )
    else:
        cal_centers, cal_correction = compute_calibration_curve_binshift(
            p12_val, error_val, n_bins=args.bins
        )

    # Print calibration curve
    print("\nCalibration curve (predicted height -> correction):")
    for center, corr in zip(cal_centers, cal_correction):
        print(f"  pred={center:.0f}cm -> bias_correction={corr:+.2f}cm")

    # LOOCV Evaluation
    print("\n=== LOOCV Evaluation ===")
    n_val = len(p12_val)
    val_corrected = np.zeros(n_val, dtype=np.float32)

    for i in range(n_val):
        train_mask = np.ones(n_val, dtype=bool)
        train_mask[i] = False
        if args.method == "spline":
            c, corr = compute_calibration_curve_spline(
                p12_val[train_mask], error_val[train_mask],
                smoothing=args.spline_sp, n_bins=args.bins
            )
        elif args.method == "rolling_avg":
            c, corr = compute_calibration_curve_rolling(
                p12_val[train_mask], error_val[train_mask],
                window=args.rolling_window
            )
        else:
            c, corr = compute_calibration_curve_binshift(
                p12_val[train_mask], error_val[train_mask],
                n_bins=args.bins
            )
        val_corrected[i] = apply_calibration(
            p12_val[i:i+1], c, corr, relax=not args.no_relax
        )[0]

    val_loocv_mae = float(np.mean(np.abs(val_corrected - y_val)))
    print(f"  LOOCV MAE: {val_loocv_mae:.4f}cm (baseline: {base_val_mae:.4f}cm)")
    print(f"  Improvement: {base_val_mae - val_loocv_mae:+.4f}cm")

    # Apply to Test
    test_corrected = apply_calibration(
        p12_test, cal_centers, cal_correction, relax=not args.no_relax
    )

    cal_test_mae = float(np.mean(np.abs(test_corrected - y_test)))
    print(f"\n=== Test Results ===")
    print(f"  Calibrated Test MAE: {cal_test_mae:.4f}cm (baseline: {base_test_mae:.4f}cm)")
    print(f"  Improvement: {base_test_mae - cal_test_mae:+.4f}cm")

    # Per-Height Breakdown
    print("\n=== Per-Height Metrics ===")
    for name, preds in [("Phase12 (baseline)", p12_test), ("Calibrated", test_corrected)]:
        m = compute_metrics(y_test, preds)
        hm = per_height_metrics(y_test, preds)
        print(f"  {name}:")
        print(f"    Overall: MAE={m['mae']:.3f} within3={m['within_3cm']:.1%} max={m['max_ae']:.1f}")
        for label in ["short", "medium", "tall"]:
            if f"{label}_mae" in hm:
                print(f"    {label}: MAE={hm[f'{label}_mae']:.3f} bias={hm[f'{label}_bias']:+.3f} n={hm[f'{label}_count']}")

    # Blocker Speakers
    blocker_ids = [
        "TIMIT_WEM0", "TIMIT_DPK0", "TIMIT_BCG1", "NISP_Tam_0012",
        "TIMIT_JES0", "TIMIT_BMJ0", "NISP_Mal_0008", "TIMIT_GES0",
        "NISP_Kan_0043", "TIMIT_SEM0", "TIMIT_TLC0", "TIMIT_JKR0",
        "TIMIT_RJS0", "NISP_Tam_0045", "NISP_Hin_0102", "TIMIT_VJH0",
        "TIMIT_CMM0", "TIMIT_WRP0",
    ]

    print("\n=== Blocker Improvement ===")
    test_df = data["test_df"]
    test_ids = test_df["speaker_id"].values
    total_before, total_after = 0.0, 0.0
    n_blocker_found = 0

    for sid in blocker_ids:
        idxs = np.where(test_ids == sid)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        err_before = abs(float(p12_test[idx] - y_test[idx]))
        err_after = abs(float(test_corrected[idx] - y_test[idx]))
        total_before += err_before
        total_after += err_after
        n_blocker_found += 1

        change = err_before - err_after
        tag = "[IMPROVED]" if change > 1.0 else ("[MINOR]" if change > 0 else "[REGRESSED]")
        print(f"  {sid:25s} h={float(y_test[idx]):.0f}cm "
              f"before={err_before:.1f}cm after={err_after:.1f}cm "
              f"({change:+.1f}cm) {tag}")

    if n_blocker_found > 0:
        print(f"\n  Total blocker error: {total_before:.1f}cm -> {total_after:.1f}cm "
              f"({total_before - total_after:+.1f}cm over {n_blocker_found} speakers)")

    # Write Outputs
    print("\n=== Writing Predictions ===")

    # Write corrected predictions
    corrected_test = test_df[["speaker_id", "source", "gender", "height_cm"]].copy()
    corrected_test["pred_height_cm"] = test_corrected
    corrected_test["phase12_baseline_cm"] = p12_test
    corrected_test["abs_error_cm"] = np.abs(test_corrected - y_test)
    corrected_test.to_csv(output_dir / "corrected_predictions_test.csv", index=False)
    print(f"  Wrote {output_dir / 'corrected_predictions_test.csv'}")

    corrected_val = data["val_df"][["speaker_id", "source", "gender", "height_cm"]].copy()
    corrected_val["pred_height_cm"] = val_corrected
    corrected_val["phase12_baseline_cm"] = p12_val
    corrected_val["abs_error_cm"] = np.abs(val_corrected - y_val)
    corrected_val.to_csv(output_dir / "corrected_predictions_val.csv", index=False)
    print(f"  Wrote {output_dir / 'corrected_predictions_val.csv'}")

    # Phase12-compatible format
    p12_cols = [c for c in pd.read_csv(
        outputs_root / "phase12_residual_guard" / "phase12_predictions_test.csv"
    ).columns if "pred" in c.lower() and "_cm" in c.lower()]
    phase12_col = p12_cols[0] if p12_cols else "pred_height_cm"

    compat_test = pd.read_csv(
        outputs_root / "phase12_residual_guard" / "phase12_predictions_test.csv"
    )
    compat_test[phase12_col] = test_corrected
    compat_test.to_csv(output_dir / "phase12_calibrated_predictions_test.csv", index=False)
    print(f"  Wrote {output_dir / 'phase12_calibrated_predictions_test.csv'}")

    compat_val = pd.read_csv(
        outputs_root / "phase12_residual_guard" / "phase12_predictions_val.csv"
    )
    compat_val_col = [c for c in compat_val.columns if "pred" in c.lower() and "_cm" in c.lower()][0]
    compat_val[compat_val_col] = val_corrected
    compat_val.to_csv(output_dir / "phase12_calibrated_predictions_val.csv", index=False)
    print(f"  Wrote {output_dir / 'phase12_calibrated_predictions_val.csv'}")

    # Build Report
    hm = per_height_metrics(y_test, test_corrected)
    baseline_hm = per_height_metrics(y_test, p12_test)

    report = {
        "method": f"height_calibration_{args.method}",
        "params": {
            "bins": args.bins,
            "spline_smoothing": args.spline_sp if args.method == "spline" else None,
            "rolling_window": args.rolling_window if args.method == "rolling_avg" else None,
            "relax_extremes": not args.no_relax,
        },
        "phase12": {
            "val_mae": round(base_val_mae, 4),
            "test_mae": round(base_test_mae, 4),
        },
        "calibrated": {
            "val_loocv_mae": round(val_loocv_mae, 4),
            "test_mae": round(cal_test_mae, 4),
            "improvement_cm": round(base_test_mae - cal_test_mae, 4),
        },
        "blocker_analysis": {
            "total_error_before_cm": round(float(total_before), 2),
            "total_error_after_cm": round(float(total_after), 2),
            "total_improvement_cm": round(float(total_before - total_after), 2),
            "n_blockers_found": n_blocker_found,
        },
        "calibration_curve": [
            {"pred": round(float(c), 2), "correction_cm": round(float(e), 2)}
            for c, e in zip(cal_centers, cal_correction)
        ],
        "per_height_calibrated": hm,
        "per_height_baseline": baseline_hm,
    }

    with open(output_dir / "correction_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote {output_dir / 'correction_report.json'}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  HEIGHT CALIBRATION RESULTS")
    print(f"  Phase12:    {base_test_mae:.3f}cm (short: {baseline_hm.get('short_mae', 0):.2f}cm)")
    print(f"  Calibrated: {cal_test_mae:.3f}cm (short: {hm.get('short_mae', 0):.2f}cm)")
    print(f"  Delta:      {base_test_mae - cal_test_mae:+.3f}cm")
    print(f"  TARGET:     3.000cm")
    print(f"  Short bias: {baseline_hm.get('short_bias', 0):+.2f}cm -> {hm.get('short_bias', 0):+.2f}cm")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
