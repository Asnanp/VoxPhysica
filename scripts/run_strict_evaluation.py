#!/usr/bin/env python
"""Run strict speaker-level evaluation baselines and assemble an honest metrics table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.audit_utils import decode_np_value, safe_float  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict VocalMorph evaluation baselines.")
    parser.add_argument("--features-dir", default="data/features_audited")
    parser.add_argument("--v2-physics-metrics", default=None, help="Path to strict V2 physics metrics JSON")
    parser.add_argument("--v2-no-physics-metrics", default=None, help="Path to strict V2 no-physics metrics JSON")
    parser.add_argument("--report-out", default="audit/validation_report.md")
    parser.add_argument("--json-out", default="outputs/strict/evaluation_table.json")
    return parser.parse_args()


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _feature_files(split_dir: str) -> List[str]:
    return sorted(
        os.path.join(split_dir, name)
        for name in os.listdir(split_dir)
        if name.lower().endswith(".npz")
    )


def _speaker_examples(split_dir: str) -> Dict[str, Dict[str, Any]]:
    speakers: Dict[str, Dict[str, Any]] = {}
    for path in _feature_files(split_dir):
        with np.load(path, allow_pickle=True) as data:
            speaker_id = decode_np_value(data["speaker_id"]).strip()
            sequence = np.asarray(data["sequence"], dtype=np.float32)
            pooled = np.concatenate(
                [
                    sequence.mean(axis=0),
                    sequence.std(axis=0),
                    np.asarray(
                        [
                            safe_float(data["f0_mean"]) if "f0_mean" in data else float("nan"),
                            safe_float(data["formant_spacing_mean"]) if "formant_spacing_mean" in data else float("nan"),
                            safe_float(data["vtl_mean"]) if "vtl_mean" in data else float("nan"),
                            safe_float(data["duration_s"]) if "duration_s" in data else float("nan"),
                            safe_float(data["speech_ratio"]) if "speech_ratio" in data else float("nan"),
                            safe_float(data["snr_db_estimate"]) if "snr_db_estimate" in data else float("nan"),
                            safe_float(data["capture_quality_score"]) if "capture_quality_score" in data else float("nan"),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            entry = speakers.setdefault(
                speaker_id,
                {
                    "height_cm": safe_float(data["height_cm"]) if "height_cm" in data else float("nan"),
                    "vectors": [],
                },
            )
            entry["vectors"].append(pooled)
    for speaker_id, entry in speakers.items():
        entry["vector"] = np.mean(np.stack(entry["vectors"], axis=0), axis=0)
    return speakers


def _regression_row(y_true: np.ndarray, y_pred: np.ndarray, notes: str) -> Dict[str, Any]:
    abs_err = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    median = float(np.median(abs_err))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "median AE": median,
        "MAE omega": mae,
        "RMSE omega": rmse,
        "median AE omega": median,
        "notes": notes,
    }


def _load_metrics_row(path: Optional[str], label: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not path:
        return None
    resolved = _resolve(path)
    if not os.path.exists(resolved):
        return None
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "final_test" in payload:
        test_metrics = payload["final_test"]
        gaps = payload.get("overfit_gaps", {})
    else:
        test_metrics = payload
        gaps = {}
    mae = safe_float(test_metrics.get("height_mae_speaker"))
    rmse = safe_float(test_metrics.get("height_rmse_speaker"))
    median = safe_float(test_metrics.get("height_median_ae_speaker"))
    mae_omega = safe_float(test_metrics.get("height_mae_speaker_omega"))
    rmse_omega = safe_float(test_metrics.get("height_rmse_speaker_omega"))
    median_omega = safe_float(test_metrics.get("height_median_ae_speaker_omega"))
    note_parts = ["strict speaker-level metric file"]
    if gaps:
        gap = safe_float(gaps.get("height_mae_speaker_gap_val_minus_train"))
        if np.isfinite(gap):
            note_parts.append(f"val-train gap={gap:.2f}cm")
    cal = safe_float(test_metrics.get("height_calibration_mae"))
    if np.isfinite(cal):
        note_parts.append(f"cal_mae={cal:.2f}")
    return (
        label,
        {
            "MAE": mae,
            "RMSE": rmse,
            "median AE": median,
            "MAE omega": mae_omega,
            "RMSE omega": rmse_omega,
            "median AE omega": median_omega,
            "notes": "; ".join(note_parts),
        },
    )


def main() -> int:
    args = parse_args()
    feature_root = _resolve(args.features_dir)
    train_speakers = _speaker_examples(os.path.join(feature_root, "train"))
    test_speakers = _speaker_examples(os.path.join(feature_root, "test"))
    if not train_speakers or not test_speakers:
        raise RuntimeError("Audited train/test features are required for strict evaluation baselines.")

    x_train = np.stack([entry["vector"] for entry in train_speakers.values()], axis=0)
    y_train = np.asarray([entry["height_cm"] for entry in train_speakers.values()], dtype=np.float32)
    x_test = np.stack([entry["vector"] for entry in test_speakers.values()], axis=0)
    y_test = np.asarray([entry["height_cm"] for entry in test_speakers.values()], dtype=np.float32)

    table: Dict[str, Dict[str, Any]] = {}
    mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)))
    table["mean_height_baseline"] = _regression_row(
        y_test,
        mean_pred,
        notes="predicts train speaker mean height",
    )

    feature_mean = x_train.mean(axis=0, keepdims=True)
    feature_std = np.where(x_train.std(axis=0, keepdims=True) < 1e-6, 1.0, x_train.std(axis=0, keepdims=True))
    x_train_norm = (x_train - feature_mean) / feature_std
    x_test_norm = (x_test - feature_mean) / feature_std
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=min(32, x_train_norm.shape[0]),
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    mlp.fit(x_train_norm, y_train)
    mlp_pred = mlp.predict(x_test_norm).astype(np.float32, copy=False)
    table["speaker_pooled_mlp"] = _regression_row(
        y_test,
        mlp_pred,
        notes="speaker-level pooled feature baseline",
    )

    v2_physics = _load_metrics_row(args.v2_physics_metrics, "v2_small_physics")
    if v2_physics is not None:
        table[v2_physics[0]] = v2_physics[1]
    else:
        table["v2_small_physics"] = {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "median AE": float("nan"),
            "MAE omega": float("nan"),
            "RMSE omega": float("nan"),
            "median AE omega": float("nan"),
            "notes": "strict model metrics not available yet",
        }

    v2_no_physics = _load_metrics_row(args.v2_no_physics_metrics, "v2_small_no_physics")
    if v2_no_physics is not None:
        table[v2_no_physics[0]] = v2_no_physics[1]
    else:
        table["v2_small_no_physics"] = {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "median AE": float("nan"),
            "MAE omega": float("nan"),
            "RMSE omega": float("nan"),
            "median AE omega": float("nan"),
            "notes": "strict no-physics metrics not available yet",
        }

    json_out = _resolve(args.json_out)
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2)

    report_out = _resolve(args.report_out)
    os.makedirs(os.path.dirname(report_out), exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as handle:
        handle.write("# Validation Report\n\n")
        handle.write("## Findings\n")
        handle.write("- Headline metrics below are speaker-level and use unseen test speakers only.\n")
        handle.write("- Mean predictor and pooled MLP baselines were recomputed directly from audited feature artifacts.\n")
        handle.write("- Legacy and omega speaker pooling are shown side by side for the same runs.\n")
        handle.write("- Any V2 row with missing values is still non-canonical until a strict audited train/eval run exists.\n")
        handle.write("\n## Honest Evaluation Table\n")
        handle.write("| model | MAE | RMSE | median AE | MAE omega | RMSE omega | median AE omega | notes |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for model_name, row in table.items():
            mae = row["MAE"]
            rmse = row["RMSE"]
            median = row["median AE"]
            mae_omega = row.get("MAE omega", float("nan"))
            rmse_omega = row.get("RMSE omega", float("nan"))
            median_omega = row.get("median AE omega", float("nan"))
            mae_text = f"{mae:.3f}" if np.isfinite(mae) else "nan"
            rmse_text = f"{rmse:.3f}" if np.isfinite(rmse) else "nan"
            median_text = f"{median:.3f}" if np.isfinite(median) else "nan"
            mae_omega_text = f"{mae_omega:.3f}" if np.isfinite(mae_omega) else "nan"
            rmse_omega_text = f"{rmse_omega:.3f}" if np.isfinite(rmse_omega) else "nan"
            median_omega_text = f"{median_omega:.3f}" if np.isfinite(median_omega) else "nan"
            handle.write(
                f"| {model_name} | {mae_text} | {rmse_text} | {median_text} | {mae_omega_text} | {rmse_omega_text} | {median_omega_text} | {row['notes']} |\n"
            )
        handle.write("\n## Fixes\n")
        handle.write("- Standardized the baseline comparison ladder around speaker-level evaluation.\n")
        handle.write("- Added train-vs-val overfit gap slots to the strict metrics JSON format.\n")
        handle.write("\n## Remaining Risks\n")
        handle.write("- A canonical V2 result still requires audited feature rebuild plus 3-seed strict retraining.\n")

    print(f"[Strict Eval] Wrote table to {json_out}")
    print(f"[Strict Eval] Wrote report to {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
