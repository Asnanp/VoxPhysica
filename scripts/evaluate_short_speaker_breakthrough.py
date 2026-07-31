"""Evaluate short speaker (< 160 cm) breakthrough performance across VoxPhysica predictions.

Computes detailed error metrics, subgroup breakdowns, within-3cm/4cm ratios, and comparison
against the 9.410 cm short-speaker baseline.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def read_prediction_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "speaker_id": str(r["speaker_id"]),
                "source": str(r["source"]),
                "gender": int(r["gender"]),
                "target_cm": float(r["height_cm"]),
                "pred_cm": float(r["pred_height_cm"]),
                "abs_error_cm": float(r["abs_error_cm"]),
            })
    return rows

def compute_slice_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = np.array([r["target_cm"] for r in rows])
    preds = np.array([r["pred_cm"] for r in rows])
    errors = np.abs(targets - preds)
    
    short_mask = targets < 160.0
    medium_mask = (targets >= 160.0) & (targets < 175.0)
    tall_mask = targets >= 175.0
    
    short_males = (targets < 160.0) & (np.array([r["gender"] for r in rows]) == 1)
    short_females = (targets < 160.0) & (np.array([r["gender"] for r in rows]) == 0)
    
    def metrics_for_mask(mask):
        if not np.any(mask):
            return {"n": 0, "mae_cm": float("nan"), "median_cm": float("nan"), "within_3cm_pct": 0.0, "within_4cm_pct": 0.0}
        sub_err = errors[mask]
        return {
            "n": int(np.sum(mask)),
            "mae_cm": float(np.mean(sub_err)),
            "median_cm": float(np.median(sub_err)),
            "within_3cm_pct": float(np.mean(sub_err <= 3.0) * 100.0),
            "within_4cm_pct": float(np.mean(sub_err <= 4.0) * 100.0),
        }
        
    return {
        "all": metrics_for_mask(np.ones(len(rows), dtype=bool)),
        "short_lt160": metrics_for_mask(short_mask),
        "medium_160_175": metrics_for_mask(medium_mask),
        "tall_ge175": metrics_for_mask(tall_mask),
        "short_males": metrics_for_mask(short_males),
        "short_females": metrics_for_mask(short_females),
    }

def generate_report(output_dir: Path, metrics: dict[str, Any]) -> str:
    lines = [
        "# VoxPhysica Short-Speaker (< 160 cm) MAE Audit & Verification Report",
        "",
        f"Generated: {output_dir}",
        "",
        "## Subgroup Performance Summary",
        "",
        "| Subgroup Slice | Speaker Count | MAE (cm) | Median Error (cm) | Within 3.0 cm % | Within 4.0 cm % |",
        "|----------------|---------------|----------|-------------------|-----------------|-----------------|",
    ]
    
    for slice_key, title in [
        ("all", "All Test Speakers"),
        ("short_lt160", "Short (< 160 cm)"),
        ("medium_160_175", "Medium (160 - 175 cm)"),
        ("tall_ge175", "Tall (>= 175 cm)"),
        ("short_males", "Short Males (< 160 cm)"),
        ("short_females", "Short Females (< 160 cm)"),
    ]:
        m = metrics.get(slice_key, {})
        lines.append(
            f"| **{title}** | {m.get('n', 0)} | **{m.get('mae_cm', 0.0):.3f} cm** | {m.get('median_cm', 0.0):.3f} cm | {m.get('within_3cm_pct', 0.0):.1f}% | {m.get('within_4cm_pct', 0.0):.1f}% |"
        )
        
    lines.extend([
        "",
        "## Short Speaker Baseline Comparison",
        "",
        f"- **Previous Baseline Short MAE**: `9.410 cm`",
        f"- **Optimized Short MAE**: `{metrics['short_lt160']['mae_cm']:.3f} cm`",
        f"- **Improvement Delta**: `{9.410 - metrics['short_lt160']['mae_cm']:+.3f} cm`",
        "",
        "---",
    ])
    
    report_text = "\n".join(lines)
    report_path = output_dir / "SHORT_SPEAKER_BREAKTHROUGH_REPORT.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return report_text

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", type=Path, default=ROOT / "outputs/strict_3cm_short_opt/test_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/strict_3cm_short_opt")
    args = parser.parse_args()
    
    rows = read_prediction_csv(args.pred_csv)
    metrics = compute_slice_breakdown(rows)
    report = generate_report(args.output_dir, metrics)
    print(report)

if __name__ == "__main__":
    main()
