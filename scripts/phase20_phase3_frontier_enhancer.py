#!/usr/bin/env python
"""Phase 20 Phase3 frontier enhancer.

This script enhances the Phase 3 final prediction head directly. It keeps
Phase3 as the anchor, then searches low-freedom validation-only transforms:

- guarded balanced mode: only deploys a global tweak if validation gain is real
- short-primary mode: targeted sigmoid lowering for likely-short speakers
- optional blends with Phase3 candidate columns and later Phase18/19 heads

The sealed test set is reported after selection. It is not used to choose the
balanced or short-primary method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


EXTERNAL_HELPERS: Tuple[Tuple[str, str, str, str], ...] = (
    ("phase18_balanced", "outputs/phase18_oof_short_rescue/phase18_predictions_oof_dev.csv", "outputs/phase18_oof_short_rescue/phase18_predictions_test.csv", "phase18_pred_cm"),
    ("phase18_short_primary", "outputs/phase18_oof_short_rescue/phase18_short_primary_predictions_oof_dev.csv", "outputs/phase18_oof_short_rescue/phase18_short_primary_predictions_test.csv", "phase18_pred_cm"),
    ("phase19_moe", "outputs/phase19_cuda_moe_residual_bagger/phase19_predictions_oof_dev.csv", "outputs/phase19_cuda_moe_residual_bagger/phase19_predictions_test.csv", "phase19_pred_cm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance Phase3 final predictions.")
    parser.add_argument("--phase3-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase20_phase3_frontier_enhancer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mae-cm", type=float, default=3.0)
    parser.add_argument("--balanced-min-val-gain", type=float, default=0.05)
    parser.add_argument("--short-cm", type=float, default=160.0)
    parser.add_argument("--tall-cm", type=float, default=175.0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sid(row: Mapping[str, Any]) -> str:
    return str(row.get("speaker_id", "")).strip()


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_external(path: Path, column: str) -> Dict[str, float]:
    if not path.exists():
        return {}
    out: Dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = sid(row)
            if key and column in row and str(row[column]).strip():
                out[key] = float(row[column])
    return out


def base_meta(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "speaker_id": sid(row),
                "source": str(row.get("source", "UNKNOWN")),
                "gender": int(float(row.get("gender", 0))),
                "height_cm": float(row["height_cm"]),
            }
        )
    return out


def tensor(values: Sequence[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float32, device=device)


def phase3_columns(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    first = rows[0]
    cols = []
    for key in first:
        if key.endswith("_pred_cm") and key not in {"final_pred_cm"}:
            cols.append(key)
    return cols


def align_external(rows: Sequence[Mapping[str, Any]], preds: Mapping[str, float]) -> Optional[List[float]]:
    values = []
    for row in rows:
        key = sid(row)
        if key not in preds:
            return None
        values.append(float(preds[key]))
    return values


def metrics(y: torch.Tensor, pred: torch.Tensor, meta: Sequence[Mapping[str, Any]], short_cm: float, tall_cm: float) -> Dict[str, float]:
    err = pred - y
    ae = err.abs()
    out = {
        "mae": float(ae.mean().item()),
        "rmse": float(torch.sqrt((err * err).mean()).item()),
        "median_ae": float(ae.median().item()),
        "p90_ae": float(torch.quantile(ae, 0.90).item()),
        "bias": float(err.mean().item()),
        "within_3cm": float((ae <= 3.0).float().mean().item()),
        "within_5cm": float((ae <= 5.0).float().mean().item()),
        "count": float(y.numel()),
    }
    bins = torch.where(y < float(short_cm), torch.zeros_like(y, dtype=torch.long), torch.where(y < float(tall_cm), torch.ones_like(y, dtype=torch.long), torch.full_like(y, 2, dtype=torch.long)))
    for label, idx in (("short", 0), ("medium", 1), ("tall", 2)):
        mask = bins == idx
        if bool(mask.any()):
            out[f"{label}_mae"] = float(ae[mask].mean().item())
            out[f"{label}_bias"] = float(err[mask].mean().item())
            out[f"{label}_n"] = float(mask.sum().item())
    for source in sorted({str(row.get("source", "UNKNOWN")) for row in meta}):
        mask = torch.tensor([str(row.get("source", "UNKNOWN")) == source for row in meta], dtype=torch.bool, device=y.device)
        if bool(mask.any()):
            out[f"source_{source.lower()}_mae"] = float(ae[mask].mean().item())
    return out


def balanced_score(m: Mapping[str, float]) -> float:
    return float(m["mae"]) + 0.04 * float(m["p90_ae"]) + 0.04 * max(0.0, float(m.get("short_mae", m["mae"])) - float(m["mae"]))


def short_score(m: Mapping[str, float], base_mae: float) -> float:
    mae = float(m["mae"])
    short = float(m.get("short_mae", mae))
    p90 = float(m["p90_ae"])
    short_bias = abs(float(m.get("short_bias", m.get("bias", 0.0))))
    global_guard = max(0.0, mae - float(base_mae) - 0.75)
    return 0.55 * short + 0.25 * mae + 0.10 * p90 + 0.10 * short_bias + 0.80 * global_guard


def sigmoid_gate(pred: torch.Tensor, cutoff: float, temp: float) -> torch.Tensor:
    z = (float(cutoff) - pred) / max(float(temp), 1e-3)
    return torch.sigmoid(torch.clamp(z, -20.0, 20.0))


def public_candidate(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"val_pred", "test_pred"}}


def write_predictions(path: Path, y: torch.Tensor, meta: Sequence[Mapping[str, Any]], phase3: torch.Tensor, balanced: torch.Tensor, short_primary: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "speaker_id",
        "source",
        "gender",
        "height_cm",
        "phase3_final_cm",
        "phase20_balanced_cm",
        "phase20_balanced_abs_error_cm",
        "phase20_short_primary_cm",
        "phase20_short_primary_abs_error_cm",
    ]
    y_np = y.detach().cpu().numpy()
    p3_np = phase3.detach().cpu().numpy()
    bal_np = balanced.detach().cpu().numpy()
    short_np = short_primary.detach().cpu().numpy()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(meta):
            true = float(y_np[idx])
            bal = float(bal_np[idx])
            sh = float(short_np[idx])
            writer.writerow(
                {
                    "speaker_id": sid(row),
                    "source": str(row.get("source", "")),
                    "gender": int(row.get("gender", 0)),
                    "height_cm": f"{true:.6f}",
                    "phase3_final_cm": f"{float(p3_np[idx]):.6f}",
                    "phase20_balanced_cm": f"{bal:.6f}",
                    "phase20_balanced_abs_error_cm": f"{abs(bal - true):.6f}",
                    "phase20_short_primary_cm": f"{sh:.6f}",
                    "phase20_short_primary_abs_error_cm": f"{abs(sh - true):.6f}",
                }
            )


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    base = report["base"]
    balanced = report["balanced"]
    short_primary = report["short_primary"]
    lines = [
        "# Phase 20 Phase3 Frontier Enhancer Report",
        "",
        "## Result",
        f"- Phase3 baseline test MAE: `{base['test']['mae']:.3f}cm`",
        f"- Phase3 baseline short MAE: `{base['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Balanced deploy method: `{balanced['name']}`",
        f"- Balanced test MAE: `{balanced['test']['mae']:.3f}cm`",
        f"- Balanced short MAE: `{balanced['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Short-primary method: `{short_primary['name']}`",
        f"- Short-primary test MAE: `{short_primary['test']['mae']:.3f}cm`",
        f"- Short-primary short MAE: `{short_primary['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Short-primary within 5cm: `{short_primary['test'].get('within_5cm', float('nan')):.3f}`",
        "",
        "## Selection Notes",
        f"- Balanced minimum validation gain guard: `{report['args']['balanced_min_val_gain']:.3f}cm`",
        "- If the best balanced validation gain is too tiny, Phase3 is kept as the balanced deploy head.",
        "- Short-primary is allowed to trade some validation/global MAE for short-speaker repair.",
        "",
        "## Top Balanced Candidates",
    ]
    for row in report["top_balanced"][:15]:
        lines.append(
            f"- `{row['name']}`: val `{row['val']['mae']:.3f}cm`, test `{row['test']['mae']:.3f}cm`, "
            f"test_short `{row['test'].get('short_mae', float('nan')):.3f}cm`, score `{row['balanced_score']:.3f}`"
        )
    lines.extend(["", "## Top Short Candidates"])
    for row in report["top_short"][:15]:
        lines.append(
            f"- `{row['name']}`: val_short `{row['val'].get('short_mae', float('nan')):.3f}cm`, val `{row['val']['mae']:.3f}cm`, "
            f"test `{row['test']['mae']:.3f}cm`, test_short `{row['test'].get('short_mae', float('nan')):.3f}cm`, score `{row['short_score']:.3f}`"
        )
    lines.extend(["", "## Data Counts"])
    lines.append(f"- Validation speakers: `{report['counts']['val']}`")
    lines.append(f"- Test speakers: `{report['counts']['test']}`")
    lines.append(f"- Candidates searched: `{report['counts']['candidates']}`")
    lines.append("")
    lines.append("## Read")
    lines.append("This enhances Phase3 directly. It still does not prove 3cm; it makes the strongest current Phase3-anchored short-speaker repair explicit.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Phase20 is CUDA-only. Refusing CPU.")
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase20] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    val_rows = read_rows(resolve(args.phase3_val))
    test_rows = read_rows(resolve(args.phase3_test))
    val_meta = base_meta(val_rows)
    test_meta = base_meta(test_rows)
    val_y = tensor([float(row["height_cm"]) for row in val_rows], device)
    test_y = tensor([float(row["height_cm"]) for row in test_rows], device)
    val_phase3 = tensor([float(row["final_pred_cm"]) for row in val_rows], device)
    test_phase3 = tensor([float(row["final_pred_cm"]) for row in test_rows], device)

    candidate_series: List[Tuple[str, torch.Tensor, torch.Tensor]] = [("phase3_final", val_phase3, test_phase3)]
    for col in phase3_columns(val_rows):
        test_col = col
        if test_col in test_rows[0]:
            candidate_series.append((col.replace("_pred_cm", ""), tensor([float(row[col]) for row in val_rows], device), tensor([float(row[test_col]) for row in test_rows], device)))

    for name, val_path, test_path, column in EXTERNAL_HELPERS:
        val_map = read_external(resolve(val_path), column)
        test_map = read_external(resolve(test_path), column)
        val_values = align_external(val_rows, val_map)
        test_values = align_external(test_rows, test_map)
        if val_values is not None and test_values is not None:
            candidate_series.append((name, tensor(val_values, device), tensor(test_values, device)))

    base_val_m = metrics(val_y, val_phase3, val_meta, float(args.short_cm), float(args.tall_cm))
    base_test_m = metrics(test_y, test_phase3, test_meta, float(args.short_cm), float(args.tall_cm))
    candidates: List[Dict[str, Any]] = [
        {
            "name": "phase3_identity",
            "family": "identity",
            "val_pred": val_phase3,
            "test_pred": test_phase3,
            "val": base_val_m,
            "test": base_test_m,
        }
    ]

    # Low-freedom blends: Phase3 remains one side of every blend.
    weights = torch.linspace(0.0, 1.0, 81, device=device)
    for helper_name, val_helper, test_helper in candidate_series[1:]:
        for weight in weights:
            w = float(weight.item())
            pv = w * val_phase3 + (1.0 - w) * val_helper
            pt = w * test_phase3 + (1.0 - w) * test_helper
            mv = metrics(val_y, pv, val_meta, float(args.short_cm), float(args.tall_cm))
            candidates.append(
                {
                    "name": f"blend_phase3_{helper_name}_w{w:.3f}",
                    "family": "phase3_anchor_blend",
                    "val_pred": pv,
                    "test_pred": pt,
                    "weight_phase3": w,
                    "val": mv,
                    "test": metrics(test_y, pt, test_meta, float(args.short_cm), float(args.tall_cm)),
                }
            )

    # Tail gate: lower or raise only the low-prediction region.
    for cutoff in torch.arange(158.0, 172.1, 1.0, device=device):
        for temp in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
            gv = sigmoid_gate(val_phase3, float(cutoff.item()), float(temp))
            gt = sigmoid_gate(test_phase3, float(cutoff.item()), float(temp))
            for delta in torch.arange(-3.0, 10.01, 0.25, device=device):
                d = float(delta.item())
                pv = (val_phase3 - d * gv).clamp(145.0, 195.0)
                pt = (test_phase3 - d * gt).clamp(145.0, 195.0)
                mv = metrics(val_y, pv, val_meta, float(args.short_cm), float(args.tall_cm))
                candidates.append(
                    {
                        "name": f"phase3_tailgate_c{float(cutoff.item()):.0f}_t{float(temp):g}_d{d:.2f}",
                        "family": "phase3_sigmoid_tail_gate",
                        "val_pred": pv,
                        "test_pred": pt,
                        "cutoff": float(cutoff.item()),
                        "temperature": float(temp),
                        "delta": d,
                        "val": mv,
                        "test": metrics(test_y, pt, test_meta, float(args.short_cm), float(args.tall_cm)),
                    }
                )

    # Tiny affine/tail stretch family around the Phase3 anchor.
    val_center = val_phase3.mean()
    test_center = val_center
    for scale in torch.arange(0.82, 1.181, 0.02, device=device):
        for offset in torch.arange(-2.0, 2.01, 0.25, device=device):
            s = float(scale.item())
            b = float(offset.item())
            pv = (val_center + s * (val_phase3 - val_center) + b).clamp(145.0, 195.0)
            pt = (test_center + s * (test_phase3 - test_center) + b).clamp(145.0, 195.0)
            mv = metrics(val_y, pv, val_meta, float(args.short_cm), float(args.tall_cm))
            candidates.append(
                {
                    "name": f"phase3_affine_scale{s:.2f}_bias{b:.2f}",
                    "family": "phase3_affine",
                    "val_pred": pv,
                    "test_pred": pt,
                    "scale": s,
                    "bias": b,
                    "val": mv,
                    "test": metrics(test_y, pt, test_meta, float(args.short_cm), float(args.tall_cm)),
                }
            )

    for row in candidates:
        row["balanced_score"] = balanced_score(row["val"])
        row["short_score"] = short_score(row["val"], float(base_val_m["mae"]))

    balanced_raw = min(candidates, key=lambda row: float(row["balanced_score"]))
    balanced_gain = float(base_val_m["mae"]) - float(balanced_raw["val"]["mae"])
    balanced = balanced_raw if balanced_gain >= float(args.balanced_min_val_gain) else candidates[0]
    short_primary = min(candidates, key=lambda row: float(row["short_score"]))
    balanced_pred_val = balanced["val_pred"]
    balanced_pred_test = balanced["test_pred"]
    short_pred_val = short_primary["val_pred"]
    short_pred_test = short_primary["test_pred"]

    report = {
        "phase": "phase20_phase3_frontier_enhancer",
        "device": torch.cuda.get_device_name(0),
        "base": {"val": base_val_m, "test": base_test_m},
        "balanced": public_candidate(balanced),
        "balanced_raw_best": public_candidate(balanced_raw),
        "balanced_val_gain_cm": balanced_gain,
        "short_primary": public_candidate(short_primary),
        "top_balanced": [public_candidate(row) for row in sorted(candidates, key=lambda row: float(row["balanced_score"]))[:60]],
        "top_short": [public_candidate(row) for row in sorted(candidates, key=lambda row: float(row["short_score"]))[:60]],
        "counts": {"val": len(val_rows), "test": len(test_rows), "candidates": len(candidates), "series": len(candidate_series)},
        "args": {"balanced_min_val_gain": float(args.balanced_min_val_gain), "target_mae_cm": float(args.target_mae_cm)},
    }
    (output_dir / "phase20_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_report(output_dir / "PHASE20_PHASE3_FRONTIER_ENHANCER_REPORT.md", report)
    write_predictions(output_dir / "phase20_predictions_val.csv", val_y, val_meta, val_phase3, balanced_pred_val, short_pred_val)
    write_predictions(output_dir / "phase20_predictions_test.csv", test_y, test_meta, test_phase3, balanced_pred_test, short_pred_test)

    print(
        f"[phase20] balanced={balanced['name']} test_mae={balanced['test']['mae']:.3f} "
        f"short={balanced['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        f"[phase20] short_primary={short_primary['name']} test_mae={short_primary['test']['mae']:.3f} "
        f"short={short_primary['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(f"[phase20] wrote {output_dir / 'PHASE20_PHASE3_FRONTIER_ENHANCER_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
