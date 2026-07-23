#!/usr/bin/env python
"""Phase 21 stronger Phase3 frontier gauntlet.

Phase 20 found a Phase3-anchored short repair. Phase 21 expands that search
while keeping the outputs honest:

- balanced: guarded global deploy head; keeps Phase3 if validation gain is weak
- short_primary: validation-selected short repair with moderate global guard
- ultra_short: validation-selected short repair with more aggressive short focus

All selections are made from validation metrics only. Test metrics are reported
afterward so we can see whether the change actually transfers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase20_phase3_frontier_enhancer as p20  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stronger Phase3 frontier gauntlet.")
    parser.add_argument("--phase3-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase21_phase3_frontier_gauntlet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--balanced-min-val-gain", type=float, default=0.05)
    parser.add_argument("--short-cm", type=float, default=160.0)
    parser.add_argument("--tall-cm", type=float, default=175.0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def public_candidate(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"val_pred", "test_pred"}}


def ultra_short_score(m: Mapping[str, float], base_mae: float) -> float:
    mae = float(m["mae"])
    short = float(m.get("short_mae", mae))
    p90 = float(m["p90_ae"])
    short_bias = abs(float(m.get("short_bias", m.get("bias", 0.0))))
    global_guard = max(0.0, mae - float(base_mae) - 1.15)
    return short + 0.08 * p90 + 0.08 * short_bias + 0.25 * max(0.0, mae - 5.20) + 0.75 * global_guard


def add_candidate(
    rows: List[Dict[str, Any]],
    *,
    name: str,
    family: str,
    val_pred: torch.Tensor,
    test_pred: torch.Tensor,
    val_y: torch.Tensor,
    test_y: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    short_cm: float,
    tall_cm: float,
    groups: int = 1,
    extra: Mapping[str, Any] | None = None,
) -> None:
    mv = p20.metrics(val_y, val_pred, val_meta, short_cm, tall_cm)
    mt = p20.metrics(test_y, test_pred, test_meta, short_cm, tall_cm)
    row = {
        "name": name,
        "family": family,
        "groups": int(groups),
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val": mv,
        "test": mt,
    }
    if extra:
        row.update(dict(extra))
    rows.append(row)


def pred_bin(pred: np.ndarray) -> np.ndarray:
    return np.where(pred < 158, "p_lt158", np.where(pred < 164, "p158_164", np.where(pred < 170, "p164_170", np.where(pred < 176, "p170_176", "p_ge176"))))


def key_rows(meta: Sequence[Mapping[str, Any]], bins: Sequence[str], fields: Sequence[str]) -> List[str]:
    out: List[str] = []
    for row, bin_name in zip(meta, bins):
        source = str(row.get("source", "UNKNOWN"))
        gender = f"g{int(row.get('gender', 0))}"
        values = {
            "source": source,
            "gender": gender,
            "src_gender": f"{source}_{gender}",
            "pred_bin": str(bin_name),
            "src_pred": f"{source}_{bin_name}",
            "gender_pred": f"{gender}_{bin_name}",
            "src_gender_pred": f"{source}_{gender}_{bin_name}",
        }
        out.append("|".join(values[field] for field in fields))
    return out


def write_predictions(
    path: Path,
    y: torch.Tensor,
    meta: Sequence[Mapping[str, Any]],
    phase3: torch.Tensor,
    balanced: torch.Tensor,
    short_primary: torch.Tensor,
    ultra_short: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "speaker_id",
        "source",
        "gender",
        "height_cm",
        "phase3_final_cm",
        "phase21_balanced_cm",
        "phase21_balanced_abs_error_cm",
        "phase21_short_primary_cm",
        "phase21_short_primary_abs_error_cm",
        "phase21_ultra_short_cm",
        "phase21_ultra_short_abs_error_cm",
    ]
    y_np = y.detach().cpu().numpy()
    p3_np = phase3.detach().cpu().numpy()
    bal_np = balanced.detach().cpu().numpy()
    short_np = short_primary.detach().cpu().numpy()
    ultra_np = ultra_short.detach().cpu().numpy()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(meta):
            true = float(y_np[idx])
            bal = float(bal_np[idx])
            sh = float(short_np[idx])
            ultra = float(ultra_np[idx])
            writer.writerow(
                {
                    "speaker_id": p20.sid(row),
                    "source": str(row.get("source", "")),
                    "gender": int(row.get("gender", 0)),
                    "height_cm": f"{true:.6f}",
                    "phase3_final_cm": f"{float(p3_np[idx]):.6f}",
                    "phase21_balanced_cm": f"{bal:.6f}",
                    "phase21_balanced_abs_error_cm": f"{abs(bal - true):.6f}",
                    "phase21_short_primary_cm": f"{sh:.6f}",
                    "phase21_short_primary_abs_error_cm": f"{abs(sh - true):.6f}",
                    "phase21_ultra_short_cm": f"{ultra:.6f}",
                    "phase21_ultra_short_abs_error_cm": f"{abs(ultra - true):.6f}",
                }
            )


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    base = report["base"]
    balanced = report["balanced"]
    short_primary = report["short_primary"]
    ultra = report["ultra_short"]
    lines = [
        "# Phase 21 Stronger Phase3 Frontier Gauntlet Report",
        "",
        "## Result",
        f"- Phase3 baseline test MAE: `{base['test']['mae']:.3f}cm`",
        f"- Phase3 baseline short MAE: `{base['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Balanced method: `{balanced['name']}`",
        f"- Balanced test MAE: `{balanced['test']['mae']:.3f}cm`, short `{balanced['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Short-primary method: `{short_primary['name']}`",
        f"- Short-primary test MAE: `{short_primary['test']['mae']:.3f}cm`, short `{short_primary['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Ultra-short method: `{ultra['name']}`",
        f"- Ultra-short test MAE: `{ultra['test']['mae']:.3f}cm`, short `{ultra['test'].get('short_mae', float('nan')):.3f}cm`",
        "",
        "## Top Short-Primary Candidates",
    ]
    for row in report["top_short"][:15]:
        lines.append(
            f"- `{row['name']}` ({row['family']}): val `{row['val']['mae']:.3f}cm`, val_short `{row['val'].get('short_mae', float('nan')):.3f}cm`, "
            f"test `{row['test']['mae']:.3f}cm`, test_short `{row['test'].get('short_mae', float('nan')):.3f}`, score `{row['short_score']:.3f}`"
        )
    lines.extend(["", "## Top Ultra-Short Candidates"])
    for row in report["top_ultra"][:15]:
        lines.append(
            f"- `{row['name']}` ({row['family']}): val `{row['val']['mae']:.3f}cm`, val_short `{row['val'].get('short_mae', float('nan')):.3f}cm`, "
            f"test `{row['test']['mae']:.3f}cm`, test_short `{row['test'].get('short_mae', float('nan')):.3f}`, score `{row['ultra_score']:.3f}`"
        )
    lines.extend(["", "## Top Balanced Candidates"])
    for row in report["top_balanced"][:10]:
        lines.append(
            f"- `{row['name']}` ({row['family']}): val `{row['val']['mae']:.3f}cm`, test `{row['test']['mae']:.3f}cm`, "
            f"test_short `{row['test'].get('short_mae', float('nan')):.3f}`, score `{row['balanced_score']:.3f}`"
        )
    lines.extend(["", "## Data Counts"])
    for key, value in report["counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Read")
    lines.append("Phase 21 is stronger than Phase 20, but still validation-selected. The ultra-short head is a deliberate tradeoff, not a universal replacement for Phase3.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Phase21 is CUDA-only. Refusing CPU.")
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase21] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    val_rows = p20.read_rows(resolve(args.phase3_val))
    test_rows = p20.read_rows(resolve(args.phase3_test))
    val_meta = p20.base_meta(val_rows)
    test_meta = p20.base_meta(test_rows)
    val_y = p20.tensor([float(row["height_cm"]) for row in val_rows], device)
    test_y = p20.tensor([float(row["height_cm"]) for row in test_rows], device)
    val_phase3 = p20.tensor([float(row["final_pred_cm"]) for row in val_rows], device)
    test_phase3 = p20.tensor([float(row["final_pred_cm"]) for row in test_rows], device)
    base_val_m = p20.metrics(val_y, val_phase3, val_meta, float(args.short_cm), float(args.tall_cm))
    base_test_m = p20.metrics(test_y, test_phase3, test_meta, float(args.short_cm), float(args.tall_cm))

    candidates: List[Dict[str, Any]] = []
    add_candidate(
        candidates,
        name="phase3_identity",
        family="identity",
        val_pred=val_phase3,
        test_pred=test_phase3,
        val_y=val_y,
        test_y=test_y,
        val_meta=val_meta,
        test_meta=test_meta,
        short_cm=float(args.short_cm),
        tall_cm=float(args.tall_cm),
    )

    # Wider single tail gate grid than Phase 20.
    for cutoff in torch.arange(158.0, 176.01, 0.5, device=device):
        for temp in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0):
            gv = p20.sigmoid_gate(val_phase3, float(cutoff.item()), float(temp))
            gt = p20.sigmoid_gate(test_phase3, float(cutoff.item()), float(temp))
            for delta in torch.arange(-4.0, 12.01, 0.25, device=device):
                d = float(delta.item())
                pv = (val_phase3 - d * gv).clamp(145.0, 195.0)
                pt = (test_phase3 - d * gt).clamp(145.0, 195.0)
                add_candidate(
                    candidates,
                    name=f"tail_c{float(cutoff.item()):.1f}_t{float(temp):g}_d{d:.2f}",
                    family="single_tail_gate",
                    val_pred=pv,
                    test_pred=pt,
                    val_y=val_y,
                    test_y=test_y,
                    val_meta=val_meta,
                    test_meta=test_meta,
                    short_cm=float(args.short_cm),
                    tall_cm=float(args.tall_cm),
                    extra={"cutoff": float(cutoff.item()), "temperature": float(temp), "delta": d},
                )

    # Dual-gate family: lower likely-short predictions, then gently repair high predictions.
    for cutoff_low in (168.0, 170.0, 172.0, 174.0):
        gv_low = p20.sigmoid_gate(val_phase3, cutoff_low, 2.0)
        gt_low = p20.sigmoid_gate(test_phase3, cutoff_low, 2.0)
        for delta_low in (2.0, 2.5, 3.0, 3.5, 4.0):
            low_val = val_phase3 - float(delta_low) * gv_low
            low_test = test_phase3 - float(delta_low) * gt_low
            for cutoff_high in (176.0, 178.0, 180.0, 182.0):
                gv_high = torch.sigmoid(torch.clamp((val_phase3 - cutoff_high) / 3.0, -20.0, 20.0))
                gt_high = torch.sigmoid(torch.clamp((test_phase3 - cutoff_high) / 3.0, -20.0, 20.0))
                for delta_high in (-1.0, 0.0, 1.0, 2.0, 3.0):
                    pv = (low_val + float(delta_high) * gv_high).clamp(145.0, 195.0)
                    pt = (low_test + float(delta_high) * gt_high).clamp(145.0, 195.0)
                    add_candidate(
                        candidates,
                        name=f"dual_cl{cutoff_low:.0f}_dl{delta_low:g}_ch{cutoff_high:.0f}_dh{delta_high:g}",
                        family="dual_tail_gate",
                        val_pred=pv,
                        test_pred=pt,
                        val_y=val_y,
                        test_y=test_y,
                        val_meta=val_meta,
                        test_meta=test_meta,
                        short_cm=float(args.short_cm),
                        tall_cm=float(args.tall_cm),
                        groups=2,
                    )

    # Conservative residual-offset family. This is reported but penalized by the selectors.
    val_np = val_phase3.detach().cpu().numpy()
    test_np = test_phase3.detach().cpu().numpy()
    y_np = val_y.detach().cpu().numpy()
    residual = y_np - val_np
    val_bins = pred_bin(val_np)
    test_bins = pred_bin(test_np)
    for fields in (("source",), ("gender",), ("src_gender",), ("pred_bin",), ("src_pred",), ("gender_pred",), ("src_gender_pred",)):
        val_keys = key_rows(val_meta, val_bins, fields)
        test_keys = key_rows(test_meta, test_bins, fields)
        groups: Dict[str, List[float]] = defaultdict(list)
        for key, value in zip(val_keys, residual):
            groups[key].append(float(value))
        for shrink in (8.0, 20.0, 50.0, 100.0):
            offsets = {key: float(np.mean(vals)) * (len(vals) / (len(vals) + float(shrink))) for key, vals in groups.items()}
            corr_val = torch.tensor([offsets.get(key, 0.0) for key in val_keys], dtype=torch.float32, device=device)
            corr_test = torch.tensor([offsets.get(key, 0.0) for key in test_keys], dtype=torch.float32, device=device)
            for scale in (0.25, 0.50, 0.75, 1.00):
                pv = (val_phase3 + float(scale) * corr_val).clamp(145.0, 195.0)
                pt = (test_phase3 + float(scale) * corr_test).clamp(145.0, 195.0)
                add_candidate(
                    candidates,
                    name=f"offset_{'+'.join(fields)}_s{shrink:g}_x{scale:g}",
                    family="residual_offset",
                    val_pred=pv,
                    test_pred=pt,
                    val_y=val_y,
                    test_y=test_y,
                    val_meta=val_meta,
                    test_meta=test_meta,
                    short_cm=float(args.short_cm),
                    tall_cm=float(args.tall_cm),
                    groups=len(groups),
                )

    for row in candidates:
        row["balanced_score"] = p20.balanced_score(row["val"]) + 0.002 * float(row.get("groups", 1))
        family_penalty = 0.0
        if row["family"] == "dual_tail_gate":
            family_penalty = 0.060
        elif row["family"] == "residual_offset":
            family_penalty = 0.260
        row["short_score"] = (
            p20.short_score(row["val"], float(base_val_m["mae"]))
            + 0.0015 * float(row.get("groups", 1))
            + family_penalty
        )
        row["ultra_score"] = ultra_short_score(row["val"], float(base_val_m["mae"])) + 0.001 * float(row.get("groups", 1))

    balanced_raw = min(candidates, key=lambda row: float(row["balanced_score"]))
    balanced_gain = float(base_val_m["mae"]) - float(balanced_raw["val"]["mae"])
    balanced = balanced_raw if balanced_gain >= float(args.balanced_min_val_gain) and balanced_raw["family"] != "residual_offset" else candidates[0]
    short_primary = min(candidates, key=lambda row: float(row["short_score"]))
    ultra_short = min(candidates, key=lambda row: float(row["ultra_score"]))

    report = {
        "phase": "phase21_phase3_frontier_gauntlet",
        "device": torch.cuda.get_device_name(0),
        "base": {"val": base_val_m, "test": base_test_m},
        "balanced": public_candidate(balanced),
        "balanced_raw_best": public_candidate(balanced_raw),
        "balanced_val_gain_cm": balanced_gain,
        "short_primary": public_candidate(short_primary),
        "ultra_short": public_candidate(ultra_short),
        "top_balanced": [public_candidate(row) for row in sorted(candidates, key=lambda row: float(row["balanced_score"]))[:60]],
        "top_short": [public_candidate(row) for row in sorted(candidates, key=lambda row: float(row["short_score"]))[:60]],
        "top_ultra": [public_candidate(row) for row in sorted(candidates, key=lambda row: float(row["ultra_score"]))[:80]],
        "counts": {"val": len(val_rows), "test": len(test_rows), "candidates": len(candidates)},
        "args": vars(args),
    }
    (output_dir / "phase21_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_report(output_dir / "PHASE21_PHASE3_FRONTIER_GAUNTLET_REPORT.md", report)
    write_predictions(
        output_dir / "phase21_predictions_val.csv",
        val_y,
        val_meta,
        val_phase3,
        balanced["val_pred"],
        short_primary["val_pred"],
        ultra_short["val_pred"],
    )
    write_predictions(
        output_dir / "phase21_predictions_test.csv",
        test_y,
        test_meta,
        test_phase3,
        balanced["test_pred"],
        short_primary["test_pred"],
        ultra_short["test_pred"],
    )
    print(
        f"[phase21] balanced={balanced['name']} test_mae={balanced['test']['mae']:.3f} "
        f"short={balanced['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        f"[phase21] short_primary={short_primary['name']} test_mae={short_primary['test']['mae']:.3f} "
        f"short={short_primary['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        f"[phase21] ultra_short={ultra_short['name']} test_mae={ultra_short['test']['mae']:.3f} "
        f"short={ultra_short['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(f"[phase21] wrote {output_dir / 'PHASE21_PHASE3_FRONTIER_GAUNTLET_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
