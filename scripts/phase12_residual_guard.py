#!/usr/bin/env python
"""Phase 12 complexity-guarded residual calibration.

Phase 9 is the current frontier. Phase 12 applies only small residual offsets
learned from validation residuals. It penalizes high-cardinality grouping rules
so the selector prefers simple, stable corrections such as age bucket over tiny
source/gender/prior cells that overfit validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase9_ecapa_prior_stack as p9  # noqa: E402
import phase11_metadata_tail_calibrator as p11  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 12 residual guard.")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--phase3-val-pred", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test-pred", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--phase9-cache", default="outputs/phase9_ecapa_prior_stack/ecapa_m6_s6p0_limit0_celeb.npz")
    parser.add_argument("--output-dir", default="outputs/phase12_residual_guard")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--complexity-penalty", type=float, default=0.010)
    parser.add_argument("--max-correction-scale", type=float, default=0.75)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_phase3(rows: Sequence[Mapping[str, Any]], path: Path) -> np.ndarray:
    pred = p11.align_pred(rows, p11.read_pred_csv(path, "final_pred_cm"))
    if pred is None:
        raise RuntimeError(f"Could not align {path}")
    return pred


def phase9_frontier(
    cache_path: Path,
    phase3_val: np.ndarray,
    phase3_test: np.ndarray,
    val_y: np.ndarray,
    val_meta: Sequence[Mapping[str, Any]],
    test_y: np.ndarray,
    test_meta: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> Mapping[str, Any]:
    return p11.rebuild_phase9_frontier(cache_path, phase3_val, phase3_test, val_y, val_meta, test_y, test_meta, device)


def key_rows(pred: np.ndarray, prior: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for idx, row in enumerate(rows):
        info = p11.token_info(row)
        p = float(pred[idx])
        d = float(prior[idx] - pred[idx])
        pred_bin = "p_lt165" if p < 165.0 else ("p_165_172" if p < 172.0 else ("p_172_180" if p < 180.0 else "p_ge180"))
        diff_bin = "prior_lo" if d < -4.0 else ("prior_hi" if d > 4.0 else "prior_mid")
        src = str(row["source"]).upper()
        gender = f"g{int(row['gender'])}"
        age = p11.age_bucket(float(row.get("age", 0.0)))
        out.append(
            {
                "source": src,
                "gender": gender,
                "age": age,
                "pred_bin": pred_bin,
                "prior_diff": diff_bin,
                "src_gender": f"{src}_{gender}",
                "src_pred": f"{src}_{pred_bin}",
                "gender_pred": f"{gender}_{pred_bin}",
                "src_gender_pred": f"{src}_{gender}_{pred_bin}",
                "src_prior": f"{src}_{diff_bin}",
                "src_gender_prior": f"{src}_{gender}_{diff_bin}",
                "dialect": str(info.get("dialect", "NA")),
                "language": str(info.get("language", "NA")),
            }
        )
    return out


def offset_candidate(
    *,
    name: str,
    fields: Sequence[str],
    shrinkage: float,
    scale: float,
    val_keys: Sequence[Mapping[str, str]],
    test_keys: Sequence[Mapping[str, str]],
    val_residual: np.ndarray,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
) -> Dict[str, Any]:
    groups: Dict[str, List[float]] = {}
    for idx, row in enumerate(val_keys):
        key = "|".join(row[field] for field in fields)
        groups.setdefault(key, []).append(float(val_residual[idx]))
    offsets = {
        key: float(np.mean(vals)) * (float(len(vals)) / (float(len(vals)) + float(shrinkage)))
        for key, vals in groups.items()
    }

    def corrections(keys: Sequence[Mapping[str, str]]) -> np.ndarray:
        vals = []
        for row in keys:
            key = "|".join(row[field] for field in fields)
            vals.append(offsets.get(key, 0.0))
        return np.asarray(vals, dtype=np.float32) * float(scale)

    return {
        "name": name,
        "val_pred": (val_pred + corrections(val_keys)).astype(np.float32),
        "test_pred": (test_pred + corrections(test_keys)).astype(np.float32),
        "fields": list(fields),
        "shrinkage": float(shrinkage),
        "scale": float(scale),
        "groups": int(len(groups)),
        "kind": "residual_offset",
    }


def complexity_score(metrics: Mapping[str, float], groups: int, penalty: float) -> float:
    mae = float(metrics["mae"])
    p90 = float(metrics.get("p90_ae", mae))
    bias = abs(float(metrics.get("bias", 0.0)))
    return mae + 0.015 * p90 + 0.020 * bias + float(penalty) * float(groups)


def candidate_rows(
    candidates: Sequence[Mapping[str, Any]],
    val_y: np.ndarray,
    test_y: np.ndarray,
    val_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    penalty: float,
) -> List[Dict[str, Any]]:
    rows = []
    for cand in candidates:
        val_m = p9.metrics_np(val_y, np.asarray(cand["val_pred"], dtype=np.float32), val_meta)
        test_m = p9.metrics_np(test_y, np.asarray(cand["test_pred"], dtype=np.float32), test_meta)
        row = {k: v for k, v in cand.items() if k not in {"val_pred", "test_pred"}}
        row["val"] = val_m
        row["test"] = test_m
        row["selection_score"] = complexity_score(val_m, int(row.get("groups", 1)), penalty)
        rows.append(row)
    rows.sort(key=lambda item: float(item["selection_score"]))
    return rows


def write_predictions(path: Path, y: np.ndarray, pred: np.ndarray, meta: Sequence[Mapping[str, Any]], extras: Mapping[str, np.ndarray]) -> None:
    fields = ["speaker_id", "source", "gender", "height_cm", "phase12_pred_cm", "phase12_abs_error_cm", *extras.keys()]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(meta):
            item = {
                "speaker_id": row["speaker_id"],
                "source": row["source"],
                "gender": row["gender"],
                "height_cm": f"{float(y[idx]):.6f}",
                "phase12_pred_cm": f"{float(pred[idx]):.6f}",
                "phase12_abs_error_cm": f"{abs(float(pred[idx]) - float(y[idx])):.6f}",
            }
            for name, values in extras.items():
                item[name] = f"{float(values[idx]):.6f}"
            writer.writerow(item)


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA required for Phase12.")
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_rows = p11.read_split(resolve(args.splits_dir) / "val_clean.csv")
    test_rows = p11.read_split(resolve(args.splits_dir) / "test_clean.csv")
    val_y = p11.y_array(val_rows)
    test_y = p11.y_array(test_rows)
    val_meta = p11.meta_for(val_rows)
    test_meta = p11.meta_for(test_rows)
    phase3_val = read_phase3(val_rows, resolve(args.phase3_val_pred))
    phase3_test = read_phase3(test_rows, resolve(args.phase3_test_pred))

    front = phase9_frontier(resolve(args.phase9_cache), phase3_val, phase3_test, val_y, val_meta, test_y, test_meta, device)
    val_pred = np.asarray(front["val_pred"], dtype=np.float32)
    test_pred = np.asarray(front["test_pred"], dtype=np.float32)
    cache = p9.load_cache(resolve(args.phase9_cache))
    prior_val = np.asarray([row["prior_mean"] for row in cache["val"]["meta"]], dtype=np.float32)
    prior_test = np.asarray([row["prior_mean"] for row in cache["test"]["meta"]], dtype=np.float32)

    val_keys = key_rows(val_pred, prior_val, val_rows)
    test_keys = key_rows(test_pred, prior_test, test_rows)
    residual = val_y - val_pred

    fields_list = [
        ("age",),
        ("dialect",),
        ("language",),
        ("source",),
        ("gender",),
        ("src_gender",),
        ("pred_bin",),
        ("src_pred",),
        ("gender_pred",),
        ("prior_diff",),
        ("src_prior",),
        ("src_gender_prior",),
        ("src_gender", "pred_bin"),
        ("src_gender", "prior_diff"),
    ]
    candidates: List[Dict[str, Any]] = [
        {"name": "phase9_frontier", "val_pred": val_pred, "test_pred": test_pred, "kind": "anchor", "groups": 1},
    ]
    scales = tuple(scale for scale in (0.25, 0.50, 0.75, 1.00, 1.25) if scale <= float(args.max_correction_scale) + 1e-8)
    if not scales:
        raise RuntimeError("No residual correction scales enabled")

    for fields in fields_list:
        for shrinkage in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
            for scale in scales:
                candidates.append(
                    offset_candidate(
                        name=f"offset_{'+'.join(fields)}_s{shrinkage:g}_x{scale:g}",
                        fields=fields,
                        shrinkage=shrinkage,
                        scale=scale,
                        val_keys=val_keys,
                        test_keys=test_keys,
                        val_residual=residual,
                        val_pred=val_pred,
                        test_pred=test_pred,
                    )
                )
    rows = candidate_rows(candidates, val_y, test_y, val_meta, test_meta, float(args.complexity_penalty))
    selected = rows[0]
    selected_cand = next(c for c in candidates if c["name"] == selected["name"])
    selected_val_pred = np.asarray(selected_cand["val_pred"], dtype=np.float32)
    selected_pred = np.asarray(selected_cand["test_pred"], dtype=np.float32)
    phase9_metrics = p9.metrics_np(test_y, test_pred, test_meta)
    phase3_metrics = p9.metrics_np(test_y, phase3_test, test_meta)

    write_predictions(
        output_dir / "phase12_predictions_val.csv",
        val_y,
        selected_val_pred,
        val_meta,
        {"phase3_pred_cm": phase3_val, "phase9_pred_cm": val_pred},
    )
    write_predictions(output_dir / "phase12_predictions_test.csv", test_y, selected_pred, test_meta, {"phase3_pred_cm": phase3_test, "phase9_pred_cm": test_pred})
    report = {
        "selected": selected,
        "phase9_reference": phase9_metrics,
        "phase3_reference": phase3_metrics,
        "candidate_count": len(candidates),
        "top_candidates": rows[:80],
        "args": vars(args),
    }
    (output_dir / "phase12_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")

    lines = [
        "# Phase 12 Residual Guard Report",
        "",
        "## Result",
        f"- Selected method: `{selected['name']}`",
        f"- Validation MAE: `{selected['val']['mae']:.3f}cm`",
        f"- Test MAE: `{selected['test']['mae']:.3f}cm`",
        f"- Test short MAE: `{selected['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Target 3cm met: `{selected['test']['mae'] <= 3.0}`",
        "",
        "## References",
        f"- Phase 3: `{phase3_metrics['mae']:.3f}cm`, short `{phase3_metrics.get('short_mae', float('nan')):.3f}cm`",
        f"- Phase 9: `{phase9_metrics['mae']:.3f}cm`, short `{phase9_metrics.get('short_mae', float('nan')):.3f}cm`",
        f"- Candidates searched: `{len(candidates)}`",
        f"- Complexity penalty per group: `{float(args.complexity_penalty):.3f}`",
        f"- Max correction scale: `{float(args.max_correction_scale):.2f}`",
        "",
        "## Top Validation Candidates",
    ]
    for row in rows[:20]:
        lines.append(
            f"- `{row['name']}`: val `{row['val']['mae']:.3f}cm`, test `{row['test']['mae']:.3f}cm`, "
            f"short `{row['test'].get('short_mae', float('nan')):.3f}cm`, groups `{row.get('groups', 1)}`, score `{row['selection_score']:.3f}`"
        )
    lines.extend(["", "## Conclusion"])
    if selected["test"]["mae"] < phase9_metrics["mae"]:
        lines.append("Complexity-guarded residual calibration improves the sealed-test frontier.")
    else:
        lines.append("Residual calibration did not beat Phase 9 under the guarded selector.")
    (output_dir / "PHASE12_RESIDUAL_GUARD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[phase12] selected={selected['name']}", flush=True)
    print(f"[phase12] test_mae={selected['test']['mae']:.3f} short={selected['test'].get('short_mae', float('nan')):.3f} phase9={phase9_metrics['mae']:.3f}", flush=True)
    print(f"[phase12] wrote {output_dir / 'PHASE12_RESIDUAL_GUARD_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
