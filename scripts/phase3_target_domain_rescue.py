#!/usr/bin/env python
"""Phase 3 target-domain rescue stacker.

This final phase consumes every speaker-level prediction head built so far and
does the most aggressive validation-only rescue that is still defensible:

- align all candidate predictions by speaker
- search CUDA convex ensembles
- search CUDA ridge residual stackers
- select only by validation score
- report sealed test once, with error-budget analysis for the 3cm target

It refuses CPU by default. The goal is not to fake 3cm; it is to squeeze the
last defensible improvement from existing evidence and expose the remaining
barrier in centimeters.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CANDIDATES: Tuple[Tuple[str, str, str, str], ...] = (
    ("combo", "outputs/speaker_gpu_combo_full_ssl_cuda/predictions_val.csv", "outputs/speaker_gpu_combo_full_ssl_cuda/predictions_test.csv", "pred_cm"),
    ("target", "outputs/speaker_gpu_target_ssl_cuda/predictions_val.csv", "outputs/speaker_gpu_target_ssl_cuda/predictions_test.csv", "pred_cm"),
    ("phase1_raw", "outputs/speaker_gpu_phase1_fullpower/predictions_val.csv", "outputs/speaker_gpu_phase1_fullpower/predictions_test.csv", "pred_cm"),
    ("phase1_cal", "outputs/speaker_gpu_phase1_fullpower/predictions_val.csv", "outputs/speaker_gpu_phase1_fullpower/predictions_test.csv", "pred_calibrated_cm"),
    ("phase2_blend", "outputs/phase2_data_forensics/phase2_predictions_val.csv", "outputs/phase2_data_forensics/phase2_predictions_test.csv", "phase2_pred_cm"),
    ("phase2_knn", "outputs/phase2_data_forensics/phase2_predictions_val.csv", "outputs/phase2_data_forensics/phase2_predictions_test.csv", "knn_pred_cm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 target-domain rescue stacker.")
    parser.add_argument("--output-dir", default="outputs/phase3_target_domain_rescue")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mae-cm", type=float, default=3.0)
    parser.add_argument("--convex-step", type=float, default=0.05)
    parser.add_argument("--short-penalty", type=float, default=0.0)
    parser.add_argument("--p90-weight", type=float, default=0.04)
    parser.add_argument("--ridge-lambdas", default="0.1,1,3,10,30,100,300,500,700,1000,1500,2000,3000,5000,10000,30000")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv_predictions(path: Path, column: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = str(row.get("speaker_id", "")).strip()
            if not sid:
                continue
            if column not in row:
                raise KeyError(f"Missing column {column!r} in {path}")
            rows[sid] = {
                "speaker_id": sid,
                "source": str(row.get("source", "UNKNOWN")),
                "gender": int(float(row.get("gender", 0))),
                "height_cm": float(row["height_cm"]),
                "n_clips": int(float(row.get("n_clips", 0))),
                "pred": float(row[column]),
            }
    return rows


def load_candidates() -> Tuple[List[str], Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Dict[str, Any]]]]:
    names: List[str] = []
    val: Dict[str, Dict[str, Dict[str, Any]]] = {}
    test: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, val_path, test_path, column in DEFAULT_CANDIDATES:
        rv = resolve(val_path)
        rt = resolve(test_path)
        if not rv.exists() or not rt.exists():
            continue
        names.append(name)
        val[name] = read_csv_predictions(rv, column)
        test[name] = read_csv_predictions(rt, column)
    if not names:
        raise RuntimeError("No candidate prediction files found.")
    return names, val, test


def align(
    names: Sequence[str],
    rows_by_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    speaker_ids = sorted(set.intersection(*(set(rows_by_model[name]) for name in names)))
    if not speaker_ids:
        raise RuntimeError("No shared speakers across candidates.")
    y = torch.tensor([float(rows_by_model[names[0]][sid]["height_cm"]) for sid in speaker_ids], dtype=torch.float32)
    pred = torch.tensor(
        [[float(rows_by_model[name][sid]["pred"]) for name in names] for sid in speaker_ids],
        dtype=torch.float32,
    )
    meta = [dict(rows_by_model[names[0]][sid]) for sid in speaker_ids]
    return speaker_ids, y, pred, meta


def height_bin(y: torch.Tensor) -> torch.Tensor:
    return torch.where(y < 160.0, torch.zeros_like(y, dtype=torch.long), torch.where(y < 175.0, torch.ones_like(y, dtype=torch.long), torch.full_like(y, 2, dtype=torch.long)))


def metrics(y: torch.Tensor, pred: torch.Tensor, meta: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    err = pred - y
    abs_err = err.abs()
    out = {
        "mae": float(abs_err.mean().item()),
        "rmse": float(torch.sqrt((err * err).mean()).item()),
        "median_ae": float(abs_err.median().item()),
        "p90_ae": float(torch.quantile(abs_err, 0.90).item()),
        "bias": float(err.mean().item()),
        "within_3cm": float((abs_err <= 3.0).float().mean().item()),
        "within_5cm": float((abs_err <= 5.0).float().mean().item()),
        "count": float(y.numel()),
    }
    bins = height_bin(y)
    for label, idx in (("short", 0), ("medium", 1), ("tall", 2)):
        mask = bins == idx
        if mask.any():
            out[f"{label}_mae"] = float(abs_err[mask].mean().item())
            out[f"{label}_n"] = float(mask.sum().item())
    for source in sorted({str(row.get("source", "UNKNOWN")) for row in meta}):
        mask = torch.tensor([str(row.get("source", "UNKNOWN")) == source for row in meta], dtype=torch.bool, device=y.device)
        if mask.any():
            out[f"source_{source.lower()}_mae"] = float(abs_err[mask].mean().item())
            out[f"source_{source.lower()}_n"] = float(mask.sum().item())
    return out


def score(m: Mapping[str, float], *, short_penalty: float, p90_weight: float) -> float:
    short = float(m.get("short_mae", m["mae"]))
    return float(m["mae"]) + float(short_penalty) * max(0.0, short - float(m["mae"])) + float(p90_weight) * float(m["p90_ae"])


def convex_weight_vectors(n: int, step: float) -> Iterable[torch.Tensor]:
    units = int(round(1.0 / float(step)))
    if units <= 0:
        raise ValueError("--convex-step must be positive")
    def rec(prefix: List[int], remaining: int, slots: int):
        if slots == 1:
            yield prefix + [remaining]
            return
        for value in range(remaining + 1):
            yield from rec(prefix + [value], remaining - value, slots - 1)
    for raw in rec([], units, n):
        yield torch.tensor(raw, dtype=torch.float32) / float(units)


def search_convex(
    names: Sequence[str],
    val_pred: torch.Tensor,
    val_y: torch.Tensor,
    test_pred: torch.Tensor,
    test_y: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None
    searched = 0
    for weights_cpu in convex_weight_vectors(len(names), float(args.convex_step)):
        weights = weights_cpu.to(val_pred.device)
        pv = (val_pred * weights.view(1, -1)).sum(dim=1)
        mv = metrics(val_y, pv, val_meta)
        sv = score(mv, short_penalty=float(args.short_penalty), p90_weight=float(args.p90_weight))
        searched += 1
        if best is None or sv < best["score"]:
            pt = (test_pred * weights.view(1, -1)).sum(dim=1)
            best = {
                "kind": "convex_blend",
                "score": sv,
                "weights": {name: float(weights_cpu[idx].item()) for idx, name in enumerate(names)},
                "val": mv,
                "test": metrics(test_y, pt, test_meta),
                "val_pred": pv.detach().cpu(),
                "test_pred": pt.detach().cpu(),
                "searched": searched,
            }
    assert best is not None
    best["searched"] = searched
    return best


def meta_matrix(pred: torch.Tensor, meta: Sequence[Mapping[str, Any]], *, residual: bool) -> torch.Tensor:
    source_nisp = torch.tensor([1.0 if str(row.get("source", "")) == "NISP" else 0.0 for row in meta], dtype=torch.float32, device=pred.device).unsqueeze(1)
    gender = torch.tensor([float(row.get("gender", 0)) for row in meta], dtype=torch.float32, device=pred.device).unsqueeze(1)
    spread = pred.std(dim=1, keepdim=True)
    if residual:
        base = pred[:, :1]
        return torch.cat([pred[:, 1:] - base, source_nisp, gender, source_nisp * gender, spread], dim=1)
    return torch.cat([torch.ones((pred.shape[0], 1), device=pred.device), pred, pred.mean(dim=1, keepdim=True), spread, source_nisp, gender, source_nisp * gender], dim=1)


def ridge_solve(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    eye = torch.eye(x.shape[1], dtype=torch.float32, device=x.device)
    return torch.linalg.solve(x.T @ x + float(lam) * eye, x.T @ y)


def search_ridge(
    names: Sequence[str],
    val_pred: torch.Tensor,
    val_y: torch.Tensor,
    test_pred: torch.Tensor,
    test_y: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    lambdas = [float(x.strip()) for x in str(args.ridge_lambdas).split(",") if x.strip()]
    out = []
    for residual in (False, True):
        xv = meta_matrix(val_pred, val_meta, residual=residual)
        xt = meta_matrix(test_pred, test_meta, residual=residual)
        target = val_y - val_pred[:, 0] if residual else val_y
        for lam in lambdas:
            coef = ridge_solve(xv, target, lam)
            pv = val_pred[:, 0] + xv @ coef if residual else xv @ coef
            pt = test_pred[:, 0] + xt @ coef if residual else xt @ coef
            mv = metrics(val_y, pv, val_meta)
            out.append(
                {
                    "kind": "ridge_residual" if residual else "ridge_direct",
                    "lambda": lam,
                    "score": score(mv, short_penalty=float(args.short_penalty), p90_weight=float(args.p90_weight)),
                    "val": mv,
                    "test": metrics(test_y, pt, test_meta),
                    "coef": [float(v) for v in coef.detach().cpu().tolist()],
                    "val_pred": pv.detach().cpu(),
                    "test_pred": pt.detach().cpu(),
                }
            )
    return out


def error_budget(y: torch.Tensor, pred: torch.Tensor, target_mae: float) -> Dict[str, Any]:
    abs_err = (pred - y).abs()
    total = float(abs_err.sum().item())
    target_total = float(target_mae) * int(y.numel())
    need = max(0.0, total - target_total)
    sorted_err = torch.sort(abs_err, descending=True).values
    cumulative = torch.cumsum(sorted_err, dim=0)
    n_perfect = int((cumulative < need).sum().item() + (1 if need > 0 else 0))
    short_mask = y < 160.0
    short_total = float(abs_err[short_mask].sum().item()) if short_mask.any() else 0.0
    after_perfect_short = (total - short_total) / float(y.numel())
    return {
        "speakers": int(y.numel()),
        "total_abs_error_cm": total,
        "target_total_abs_error_cm": target_total,
        "absolute_error_reduction_needed_cm": need,
        "worst_speakers_needed_if_perfectly_fixed": n_perfect,
        "short_total_abs_error_cm": short_total,
        "mae_if_all_short_speakers_were_perfect": after_perfect_short,
    }


def top_failures(y: torch.Tensor, pred: torch.Tensor, meta: Sequence[Mapping[str, Any]], limit: int = 25) -> List[Dict[str, Any]]:
    abs_err = (pred - y).abs().detach().cpu().numpy()
    order = np.argsort(-abs_err)[:limit]
    rows = []
    for idx in order:
        row = dict(meta[int(idx)])
        row["pred_cm"] = float(pred[int(idx)].item())
        row["abs_error_cm"] = float(abs_err[int(idx)])
        rows.append(row)
    return rows


def write_predictions(path: Path, names: Sequence[str], y: torch.Tensor, candidate_pred: torch.Tensor, final_pred: torch.Tensor, meta: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for idx, row in enumerate(meta):
        true = float(y[idx].item())
        final = float(final_pred[idx].item())
        item = {
            "speaker_id": row["speaker_id"],
            "source": row.get("source", "UNKNOWN"),
            "gender": row.get("gender", 0),
            "height_cm": f"{true:.6f}",
            "final_pred_cm": f"{final:.6f}",
            "final_abs_error_cm": f"{abs(final - true):.6f}",
        }
        for j, name in enumerate(names):
            pred = float(candidate_pred[idx, j].item())
            item[f"{name}_pred_cm"] = f"{pred:.6f}"
            item[f"{name}_abs_error_cm"] = f"{abs(pred - true):.6f}"
        rows.append(item)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Phase 3 Target-Domain Rescue Report",
        "",
        "## Result",
        f"- Selected method: `{selected['kind']}`",
        f"- Baseline combo test MAE: `{report['candidate_metrics']['combo']['test']['mae']:.3f}cm`",
        f"- Phase 3 selected test MAE: `{selected['test']['mae']:.3f}cm`",
        f"- Phase 3 short MAE: `{selected['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Target 3cm met: `{report['target_met']}`",
        "",
        "## Selected Parameters",
        f"- `{json.dumps(report['selected_public'], sort_keys=True)}`",
        "",
        "## Aggressive Validation-Only Diagnostic",
        f"- Best validation method: `{report['aggressive_best_validation']['kind']}`",
        f"- Best validation method test MAE: `{report['aggressive_best_validation']['test']['mae']:.3f}cm`",
        "- This method is not selected when it is a direct ridge stacker because it has too much freedom for only 97 validation speakers.",
        "",
        "## Error Budget",
        f"- Total abs error: `{report['error_budget']['total_abs_error_cm']:.1f}cm`",
        f"- Needed for 3cm: `{report['error_budget']['absolute_error_reduction_needed_cm']:.1f}cm` less error",
        f"- Worst speakers needing perfect repair: `{report['error_budget']['worst_speakers_needed_if_perfectly_fixed']}`",
        f"- MAE if every short speaker were perfect: `{report['error_budget']['mae_if_all_short_speakers_were_perfect']:.3f}cm`",
        "",
        "## Candidate Test MAE",
    ]
    for name, row in report["candidate_metrics"].items():
        lines.append(f"- `{name}`: `{row['test']['mae']:.3f}cm`, short `{row['test'].get('short_mae', float('nan')):.3f}cm`")
    lines.extend(["", "## Worst Selected-Test Failures"])
    for row in report["top_failures"][:15]:
        lines.append(
            f"- `{row['speaker_id']}` source `{row.get('source', '')}` true `{float(row['height_cm']):.2f}` "
            f"pred `{float(row['pred_cm']):.2f}` err `{float(row['abs_error_cm']):.2f}`"
        )
    lines.extend(
        [
            "",
            "## Phase 3 Conclusion",
            "This phase squeezes the best validation-only ensemble from the existing evidence. It still does not reach 3cm. The remaining gap requires new or corrected height supervision, especially for short speakers, not another blind architecture increase.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def public_result(row: Mapping[str, Any]) -> Dict[str, Any]:
    keep = {k: v for k, v in row.items() if k not in {"val_pred", "test_pred", "coef"}}
    if "coef" in row:
        keep["coef_l2"] = float(torch.tensor(row["coef"]).norm().item())
    return keep


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Phase 3. Refusing CPU.")
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names, val_rows, test_rows = load_candidates()
    val_sids, val_y_cpu, val_pred_cpu, val_meta = align(names, val_rows)
    test_sids, test_y_cpu, test_pred_cpu, test_meta = align(names, test_rows)
    val_y = val_y_cpu.to(device)
    test_y = test_y_cpu.to(device)
    val_pred = val_pred_cpu.to(device)
    test_pred = test_pred_cpu.to(device)

    candidate_metrics = {}
    for idx, name in enumerate(names):
        candidate_metrics[name] = {
            "val": metrics(val_y, val_pred[:, idx], val_meta),
            "test": metrics(test_y, test_pred[:, idx], test_meta),
        }

    convex = search_convex(names, val_pred, val_y, test_pred, test_y, val_meta, test_meta, args)
    ridge_rows = search_ridge(names, val_pred, val_y, test_pred, test_y, val_meta, test_meta, args)
    all_methods = [convex] + ridge_rows
    aggressive_best = min(all_methods, key=lambda row: row["score"])
    # The direct ridge stacker is intentionally reported but not deployed. With
    # 97 validation speakers it can fit validation residuals too tightly. The
    # guarded final head uses the best convex blend of already trained models,
    # which is much less free to invent a new mapping from tiny validation data.
    selected = convex
    selected_pred_test = selected["test_pred"].to(device)
    selected_pred_val = selected["val_pred"].to(device)

    report = {
        "phase": "phase3_target_domain_rescue",
        "device": torch.cuda.get_device_name(0),
        "target_mae_cm": float(args.target_mae_cm),
        "target_met": bool(selected["test"]["mae"] <= float(args.target_mae_cm)),
        "candidate_names": names,
        "n_val": len(val_sids),
        "n_test": len(test_sids),
        "candidate_metrics": candidate_metrics,
        "selected": public_result(selected),
        "selected_public": public_result(selected),
        "aggressive_best_validation": public_result(aggressive_best),
        "convex_best": public_result(convex),
        "ridge_top10": [public_result(row) for row in sorted(ridge_rows, key=lambda row: row["score"])[:10]],
        "error_budget": error_budget(test_y, selected_pred_test, float(args.target_mae_cm)),
        "top_failures": top_failures(test_y, selected_pred_test, test_meta),
    }
    (output_dir / "phase3_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_markdown(output_dir / "PHASE3_REPORT.md", report)
    write_predictions(output_dir / "phase3_predictions_val.csv", names, val_y, val_pred, selected_pred_val, val_meta)
    write_predictions(output_dir / "phase3_predictions_test.csv", names, test_y, test_pred, selected_pred_test, test_meta)

    print(
        "[phase3] selected "
        f"{selected['kind']} val_mae={selected['val']['mae']:.3f} test_mae={selected['test']['mae']:.3f} "
        f"short={selected['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        "[phase3] 3cm budget "
        f"need_reduce={report['error_budget']['absolute_error_reduction_needed_cm']:.1f}cm "
        f"perfect_short_mae={report['error_budget']['mae_if_all_short_speakers_were_perfect']:.3f}cm",
        flush=True,
    )
    print(f"[phase3] wrote {output_dir / 'PHASE3_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
