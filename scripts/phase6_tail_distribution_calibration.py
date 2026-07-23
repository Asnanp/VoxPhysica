#!/usr/bin/env python
"""Phase 6 tail/distribution calibration.

The best model family overestimates short speakers and underestimates very tall
speakers: classic regression-to-the-mean. This phase tests validation-only
monotonic and distributional calibrators on top of the current best speaker
predictions. It never selects by test labels; test is reported after selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = (
    ("phase3_final", "outputs/phase3_target_domain_rescue/phase3_predictions_val.csv", "outputs/phase3_target_domain_rescue/phase3_predictions_test.csv", "final_pred_cm"),
    ("combo", "outputs/speaker_gpu_combo_full_ssl_cuda/predictions_val.csv", "outputs/speaker_gpu_combo_full_ssl_cuda/predictions_test.csv", "pred_cm"),
    ("phase2_blend", "outputs/phase2_data_forensics/phase2_predictions_val.csv", "outputs/phase2_data_forensics/phase2_predictions_test.csv", "phase2_pred_cm"),
    ("target", "outputs/speaker_gpu_target_ssl_cuda/predictions_val.csv", "outputs/speaker_gpu_target_ssl_cuda/predictions_test.csv", "pred_cm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 validation-only tail calibration.")
    parser.add_argument("--output-dir", default="outputs/phase6_tail_distribution_calibration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mae-cm", type=float, default=3.0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_pred_csv(path: Path, pred_col: str) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
    rows: List[Dict[str, Any]] = []
    y, p = [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pred_value = row.get(pred_col)
            if pred_value is None:
                pred_value = row.get("final_pred_cm") or row.get("phase4_pred_cm") or row.get("pred_cm")
            height = float(row["height_cm"])
            pred = float(pred_value)
            item = dict(row)
            item["height_cm"] = height
            item["pred_cm"] = pred
            rows.append(item)
            y.append(height)
            p.append(pred)
    return rows, torch.tensor(y, dtype=torch.float32), torch.tensor(p, dtype=torch.float32)


def height_bin(y: torch.Tensor) -> torch.Tensor:
    return torch.where(y < 160.0, torch.zeros_like(y, dtype=torch.long), torch.where(y < 175.0, torch.ones_like(y, dtype=torch.long), torch.full_like(y, 2, dtype=torch.long)))


def metrics(y: torch.Tensor, pred: torch.Tensor, meta: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
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
    }
    bins = height_bin(y)
    for label, idx in (("short", 0), ("medium", 1), ("tall", 2)):
        mask = bins == idx
        if mask.any():
            out[f"{label}_mae"] = float(ae[mask].mean().item())
            out[f"{label}_n"] = float(mask.sum().item())
    for source in sorted({str(row.get("source", "UNKNOWN")) for row in meta}):
        mask = torch.tensor([str(row.get("source", "UNKNOWN")) == source for row in meta], dtype=torch.bool, device=y.device)
        if mask.any():
            out[f"source_{source.lower()}_mae"] = float(ae[mask].mean().item())
    return out


def score(m: Mapping[str, float]) -> float:
    return float(m["mae"]) + 0.04 * float(m["p90_ae"]) + 0.18 * max(0.0, float(m.get("short_mae", m["mae"])) - float(m["mae"]))


def linear_fit(x: torch.Tensor, y: torch.Tensor, lam: float = 0.0) -> Tuple[float, float]:
    X = torch.stack([x, torch.ones_like(x)], dim=1)
    eye = torch.eye(2, dtype=torch.float32, device=x.device)
    eye[1, 1] = 0.0
    coef = torch.linalg.solve(X.T @ X + float(lam) * eye, X.T @ y)
    return float(coef[0].item()), float(coef[1].item())


def apply_affine(pred: torch.Tensor, a: float, b: float, shrink: float) -> torch.Tensor:
    calibrated = float(a) * pred + float(b)
    return pred + float(shrink) * (calibrated - pred)


def pava_fit(x: torch.Tensor, y: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    order = torch.argsort(x).detach().cpu().numpy()
    xs = x.detach().cpu().numpy()[order]
    ys = y.detach().cpu().numpy()[order]
    blocks: List[Tuple[float, float, int, float, float]] = []
    for xi, yi in zip(xs, ys):
        blocks.append((float(yi), float(yi), 1, float(xi), float(xi)))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            m1, s1, n1, x1a, x1b = blocks[-2]
            m2, s2, n2, x2a, x2b = blocks[-1]
            n = n1 + n2
            s = s1 + s2
            blocks[-2:] = [(s / n, s, n, x1a, x2b)]
    xp, fp = [], []
    for mean, _s, _n, x0, x1 in blocks:
        xp.extend([x0, x1])
        fp.extend([mean, mean])
    return np.asarray(xp, dtype=np.float32), np.asarray(fp, dtype=np.float32)


def interp_apply(pred: torch.Tensor, xp: np.ndarray, fp: np.ndarray) -> torch.Tensor:
    out = np.interp(pred.detach().cpu().numpy(), xp, fp, left=fp[0], right=fp[-1]).astype(np.float32)
    return torch.tensor(out, dtype=torch.float32, device=pred.device)


def quantile_map_fit(pred: torch.Tensor, target: torch.Tensor, n: int = 17) -> Tuple[np.ndarray, np.ndarray]:
    qs = torch.linspace(0.0, 1.0, steps=n, device=pred.device)
    xp = torch.quantile(pred, qs).detach().cpu().numpy()
    fp = torch.quantile(target, qs).detach().cpu().numpy()
    keep = np.concatenate([[True], np.diff(xp) > 1e-5])
    return xp[keep].astype(np.float32), fp[keep].astype(np.float32)


def source_gender_calibrate(
    val_pred: torch.Tensor,
    val_y: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    target_pred: torch.Tensor,
    target_meta: Sequence[Mapping[str, Any]],
    shrink: float,
) -> torch.Tensor:
    global_bias = float(torch.median(val_y - val_pred).item())
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for pred, y, row in zip(val_pred.detach().cpu().tolist(), val_y.detach().cpu().tolist(), val_meta):
        key = (str(row.get("source", "")), str(row.get("gender", "")))
        buckets.setdefault(key, []).append(float(y - pred))
    bias = {k: float(np.median(v)) for k, v in buckets.items() if v}
    values = []
    for pred, row in zip(target_pred.detach().cpu().tolist(), target_meta):
        key = (str(row.get("source", "")), str(row.get("gender", "")))
        values.append(float(pred) + float(shrink) * bias.get(key, global_bias))
    return torch.tensor(values, dtype=torch.float32, device=target_pred.device)


def write_predictions(path: Path, y: torch.Tensor, pred: torch.Tensor, meta: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for idx, row in enumerate(meta):
        true = float(y[idx].item())
        value = float(pred[idx].item())
        rows.append(
            {
                "speaker_id": row.get("speaker_id", ""),
                "source": row.get("source", ""),
                "gender": row.get("gender", ""),
                "height_cm": f"{true:.6f}",
                "phase6_pred_cm": f"{value:.6f}",
                "phase6_abs_error_cm": f"{abs(value - true):.6f}",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def public(row: Mapping[str, Any], val_y: torch.Tensor, val_meta: Sequence[Mapping[str, Any]], test_y: torch.Tensor, test_meta: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "name": row["name"],
        "base": row["base"],
        "kind": row["kind"],
        "params": row.get("params", {}),
        "val": metrics(val_y, row["val_pred"], val_meta),
        "test": metrics(test_y, row["test_pred"], test_meta),
        "score": score(metrics(val_y, row["val_pred"], val_meta)),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    sel = report["selected"]
    lines = [
        "# Phase 6 Tail/Distribution Calibration Report",
        "",
        "## Result",
        f"- Selected method: `{sel['name']}`",
        f"- Selected test MAE: `{sel['test']['mae']:.3f}cm`",
        f"- Selected short MAE: `{sel['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Target 3cm met: `{report['target_met']}`",
        "",
        "## Top By Validation Score",
    ]
    for row in report["top_by_validation"][:15]:
        lines.append(f"- `{row['name']}`: val `{row['val']['mae']:.3f}cm`, test `{row['test']['mae']:.3f}cm`, short `{row['test'].get('short_mae', float('nan')):.3f}cm`")
    lines.extend(["", "## Top By Test Diagnostic"])
    for row in report["top_by_test"][:15]:
        lines.append(f"- `{row['name']}`: test `{row['test']['mae']:.3f}cm`, val `{row['val']['mae']:.3f}cm`, short `{row['test'].get('short_mae', float('nan')):.3f}cm`")
    lines.extend(
        [
            "",
            "## Conclusion",
            "This phase tests whether validation-only tail expansion can overcome regression-to-the-mean. If validation improves while test worsens, the calibrator is not deployable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.device).lower() == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    device = torch.device("cuda" if str(args.device).lower() == "cuda" else "cpu")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for base_name, val_path, test_path, col in CANDIDATES:
        rv, rt = resolve(val_path), resolve(test_path)
        if not rv.exists() or not rt.exists():
            continue
        val_meta, val_y_cpu, val_pred_cpu = read_pred_csv(rv, col)
        test_meta, test_y_cpu, test_pred_cpu = read_pred_csv(rt, col)
        val_y, val_pred = val_y_cpu.to(device), val_pred_cpu.to(device)
        test_y, test_pred = test_y_cpu.to(device), test_pred_cpu.to(device)
        candidates.append({"name": f"{base_name}_raw", "base": base_name, "kind": "raw", "val_pred": val_pred, "test_pred": test_pred, "params": {}})

        for lam in (0.0, 1.0, 5.0, 20.0, 80.0):
            a, b = linear_fit(val_pred, val_y, lam)
            for shrink in (0.25, 0.50, 0.75, 1.0):
                candidates.append(
                    {
                        "name": f"{base_name}_affine_l{lam:g}_s{shrink:g}",
                        "base": base_name,
                        "kind": "affine",
                        "val_pred": apply_affine(val_pred, a, b, shrink),
                        "test_pred": apply_affine(test_pred, a, b, shrink),
                        "params": {"lambda": lam, "slope": a, "bias": b, "shrink": shrink},
                    }
                )

        xp, fp = pava_fit(val_pred, val_y)
        candidates.append({"name": f"{base_name}_isotonic", "base": base_name, "kind": "isotonic", "val_pred": interp_apply(val_pred, xp, fp), "test_pred": interp_apply(test_pred, xp, fp), "params": {}})
        for n in (7, 11, 17, 25):
            xp, fp = quantile_map_fit(val_pred, val_y, n)
            candidates.append({"name": f"{base_name}_qmap_{n}", "base": base_name, "kind": "quantile_map", "val_pred": interp_apply(val_pred, xp, fp), "test_pred": interp_apply(test_pred, xp, fp), "params": {"quantiles": n}})
        for shrink in (0.25, 0.50, 0.75, 1.0):
            candidates.append(
                {
                    "name": f"{base_name}_source_gender_bias_s{shrink:g}",
                    "base": base_name,
                    "kind": "source_gender_bias",
                    "val_pred": source_gender_calibrate(val_pred, val_y, val_meta, val_pred, val_meta, shrink),
                    "test_pred": source_gender_calibrate(val_pred, val_y, val_meta, test_pred, test_meta, shrink),
                    "params": {"shrink": shrink},
                }
            )

    if not candidates:
        raise RuntimeError("No candidate prediction files found.")
    val_y = read_pred_csv(resolve(CANDIDATES[0][1]), CANDIDATES[0][3])[1].to(device)
    test_meta, test_y_cpu, _ = read_pred_csv(resolve(CANDIDATES[0][2]), CANDIDATES[0][3])
    val_meta = read_pred_csv(resolve(CANDIDATES[0][1]), CANDIDATES[0][3])[0]
    test_y = test_y_cpu.to(device)

    rows = [public(row, val_y, val_meta, test_y, test_meta) for row in candidates]
    top_val = sorted(rows, key=lambda r: r["score"])
    top_test = sorted(rows, key=lambda r: r["test"]["mae"])
    selected_public = top_val[0]
    selected_internal = next(row for row in candidates if row["name"] == selected_public["name"])
    report = {
        "phase": "phase6_tail_distribution_calibration",
        "device": str(device),
        "target_mae_cm": float(args.target_mae_cm),
        "target_met": bool(selected_public["test"]["mae"] <= float(args.target_mae_cm)),
        "selected": selected_public,
        "top_by_validation": top_val,
        "top_by_test": top_test,
        "n_candidates": len(candidates),
    }
    (output_dir / "phase6_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_markdown(output_dir / "PHASE6_TAIL_CALIBRATION_REPORT.md", report)
    write_predictions(output_dir / "phase6_predictions_test.csv", test_y, selected_internal["test_pred"], test_meta)
    print(
        f"[phase6] selected={selected_public['name']} val={selected_public['val']['mae']:.3f} "
        f"test={selected_public['test']['mae']:.3f} short={selected_public['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(f"[phase6] best_test_diagnostic={top_test[0]['name']} test={top_test[0]['test']['mae']:.3f}", flush=True)
    print(f"[phase6] wrote {output_dir / 'PHASE6_TAIL_CALIBRATION_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
