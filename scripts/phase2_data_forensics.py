#!/usr/bin/env python
"""Phase 2 data/label/domain forensics for the 3cm MAE push.

Phase 1 showed that a bigger speaker model is not enough. This script attacks
the data side: split balance, short-speaker support, nearest-neighbor evidence,
and validation-only model blending. It refuses CPU by default because the KNN
support search is a CUDA operation over the speaker cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 data/label/domain forensics.")
    parser.add_argument("--speaker-cache", default="outputs/speaker_gpu_combo_full_ssl_cuda/speaker_gpu_cache.pt")
    parser.add_argument("--predictions-val", default="outputs/speaker_gpu_combo_full_ssl_cuda/predictions_val.csv")
    parser.add_argument("--predictions-test", default="outputs/speaker_gpu_combo_full_ssl_cuda/predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase2_data_forensics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mae-cm", type=float, default=3.0)
    parser.add_argument("--short-cm", type=float, default=160.0)
    parser.add_argument("--tall-cm", type=float, default=175.0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-k", type=int, default=120)
    parser.add_argument("--val-short-penalty", type=float, default=0.65)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_prediction_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = str(row.get("speaker_id", "")).strip()
            if not sid:
                continue
            item: Dict[str, Any] = dict(row)
            for key in ("height_cm", "pred_cm", "abs_error_cm"):
                if key in item:
                    item[key] = float(item[key])
            for key in ("gender", "height_bin", "n_clips"):
                if key in item:
                    item[key] = int(float(item[key]))
            rows[sid] = item
    return rows


def height_bin(height: float, short_cm: float, tall_cm: float) -> str:
    if height < short_cm:
        return "short"
    if height < tall_cm:
        return "medium"
    return "tall"


def finite_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float32)
    if arr.size == 0:
        return {"count": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def source_id(source: str) -> int:
    text = str(source or "").upper()
    if text == "TIMIT":
        return 0
    if text == "NISP":
        return 1
    if text in {"CELEB", "VOXCELEB"}:
        return 2
    return 3


def metadata_vectors(meta: Sequence[Mapping[str, Any]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    genders = torch.tensor([int(row.get("gender", 0)) for row in meta], dtype=torch.long, device=device)
    sources = torch.tensor([source_id(str(row.get("source", "UNKNOWN"))) for row in meta], dtype=torch.long, device=device)
    return genders, sources


def metrics(y_true: torch.Tensor, y_pred: torch.Tensor, meta: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = err.abs()
    out = {
        "mae": float(abs_err.mean().item()),
        "rmse": float(torch.sqrt((err * err).mean()).item()),
        "median_ae": float(abs_err.median().item()),
        "p90_ae": float(torch.quantile(abs_err, 0.90).item()),
        "bias": float(err.mean().item()),
        "within_3cm": float((abs_err <= 3.0).float().mean().item()),
        "within_5cm": float((abs_err <= 5.0).float().mean().item()),
        "count": float(y_true.numel()),
    }
    for label in ("short", "medium", "tall"):
        mask = torch.tensor(
            [height_bin(float(row["height_cm"]), args.short_cm, args.tall_cm) == label for row in meta],
            dtype=torch.bool,
            device=y_true.device,
        )
        if mask.any():
            out[f"{label}_mae"] = float(abs_err[mask].mean().item())
            out[f"{label}_n"] = float(mask.sum().item())
    for source in sorted({str(row.get("source", "UNKNOWN")) for row in meta}):
        mask = torch.tensor([str(row.get("source", "UNKNOWN")) == source for row in meta], dtype=torch.bool, device=y_true.device)
        if mask.any():
            key = source.lower()
            out[f"source_{key}_mae"] = float(abs_err[mask].mean().item())
            out[f"source_{key}_n"] = float(mask.sum().item())
    return out


def score_for_selection(m: Mapping[str, float], args: argparse.Namespace) -> float:
    short = float(m.get("short_mae", m["mae"]))
    return float(m["mae"]) + float(args.val_short_penalty) * max(0.0, short - float(m["mae"])) + 0.04 * float(m["p90_ae"])


def robust_standardize(train_x: torch.Tensor, *others: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    center = torch.median(train_x, dim=0, keepdim=True).values
    q25 = torch.quantile(train_x, 0.25, dim=0, keepdim=True)
    q75 = torch.quantile(train_x, 0.75, dim=0, keepdim=True)
    iqr = (q75 - q25).clamp_min(1e-4)
    return tuple(torch.nan_to_num(((x - center) / iqr).clamp(-8.0, 8.0), nan=0.0, posinf=0.0, neginf=0.0) for x in (train_x,) + others)


@torch.no_grad()
def knn_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_meta: Sequence[Mapping[str, Any]],
    query_x: torch.Tensor,
    query_meta: Sequence[Mapping[str, Any]],
    *,
    k: int,
    power: float,
    same_source_boost: float,
    same_gender_boost: float,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_gender, train_source = metadata_vectors(train_meta, train_x.device)
    query_gender, query_source = metadata_vectors(query_meta, train_x.device)
    preds: List[torch.Tensor] = []
    top_dists: List[torch.Tensor] = []
    top_indices: List[torch.Tensor] = []
    k_eff = min(int(k), train_x.shape[0])

    for start in range(0, query_x.shape[0], int(chunk_size)):
        q = query_x[start : start + int(chunk_size)]
        dists = torch.cdist(q, train_x)
        vals, idx = torch.topk(dists, k=k_eff, dim=1, largest=False)
        neighbor_y = train_y[idx]
        weights = 1.0 / (vals.clamp_min(1e-4) ** float(power))
        q_src = query_source[start : start + q.shape[0]].unsqueeze(1)
        q_gender = query_gender[start : start + q.shape[0]].unsqueeze(1)
        weights = weights * torch.where(train_source[idx] == q_src, float(same_source_boost), 1.0)
        weights = weights * torch.where(train_gender[idx] == q_gender, float(same_gender_boost), 1.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        preds.append((weights * neighbor_y).sum(dim=1))
        top_dists.append(vals[:, 0])
        top_indices.append(idx[:, 0])

    return torch.cat(preds, dim=0), torch.cat(top_dists, dim=0), torch.cat(top_indices, dim=0)


def split_summary(split: str, y: torch.Tensor, meta: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    by_source = Counter(str(row.get("source", "UNKNOWN")) for row in meta)
    by_gender = Counter(str(row.get("gender", 0)) for row in meta)
    by_bin = Counter(height_bin(float(row["height_cm"]), args.short_cm, args.tall_cm) for row in meta)
    rows_by_group: Dict[str, List[float]] = defaultdict(list)
    for row in meta:
        rows_by_group[f"bin:{height_bin(float(row['height_cm']), args.short_cm, args.tall_cm)}"].append(float(row["height_cm"]))
        rows_by_group[f"source:{row.get('source', 'UNKNOWN')}"].append(float(row["height_cm"]))
    return {
        "split": split,
        "speakers": int(len(meta)),
        "height": finite_stats([float(v) for v in y.detach().cpu().numpy().tolist()]),
        "by_source": dict(sorted(by_source.items())),
        "by_gender": dict(sorted(by_gender.items())),
        "by_height_bin": dict(sorted(by_bin.items())),
        "height_by_group": {key: finite_stats(values) for key, values in sorted(rows_by_group.items())},
    }


def leakage_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    sets = {split: set(map(str, payload[split]["speaker_ids"])) for split in ("train", "val", "test")}
    out = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sorted(sets[left] & sets[right])
        out[f"{left}_{right}"] = {"count": len(shared), "examples": shared[:20]}
    return out


def attach_predictions(
    pred_rows: Mapping[str, Mapping[str, Any]],
    meta: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    values = []
    missing = []
    for row in meta:
        sid = str(row["speaker_id"])
        if sid not in pred_rows:
            missing.append(sid)
            values.append(float("nan"))
        else:
            values.append(float(pred_rows[sid]["pred_cm"]))
    if missing:
        raise RuntimeError(f"Missing prediction rows for {len(missing)} speakers, examples={missing[:5]}")
    return torch.tensor(values, dtype=torch.float32, device=device)


def top_failures(
    y_true: torch.Tensor,
    pred: torch.Tensor,
    meta: Sequence[Mapping[str, Any]],
    *,
    nearest_dist: Optional[torch.Tensor] = None,
    nearest_index: Optional[torch.Tensor] = None,
    train_meta: Optional[Sequence[Mapping[str, Any]]] = None,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    abs_err = (pred - y_true).abs().detach().cpu().numpy()
    order = np.argsort(-abs_err)[:limit]
    rows = []
    nearest_dist_np = nearest_dist.detach().cpu().numpy() if nearest_dist is not None else None
    nearest_idx_np = nearest_index.detach().cpu().numpy() if nearest_index is not None else None
    for idx in order:
        row = dict(meta[int(idx)])
        row["pred_cm"] = float(pred[int(idx)].item())
        row["abs_error_cm"] = float(abs_err[int(idx)])
        if nearest_dist_np is not None:
            row["nearest_train_distance"] = float(nearest_dist_np[int(idx)])
        if nearest_idx_np is not None and train_meta is not None:
            nrow = train_meta[int(nearest_idx_np[int(idx)])]
            row["nearest_train_speaker"] = str(nrow.get("speaker_id", ""))
            row["nearest_train_height_cm"] = float(nrow.get("height_cm", float("nan")))
            row["nearest_train_source"] = str(nrow.get("source", ""))
        rows.append(row)
    return rows


def write_prediction_csv(path: Path, y_true: torch.Tensor, base: torch.Tensor, knn: torch.Tensor, phase2: torch.Tensor, meta: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for idx, row in enumerate(meta):
        true = float(y_true[idx].item())
        base_pred = float(base[idx].item())
        knn_pred = float(knn[idx].item())
        phase2_pred = float(phase2[idx].item())
        rows.append(
            {
                "speaker_id": row["speaker_id"],
                "source": row.get("source", "UNKNOWN"),
                "gender": row.get("gender", 0),
                "height_cm": f"{true:.6f}",
                "height_bin": height_bin(true, 160.0, 175.0),
                "n_clips": row.get("n_clips", 0),
                "base_pred_cm": f"{base_pred:.6f}",
                "knn_pred_cm": f"{knn_pred:.6f}",
                "phase2_pred_cm": f"{phase2_pred:.6f}",
                "base_abs_error_cm": f"{abs(base_pred - true):.6f}",
                "knn_abs_error_cm": f"{abs(knn_pred - true):.6f}",
                "phase2_abs_error_cm": f"{abs(phase2_pred - true):.6f}",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Phase 2 Data Forensics Report",
        "",
        "## Result",
        f"- Baseline test MAE: `{report['baseline_test']['mae']:.3f}cm`",
        f"- Best KNN test MAE: `{report['best_knn_test']['mae']:.3f}cm`",
        f"- Phase 2 blend test MAE: `{report['phase2_test']['mae']:.3f}cm`",
        f"- Phase 2 short-speaker MAE: `{report['phase2_test'].get('short_mae', float('nan')):.3f}cm`",
        "",
        "## Best Validation Settings",
        f"- KNN: `{json.dumps(report['best_knn_config'], sort_keys=True)}`",
        f"- Blend alpha on baseline: `{report['best_blend']['alpha_baseline']:.2f}`",
        "",
        "## Split Counts",
        f"- Train: `{report['split_summary']['train']['speakers']}` speakers",
        f"- Val: `{report['split_summary']['val']['speakers']}` speakers",
        f"- Test: `{report['split_summary']['test']['speakers']}` speakers",
        "",
        "## Leakage",
        f"- `{json.dumps(report['leakage'], sort_keys=True)}`",
        "",
        "## Phase 2 Diagnosis",
    ]
    if report["phase2_test"]["mae"] < report["baseline_test"]["mae"]:
        lines.append("- Validation-only blending improved sealed test. Keep Phase 2 blend as a candidate inference head.")
    else:
        lines.append("- Validation-only blending did not beat baseline. This means local feature support is not enough to repair the sealed split.")
    if report["phase2_test"].get("short_mae", 999.0) > 6.0:
        lines.append("- Short speakers remain the primary blocker. Phase 3 must focus on verified short-height labels and source-balanced split support.")
    lines.extend(
        [
            "",
            "## Worst Phase 2 Test Failures",
        ]
    )
    for row in report["top_phase2_failures"][:12]:
        lines.append(
            f"- `{row['speaker_id']}` true `{float(row['height_cm']):.2f}` pred `{float(row['pred_cm']):.2f}` "
            f"err `{float(row['abs_error_cm']):.2f}` source `{row.get('source', '')}` nearest `{row.get('nearest_train_speaker', '')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Phase 2 KNN forensics. Refusing CPU.")
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(resolve(args.speaker_cache), map_location=device, weights_only=False)
    for split in ("train", "val", "test"):
        payload[split]["x"] = payload[split]["x"].to(device).float()
        payload[split]["y"] = payload[split]["y"].to(device).float()

    train_x, val_x, test_x = robust_standardize(payload["train"]["x"], payload["val"]["x"], payload["test"]["x"])
    train_y = payload["train"]["y"]
    val_y = payload["val"]["y"]
    test_y = payload["test"]["y"]
    train_meta = payload["train"]["metadata"]
    val_meta = payload["val"]["metadata"]
    test_meta = payload["test"]["metadata"]

    val_rows = read_prediction_csv(resolve(args.predictions_val))
    test_rows = read_prediction_csv(resolve(args.predictions_test))
    base_val = attach_predictions(val_rows, val_meta, device)
    base_test = attach_predictions(test_rows, test_meta, device)
    baseline_val = metrics(val_y, base_val, val_meta, args)
    baseline_test = metrics(test_y, base_test, test_meta, args)

    k_values = sorted({1, 3, 5, 7, 11, 15, 25, 35, 50, 75, min(int(args.max_k), train_x.shape[0])})
    configs = []
    for k in k_values:
        for power in (1.0, 1.5, 2.0):
            for source_boost in (1.0, 1.35, 1.80):
                for gender_boost in (1.0, 1.20):
                    configs.append(
                        {
                            "k": int(k),
                            "power": float(power),
                            "same_source_boost": float(source_boost),
                            "same_gender_boost": float(gender_boost),
                        }
                    )

    best_knn: Optional[Dict[str, Any]] = None
    best_val_pred: Optional[torch.Tensor] = None
    best_test_pred: Optional[torch.Tensor] = None
    best_test_dist: Optional[torch.Tensor] = None
    best_test_idx: Optional[torch.Tensor] = None
    sweep_rows = []
    for idx, cfg in enumerate(configs, start=1):
        val_pred, _, _ = knn_predict(train_x, train_y, train_meta, val_x, val_meta, chunk_size=int(args.chunk_size), **cfg)
        val_m = metrics(val_y, val_pred, val_meta, args)
        val_s = score_for_selection(val_m, args)
        sweep_rows.append({"config": cfg, "val": val_m, "score": val_s})
        if best_knn is None or val_s < best_knn["score"]:
            test_pred, test_dist, test_idx = knn_predict(
                train_x,
                train_y,
                train_meta,
                test_x,
                test_meta,
                chunk_size=int(args.chunk_size),
                **cfg,
            )
            best_knn = {"config": cfg, "val": val_m, "test": metrics(test_y, test_pred, test_meta, args), "score": val_s}
            best_val_pred = val_pred
            best_test_pred = test_pred
            best_test_dist = test_dist
            best_test_idx = test_idx
        if idx % 25 == 0:
            print(f"[phase2] searched {idx}/{len(configs)} KNN configs | best_val_score={best_knn['score']:.3f}", flush=True)

    assert best_knn is not None and best_val_pred is not None and best_test_pred is not None
    best_blend: Optional[Dict[str, Any]] = None
    phase2_val_pred = best_val_pred
    phase2_test_pred = best_test_pred
    for alpha in np.linspace(0.0, 1.0, 41):
        alpha_t = float(alpha)
        val_blend = alpha_t * base_val + (1.0 - alpha_t) * best_val_pred
        val_m = metrics(val_y, val_blend, val_meta, args)
        val_s = score_for_selection(val_m, args)
        if best_blend is None or val_s < best_blend["score"]:
            test_blend = alpha_t * base_test + (1.0 - alpha_t) * best_test_pred
            best_blend = {
                "alpha_baseline": alpha_t,
                "alpha_knn": 1.0 - alpha_t,
                "score": val_s,
                "val": val_m,
                "test": metrics(test_y, test_blend, test_meta, args),
            }
            phase2_val_pred = val_blend
            phase2_test_pred = test_blend

    assert best_blend is not None
    report = {
        "phase": "phase2_data_label_domain_forensics",
        "speaker_cache": str(resolve(args.speaker_cache)),
        "predictions_val": str(resolve(args.predictions_val)),
        "predictions_test": str(resolve(args.predictions_test)),
        "device": torch.cuda.get_device_name(0),
        "target_mae_cm": float(args.target_mae_cm),
        "target_met": bool(best_blend["test"]["mae"] <= float(args.target_mae_cm)),
        "split_summary": {
            "train": split_summary("train", train_y, train_meta, args),
            "val": split_summary("val", val_y, val_meta, args),
            "test": split_summary("test", test_y, test_meta, args),
        },
        "leakage": leakage_summary(payload),
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "best_knn_config": best_knn["config"],
        "best_knn_val": best_knn["val"],
        "best_knn_test": best_knn["test"],
        "best_blend": best_blend,
        "phase2_val": best_blend["val"],
        "phase2_test": best_blend["test"],
        "top_baseline_failures": top_failures(test_y, base_test, test_meta, nearest_dist=best_test_dist, nearest_index=best_test_idx, train_meta=train_meta),
        "top_phase2_failures": top_failures(test_y, phase2_test_pred, test_meta, nearest_dist=best_test_dist, nearest_index=best_test_idx, train_meta=train_meta),
        "knn_sweep_top10": sorted(sweep_rows, key=lambda row: row["score"])[:10],
    }

    (output_dir / "phase2_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_markdown(output_dir / "PHASE2_REPORT.md", report)
    write_prediction_csv(output_dir / "phase2_predictions_val.csv", val_y, base_val, best_val_pred, phase2_val_pred, val_meta)
    write_prediction_csv(output_dir / "phase2_predictions_test.csv", test_y, base_test, best_test_pred, phase2_test_pred, test_meta)

    print(
        "[phase2] baseline test "
        f"mae={baseline_test['mae']:.3f} short={baseline_test.get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        "[phase2] best knn test "
        f"mae={best_knn['test']['mae']:.3f} short={best_knn['test'].get('short_mae', float('nan')):.3f} "
        f"config={best_knn['config']}",
        flush=True,
    )
    print(
        "[phase2] blend test "
        f"mae={best_blend['test']['mae']:.3f} short={best_blend['test'].get('short_mae', float('nan')):.3f} "
        f"alpha_baseline={best_blend['alpha_baseline']:.2f}",
        flush=True,
    )
    print(f"[phase2] wrote {output_dir / 'PHASE2_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
