#!/usr/bin/env python
"""Phase 7 target-domain repeated CV trainer.

Phase 6 proved that a single 97-speaker validation split is too easy to overfit.
This phase creates a stronger validation design:

- development set = target-domain train speakers + validation speakers
- sealed test = original test speakers, evaluated after CV selection
- repeated stratified CV by source, gender, and height bin
- search target-only and all-domain support models
- select by out-of-fold validation score, then fit once on full development set

This does not claim 3cm by leakage. It gives the next honest frontier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7 repeated target-domain CV trainer.")
    parser.add_argument("--speaker-cache", default="outputs/speaker_gpu_combo_full_ssl_cuda/speaker_gpu_cache.pt")
    parser.add_argument("--phase3-test-pred", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase7_target_cv_trainer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--target-mae-cm", type=float, default=3.0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def is_target(row: Mapping[str, Any]) -> bool:
    return str(row.get("source", "")).upper() in {"NISP", "TIMIT"}


def source_id(row: Mapping[str, Any]) -> int:
    source = str(row.get("source", "")).upper()
    if source == "TIMIT":
        return 0
    if source == "NISP":
        return 1
    if source in {"CELEB", "VOXCELEB"}:
        return 2
    return 3


def height_bin_value(height: float) -> int:
    if height < 160.0:
        return 0
    if height < 175.0:
        return 1
    return 2


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
        "count": float(y.numel()),
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


def selection_score(m: Mapping[str, float]) -> float:
    short = float(m.get("short_mae", m["mae"]))
    return float(m["mae"]) + 0.05 * float(m["p90_ae"]) + 0.20 * max(0.0, short - float(m["mae"]))


def robust_standardize(train_x: torch.Tensor, query_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    center = torch.quantile(train_x, 0.50, dim=0)
    q25 = torch.quantile(train_x, 0.25, dim=0)
    q75 = torch.quantile(train_x, 0.75, dim=0)
    scale = (q75 - q25).clamp_min(1e-3)
    return (
        torch.nan_to_num((train_x - center) / scale, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0),
        torch.nan_to_num((query_x - center) / scale, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0),
    )


def meta_tensor(meta: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    rows = []
    for row in meta:
        src = source_id(row)
        gender = float(row.get("gender", 0))
        n_clips = math.log1p(float(row.get("n_clips", 0)))
        rows.append([gender, n_clips, *[1.0 if src == idx else 0.0 for idx in range(4)]])
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    return torch.nan_to_num(x, nan=0.0)


def make_folds(meta: Sequence[Mapping[str, Any]], y: torch.Tensor, folds: int, repeats: int, seed: int) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
    buckets: Dict[Tuple[str, int, int], List[int]] = defaultdict(list)
    for idx, row in enumerate(meta):
        key = (str(row.get("source", "UNKNOWN")), int(row.get("gender", 0)), height_bin_value(float(y[idx].item())))
        buckets[key].append(idx)
    all_folds = []
    for rep in range(int(repeats)):
        fold_lists = [[] for _ in range(int(folds))]
        rng = random.Random(int(seed) + rep * 1009)
        for indices in buckets.values():
            local = list(indices)
            rng.shuffle(local)
            for pos, idx in enumerate(local):
                fold_lists[pos % int(folds)].append(idx)
        all_indices = set(range(len(meta)))
        for fold_idx, val_list in enumerate(fold_lists):
            val_idx = sorted(val_list)
            train_idx = sorted(all_indices - set(val_idx))
            all_folds.append(
                (
                    torch.tensor(train_idx, dtype=torch.long, device=y.device),
                    torch.tensor(val_idx, dtype=torch.long, device=y.device),
                    f"rep{rep + 1}_fold{fold_idx + 1}",
                )
            )
    return all_folds


def group_median_predict(train_y: torch.Tensor, train_meta: Sequence[Mapping[str, Any]], query_meta: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    global_median = float(torch.median(train_y).item())
    buckets: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for y, row in zip(train_y.detach().cpu().tolist(), train_meta):
        buckets[(str(row.get("source", "UNKNOWN")), int(row.get("gender", 0)))].append(float(y))
    medians = {key: float(np.median(vals)) for key, vals in buckets.items()}
    values = [medians.get((str(row.get("source", "UNKNOWN")), int(row.get("gender", 0))), global_median) for row in query_meta]
    return torch.tensor(values, dtype=torch.float32, device=device)


def knn_predict(
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    support_meta: Sequence[Mapping[str, Any]],
    query_x: torch.Tensor,
    query_meta: Sequence[Mapping[str, Any]],
    *,
    k: int,
    temperature: float,
    source_boost: float,
    gender_boost: float,
) -> torch.Tensor:
    support_z, query_z = robust_standardize(support_x, query_x)
    support_z = F.normalize(support_z, dim=1)
    query_z = F.normalize(query_z, dim=1)
    sim = query_z @ support_z.T
    top_sim, top_idx = torch.topk(sim, k=min(int(k), support_z.shape[0]), dim=1)
    weights = torch.softmax(top_sim / float(temperature), dim=1)
    src = torch.tensor([source_id(row) for row in support_meta], dtype=torch.long, device=support_x.device)
    gen = torch.tensor([int(row.get("gender", 0)) for row in support_meta], dtype=torch.long, device=support_x.device)
    qsrc = torch.tensor([source_id(row) for row in query_meta], dtype=torch.long, device=support_x.device).unsqueeze(1)
    qgen = torch.tensor([int(row.get("gender", 0)) for row in query_meta], dtype=torch.long, device=support_x.device).unsqueeze(1)
    weights = weights * torch.where(src[top_idx] == qsrc, float(source_boost), 1.0)
    weights = weights * torch.where(gen[top_idx] == qgen, float(gender_boost), 1.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (support_y[top_idx] * weights).sum(dim=1)


def add_intercept(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device), x], dim=1)


def random_features(x: torch.Tensor, meta_x: torch.Tensor, dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    proj = torch.randn((x.shape[1], int(dim)), dtype=torch.float32, device=x.device, generator=gen) / math.sqrt(float(dim))
    z = torch.cat([x @ proj, meta_x], dim=1)
    return add_intercept(torch.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0))


def ridge_predict(
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    support_meta_x: torch.Tensor,
    query_x: torch.Tensor,
    query_meta_x: torch.Tensor,
    *,
    dim: int,
    lam: float,
    seed: int,
) -> torch.Tensor:
    support_z, query_z = robust_standardize(support_x, query_x)
    xs = random_features(support_z, support_meta_x, int(dim), int(seed))
    xq = random_features(query_z, query_meta_x, int(dim), int(seed))
    eye = torch.eye(xs.shape[1], dtype=torch.float32, device=xs.device)
    eye[0, 0] = 0.0
    coef = torch.linalg.solve(xs.T @ xs + float(lam) * eye, xs.T @ support_y)
    return xq @ coef


def config_grid(seed: int) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = [{"kind": "group_median", "support": "target"}]
    for support in ("target", "all"):
        for k in (5, 9, 15, 25, 35, 55, 75):
            for temp in (0.03, 0.05, 0.08, 0.12):
                for source_boost in (1.0, 1.35):
                    configs.append({"kind": "knn", "support": support, "k": k, "temperature": temp, "source_boost": source_boost, "gender_boost": 1.10})
        for dim in (96, 192, 384, 768):
            for lam in (1.0, 10.0, 100.0, 1000.0, 5000.0):
                for rp_seed in (seed, seed + 17):
                    configs.append({"kind": "ridge", "support": support, "dim": dim, "lambda": lam, "seed": rp_seed})
    return configs


def support_from_config(
    cfg: Mapping[str, Any],
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    dev_meta: Sequence[Mapping[str, Any]],
    dev_meta_x: torch.Tensor,
    train_idx: torch.Tensor,
    celeb_x: torch.Tensor,
    celeb_y: torch.Tensor,
    celeb_meta: Sequence[Mapping[str, Any]],
    celeb_meta_x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[Mapping[str, Any]], torch.Tensor]:
    x = dev_x[train_idx]
    y = dev_y[train_idx]
    meta = [dev_meta[int(i)] for i in train_idx.detach().cpu().tolist()]
    meta_x = dev_meta_x[train_idx]
    if cfg.get("support") == "all" and celeb_x.numel() > 0:
        x = torch.cat([x, celeb_x], dim=0)
        y = torch.cat([y, celeb_y], dim=0)
        meta = [*meta, *celeb_meta]
        meta_x = torch.cat([meta_x, celeb_meta_x], dim=0)
    return x, y, meta, meta_x


def predict_config(
    cfg: Mapping[str, Any],
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    support_meta: Sequence[Mapping[str, Any]],
    support_meta_x: torch.Tensor,
    query_x: torch.Tensor,
    query_meta: Sequence[Mapping[str, Any]],
    query_meta_x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    kind = str(cfg["kind"])
    if kind == "group_median":
        return group_median_predict(support_y, support_meta, query_meta, device)
    if kind == "knn":
        return knn_predict(
            support_x,
            support_y,
            support_meta,
            query_x,
            query_meta,
            k=int(cfg["k"]),
            temperature=float(cfg["temperature"]),
            source_boost=float(cfg["source_boost"]),
            gender_boost=float(cfg["gender_boost"]),
        )
    if kind == "ridge":
        return ridge_predict(
            support_x,
            support_y,
            support_meta_x,
            query_x,
            query_meta_x,
            dim=int(cfg["dim"]),
            lam=float(cfg["lambda"]),
            seed=int(cfg["seed"]),
        )
    raise ValueError(f"Unknown config kind: {kind}")


def read_phase3_reference(path: Path, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]]:
    if not path.exists():
        return None
    rows, y, pred = [], [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            y.append(float(row["height_cm"]))
            pred.append(float(row["final_pred_cm"]))
    return torch.tensor(y, dtype=torch.float32, device=device), torch.tensor(pred, dtype=torch.float32, device=device), rows


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
                "phase7_pred_cm": f"{value:.6f}",
                "phase7_abs_error_cm": f"{abs(value - true):.6f}",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def public_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in {"oof_pred", "test_pred"}}


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    sel = report["selected"]
    lines = [
        "# Phase 7 Target-Domain CV Trainer Report",
        "",
        "## Result",
        f"- Selected config: `{sel['config']}`",
        f"- CV OOF MAE: `{sel['cv_metrics']['mae']:.3f}cm`",
        f"- Sealed test MAE: `{sel['test_metrics']['mae']:.3f}cm`",
        f"- Sealed short MAE: `{sel['test_metrics'].get('short_mae', float('nan')):.3f}cm`",
        f"- Target 3cm met: `{report['target_met']}`",
        "",
        "## Reference",
    ]
    if report.get("phase3_reference"):
        ref = report["phase3_reference"]
        lines.append(f"- Phase 3 frontier test MAE: `{ref['mae']:.3f}cm`, short `{ref.get('short_mae', float('nan')):.3f}cm`")
    lines.extend(["", "## Top CV Configs"])
    for row in report["top_configs"][:15]:
        lines.append(
            f"- `{row['config']}`: CV `{row['cv_metrics']['mae']:.3f}cm`, "
            f"test `{row['test_metrics']['mae']:.3f}cm`, short `{row['test_metrics'].get('short_mae', float('nan')):.3f}cm`"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "This phase replaces the fragile single validation split with repeated target-domain CV. If the selected CV winner still fails on sealed test, the remaining gap is true generalization/data signal, not one unlucky validation split.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Phase 7. Refusing CPU.")
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    payload = torch.load(resolve(args.speaker_cache), map_location=device, weights_only=False)
    for split in ("train", "val", "test"):
        payload[split]["x"] = payload[split]["x"].to(device).float()
        payload[split]["y"] = payload[split]["y"].to(device).float()

    train_meta = payload["train"]["metadata"]
    target_train_mask = torch.tensor([is_target(row) for row in train_meta], dtype=torch.bool, device=device)
    celeb_mask = ~target_train_mask
    dev_x = torch.cat([payload["train"]["x"][target_train_mask], payload["val"]["x"]], dim=0)
    dev_y = torch.cat([payload["train"]["y"][target_train_mask], payload["val"]["y"]], dim=0)
    dev_meta = [row for row in train_meta if is_target(row)] + list(payload["val"]["metadata"])
    dev_meta_x = meta_tensor(dev_meta, device)
    celeb_x = payload["train"]["x"][celeb_mask]
    celeb_y = payload["train"]["y"][celeb_mask]
    celeb_meta = [row for row in train_meta if not is_target(row)]
    celeb_meta_x = meta_tensor(celeb_meta, device) if celeb_meta else torch.empty((0, dev_meta_x.shape[1]), dtype=torch.float32, device=device)
    test_x = payload["test"]["x"]
    test_y = payload["test"]["y"]
    test_meta = payload["test"]["metadata"]
    test_meta_x = meta_tensor(test_meta, device)

    folds = make_folds(dev_meta, dev_y, int(args.folds), int(args.repeats), int(args.seed))
    configs = config_grid(int(args.seed))
    print(f"[phase7] dev_target_speakers={len(dev_meta)} celeb_support={len(celeb_meta)} test={len(test_meta)}", flush=True)
    print(f"[phase7] configs={len(configs)} folds={len(folds)}", flush=True)

    results = []
    for cidx, cfg in enumerate(configs, start=1):
        oof_sum = torch.zeros_like(dev_y)
        oof_count = torch.zeros_like(dev_y)
        fold_rows = []
        failed = False
        for train_idx, val_idx, fold_name in folds:
            try:
                support_x, support_y, support_meta, support_meta_x = support_from_config(
                    cfg, dev_x, dev_y, dev_meta, dev_meta_x, train_idx, celeb_x, celeb_y, celeb_meta, celeb_meta_x
                )
                q_meta = [dev_meta[int(i)] for i in val_idx.detach().cpu().tolist()]
                pred = predict_config(cfg, support_x, support_y, support_meta, support_meta_x, dev_x[val_idx], q_meta, dev_meta_x[val_idx], device)
                oof_sum[val_idx] += pred
                oof_count[val_idx] += 1.0
                fold_rows.append(metrics(dev_y[val_idx], pred, q_meta))
            except Exception as exc:
                failed = True
                fold_rows.append({"error": str(exc)})
                break
        if failed or (oof_count <= 0).any():
            continue
        oof_pred = oof_sum / oof_count.clamp_min(1.0)
        cv_m = metrics(dev_y, oof_pred, dev_meta)
        full_support_x = dev_x
        full_support_y = dev_y
        full_support_meta = list(dev_meta)
        full_support_meta_x = dev_meta_x
        if cfg.get("support") == "all" and celeb_x.numel() > 0:
            full_support_x = torch.cat([full_support_x, celeb_x], dim=0)
            full_support_y = torch.cat([full_support_y, celeb_y], dim=0)
            full_support_meta = [*full_support_meta, *celeb_meta]
            full_support_meta_x = torch.cat([full_support_meta_x, celeb_meta_x], dim=0)
        test_pred = predict_config(cfg, full_support_x, full_support_y, full_support_meta, full_support_meta_x, test_x, test_meta, test_meta_x, device)
        test_m = metrics(test_y, test_pred, test_meta)
        results.append(
            {
                "config": public_config(cfg),
                "score": selection_score(cv_m),
                "cv_metrics": cv_m,
                "test_metrics": test_m,
                "test_pred": test_pred.detach().cpu(),
            }
        )
        if cidx % 40 == 0:
            best = min(results, key=lambda row: row["score"]) if results else None
            print(f"[phase7] searched {cidx}/{len(configs)} best_cv={best['cv_metrics']['mae']:.3f} test={best['test_metrics']['mae']:.3f}", flush=True)

    if not results:
        raise RuntimeError("No Phase 7 configs completed.")
    results.sort(key=lambda row: row["score"])
    selected = results[0]
    phase3_ref = None
    ref = read_phase3_reference(resolve(args.phase3_test_pred), device)
    if ref is not None:
        ref_y, ref_pred, ref_meta = ref
        phase3_ref = metrics(ref_y, ref_pred, ref_meta)

    report = {
        "phase": "phase7_target_domain_repeated_cv",
        "device": torch.cuda.get_device_name(0),
        "target_mae_cm": float(args.target_mae_cm),
        "target_met": bool(selected["test_metrics"]["mae"] <= float(args.target_mae_cm)),
        "speaker_counts": {
            "dev_target": len(dev_meta),
            "celeb_support": len(celeb_meta),
            "sealed_test": len(test_meta),
            "folds": len(folds),
            "configs": len(configs),
        },
        "selected": {k: v for k, v in selected.items() if k != "test_pred"},
        "phase3_reference": phase3_ref,
        "top_configs": [{k: v for k, v in row.items() if k != "test_pred"} for row in results[:25]],
    }
    (output_dir / "phase7_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_markdown(output_dir / "PHASE7_TARGET_CV_REPORT.md", report)
    write_predictions(output_dir / "phase7_predictions_test.csv", test_y, selected["test_pred"].to(device), test_meta)
    print(
        f"[phase7] selected={selected['config']} cv={selected['cv_metrics']['mae']:.3f} "
        f"test={selected['test_metrics']['mae']:.3f} short={selected['test_metrics'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    if phase3_ref:
        print(f"[phase7] phase3_reference test={phase3_ref['mae']:.3f}", flush=True)
    print(f"[phase7] wrote {output_dir / 'PHASE7_TARGET_CV_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
