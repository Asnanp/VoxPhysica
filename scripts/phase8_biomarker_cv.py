#!/usr/bin/env python
"""Phase 8 biology/voice-biomarker CV.

This is deliberately different from the previous embedding stackers. It uses
only interpretable acoustic/body proxies from NPZ files:

- F0
- formant spacing
- vocal tract length estimate
- jitter/shimmer/HNR
- voicing, duration, SNR, capture quality

It trains CUDA ridge models with repeated target-domain CV, reports feature
correlations, and evaluates the sealed test once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

BASE_SCALARS = (
    "f0_mean",
    "formant_spacing_mean",
    "vtl_mean",
    "jitter",
    "shimmer",
    "hnr",
    "duration_s",
    "voiced_ratio",
    "invalid_spacing_rate",
    "invalid_vtl_rate",
    "speech_ratio",
    "snr_db_estimate",
    "capture_quality_score",
    "distance_cm_estimate",
    "distance_confidence",
    "clipped_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 biomarker-only repeated CV.")
    parser.add_argument("--features-root", default="data/features_v4_combo_full_ssl")
    parser.add_argument("--phase3-test-pred", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase8_biomarker_cv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--include-augmented", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return decode(value.reshape(-1)[0])
    return str(value)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(np.asarray(value).reshape(-1)[0])
        return out if math.isfinite(out) else default
    except Exception:
        return default


def source_id(source: str, sid: str = "") -> int:
    text = str(source or "").upper()
    sid = str(sid or "").upper()
    if text == "TIMIT" or sid.startswith("TIMIT_"):
        return 0
    if text == "NISP" or sid.startswith("NISP_"):
        return 1
    if text in {"CELEB", "VOXCELEB"} or sid.startswith("CELEB_"):
        return 2
    return 3


def is_target(row: Mapping[str, Any]) -> bool:
    return str(row.get("source", "")).upper() in {"NISP", "TIMIT"}


def gender_id(raw: Any) -> int:
    text = decode(raw).strip().lower()
    if text == "male":
        return 1
    if text == "female":
        return 0
    try:
        return int(float(text))
    except Exception:
        return 0


def clip_quality(row: Mapping[str, float]) -> float:
    q = 1.0
    capture = row.get("capture_quality_score", float("nan"))
    speech = row.get("speech_ratio", float("nan"))
    clipped = row.get("clipped_ratio", float("nan"))
    if math.isfinite(capture):
        q *= float(np.clip(capture, 0.10, 1.25))
    if math.isfinite(speech):
        q *= float(np.clip(speech, 0.20, 1.10))
    if math.isfinite(clipped):
        q *= float(np.clip(1.0 - 4.0 * clipped, 0.20, 1.0))
    return float(np.clip(q, 0.05, 1.50))


def derived_scalars(raw: Mapping[str, float]) -> Dict[str, float]:
    f0 = raw.get("f0_mean", float("nan"))
    spacing = raw.get("formant_spacing_mean", float("nan"))
    vtl = raw.get("vtl_mean", float("nan"))
    out = dict(raw)
    out["log_f0"] = math.log(max(f0, 1e-3)) if math.isfinite(f0) else float("nan")
    out["log_formant_spacing"] = math.log(max(spacing, 1e-3)) if math.isfinite(spacing) else float("nan")
    out["log_vtl"] = math.log(max(vtl, 1e-3)) if math.isfinite(vtl) else float("nan")
    out["spacing_inv"] = 1.0 / max(spacing, 1e-3) if math.isfinite(spacing) else float("nan")
    out["f0_vtl_ratio"] = f0 / max(vtl, 1e-3) if math.isfinite(f0) and math.isfinite(vtl) else float("nan")
    out["vtl_spacing_product"] = vtl * spacing if math.isfinite(vtl) and math.isfinite(spacing) else float("nan")
    return out


def read_clip(path: Path) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    with np.load(path, allow_pickle=True) as data:
        sid = decode(data["speaker_id"]) if "speaker_id" in data else path.stem.rsplit("_", 1)[0]
        source = decode(data["source"]).upper() if "source" in data else "UNKNOWN"
        raw = {key: safe_float(data[key]) if key in data else float("nan") for key in BASE_SCALARS}
        meta = {
            "speaker_id": sid,
            "height_cm": safe_float(data["height_cm"]) if "height_cm" in data else float("nan"),
            "gender": gender_id(data["gender"]) if "gender" in data else 0,
            "source": source,
            "is_augmented": int(safe_float(data["is_augmented"], 0.0)) if "is_augmented" in data else (1 if "_aug" in path.stem else 0),
        }
        return sid, meta, derived_scalars(raw)


def aggregate_speaker(rows: Sequence[Tuple[Dict[str, Any], Dict[str, float]]], feature_keys: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta_rows = [row[0] for row in rows]
    scalar_rows = [row[1] for row in rows]
    height = float(np.median([m["height_cm"] for m in meta_rows]))
    gender = int(round(float(np.median([m["gender"] for m in meta_rows]))))
    source = Counter(str(m["source"]) for m in meta_rows).most_common(1)[0][0]
    mat = np.asarray([[r.get(k, np.nan) for k in feature_keys] for r in scalar_rows], dtype=np.float32)
    qualities = np.asarray([clip_quality(r) for r in scalar_rows], dtype=np.float32)
    weights = qualities / max(float(qualities.sum()), 1e-6)
    with np.errstate(all="ignore"):
        weighted_mean = np.nansum(mat * weights[:, None], axis=0)
        stats = [
            weighted_mean,
            np.nanmedian(mat, axis=0),
            np.nanstd(mat, axis=0),
            np.nanpercentile(mat, 10, axis=0),
            np.nanpercentile(mat, 90, axis=0),
            np.nanmin(mat, axis=0),
            np.nanmax(mat, axis=0),
        ]
    src = source_id(source)
    meta_vec = np.asarray(
        [
            float(len(rows)),
            float(np.nanmean(qualities)),
            float(np.nanstd(qualities)),
            float(gender),
            *[1.0 if src == i else 0.0 for i in range(4)],
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([*stats, meta_vec]).astype(np.float32)
    meta = {"speaker_id": meta_rows[0]["speaker_id"], "height_cm": height, "gender": gender, "source": source, "n_clips": len(rows)}
    return vector, meta


def load_split(features_root: Path, split: str, include_augmented: bool) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[str]]:
    grouped: Dict[str, List[Tuple[Dict[str, Any], Dict[str, float]]]] = defaultdict(list)
    feature_keys: List[str] = []
    for idx, path in enumerate(sorted((features_root / split).glob("*.npz")), start=1):
        try:
            sid, meta, scalars = read_clip(path)
        except Exception:
            continue
        if not include_augmented and int(meta.get("is_augmented", 0)):
            continue
        if not math.isfinite(float(meta["height_cm"])):
            continue
        if not feature_keys:
            feature_keys = sorted(scalars.keys())
        grouped[sid].append((meta, scalars))
        if idx % 10000 == 0:
            print(f"[phase8] {split}: read {idx} clips", flush=True)
    rows, y, meta = [], [], []
    for sid, values in sorted(grouped.items()):
        vec, m = aggregate_speaker(values, feature_keys)
        rows.append(vec)
        y.append(float(m["height_cm"]))
        meta.append(m)
    names = []
    for stat in ("wmean", "median", "std", "p10", "p90", "min", "max"):
        names.extend(f"{stat}_{key}" for key in feature_keys)
    names.extend(["n_clips", "quality_mean", "quality_std", "gender", "source_timit", "source_nisp", "source_celeb", "source_unknown"])
    return np.stack(rows).astype(np.float32), np.asarray(y, dtype=np.float32), meta, names


def height_bin_value(height: float) -> int:
    if height < 160:
        return 0
    if height < 175:
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
    return float(m["mae"]) + 0.05 * float(m["p90_ae"]) + 0.15 * max(0.0, float(m.get("short_mae", m["mae"])) - float(m["mae"]))


def make_folds(meta: Sequence[Mapping[str, Any]], y: np.ndarray, folds: int, repeats: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    buckets: Dict[Tuple[str, int, int], List[int]] = defaultdict(list)
    for i, row in enumerate(meta):
        buckets[(str(row["source"]), int(row["gender"]), height_bin_value(float(y[i])))].append(i)
    out = []
    all_idx = set(range(len(meta)))
    for rep in range(repeats):
        rng = random.Random(seed + rep * 917)
        fold_lists = [[] for _ in range(folds)]
        for ids in buckets.values():
            ids = list(ids)
            rng.shuffle(ids)
            for pos, idx in enumerate(ids):
                fold_lists[pos % folds].append(idx)
        for fold in fold_lists:
            val = np.asarray(sorted(fold), dtype=np.int64)
            train = np.asarray(sorted(all_idx - set(fold)), dtype=np.int64)
            out.append((train, val))
    return out


def robust_scale(train: torch.Tensor, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    center = torch.quantile(train, 0.50, dim=0)
    q25 = torch.quantile(train, 0.25, dim=0)
    q75 = torch.quantile(train, 0.75, dim=0)
    scale = (q75 - q25).clamp_min(1e-3)
    return torch.nan_to_num((train - center) / scale).clamp(-8, 8), torch.nan_to_num((query - center) / scale).clamp(-8, 8)


def ridge_fit_predict(train_x: torch.Tensor, train_y: torch.Tensor, query_x: torch.Tensor, lam: float) -> torch.Tensor:
    xs, xq = robust_scale(train_x, query_x)
    xs = torch.cat([torch.ones((xs.shape[0], 1), device=xs.device), xs], dim=1)
    xq = torch.cat([torch.ones((xq.shape[0], 1), device=xq.device), xq], dim=1)
    eye = torch.eye(xs.shape[1], dtype=torch.float32, device=xs.device)
    eye[0, 0] = 0.0
    coef = torch.linalg.solve(xs.T @ xs + float(lam) * eye, xs.T @ train_y)
    return xq @ coef


def subset_indices(names: Sequence[str], subset: str) -> List[int]:
    if subset == "core":
        needles = ("f0", "formant", "vtl", "spacing")
        return [i for i, n in enumerate(names) if any(k in n for k in needles) or n in {"gender", "source_timit", "source_nisp"}]
    if subset == "core_quality":
        needles = ("f0", "formant", "vtl", "spacing", "voiced", "quality", "snr", "duration", "hnr")
        return [i for i, n in enumerate(names) if any(k in n for k in needles) or n in {"gender", "source_timit", "source_nisp"}]
    if subset == "no_source":
        return [i for i, n in enumerate(names) if not n.startswith("source_")]
    return list(range(len(names)))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks


def correlations(x: np.ndarray, y: np.ndarray, names: Sequence[str], limit: int = 80) -> List[Dict[str, Any]]:
    rows = []
    yr = rankdata(y)
    for i, name in enumerate(names):
        col = x[:, i]
        mask = np.isfinite(col) & np.isfinite(y)
        if mask.sum() < 8 or float(np.nanstd(col[mask])) < 1e-8:
            continue
        pear = float(np.corrcoef(col[mask], y[mask])[0, 1])
        spear = float(np.corrcoef(rankdata(col[mask]), yr[mask])[0, 1])
        rows.append({"feature": name, "pearson": pear, "spearman": spear, "abs_spearman": abs(spear)})
    rows.sort(key=lambda r: r["abs_spearman"], reverse=True)
    return rows[:limit]


def read_phase3(path: Path, device: torch.device) -> Dict[str, float] | None:
    if not path.exists():
        return None
    y, p, meta = [], [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y.append(float(row["height_cm"]))
            p.append(float(row["final_pred_cm"]))
            meta.append(dict(row))
    return metrics(torch.tensor(y, dtype=torch.float32, device=device), torch.tensor(p, dtype=torch.float32, device=device), meta)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_predictions(path: Path, y: torch.Tensor, pred: torch.Tensor, meta: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for i, row in enumerate(meta):
        true = float(y[i].item())
        value = float(pred[i].item())
        rows.append({"speaker_id": row["speaker_id"], "source": row["source"], "gender": row["gender"], "height_cm": f"{true:.6f}", "phase8_pred_cm": f"{value:.6f}", "phase8_abs_error_cm": f"{abs(value - true):.6f}"})
    write_csv(path, rows, ["speaker_id", "source", "gender", "height_cm", "phase8_pred_cm", "phase8_abs_error_cm"])


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    sel = report["selected"]
    lines = [
        "# Phase 8 Biomarker CV Report",
        "",
        "## Result",
        f"- Selected biomarker model: `{sel['subset']}` lambda `{sel['lambda']}`",
        f"- CV OOF MAE: `{sel['cv_metrics']['mae']:.3f}cm`",
        f"- Sealed test MAE: `{sel['test_metrics']['mae']:.3f}cm`",
        f"- Sealed short MAE: `{sel['test_metrics'].get('short_mae', float('nan')):.3f}cm`",
        "",
        "## Reference",
    ]
    if report.get("phase3_reference"):
        ref = report["phase3_reference"]
        lines.append(f"- Phase 3 frontier: `{ref['mae']:.3f}cm`, short `{ref.get('short_mae', float('nan')):.3f}cm`")
    lines.extend(["", "## Top Biomarker Features"])
    for row in report["top_correlations"][:20]:
        lines.append(f"- `{row['feature']}` spearman `{row['spearman']:.3f}` pearson `{row['pearson']:.3f}`")
    lines.extend(["", "## Top CV Models"])
    for row in report["top_models"][:12]:
        lines.append(f"- `{row['subset']}` lambda `{row['lambda']}`: CV `{row['cv_metrics']['mae']:.3f}cm`, test `{row['test_metrics']['mae']:.3f}cm`")
    lines.extend(["", "## Conclusion", "This phase tests a different hypothesis: interpretable biological acoustic markers. If it underperforms Phase 3, the path forward is improving biomarker extraction or adding stronger pretrained audio representations, not more regressors on the same signal."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Phase 8. Refusing CPU model fitting.")
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_root = resolve(args.features_root)

    tables = {}
    for split in ("train", "val", "test"):
        x, y, meta, names = load_split(features_root, split, bool(args.include_augmented))
        tables[split] = {"x": x, "y": y, "meta": meta}
        print(f"[phase8] {split}: speakers={len(meta)} dim={x.shape[1]}", flush=True)
    target_train = np.asarray([is_target(row) for row in tables["train"]["meta"]], dtype=bool)
    dev_x = np.concatenate([tables["train"]["x"][target_train], tables["val"]["x"]], axis=0)
    dev_y = np.concatenate([tables["train"]["y"][target_train], tables["val"]["y"]], axis=0)
    dev_meta = [row for row in tables["train"]["meta"] if is_target(row)] + list(tables["val"]["meta"])
    test_x, test_y_np, test_meta = tables["test"]["x"], tables["test"]["y"], tables["test"]["meta"]
    folds = make_folds(dev_meta, dev_y, int(args.folds), int(args.repeats), int(args.seed))
    top_corr = correlations(dev_x, dev_y, names)
    write_csv(output_dir / "biomarker_correlations.csv", top_corr, ["feature", "pearson", "spearman", "abs_spearman"])

    dev_x_t = torch.tensor(dev_x, dtype=torch.float32, device=device)
    dev_y_t = torch.tensor(dev_y, dtype=torch.float32, device=device)
    test_x_t = torch.tensor(test_x, dtype=torch.float32, device=device)
    test_y_t = torch.tensor(test_y_np, dtype=torch.float32, device=device)
    results = []
    for subset in ("core", "core_quality", "no_source", "all"):
        idx = subset_indices(names, subset)
        for lam in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
            oof = torch.zeros_like(dev_y_t)
            count = torch.zeros_like(dev_y_t)
            for tr, va in folds:
                tr_t = torch.tensor(tr, dtype=torch.long, device=device)
                va_t = torch.tensor(va, dtype=torch.long, device=device)
                pred = ridge_fit_predict(dev_x_t[tr_t][:, idx], dev_y_t[tr_t], dev_x_t[va_t][:, idx], lam)
                oof[va_t] += pred
                count[va_t] += 1
            oof = oof / count.clamp_min(1.0)
            cv_m = metrics(dev_y_t, oof, dev_meta)
            test_pred = ridge_fit_predict(dev_x_t[:, idx], dev_y_t, test_x_t[:, idx], lam)
            test_m = metrics(test_y_t, test_pred, test_meta)
            results.append({"subset": subset, "lambda": lam, "score": score(cv_m), "cv_metrics": cv_m, "test_metrics": test_m, "test_pred": test_pred.detach().cpu()})
    results.sort(key=lambda r: r["score"])
    selected = results[0]
    phase3 = read_phase3(resolve(args.phase3_test_pred), device)
    report = {
        "phase": "phase8_biomarker_cv",
        "features_root": str(features_root),
        "speaker_counts": {"dev_target": len(dev_meta), "test": len(test_meta)},
        "selected": {k: v for k, v in selected.items() if k != "test_pred"},
        "top_models": [{k: v for k, v in row.items() if k != "test_pred"} for row in results[:20]],
        "top_correlations": top_corr,
        "phase3_reference": phase3,
        "target_met": bool(selected["test_metrics"]["mae"] <= 3.0),
    }
    (output_dir / "phase8_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_markdown(output_dir / "PHASE8_BIOMARKER_CV_REPORT.md", report)
    write_predictions(output_dir / "phase8_predictions_test.csv", test_y_t, selected["test_pred"].to(device), test_meta)
    print(f"[phase8] selected subset={selected['subset']} lambda={selected['lambda']} cv={selected['cv_metrics']['mae']:.3f} test={selected['test_metrics']['mae']:.3f}", flush=True)
    if phase3:
        print(f"[phase8] phase3_reference test={phase3['mae']:.3f}", flush=True)
    print(f"[phase8] wrote {output_dir / 'PHASE8_BIOMARKER_CV_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
