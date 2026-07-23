#!/usr/bin/env python
"""Phase 13 target-CV residual stack.

Phase 12 improved the sealed test frontier, but it learned residual offsets
from only the 97-speaker validation split. Phase 13 moves that calibration to a
larger, more honest design:

- development speakers = canonical target train speakers + validation speakers
- support speakers = CELEB / VoxCeleb-height rows from the Phase 9 ECAPA cache
- model selection = repeated out-of-fold scoring on the development speakers
- sealed test = original test speakers, used only for final reporting

The point is not to claim a magic 3cm result. The point is to spend the signal
budget correctly: stronger ECAPA/biomarker features, target-domain CV, and
guarded residual corrections selected on out-of-fold predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase9_ecapa_prior_stack as p9  # noqa: E402
import phase11_metadata_tail_calibrator as p11  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 13 target-CV residual stack.")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--phase9-cache", default="outputs/phase9_ecapa_prior_stack/ecapa_m6_s6p0_limit0_celeb.npz")
    parser.add_argument("--biomarker-cache", default="outputs/phase10_super_stack/phase10_biomarker_cache.npz")
    parser.add_argument("--phase12-test-pred", default="outputs/phase12_residual_guard/phase12_predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase13_target_cv_residual_stack")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--complexity-penalty", type=float, default=0.006)
    parser.add_argument("--top-base-for-residuals", type=int, default=10)
    parser.add_argument("--top-for-blends", type=int, default=14)
    parser.add_argument("--max-pair-blends", type=int, default=120)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def sid(row: Mapping[str, Any]) -> str:
    return str(row.get("speaker_id", "")).strip()


def target_source(row: Mapping[str, Any]) -> bool:
    return str(row.get("source", "")).upper() in {"NISP", "TIMIT"}


def height_bin(height: float) -> int:
    if float(height) < 160.0:
        return 0
    if float(height) < 175.0:
        return 1
    return 2


def read_phase12_predictions(path: Path, meta: Sequence[Mapping[str, Any]]) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    preds: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("speaker_id", "")).strip()
            value = p9.safe_float(row.get("phase12_pred_cm", row.get("final_pred_cm", "")))
            if key and math.isfinite(value):
                preds[key] = float(value)
    values: List[float] = []
    for row in meta:
        key = sid(row)
        if key not in preds:
            return None
        values.append(preds[key])
    return np.asarray(values, dtype=np.float32)


def split_lookup(splits_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in ("train", "val", "test"):
        for row in p11.read_split(splits_dir / f"{name}_clean.csv"):
            out[sid(row)] = dict(row)
    return out


def enrich_meta(meta: Sequence[Mapping[str, Any]], row_by_sid: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in meta:
        item = dict(row)
        raw = row_by_sid.get(sid(row))
        if raw is not None:
            item["age"] = float(raw.get("age", 0.0))
            item["audio_paths"] = list(raw.get("audio_paths") or [])
        else:
            item.setdefault("age", 0.0)
            item.setdefault("audio_paths", [])
        item["height_bin"] = height_bin(float(item.get("height_cm", 0.0)))
        out.append(item)
    return out


def meta_for_metrics(meta: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in meta:
        h = float(row["height_cm"])
        out.append(
            {
                "speaker_id": sid(row),
                "source": str(row.get("source", "")).upper(),
                "gender": int(row.get("gender", 0)),
                "height_cm": h,
                "height_bin": height_bin(h),
            }
        )
    return out


def load_biomarker_cache(path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return None
    out: Dict[str, Dict[str, Any]] = {}
    with np.load(path, allow_pickle=True) as data:
        names = json.loads(str(np.asarray(data["names_json"]).item()))
        for split in ("train", "val", "test"):
            out[split] = {
                "x": np.asarray(data[f"{split}_x"], dtype=np.float32),
                "y": np.asarray(data[f"{split}_y"], dtype=np.float32),
                "meta": json.loads(str(np.asarray(data[f"{split}_meta_json"]).item())),
                "names": names,
            }
    return out


def align_features(
    ref_meta: Sequence[Mapping[str, Any]],
    source_x: np.ndarray,
    source_meta: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    by_sid = {sid(row): idx for idx, row in enumerate(source_meta)}
    dim = int(source_x.shape[1])
    out = np.zeros((len(ref_meta), dim), dtype=np.float32)
    present = np.zeros((len(ref_meta), 1), dtype=np.float32)
    for idx, row in enumerate(ref_meta):
        key = sid(row)
        if key in by_sid:
            out[idx] = source_x[by_sid[key]]
            present[idx, 0] = 1.0
    return out, present


def metadata_matrix(meta: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows: List[List[float]] = []
    for row in meta:
        source = str(row.get("source", "")).upper()
        gender = float(int(row.get("gender", 0)))
        age = float(row.get("age", 0.0))
        age_scaled = np.clip(age / 80.0, 0.0, 1.4) if math.isfinite(age) and age > 0.0 else 0.0
        n_clips = math.log1p(float(row.get("n_clips", len(row.get("audio_paths") or [])))) / math.log(80.0)
        prior = float(row.get("prior_mean", 0.0))
        prior_delta = prior - 170.0 if math.isfinite(prior) else 0.0
        rows.append(
            [
                gender,
                age_scaled,
                n_clips,
                prior_delta / 20.0,
                1.0 if source == "TIMIT" else 0.0,
                1.0 if source == "NISP" else 0.0,
                1.0 if source == "CELEB" else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def append_features(*parts: np.ndarray) -> np.ndarray:
    clean = []
    for part in parts:
        arr = np.asarray(part, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        clean.append(arr)
    return np.concatenate(clean, axis=1).astype(np.float32)


def make_folds(
    meta: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    folds: int,
    repeats: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    buckets: Dict[Tuple[str, int, int], List[int]] = defaultdict(list)
    for idx, row in enumerate(meta):
        key = (str(row.get("source", "UNKNOWN")).upper(), int(row.get("gender", 0)), height_bin(float(y[idx])))
        buckets[key].append(idx)
    all_indices = set(range(len(meta)))
    out: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for rep in range(int(repeats)):
        rng = random.Random(int(seed) + 1009 * rep)
        fold_lists: List[List[int]] = [[] for _ in range(int(folds))]
        for indices in buckets.values():
            local = list(indices)
            rng.shuffle(local)
            for pos, idx in enumerate(local):
                fold_lists[pos % int(folds)].append(idx)
        for fold_idx, val_list in enumerate(fold_lists):
            val_idx = np.asarray(sorted(val_list), dtype=np.int64)
            train_idx = np.asarray(sorted(all_indices - set(val_idx.tolist())), dtype=np.int64)
            out.append((train_idx, val_idx, f"rep{rep + 1}_fold{fold_idx + 1}"))
    return out


def robust_standardize(train_x: np.ndarray, query_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_x, dtype=np.float32)
    med = np.nanmedian(train, axis=0)
    q25 = np.nanpercentile(train, 25, axis=0)
    q75 = np.nanpercentile(train, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4)

    def transform(x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=np.float32).copy()
        bad = ~np.isfinite(z)
        if bad.any():
            z[bad] = np.take(med, np.where(bad)[1])
        z = (z - med) / scale
        return np.clip(z, -8.0, 8.0).astype(np.float32)

    return transform(train), transform(query_x)


def weighted_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    weights: np.ndarray,
    lam: float,
    device: torch.device,
) -> np.ndarray:
    x_train, x_query = robust_standardize(train_x, query_x)
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    x_query = np.concatenate([x_query, np.ones((x_query.shape[0], 1), dtype=np.float32)], axis=1)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    xq = torch.tensor(x_query, dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(train_y, dtype=np.float32), dtype=torch.float32, device=device)
    w = torch.tensor(np.asarray(weights, dtype=np.float32), dtype=torch.float32, device=device).clamp_min(1e-4)
    y_mean = (y * w).sum() / w.sum().clamp_min(1e-6)
    yc = y - y_mean
    sqrt_w = torch.sqrt(w)[:, None]
    xw = xt * sqrt_w
    yw = yc * sqrt_w.squeeze(1)
    eye = torch.eye(xw.shape[1], dtype=torch.float32, device=device)
    eye[-1, -1] = 0.0
    beta = torch.linalg.solve(xw.T @ xw + float(lam) * eye, xw.T @ yw)
    return (xq @ beta + y_mean).detach().cpu().numpy().astype(np.float32)


def sample_weights(meta: Sequence[Mapping[str, Any]], *, celeb_weight: float, short_boost: float) -> np.ndarray:
    vals = []
    for row in meta:
        source = str(row.get("source", "")).upper()
        w = float(celeb_weight) if source == "CELEB" else 1.0
        if float(row.get("height_cm", 0.0)) < 160.0:
            w *= float(short_boost)
        vals.append(w)
    arr = np.asarray(vals, dtype=np.float32)
    return arr / max(float(arr.mean()), 1e-6)


def cv_ridge_candidate(
    *,
    name: str,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    dev_meta: Sequence[Mapping[str, Any]],
    support_x: np.ndarray,
    support_y: np.ndarray,
    support_meta: Sequence[Mapping[str, Any]],
    test_x: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray, str]],
    lam: float,
    celeb_weight: float,
    short_boost: float,
    device: torch.device,
) -> Dict[str, Any]:
    oof_sum = np.zeros(len(dev_y), dtype=np.float32)
    oof_count = np.zeros(len(dev_y), dtype=np.float32)
    use_support = support_x.size > 0 and float(celeb_weight) > 0.0
    for train_idx, val_idx, _fold_name in folds:
        x_train = dev_x[train_idx]
        y_train = dev_y[train_idx]
        meta_train = [dev_meta[int(i)] for i in train_idx.tolist()]
        if use_support:
            x_train = np.concatenate([x_train, support_x], axis=0)
            y_train = np.concatenate([y_train, support_y], axis=0)
            meta_train = meta_train + list(support_meta)
        w = sample_weights(meta_train, celeb_weight=float(celeb_weight), short_boost=float(short_boost))
        fold_pred = weighted_ridge(x_train, y_train, dev_x[val_idx], w, float(lam), device)
        oof_sum[val_idx] += fold_pred
        oof_count[val_idx] += 1.0
    if np.any(oof_count <= 0):
        raise RuntimeError(f"OOF coverage failed for {name}")
    oof = oof_sum / np.maximum(oof_count, 1.0)

    full_x = dev_x
    full_y = dev_y
    full_meta: List[Mapping[str, Any]] = list(dev_meta)
    if use_support:
        full_x = np.concatenate([full_x, support_x], axis=0)
        full_y = np.concatenate([full_y, support_y], axis=0)
        full_meta = full_meta + list(support_meta)
    full_w = sample_weights(full_meta, celeb_weight=float(celeb_weight), short_boost=float(short_boost))
    test_pred = weighted_ridge(full_x, full_y, test_x, full_w, float(lam), device)
    return {
        "name": name,
        "oof_pred": oof.astype(np.float32),
        "test_pred": test_pred.astype(np.float32),
        "kind": "cv_ridge",
        "lambda": float(lam),
        "celeb_weight": float(celeb_weight),
        "short_boost": float(short_boost),
        "groups": 1,
    }


def selection_score(metrics: Mapping[str, float], groups: int, complexity_penalty: float) -> float:
    mae = float(metrics.get("mae", 999.0))
    p90 = float(metrics.get("p90_ae", mae))
    bias = abs(float(metrics.get("bias", 0.0)))
    short = float(metrics.get("short_mae", mae))
    bin_vals = [float(metrics.get(k, mae)) for k in ("short_mae", "medium_mae", "tall_mae")]
    bin_mean = float(np.mean(bin_vals))
    return (
        0.62 * mae
        + 0.20 * bin_mean
        + 0.10 * p90
        + 0.06 * max(0.0, short - mae)
        + 0.02 * bias
        + float(complexity_penalty) * float(groups)
    )


def candidate_rows(
    candidates: Sequence[Mapping[str, Any]],
    dev_y: np.ndarray,
    test_y: np.ndarray,
    dev_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    complexity_penalty: float,
) -> List[Dict[str, Any]]:
    rows = []
    for cand in candidates:
        dev_m = p9.metrics_np(dev_y, np.asarray(cand["oof_pred"], dtype=np.float32), dev_meta)
        test_m = p9.metrics_np(test_y, np.asarray(cand["test_pred"], dtype=np.float32), test_meta)
        row = {k: v for k, v in cand.items() if k not in {"oof_pred", "test_pred"}}
        row["oof"] = dev_m
        row["test"] = test_m
        row["selection_score"] = selection_score(dev_m, int(row.get("groups", 1)), float(complexity_penalty))
        rows.append(row)
    rows.sort(key=lambda item: float(item["selection_score"]))
    return rows


def prior_array(meta: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([float(row.get("prior_mean", 170.0)) for row in meta], dtype=np.float32)


def key_rows(pred: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    prior = prior_array(rows)
    out: List[Dict[str, str]] = []
    for idx, row in enumerate(rows):
        info = p11.token_info(row)
        p = float(pred[idx])
        d = float(prior[idx] - p)
        pred_bin = "p_lt160" if p < 160.0 else ("p_160_168" if p < 168.0 else ("p_168_176" if p < 176.0 else "p_ge176"))
        diff_bin = "prior_lo" if d < -4.0 else ("prior_hi" if d > 4.0 else "prior_mid")
        src = str(row.get("source", "")).upper()
        gender = f"g{int(row.get('gender', 0))}"
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
                "src_prior": f"{src}_{diff_bin}",
                "src_gender_prior": f"{src}_{gender}_{diff_bin}",
                "dialect": str(info.get("dialect", "NA")),
                "language": str(info.get("language", "NA")),
                "n_paths": str(info.get("n_paths_bucket", "NA")),
            }
        )
    return out


def residual_offset_candidate(
    *,
    base: Mapping[str, Any],
    fields: Sequence[str],
    shrinkage: float,
    scale: float,
    dev_y: np.ndarray,
    dev_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    oof_pred = np.asarray(base["oof_pred"], dtype=np.float32)
    test_pred = np.asarray(base["test_pred"], dtype=np.float32)
    dev_keys = key_rows(oof_pred, dev_meta)
    test_keys = key_rows(test_pred, test_meta)
    residual = np.asarray(dev_y, dtype=np.float32) - oof_pred
    groups: Dict[str, List[float]] = defaultdict(list)
    for idx, row in enumerate(dev_keys):
        key = "|".join(row[field] for field in fields)
        groups[key].append(float(residual[idx]))
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

    field_name = "+".join(fields)
    return {
        "name": f"{base['name']}__resid_{field_name}_s{float(shrinkage):g}_x{float(scale):g}",
        "oof_pred": (oof_pred + corrections(dev_keys)).astype(np.float32),
        "test_pred": (test_pred + corrections(test_keys)).astype(np.float32),
        "kind": "oof_residual_offset",
        "base": str(base["name"]),
        "fields": list(fields),
        "shrinkage": float(shrinkage),
        "scale": float(scale),
        "groups": int(len(groups)),
    }


def affine_candidate(base: Mapping[str, Any], dev_y: np.ndarray) -> Dict[str, Any]:
    oof = np.asarray(base["oof_pred"], dtype=np.float32)
    test = np.asarray(base["test_pred"], dtype=np.float32)
    xm = float(oof.mean())
    ym = float(np.asarray(dev_y, dtype=np.float32).mean())
    denom = float(np.sum((oof - xm) ** 2)) + 1e-6
    slope = float(np.sum((oof - xm) * (np.asarray(dev_y, dtype=np.float32) - ym)) / denom)
    slope = float(np.clip(slope, 0.65, 1.35))
    intercept = float(ym - slope * xm)
    return {
        "name": f"{base['name']}__affine_s{slope:.3f}",
        "oof_pred": (slope * oof + intercept).astype(np.float32),
        "test_pred": (slope * test + intercept).astype(np.float32),
        "kind": "oof_affine",
        "base": str(base["name"]),
        "slope": float(slope),
        "intercept": float(intercept),
        "groups": 2,
    }


def add_residual_family(
    bases: Sequence[Mapping[str, Any]],
    dev_y: np.ndarray,
    dev_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    fields_list = [
        ("age",),
        ("source",),
        ("gender",),
        ("src_gender",),
        ("pred_bin",),
        ("prior_diff",),
        ("src_pred",),
        ("gender_pred",),
        ("src_prior",),
        ("src_gender_prior",),
        ("dialect",),
        ("language",),
        ("n_paths",),
    ]
    for base in bases:
        out.append(affine_candidate(base, dev_y))
        for fields in fields_list:
            for shrinkage in (8.0, 20.0, 50.0, 100.0):
                for scale in (0.25, 0.50, 0.75):
                    out.append(
                        residual_offset_candidate(
                            base=base,
                            fields=fields,
                            shrinkage=shrinkage,
                            scale=scale,
                            dev_y=dev_y,
                            dev_meta=dev_meta,
                            test_meta=test_meta,
                        )
                    )
    return out


def pair_blends(
    candidates: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    dev_y: np.ndarray,
    dev_meta: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
    max_pairs: int,
) -> List[Dict[str, Any]]:
    by_name = {str(c["name"]): c for c in candidates}
    top = [row for row in rows[: int(top_n)] if str(row["name"]) in by_name]
    out: List[Dict[str, Any]] = []
    weights = np.arange(0.10, 0.91, 0.05, dtype=np.float32)
    pair_count = 0
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            if pair_count >= int(max_pairs):
                return out
            a = by_name[str(top[i]["name"])]
            b = by_name[str(top[j]["name"])]
            best = None
            av = np.asarray(a["oof_pred"], dtype=np.float32)
            bv = np.asarray(b["oof_pred"], dtype=np.float32)
            at = np.asarray(a["test_pred"], dtype=np.float32)
            bt = np.asarray(b["test_pred"], dtype=np.float32)
            for w in weights:
                oof = w * av + (1.0 - w) * bv
                metric = p9.metrics_np(dev_y, oof, dev_meta)
                score = selection_score(metric, 3, 0.0)
                if best is None or score < best["score"]:
                    best = {"w": float(w), "oof": oof.astype(np.float32), "score": float(score)}
            if best is not None:
                pair_count += 1
                w = float(best["w"])
                out.append(
                    {
                        "name": f"blend_{a['name']}__{b['name']}_w{w:.2f}",
                        "oof_pred": best["oof"],
                        "test_pred": (w * at + (1.0 - w) * bt).astype(np.float32),
                        "kind": "pair_blend",
                        "a": str(a["name"]),
                        "b": str(b["name"]),
                        "weight_a": w,
                        "groups": 3,
                    }
                )
    return out


def write_predictions(
    path: Path,
    y: np.ndarray,
    pred: np.ndarray,
    meta: Sequence[Mapping[str, Any]],
    extras: Mapping[str, Optional[np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["speaker_id", "source", "gender", "height_cm", "phase13_pred_cm", "phase13_abs_error_cm"]
    for name, values in extras.items():
        if values is not None:
            fields.append(name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(meta):
            item = {
                "speaker_id": sid(row),
                "source": str(row.get("source", "")),
                "gender": int(row.get("gender", 0)),
                "height_cm": f"{float(y[idx]):.6f}",
                "phase13_pred_cm": f"{float(pred[idx]):.6f}",
                "phase13_abs_error_cm": f"{abs(float(pred[idx]) - float(y[idx])):.6f}",
            }
            for name, values in extras.items():
                if values is not None:
                    item[name] = f"{float(values[idx]):.6f}"
            writer.writerow(item)


def write_report(
    output_dir: Path,
    selected: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    phase12_metrics: Optional[Mapping[str, float]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Phase 13 Target-CV Residual Stack Report",
        "",
        "## Result",
        f"- Selected method: `{selected['name']}`",
        f"- Development OOF MAE: `{selected['oof']['mae']:.3f}cm`",
        f"- Test MAE: `{selected['test']['mae']:.3f}cm`",
        f"- Test short MAE: `{selected['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Within 3cm: `{selected['test'].get('within_3cm', float('nan')):.3f}`",
        f"- Target 4.5cm met: `{selected['test']['mae'] <= 4.5}`",
        "",
        "## Reference",
    ]
    if phase12_metrics is not None:
        delta = float(selected["test"]["mae"]) - float(phase12_metrics["mae"])
        lines.append(
            f"- Phase 12 frontier: `{phase12_metrics['mae']:.3f}cm`, short `{phase12_metrics.get('short_mae', float('nan')):.3f}cm`, delta `{delta:+.3f}cm`"
        )
    lines.extend(
        [
            f"- Folds: `{int(args.folds)}`",
            f"- Repeats: `{int(args.repeats)}`",
            f"- Candidates searched: `{len(rows)}`",
            f"- Complexity penalty per group: `{float(args.complexity_penalty):.4f}`",
            "",
            "## Top OOF Candidates",
        ]
    )
    for row in rows[:30]:
        lines.append(
            f"- `{row['name']}`: oof `{row['oof']['mae']:.3f}cm`, test `{row['test']['mae']:.3f}cm`, "
            f"short `{row['test'].get('short_mae', float('nan')):.3f}cm`, score `{row['selection_score']:.3f}`"
        )
    lines.extend(["", "## Conclusion"])
    if phase12_metrics is not None and float(selected["test"]["mae"]) < float(phase12_metrics["mae"]):
        lines.append("Phase 13 improves the sealed-test frontier using target-domain OOF calibration.")
    else:
        lines.append("Phase 13 did not beat the guarded Phase 12 frontier; keep Phase 12 as the best measured checkpoint.")
    (output_dir / "PHASE13_TARGET_CV_RESIDUAL_STACK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = torch.device(str(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Phase13 is CUDA-only for this run.")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase13] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    phase9_cache = p9.load_cache(resolve(args.phase9_cache))
    row_by_sid = split_lookup(resolve(args.splits_dir))
    train_meta_all = enrich_meta(phase9_cache["train"]["meta"], row_by_sid)
    val_meta = enrich_meta(phase9_cache["val"]["meta"], row_by_sid)
    test_meta = enrich_meta(phase9_cache["test"]["meta"], row_by_sid)
    target_train_idx = np.asarray([i for i, row in enumerate(train_meta_all) if target_source(row)], dtype=np.int64)
    support_idx = np.asarray([i for i, row in enumerate(train_meta_all) if not target_source(row)], dtype=np.int64)

    dev_y = np.concatenate([phase9_cache["train"]["y"][target_train_idx], phase9_cache["val"]["y"]], axis=0).astype(np.float32)
    test_y = np.asarray(phase9_cache["test"]["y"], dtype=np.float32)
    dev_meta_full = [train_meta_all[int(i)] for i in target_train_idx.tolist()] + list(val_meta)
    support_meta = [train_meta_all[int(i)] for i in support_idx.tolist()]
    test_meta_metrics = meta_for_metrics(test_meta)
    dev_meta_metrics = meta_for_metrics(dev_meta_full)

    train_ecapa_target = phase9_cache["train"]["x"][target_train_idx]
    support_ecapa = phase9_cache["train"]["x"][support_idx]
    val_ecapa = phase9_cache["val"]["x"]
    test_ecapa = phase9_cache["test"]["x"]
    dev_ecapa = np.concatenate([train_ecapa_target, val_ecapa], axis=0).astype(np.float32)
    support_y = phase9_cache["train"]["y"][support_idx].astype(np.float32)

    feature_sets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "ecapa": (dev_ecapa, support_ecapa.astype(np.float32), test_ecapa.astype(np.float32)),
        "ecapa_meta": (
            append_features(dev_ecapa, metadata_matrix(dev_meta_full)),
            append_features(support_ecapa, metadata_matrix(support_meta)),
            append_features(test_ecapa, metadata_matrix(test_meta)),
        ),
    }
    biomarker = load_biomarker_cache(resolve(args.biomarker_cache))
    if biomarker is not None:
        bio_train_all, bio_train_present = align_features(train_meta_all, biomarker["train"]["x"], biomarker["train"]["meta"])
        bio_val, bio_val_present = align_features(val_meta, biomarker["val"]["x"], biomarker["val"]["meta"])
        bio_test, bio_test_present = align_features(test_meta, biomarker["test"]["x"], biomarker["test"]["meta"])
        dev_bio = np.concatenate([bio_train_all[target_train_idx], bio_val], axis=0)
        dev_bio_present = np.concatenate([bio_train_present[target_train_idx], bio_val_present], axis=0)
        support_bio = bio_train_all[support_idx]
        support_bio_present = bio_train_present[support_idx]
        feature_sets["ecapa_bio_meta"] = (
            append_features(dev_ecapa, dev_bio, dev_bio_present, metadata_matrix(dev_meta_full)),
            append_features(support_ecapa, support_bio, support_bio_present, metadata_matrix(support_meta)),
            append_features(test_ecapa, bio_test, bio_test_present, metadata_matrix(test_meta)),
        )
        print("[phase13] biomarker cache loaded and aligned", flush=True)
    else:
        print("[phase13] biomarker cache unavailable; continuing with ECAPA features", flush=True)

    folds = make_folds(dev_meta_full, dev_y, int(args.folds), int(args.repeats), int(args.seed))
    print(f"[phase13] dev target speakers={len(dev_y)} support={len(support_y)} test={len(test_y)} folds={len(folds)}", flush=True)

    candidates: List[Dict[str, Any]] = []
    lambdas = (30.0, 100.0, 300.0, 1000.0, 3000.0)
    short_boosts = (1.0, 1.15, 1.35)
    for feat_name, (dev_x, support_x, test_x) in feature_sets.items():
        print(f"[phase13] feature set {feat_name}: dim={dev_x.shape[1]}", flush=True)
        for support_mode in ("target", "celeb"):
            celeb_weights = (0.0,) if support_mode == "target" else (0.06, 0.12, 0.24)
            for celeb_weight in celeb_weights:
                for short_boost in short_boosts:
                    for lam in lambdas:
                        name = f"{feat_name}_{support_mode}_ridge_lam{lam:g}_cw{celeb_weight:g}_sb{short_boost:g}"
                        candidates.append(
                            cv_ridge_candidate(
                                name=name,
                                dev_x=dev_x,
                                dev_y=dev_y,
                                dev_meta=dev_meta_full,
                                support_x=support_x,
                                support_y=support_y,
                                support_meta=support_meta,
                                test_x=test_x,
                                folds=folds,
                                lam=lam,
                                celeb_weight=celeb_weight,
                                short_boost=short_boost,
                                device=device,
                            )
                        )
                        if len(candidates) % 25 == 0:
                            print(f"[phase13] base candidates searched={len(candidates)}", flush=True)

    rows = candidate_rows(candidates, dev_y, test_y, dev_meta_metrics, test_meta_metrics, float(args.complexity_penalty))
    print(f"[phase13] best base oof={rows[0]['oof']['mae']:.3f} test={rows[0]['test']['mae']:.3f} {rows[0]['name']}", flush=True)
    by_name = {str(c["name"]): c for c in candidates}
    top_bases = [by_name[str(row["name"])] for row in rows[: int(args.top_base_for_residuals)]]
    residuals = add_residual_family(top_bases, dev_y, dev_meta_full, test_meta)
    print(f"[phase13] residual candidates={len(residuals)}", flush=True)
    candidates.extend(residuals)
    rows = candidate_rows(candidates, dev_y, test_y, dev_meta_metrics, test_meta_metrics, float(args.complexity_penalty))
    blends = pair_blends(candidates, rows, dev_y, dev_meta_metrics, top_n=int(args.top_for_blends), max_pairs=int(args.max_pair_blends))
    print(f"[phase13] pair blends={len(blends)}", flush=True)
    candidates.extend(blends)
    rows = candidate_rows(candidates, dev_y, test_y, dev_meta_metrics, test_meta_metrics, float(args.complexity_penalty))
    selected = rows[0]
    selected_cand = next(c for c in candidates if str(c["name"]) == str(selected["name"]))
    selected_pred = np.asarray(selected_cand["test_pred"], dtype=np.float32)

    phase12_pred = read_phase12_predictions(resolve(args.phase12_test_pred), test_meta)
    phase12_metrics = p9.metrics_np(test_y, phase12_pred, test_meta_metrics) if phase12_pred is not None else None
    write_predictions(
        output_dir / "phase13_predictions_test.csv",
        test_y,
        selected_pred,
        test_meta,
        {"phase12_pred_cm": phase12_pred},
    )
    report = {
        "selected": selected,
        "phase12_reference": phase12_metrics,
        "candidate_count": len(candidates),
        "top_candidates": rows[:120],
        "args": vars(args),
    }
    (output_dir / "phase13_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_report(output_dir, selected, rows, phase12_metrics, args)
    print(
        f"[phase13] selected={selected['name']} test_mae={selected['test']['mae']:.3f} "
        f"short={selected['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    if phase12_metrics is not None:
        print(f"[phase13] phase12_reference={phase12_metrics['mae']:.3f}", flush=True)
    print(f"[phase13] wrote {output_dir / 'PHASE13_TARGET_CV_RESIDUAL_STACK_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
