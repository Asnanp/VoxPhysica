#!/usr/bin/env python
"""Phase 11 metadata-aware tail calibrator.

This phase targets the persistent short-speaker failure. It uses only metadata
available in the canonical split CSVs: source, gender, age, TIMIT dialect
region, and NISP language/path family. It builds leave-one-out group priors for
train speakers, trains conservative calibrators, then blends with the Phase 9
frontier through validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 11 metadata tail calibrator.")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--phase3-val-pred", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test-pred", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--phase9-val-rebuild-cache", default="outputs/phase9_ecapa_prior_stack/ecapa_m6_s6p0_limit0_celeb.npz")
    parser.add_argument("--output-dir", default="outputs/phase11_metadata_tail_calibrator")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--skip-xgb", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def gender_id(raw: str) -> int:
    text = str(raw or "").strip().lower()
    if text == "male":
        return 1
    if text == "female":
        return 0
    return int(safe_float(text, 0.0) >= 0.5)


def read_split(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sid = str(row.get("speaker_id", "")).strip()
            height = safe_float(row.get("height_cm"))
            if not sid or not math.isfinite(height):
                continue
            paths = [part.strip() for part in str(row.get("audio_paths", "")).split("|") if part.strip()]
            rows.append(
                {
                    "speaker_id": sid,
                    "source": str(row.get("source", "")).upper(),
                    "gender": gender_id(str(row.get("gender", ""))),
                    "height_cm": float(height),
                    "age": safe_float(row.get("age"), 0.0),
                    "audio_paths": paths,
                }
            )
    return rows


def token_info(row: Mapping[str, Any]) -> Dict[str, Any]:
    sid = str(row["speaker_id"])
    paths = list(row.get("audio_paths") or [])
    joined = "|".join(paths)
    source = str(row.get("source", "")).upper()
    out = {
        "source": source,
        "gender": int(row.get("gender", 0)),
        "age_bucket": age_bucket(float(row.get("age", 0.0))),
        "n_paths_bucket": n_bucket(len(paths)),
        "dialect": "NA",
        "subset": "NA",
        "language": "NA",
        "speaker_prefix": sid.split("_", 1)[1][:3] if "_" in sid else sid[:3],
    }
    if source == "TIMIT":
        m = re.search(r"[\\/](DR[1-8])[\\/]", joined, flags=re.IGNORECASE)
        out["dialect"] = m.group(1).upper() if m else "DRX"
        out["subset"] = "TIMIT_TEST" if re.search(r"[\\/]TEST[\\/]", joined, flags=re.IGNORECASE) else "TIMIT_TRAIN"
    elif source == "NISP":
        prefix = sid.split("_", 1)[1].split("_", 1)[0] if "_" in sid else sid[:3]
        out["language"] = prefix[:3].title()
        for name in ("Hindi", "Tamil", "Kannada", "Malayalam"):
            if name.lower() in joined.lower():
                out["language"] = name[:3].title()
                break
    return out


def age_bucket(age: float) -> str:
    if not math.isfinite(age) or age <= 0:
        return "age_unknown"
    if age < 22:
        return "age_lt22"
    if age < 28:
        return "age_22_27"
    if age < 36:
        return "age_28_35"
    if age < 50:
        return "age_36_49"
    return "age_50p"


def n_bucket(n: int) -> str:
    if n <= 4:
        return "clips_0_4"
    if n <= 8:
        return "clips_5_8"
    if n <= 20:
        return "clips_9_20"
    if n <= 50:
        return "clips_21_50"
    return "clips_50p"


GROUP_KEYS = (
    ("source",),
    ("gender",),
    ("source", "gender"),
    ("source", "gender", "age_bucket"),
    ("source", "gender", "dialect"),
    ("source", "gender", "language"),
    ("source", "gender", "n_paths_bucket"),
    ("source", "dialect"),
    ("source", "language"),
    ("gender", "age_bucket"),
)


def group_name(info: Mapping[str, Any], keys: Sequence[str]) -> str:
    return "::".join(str(info.get(k, "NA")) for k in keys)


def shrink_mean(values: Sequence[float], global_mean: float, shrinkage: float) -> float:
    if not values:
        return float(global_mean)
    vals = np.asarray(values, dtype=np.float32)
    n = float(len(vals))
    return float((float(vals.mean()) * n + float(global_mean) * shrinkage) / (n + shrinkage))


def build_group_tables(train_rows: Sequence[Mapping[str, Any]], shrinkage: float) -> Tuple[Dict[str, Dict[str, List[float]]], float]:
    global_mean = float(np.mean([float(r["height_cm"]) for r in train_rows]))
    tables: Dict[str, Dict[str, List[float]]] = {"+".join(keys): defaultdict(list) for keys in GROUP_KEYS}
    for row in train_rows:
        info = token_info(row)
        for keys in GROUP_KEYS:
            tables["+".join(keys)][group_name(info, keys)].append(float(row["height_cm"]))
    return tables, global_mean


def prior_features(
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    *,
    leave_one_out: bool,
    shrinkage: float,
) -> Tuple[np.ndarray, List[str]]:
    tables, global_mean = build_group_tables(train_rows, shrinkage)
    names: List[str] = []
    matrix: List[List[float]] = []
    for row in rows:
        info = token_info(row)
        values: List[float] = []
        for keys in GROUP_KEYS:
            table_key = "+".join(keys)
            if not names or len(names) < len(GROUP_KEYS):
                names.append("prior_" + table_key)
            bucket = group_name(info, keys)
            group_vals = list(tables[table_key].get(bucket, []))
            if leave_one_out:
                h = float(row["height_cm"])
                for i, v in enumerate(group_vals):
                    if abs(float(v) - h) < 1e-5:
                        group_vals.pop(i)
                        break
            values.append(shrink_mean(group_vals, global_mean, shrinkage))
        values.extend(
            [
                float(row.get("gender", 0)),
                float(row.get("age", 0.0)) / 100.0,
                float(len(row.get("audio_paths") or [])) / 100.0,
                1.0 if str(row.get("source", "")).upper() == "TIMIT" else 0.0,
                1.0 if str(row.get("source", "")).upper() == "NISP" else 0.0,
            ]
        )
        matrix.append(values)
    names.extend(["gender", "age_scaled", "n_paths_scaled", "is_timit", "is_nisp"])
    return np.asarray(matrix, dtype=np.float32), names


def one_hot_features(rows: Sequence[Mapping[str, Any]], vocab: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    if vocab is None:
        vocab = {}
        for row in rows:
            info = token_info(row)
            for key, value in info.items():
                token = f"{key}={value}"
                if token not in vocab:
                    vocab[token] = len(vocab)
    x = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for i, row in enumerate(rows):
        info = token_info(row)
        for key, value in info.items():
            token = f"{key}={value}"
            if token in vocab:
                x[i, vocab[token]] = 1.0
    return x, vocab


def y_array(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([float(r["height_cm"]) for r in rows], dtype=np.float32)


def meta_for(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        h = float(row["height_cm"])
        out.append(
            {
                "speaker_id": row["speaker_id"],
                "source": row["source"],
                "gender": row["gender"],
                "height_cm": h,
                "height_bin": 0 if h < 160 else (1 if h < 175 else 2),
            }
        )
    return out


def read_pred_csv(path: Path, column: str = "final_pred_cm") -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sid = str(row.get("speaker_id", "")).strip()
            pred = safe_float(row.get(column))
            if sid and math.isfinite(pred):
                out[sid] = pred
    return out


def align_pred(rows: Sequence[Mapping[str, Any]], preds: Mapping[str, float]) -> Optional[np.ndarray]:
    vals = []
    for row in rows:
        sid = str(row["speaker_id"])
        if sid not in preds:
            return None
        vals.append(float(preds[sid]))
    return np.asarray(vals, dtype=np.float32)


def source_weights(rows: Sequence[Mapping[str, Any]], short_boost: float = 1.0) -> np.ndarray:
    weights = []
    for row in rows:
        w = 1.0
        if float(row["height_cm"]) < 160.0:
            w *= short_boost
        weights.append(w)
    arr = np.asarray(weights, dtype=np.float32)
    return arr / max(float(arr.mean()), 1e-6)


def calibrate_affine(val_y: np.ndarray, val_pred: np.ndarray, test_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x = np.asarray(val_pred, dtype=np.float32)
    y = np.asarray(val_y, dtype=np.float32)
    xm = float(x.mean())
    ym = float(y.mean())
    slope = float(np.sum((x - xm) * (y - ym)) / (np.sum((x - xm) ** 2) + 1e-6))
    slope = float(np.clip(slope, 0.45, 1.65))
    intercept = ym - slope * xm
    return (slope * val_pred + intercept).astype(np.float32), (slope * test_pred + intercept).astype(np.float32), {"slope": slope, "intercept": intercept}


def score(metrics: Mapping[str, float]) -> float:
    mae = float(metrics["mae"])
    short = float(metrics.get("short_mae", mae))
    medium = float(metrics.get("medium_mae", mae))
    tall = float(metrics.get("tall_mae", mae))
    p90 = float(metrics.get("p90_ae", mae))
    return 0.58 * mae + 0.28 * float(np.mean([short, medium, tall])) + 0.08 * p90 + 0.06 * max(0.0, short - mae)


def candidate_rows(candidates: Sequence[Mapping[str, Any]], val_y: np.ndarray, test_y: np.ndarray, val_meta: Sequence[Mapping[str, Any]], test_meta: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for cand in candidates:
        val_m = p9.metrics_np(val_y, cand["val_pred"], val_meta)
        test_m = p9.metrics_np(test_y, cand["test_pred"], test_meta)
        row = {k: v for k, v in cand.items() if k not in {"val_pred", "test_pred"}}
        row["val"] = val_m
        row["test"] = test_m
        row["score"] = score(val_m)
        rows.append(row)
    rows.sort(key=lambda x: float(x["score"]))
    return rows


def simplex_blends(anchors: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]], val_y: np.ndarray, val_meta: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    a0, a1 = anchors
    a0v, a0t = np.asarray(a0["val_pred"], dtype=np.float32), np.asarray(a0["test_pred"], dtype=np.float32)
    a1v, a1t = np.asarray(a1["val_pred"], dtype=np.float32), np.asarray(a1["test_pred"], dtype=np.float32)
    grid = np.arange(0.0, 1.0001, 0.02, dtype=np.float32)
    for cand in candidates:
        cv, ct = np.asarray(cand["val_pred"], dtype=np.float32), np.asarray(cand["test_pred"], dtype=np.float32)
        best = None
        for w0 in grid:
            for w1 in grid:
                if float(w0 + w1) > 1.0001:
                    continue
                w2 = np.float32(1.0) - w0 - w1
                val_pred = w0 * a0v + w1 * a1v + w2 * cv
                s = score(p9.metrics_np(val_y, val_pred, val_meta))
                if best is None or s < best["score"]:
                    best = {
                        "name": f"simplex_{a0['name']}__{a1['name']}__{cand['name']}_w{float(w0):.2f}_{float(w1):.2f}_{float(w2):.2f}",
                        "val_pred": val_pred.astype(np.float32),
                        "test_pred": (w0 * a0t + w1 * a1t + w2 * ct).astype(np.float32),
                        "weights": {a0["name"]: float(w0), a1["name"]: float(w1), cand["name"]: float(w2)},
                        "kind": "simplex",
                        "score": float(s),
                    }
        if best is not None:
            out.append(best)
    return out


def rebuild_phase9_frontier(cache_path: Path, phase3_val: np.ndarray, phase3_test: np.ndarray, val_y: np.ndarray, val_meta: Sequence[Mapping[str, Any]], test_y: np.ndarray, test_meta: Sequence[Mapping[str, Any]], device: torch.device) -> Mapping[str, Any]:
    data = p9.load_cache(cache_path)
    train_x = data["train"]["x"]
    train_y = data["train"]["y"]
    val_x = data["val"]["x"]
    test_x = data["test"]["x"]
    train_meta = data["train"]["meta"]
    target_mask = np.asarray([p9.is_target_source(row["source"]) for row in train_meta], dtype=bool)
    candidates = []
    candidates.extend(
        p9.weighted_ridge_predict(train_x, train_y, val_x, test_x, sample_weight=np.ones_like(train_y, dtype=np.float32), lambdas=[10.0, 30.0, 100.0, 300.0, 1000.0], device=device, label="p9_all")
    )
    candidates.extend(
        p9.weighted_ridge_predict(train_x, train_y, val_x, test_x, sample_weight=np.where(target_mask, 1.0, 0.25).astype(np.float32), lambdas=[10.0, 30.0, 100.0, 300.0, 1000.0], device=device, label="p9_target_weighted")
    )
    candidates.extend(p9.choose_blends(candidates[:], phase3_val, phase3_test, val_y, val_meta))
    rows = candidate_rows(candidates, val_y, test_y, val_meta, test_meta)
    name = rows[0]["name"]
    for cand in candidates:
        if cand["name"] == name:
            out = dict(cand)
            out["name"] = "phase9_frontier_rebuilt"
            return out
    raise RuntimeError("Could not rebuild Phase9 frontier")


def xgb_candidates(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, test_x: np.ndarray, train_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    try:
        import xgboost as xgb
    except Exception:
        return []
    tr, va, te = p9.robust_standardize(train_x, val_x, test_x)
    configs = [
        {"max_depth": 1, "learning_rate": 0.035, "n_estimators": 260, "reg_lambda": 8.0, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 4.0},
        {"max_depth": 2, "learning_rate": 0.025, "n_estimators": 320, "reg_lambda": 18.0, "subsample": 0.85, "colsample_bytree": 0.7, "min_child_weight": 8.0},
        {"max_depth": 2, "learning_rate": 0.018, "n_estimators": 460, "reg_lambda": 35.0, "subsample": 0.8, "colsample_bytree": 0.55, "min_child_weight": 12.0},
    ]
    out = []
    for short_boost in (1.0, 1.35, 1.7):
        w = source_weights(train_rows, short_boost)
        for idx, cfg in enumerate(configs):
            model = xgb.XGBRegressor(objective="reg:absoluteerror", tree_method="hist", device="cuda", random_state=200 + idx, n_jobs=1, **cfg)
            model.fit(tr, train_y, sample_weight=w, verbose=False)
            out.append({"name": f"metadata_xgb{idx}_sb{short_boost:g}", "val_pred": model.predict(va).astype(np.float32), "test_pred": model.predict(te).astype(np.float32), "kind": "xgb", "config": cfg})
    return out


def write_predictions(path: Path, y: np.ndarray, pred: np.ndarray, meta: Sequence[Mapping[str, Any]], extras: Mapping[str, np.ndarray]) -> None:
    fields = ["speaker_id", "source", "gender", "height_cm", "phase11_pred_cm", "phase11_abs_error_cm", *extras.keys()]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(meta):
            item = {
                "speaker_id": row["speaker_id"],
                "source": row["source"],
                "gender": row["gender"],
                "height_cm": f"{float(y[i]):.6f}",
                "phase11_pred_cm": f"{float(pred[i]):.6f}",
                "phase11_abs_error_cm": f"{abs(float(pred[i]) - float(y[i])):.6f}",
            }
            for name, values in extras.items():
                item[name] = f"{float(values[i]):.6f}"
            writer.writerow(item)


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA required for Phase11.")
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = resolve(args.splits_dir)
    train_rows = read_split(splits / "train_clean.csv")
    val_rows = read_split(splits / "val_clean.csv")
    test_rows = read_split(splits / "test_clean.csv")
    train_y, val_y, test_y = y_array(train_rows), y_array(val_rows), y_array(test_rows)
    val_meta, test_meta = meta_for(val_rows), meta_for(test_rows)

    train_oh, vocab = one_hot_features(train_rows)
    val_oh, _ = one_hot_features(val_rows, vocab)
    test_oh, _ = one_hot_features(test_rows, vocab)

    candidates: List[Dict[str, Any]] = []
    for shrinkage in (2.0, 5.0, 10.0, 20.0, 40.0):
        train_prior, _ = prior_features(train_rows, train_rows, leave_one_out=True, shrinkage=shrinkage)
        val_prior, _ = prior_features(val_rows, train_rows, leave_one_out=False, shrinkage=shrinkage)
        test_prior, _ = prior_features(test_rows, train_rows, leave_one_out=False, shrinkage=shrinkage)
        for idx in range(train_prior.shape[1] - 5):
            val_pred = val_prior[:, idx].astype(np.float32)
            test_pred = test_prior[:, idx].astype(np.float32)
            cal_val, cal_test, cal = calibrate_affine(val_y, val_pred, test_pred)
            candidates.append({"name": f"group_prior{idx}_s{shrinkage:g}", "val_pred": val_pred, "test_pred": test_pred, "kind": "group_prior"})
            candidates.append({"name": f"group_prior{idx}_s{shrinkage:g}_affine", "val_pred": cal_val, "test_pred": cal_test, "kind": "group_prior_affine", "calibration": cal})
        train_x = np.concatenate([train_prior, train_oh], axis=1)
        val_x = np.concatenate([val_prior, val_oh], axis=1)
        test_x = np.concatenate([test_prior, test_oh], axis=1)
        for short_boost in (1.0, 1.3, 1.7):
            candidates.extend(
                p9.weighted_ridge_predict(
                    train_x,
                    train_y,
                    val_x,
                    test_x,
                    sample_weight=source_weights(train_rows, short_boost),
                    lambdas=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
                    device=device,
                    label=f"metadata_s{shrinkage:g}_sb{short_boost:g}",
                )
            )
        if not args.skip_xgb and shrinkage in {5.0, 10.0, 20.0}:
            candidates.extend(xgb_candidates(train_x, train_y, val_x, test_x, train_rows))

    phase3_val = align_pred(val_rows, read_pred_csv(resolve(args.phase3_val_pred), "final_pred_cm"))
    phase3_test = align_pred(test_rows, read_pred_csv(resolve(args.phase3_test_pred), "final_pred_cm"))
    if phase3_val is None or phase3_test is None:
        raise RuntimeError("Phase3 predictions did not align")
    phase9_frontier = rebuild_phase9_frontier(resolve(args.phase9_val_rebuild_cache), phase3_val, phase3_test, val_y, val_meta, test_y, test_meta, device)
    anchors = [
        {"name": "phase3_frontier", "val_pred": phase3_val, "test_pred": phase3_test, "kind": "anchor"},
        phase9_frontier,
    ]
    all_candidates = anchors + candidates + p9.choose_blends(candidates, phase3_val, phase3_test, val_y, val_meta) + simplex_blends(anchors, candidates, val_y, val_meta)
    rows = candidate_rows(all_candidates, val_y, test_y, val_meta, test_meta)
    selected = rows[0]
    selected_cand = next(c for c in all_candidates if c["name"] == selected["name"])
    selected_pred = np.asarray(selected_cand["test_pred"], dtype=np.float32)
    selected_val_pred = np.asarray(selected_cand["val_pred"], dtype=np.float32)
    phase9_metrics = p9.metrics_np(test_y, np.asarray(phase9_frontier["test_pred"], dtype=np.float32), test_meta)
    phase3_metrics = p9.metrics_np(test_y, phase3_test, test_meta)

    write_predictions(
        output_dir / "phase11_predictions_val.csv",
        val_y,
        selected_val_pred,
        val_meta,
        {"phase3_pred_cm": phase3_val, "phase9_pred_cm": np.asarray(phase9_frontier["val_pred"], dtype=np.float32)},
    )
    write_predictions(output_dir / "phase11_predictions_test.csv", test_y, selected_pred, test_meta, {"phase3_pred_cm": phase3_test, "phase9_pred_cm": np.asarray(phase9_frontier["test_pred"], dtype=np.float32)})
    report = {"selected": selected, "phase9_reference": phase9_metrics, "phase3_reference": phase3_metrics, "candidate_count": len(all_candidates), "top_candidates": rows[:80], "args": vars(args)}
    (output_dir / "phase11_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")

    lines = [
        "# Phase 11 Metadata Tail Calibrator Report",
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
        f"- Candidates searched: `{len(all_candidates)}`",
        "",
        "## Top Validation Candidates",
    ]
    for row in rows[:20]:
        lines.append(f"- `{row['name']}`: val `{row['val']['mae']:.3f}cm`, test `{row['test']['mae']:.3f}cm`, short `{row['test'].get('short_mae', float('nan')):.3f}cm`, score `{row['score']:.3f}`")
    lines.extend(["", "## Conclusion"])
    if selected["test"]["mae"] < phase9_metrics["mae"]:
        lines.append("Metadata-aware tail calibration improves the sealed-test frontier.")
    else:
        lines.append("Metadata-aware tail calibration did not beat Phase 9 under validation selection; keep Phase 9 as frontier.")
    (output_dir / "PHASE11_METADATA_TAIL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[phase11] selected={selected['name']}", flush=True)
    print(f"[phase11] test_mae={selected['test']['mae']:.3f} short={selected['test'].get('short_mae', float('nan')):.3f} phase9={phase9_metrics['mae']:.3f}", flush=True)
    print(f"[phase11] wrote {output_dir / 'PHASE11_METADATA_TAIL_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
