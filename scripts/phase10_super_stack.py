#!/usr/bin/env python
"""Phase 10 super stack over the current frontier.

Phase 9 finally moved the sealed speaker MAE meaningfully by adding ECAPA and a
VoxCeleb height prior. Phase 10 keeps that signal and adds:

- speaker-level biomarker aggregates from Phase 8,
- ECAPA + biomarker joint ridge/KRR/KNN candidates,
- optional GPU XGBoost candidates,
- simplex blends against Phase 3 and the Phase 9 frontier.

Selection is validation-gated. The sealed test is evaluated once per candidate
for reporting, but the selected method is the best validation score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase8_biomarker_cv as p8  # noqa: E402
import phase9_ecapa_prior_stack as p9  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 10 ECAPA+biomarker super stack.")
    parser.add_argument("--phase9-cache", default="outputs/phase9_ecapa_prior_stack/ecapa_m6_s6p0_limit0_celeb.npz")
    parser.add_argument("--features-root", default="data/features_v4_combo_full_ssl")
    parser.add_argument("--phase3-val-pred", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test-pred", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase10_super_stack")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rebuild-biomarker-cache", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def load_biomarker_tables(features_root: Path, output_dir: Path, rebuild: bool) -> Dict[str, Dict[str, Any]]:
    cache = output_dir / "phase10_biomarker_cache.npz"
    if cache.exists() and not rebuild:
        out: Dict[str, Dict[str, Any]] = {}
        with np.load(cache, allow_pickle=True) as data:
            names = json.loads(str(np.asarray(data["names_json"]).item()))
            for split in ("train", "val", "test"):
                out[split] = {
                    "x": np.asarray(data[f"{split}_x"], dtype=np.float32),
                    "y": np.asarray(data[f"{split}_y"], dtype=np.float32),
                    "meta": json.loads(str(np.asarray(data[f"{split}_meta_json"]).item())),
                    "names": names,
                }
        print(f"[phase10] loaded biomarker cache {cache}", flush=True)
        return out

    out = {}
    names: List[str] = []
    for split in ("train", "val", "test"):
        x, y, meta, names = p8.load_split(features_root, split, include_augmented=False)
        out[split] = {"x": x, "y": y, "meta": meta, "names": names}
        print(f"[phase10] biomarker {split}: speakers={len(meta)} dim={x.shape[1]}", flush=True)
    payload: Dict[str, Any] = {"names_json": np.asarray(json.dumps(names))}
    for split in ("train", "val", "test"):
        payload[f"{split}_x"] = out[split]["x"]
        payload[f"{split}_y"] = out[split]["y"]
        payload[f"{split}_meta_json"] = np.asarray(json.dumps(out[split]["meta"], ensure_ascii=False))
    np.savez_compressed(cache, **payload)
    print(f"[phase10] saved biomarker cache {cache}", flush=True)
    return out


def sid(row: Mapping[str, Any]) -> str:
    return str(row.get("speaker_id", "")).strip()


def align_table(
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


def read_pred(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("speaker_id", "")).strip()
            val = p9.safe_float(row.get("final_pred_cm", row.get("phase9_pred_cm", "")))
            if key and math.isfinite(val):
                out[key] = float(val)
    return out


def align_pred(meta: Sequence[Mapping[str, Any]], preds: Mapping[str, float]) -> Optional[np.ndarray]:
    vals: List[float] = []
    for row in meta:
        key = sid(row)
        if key not in preds:
            return None
        vals.append(float(preds[key]))
    return np.asarray(vals, dtype=np.float32)


def metric_score(metrics: Mapping[str, float], mode: str = "balanced") -> float:
    mae = float(metrics.get("mae", 999.0))
    short = float(metrics.get("short_mae", mae))
    medium = float(metrics.get("medium_mae", mae))
    tall = float(metrics.get("tall_mae", mae))
    p90 = float(metrics.get("p90_ae", mae))
    bias = abs(float(metrics.get("bias", 0.0)))
    bin_mean = float(np.mean([short, medium, tall]))
    if mode == "mae":
        return mae
    if mode == "tail":
        return 0.55 * mae + 0.35 * bin_mean + 0.06 * p90 + 0.12 * max(0.0, short - mae) + 0.04 * bias
    if mode == "short":
        return 0.45 * mae + 0.35 * short + 0.15 * bin_mean + 0.05 * p90
    return 0.60 * mae + 0.28 * bin_mean + 0.06 * p90 + 0.06 * bias


def candidate_rows(
    candidates: Sequence[Mapping[str, Any]],
    val_y: np.ndarray,
    test_y: np.ndarray,
    val_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    score_mode: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        val_m = p9.metrics_np(val_y, np.asarray(cand["val_pred"], dtype=np.float32), val_meta)
        test_m = p9.metrics_np(test_y, np.asarray(cand["test_pred"], dtype=np.float32), test_meta)
        row = {k: v for k, v in cand.items() if k not in {"val_pred", "test_pred"}}
        row["val"] = val_m
        row["test"] = test_m
        row["score"] = metric_score(val_m, score_mode)
        rows.append(row)
    rows.sort(key=lambda item: float(item["score"]))
    return rows


def get_candidate(candidates: Sequence[Mapping[str, Any]], name: str) -> Optional[Mapping[str, Any]]:
    for cand in candidates:
        if cand.get("name") == name:
            return cand
    return None


def source_weights(meta: Sequence[Mapping[str, Any]], celeb_weight: float, short_boost: float = 1.0) -> np.ndarray:
    weights = []
    for row in meta:
        source = str(row.get("source", "")).upper()
        w = float(celeb_weight) if source == "CELEB" else 1.0
        if float(row.get("height_cm", 0.0)) < 160.0:
            w *= float(short_boost)
        weights.append(w)
    arr = np.asarray(weights, dtype=np.float32)
    return arr / max(float(arr.mean()), 1e-6)


def xgb_candidates(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_meta: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> List[Dict[str, Any]]:
    try:
        import xgboost as xgb
    except Exception as exc:
        print(f"[phase10] XGBoost unavailable: {exc}", flush=True)
        return []

    x_train, x_val, x_test = p9.robust_standardize(train_x, val_x, test_x)
    configs = [
        {"max_depth": 2, "learning_rate": 0.025, "n_estimators": 420, "subsample": 0.78, "colsample_bytree": 0.45, "reg_lambda": 10.0, "min_child_weight": 8.0},
        {"max_depth": 2, "learning_rate": 0.018, "n_estimators": 650, "subsample": 0.85, "colsample_bytree": 0.55, "reg_lambda": 25.0, "min_child_weight": 12.0},
        {"max_depth": 3, "learning_rate": 0.018, "n_estimators": 520, "subsample": 0.75, "colsample_bytree": 0.38, "reg_lambda": 35.0, "min_child_weight": 16.0},
        {"max_depth": 1, "learning_rate": 0.030, "n_estimators": 500, "subsample": 0.90, "colsample_bytree": 0.65, "reg_lambda": 5.0, "min_child_weight": 5.0},
    ]
    out: List[Dict[str, Any]] = []
    for celeb_weight in (0.08, 0.18, 0.32):
        w = source_weights(train_meta, celeb_weight=celeb_weight, short_boost=1.15)
        for idx, cfg in enumerate(configs):
            model = xgb.XGBRegressor(
                objective="reg:absoluteerror",
                tree_method="hist",
                device="cuda",
                random_state=1000 + idx,
                n_jobs=1,
                **cfg,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, train_y, sample_weight=w, verbose=False)
            out.append(
                {
                    "name": f"{label}_xgb{idx}_cw{celeb_weight:g}",
                    "val_pred": model.predict(x_val).astype(np.float32),
                    "test_pred": model.predict(x_test).astype(np.float32),
                    "kind": "xgb",
                    "celeb_weight": float(celeb_weight),
                    "config": cfg,
                }
            )
    return out


def simplex_blends(
    anchors: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    val_y: np.ndarray,
    val_meta: Sequence[Mapping[str, Any]],
    *,
    step: float = 0.04,
    score_mode: str = "balanced",
) -> List[Dict[str, Any]]:
    if len(anchors) < 2:
        return []
    out: List[Dict[str, Any]] = []
    grid = np.arange(0.0, 1.0001, step, dtype=np.float32)
    a0, a1 = anchors[0], anchors[1]
    a0v = np.asarray(a0["val_pred"], dtype=np.float32)
    a0t = np.asarray(a0["test_pred"], dtype=np.float32)
    a1v = np.asarray(a1["val_pred"], dtype=np.float32)
    a1t = np.asarray(a1["test_pred"], dtype=np.float32)
    for cand in candidates:
        cv = np.asarray(cand["val_pred"], dtype=np.float32)
        ct = np.asarray(cand["test_pred"], dtype=np.float32)
        best = None
        for w0 in grid:
            for w1 in grid:
                if float(w0 + w1) > 1.0001:
                    continue
                w2 = np.float32(1.0) - w0 - w1
                val_pred = w0 * a0v + w1 * a1v + w2 * cv
                score = metric_score(p9.metrics_np(val_y, val_pred, val_meta), score_mode)
                if best is None or score < best["score"]:
                    test_pred = w0 * a0t + w1 * a1t + w2 * ct
                    best = {
                        "name": f"simplex_{a0['name']}__{a1['name']}__{cand['name']}_w{float(w0):.2f}_{float(w1):.2f}_{float(w2):.2f}",
                        "val_pred": val_pred.astype(np.float32),
                        "test_pred": test_pred.astype(np.float32),
                        "kind": "simplex_blend",
                        "weights": {a0["name"]: float(w0), a1["name"]: float(w1), cand["name"]: float(w2)},
                        "score": float(score),
                    }
        if best is not None:
            out.append(best)
    return out


def write_predictions(path: Path, y: np.ndarray, pred: np.ndarray, meta: Sequence[Mapping[str, Any]], extras: Mapping[str, np.ndarray]) -> None:
    fields = ["speaker_id", "source", "gender", "height_cm", "phase10_pred_cm", "phase10_abs_error_cm", *extras.keys()]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(meta):
            item = {
                "speaker_id": row["speaker_id"],
                "source": row["source"],
                "gender": row["gender"],
                "height_cm": f"{float(y[idx]):.6f}",
                "phase10_pred_cm": f"{float(pred[idx]):.6f}",
                "phase10_abs_error_cm": f"{abs(float(pred[idx]) - float(y[idx])):.6f}",
            }
            for name, values in extras.items():
                item[name] = f"{float(values[idx]):.6f}"
            writer.writerow(item)


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Phase10. Refusing CPU run.")
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase10] device={device} gpu={torch.cuda.get_device_name(0)}", flush=True)
    p9_data = p9.load_cache(resolve(args.phase9_cache))
    bio = load_biomarker_tables(resolve(args.features_root), output_dir, bool(args.rebuild_biomarker_cache))

    train_meta = p9_data["train"]["meta"]
    val_meta = p9_data["val"]["meta"]
    test_meta = p9_data["test"]["meta"]
    train_y = p9_data["train"]["y"]
    val_y = p9_data["val"]["y"]
    test_y = p9_data["test"]["y"]

    bio_train, bio_train_ok = align_table(train_meta, bio["train"]["x"], bio["train"]["meta"])
    bio_val, bio_val_ok = align_table(val_meta, bio["val"]["x"], bio["val"]["meta"])
    bio_test, bio_test_ok = align_table(test_meta, bio["test"]["x"], bio["test"]["meta"])
    names = bio["train"]["names"]
    core_idx = p8.subset_indices(names, "core_quality")

    feature_sets = {
        "ecapa": (p9_data["train"]["x"], p9_data["val"]["x"], p9_data["test"]["x"]),
        "bio_core": (np.concatenate([bio_train[:, core_idx], bio_train_ok], axis=1), np.concatenate([bio_val[:, core_idx], bio_val_ok], axis=1), np.concatenate([bio_test[:, core_idx], bio_test_ok], axis=1)),
        "ecapa_bio_core": (
            np.concatenate([p9_data["train"]["x"], bio_train[:, core_idx], bio_train_ok], axis=1),
            np.concatenate([p9_data["val"]["x"], bio_val[:, core_idx], bio_val_ok], axis=1),
            np.concatenate([p9_data["test"]["x"], bio_test[:, core_idx], bio_test_ok], axis=1),
        ),
        "ecapa_bio_all": (
            np.concatenate([p9_data["train"]["x"], bio_train, bio_train_ok], axis=1),
            np.concatenate([p9_data["val"]["x"], bio_val, bio_val_ok], axis=1),
            np.concatenate([p9_data["test"]["x"], bio_test, bio_test_ok], axis=1),
        ),
    }

    phase3_val = align_pred(val_meta, read_pred(resolve(args.phase3_val_pred)))
    phase3_test = align_pred(test_meta, read_pred(resolve(args.phase3_test_pred)))
    if phase3_val is None or phase3_test is None:
        raise RuntimeError("Could not align Phase3 val/test predictions")

    candidates: List[Dict[str, Any]] = [
        {"name": "phase3_frontier", "val_pred": phase3_val, "test_pred": phase3_test, "kind": "anchor"},
    ]

    # Rebuild the Phase9 frontier from cache so Phase10 can use it as an anchor.
    lambdas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    target_mask = np.asarray([p9.is_target_source(row["source"]) for row in train_meta], dtype=bool)
    phase9_rebuilt: List[Dict[str, Any]] = []
    phase9_rebuilt.extend(
        p9.weighted_ridge_predict(
            p9_data["train"]["x"],
            train_y,
            p9_data["val"]["x"],
            p9_data["test"]["x"],
            sample_weight=np.ones_like(train_y, dtype=np.float32),
            lambdas=lambdas,
            device=device,
            label="phase9_all",
        )
    )
    phase9_rebuilt.extend(p9.choose_blends(phase9_rebuilt[:], phase3_val, phase3_test, val_y, val_meta))
    phase9_rows = candidate_rows(phase9_rebuilt, val_y, test_y, val_meta, test_meta, "balanced")
    phase9_selected_name = phase9_rows[0]["name"]
    phase9_selected = get_candidate(phase9_rebuilt, phase9_selected_name)
    if phase9_selected is None:
        raise RuntimeError("Failed to rebuild Phase9 selected candidate")
    phase9_selected = dict(phase9_selected)
    phase9_selected["name"] = "phase9_frontier_rebuilt"
    candidates.append(phase9_selected)
    print(f"[phase10] rebuilt Phase9 frontier: {phase9_rows[0]['test']['mae']:.3f}cm", flush=True)

    new_candidates: List[Dict[str, Any]] = []
    for label, (tr_x, va_x, te_x) in feature_sets.items():
        for celeb_weight in (0.08, 0.18, 0.32, 0.50):
            for short_boost in (1.0, 1.25):
                new_candidates.extend(
                    p9.weighted_ridge_predict(
                        tr_x,
                        train_y,
                        va_x,
                        te_x,
                        sample_weight=source_weights(train_meta, celeb_weight=celeb_weight, short_boost=short_boost),
                        lambdas=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
                        device=device,
                        label=f"{label}_cw{celeb_weight:g}_sb{short_boost:g}",
                    )
                )
        if label in {"ecapa_bio_core", "ecapa_bio_all"}:
            new_candidates.extend(
                p9.kernel_ridge_predict(
                    tr_x[target_mask],
                    train_y[target_mask],
                    va_x,
                    te_x,
                    lambdas=[0.03, 0.1, 0.3, 1.0, 3.0],
                    gammas=[0.15, 0.3, 0.6, 1.2],
                    device=device,
                    label=f"{label}_target",
                )
            )
            new_candidates.extend(
                p9.knn_predict(
                    tr_x[target_mask],
                    train_y[target_mask],
                    va_x,
                    te_x,
                    ks=[6, 12, 24, 48, 96],
                    temps=[0.03, 0.06, 0.10],
                    device=device,
                    label=f"{label}_target",
                )
            )

    if not args.skip_xgb:
        for label in ("bio_core", "ecapa_bio_core", "ecapa_bio_all"):
            tr_x, va_x, te_x = feature_sets[label]
            new_candidates.extend(xgb_candidates(tr_x, train_y, va_x, te_x, train_meta, label=label))

    candidates.extend(new_candidates)
    candidates.extend(p9.choose_blends(new_candidates, phase3_val, phase3_test, val_y, val_meta))
    candidates.extend(simplex_blends([candidates[0], candidates[1]], new_candidates, val_y, val_meta, step=0.04, score_mode="balanced"))
    candidates.extend(simplex_blends([candidates[0], candidates[1]], new_candidates, val_y, val_meta, step=0.04, score_mode="short"))

    rows = candidate_rows(candidates, val_y, test_y, val_meta, test_meta, "balanced")
    selected = rows[0]
    selected_cand = get_candidate(candidates, selected["name"])
    if selected_cand is None:
        raise RuntimeError("Selected candidate missing from candidate list")
    selected_pred = np.asarray(selected_cand["test_pred"], dtype=np.float32)

    phase9_metrics = p9.metrics_np(test_y, np.asarray(candidates[1]["test_pred"], dtype=np.float32), test_meta)
    phase3_metrics = p9.metrics_np(test_y, phase3_test, test_meta)
    extras = {
        "phase3_pred_cm": phase3_test,
        "phase9_pred_cm": np.asarray(candidates[1]["test_pred"], dtype=np.float32),
    }
    write_predictions(output_dir / "phase10_predictions_test.csv", test_y, selected_pred, test_meta, extras)

    report = {
        "selected": selected,
        "phase9_reference": phase9_metrics,
        "phase3_reference": phase3_metrics,
        "top_candidates": rows[:80],
        "candidate_count": len(candidates),
        "args": vars(args),
    }
    (output_dir / "phase10_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")

    md = [
        "# Phase 10 Super Stack Report",
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
        "",
        "## Top Validation Candidates",
    ]
    for row in rows[:20]:
        md.append(
            f"- `{row['name']}`: val `{row['val']['mae']:.3f}cm`, "
            f"test `{row['test']['mae']:.3f}cm`, short `{row['test'].get('short_mae', float('nan')):.3f}cm`, score `{row['score']:.3f}`"
        )
    md.extend(["", "## Conclusion"])
    if selected["test"]["mae"] < phase9_metrics["mae"]:
        md.append("Phase 10 improves the current sealed-test frontier.")
    else:
        md.append("Phase 10 did not beat the Phase 9 sealed-test frontier under validation selection; keep Phase 9 as the safer deployed frontier.")
    (output_dir / "PHASE10_SUPER_STACK_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[phase10] selected={selected['name']}", flush=True)
    print(
        f"[phase10] test_mae={selected['test']['mae']:.3f} short={selected['test'].get('short_mae', float('nan')):.3f} "
        f"phase9={phase9_metrics['mae']:.3f}",
        flush=True,
    )
    print(f"[phase10] wrote {output_dir / 'PHASE10_SUPER_STACK_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
