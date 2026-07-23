#!/usr/bin/env python
"""Phase 15 selective confidence layer.

The previous phases showed a hard ceiling around 5cm sealed-test MAE for
always-on height prediction. This phase adds the missing real-world layer:

- build a focused target-domain candidate pool using the Phase14 feature stack
- select the height predictor from out-of-fold development evidence only
- train error/confidence models on out-of-fold absolute errors
- calibrate accept/reject thresholds on development speakers only
- report sealed-test selective MAE at fixed coverage levels

This script does not tune thresholds on the test set. If it reaches 3cm only at
low coverage, the report says that plainly.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase9_ecapa_prior_stack as p9  # noqa: E402
import phase13_target_cv_residual_stack as p13  # noqa: E402
import phase14_full_research_gauntlet as p14  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 15 selective confidence gate.")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--phase9-cache", default="outputs/phase9_ecapa_prior_stack/ecapa_m6_s6p0_limit0_celeb.npz")
    parser.add_argument("--extra-ecapa-cache", default="outputs/phase9_ecapa_prior_stack_m12s8/ecapa_m12_s8p0_limit0_celeb.npz")
    parser.add_argument("--biomarker-cache", default="outputs/phase10_super_stack/phase10_biomarker_cache.npz")
    parser.add_argument("--phase12-test-pred", default="outputs/phase12_residual_guard/phase12_predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase15_selective_confidence")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--top-confidence-candidates", type=int, default=80)
    parser.add_argument("--top-residual-bases", type=int, default=8)
    parser.add_argument("--skip-error-mlp", action="store_true")
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def sid(row: Mapping[str, Any]) -> str:
    return str(row.get("speaker_id", "")).strip()


def prior_array(meta: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([float(row.get("prior_mean", 170.0)) for row in meta], dtype=np.float32)


def clip_height(pred: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=np.float32), 140.0, 205.0).astype(np.float32)


def subset_meta(meta: Sequence[Mapping[str, Any]], mask: np.ndarray) -> List[Mapping[str, Any]]:
    return [row for row, keep in zip(meta, mask.tolist()) if bool(keep)]


def safe_metrics(y: np.ndarray, pred: np.ndarray, meta: Sequence[Mapping[str, Any]], mask: np.ndarray) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return {"count": 0.0, "coverage": 0.0, "mae": float("nan")}
    metrics = p9.metrics_np(np.asarray(y)[mask], np.asarray(pred)[mask], subset_meta(meta, mask))
    metrics["coverage"] = float(mask.mean())
    return metrics


def canonical_feature_name(feature_sets: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> str:
    for name in ("ecapa_dual_bio_meta", "ecapa_bio_meta", "ecapa_meta"):
        if name in feature_sets:
            return name
    return next(iter(feature_sets.keys()))


def build_candidate_pool(
    data: Mapping[str, Any],
    folds: Sequence[Tuple[np.ndarray, np.ndarray, str]],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dev_y = np.asarray(data["dev_y"], dtype=np.float32)
    test_y = np.asarray(data["test_y"], dtype=np.float32)
    support_y = np.asarray(data["support_y"], dtype=np.float32)
    dev_meta = data["dev_meta"]
    support_meta = data["support_meta"]
    test_meta = data["test_meta"]
    dev_meta_metrics = data["dev_meta_metrics"]
    test_meta_metrics = data["test_meta_metrics"]

    candidates: List[Dict[str, Any]] = []
    feature_sets = data["feature_sets"]
    preferred = [canonical_feature_name(feature_sets)]
    if "ecapa_bio_meta" in feature_sets and "ecapa_bio_meta" not in preferred:
        preferred.append("ecapa_bio_meta")

    for feat_name in preferred[:2]:
        dev_x, support_x, test_x = feature_sets[feat_name]
        print(f"[phase15] feature={feat_name} dim={dev_x.shape[1]}", flush=True)
        for lam in (1000.0, 3000.0, 10000.0):
            for cw in (0.08, 0.18, 0.32):
                for sb in (1.0, 1.25, 1.55):
                    candidates.append(
                        p13.cv_ridge_candidate(
                            name=f"{feat_name}_ridge_lam{lam:g}_cw{cw:g}_sb{sb:g}",
                            dev_x=dev_x,
                            dev_y=dev_y,
                            dev_meta=dev_meta,
                            support_x=support_x,
                            support_y=support_y,
                            support_meta=support_meta,
                            test_x=test_x,
                            folds=folds,
                            lam=lam,
                            celeb_weight=cw,
                            short_boost=sb,
                            device=device,
                        )
                    )

    feat_name = preferred[0]
    dev_x, support_x, test_x = feature_sets[feat_name]
    for k in (31, 55):
        for temp in (0.06, 0.10):
            candidates.append(
                p14.cv_knn_candidate(
                    f"{feat_name}_knn_k{k}_t{temp:g}",
                    dev_x,
                    dev_y,
                    support_x,
                    support_y,
                    test_x,
                    folds,
                    k=k,
                    temp=temp,
                    use_support=False,
                    device=device,
                )
            )
            candidates.append(
                p14.cv_knn_candidate(
                    f"{feat_name}_knn_support_k{k}_t{temp:g}",
                    dev_x,
                    dev_y,
                    support_x,
                    support_y,
                    test_x,
                    folds,
                    k=k,
                    temp=temp,
                    use_support=True,
                    device=device,
                )
            )

    rows = p14.phase14_rows(candidates, dev_y, test_y, dev_meta_metrics, test_meta_metrics)
    print(
        f"[phase15] raw best={rows[0]['name']} oof={rows[0]['oof']['mae']:.3f} "
        f"test={rows[0]['test']['mae']:.3f}",
        flush=True,
    )
    candidates.extend(
        p14.residual_family(
            candidates,
            rows,
            dev_y,
            dev_meta,
            test_meta,
            top_n=int(args.top_residual_bases),
        )
    )
    rows = p14.phase14_rows(candidates, dev_y, test_y, dev_meta_metrics, test_meta_metrics)
    candidates.extend(p14.convex_stack(candidates, rows, dev_y, top_n=12, seed=int(args.seed)))
    rows = p14.phase14_rows(candidates, dev_y, test_y, dev_meta_metrics, test_meta_metrics)
    print(
        f"[phase15] selected={rows[0]['name']} oof={rows[0]['oof']['mae']:.3f} "
        f"test={rows[0]['test']['mae']:.3f}",
        flush=True,
    )
    return candidates, rows


def prediction_matrix(
    candidates: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    limit: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    by_name = {str(c["name"]): c for c in candidates}
    chosen: List[Mapping[str, Any]] = []
    names: List[str] = []
    for row in rows:
        name = str(row["name"])
        if name in by_name and name not in names:
            chosen.append(by_name[name])
            names.append(name)
        if len(chosen) >= int(limit):
            break
    dev = np.stack([clip_height(c["oof_pred"]) for c in chosen], axis=1)
    test = np.stack([clip_height(c["test_pred"]) for c in chosen], axis=1)
    return dev.astype(np.float32), test.astype(np.float32), names


def source_gender_matrix(meta: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows = []
    for row in meta:
        src = str(row.get("source", "")).upper()
        g = int(row.get("gender", 0))
        age = float(row.get("age", 0.0))
        n_clips = float(row.get("n_clips", len(row.get("audio_paths") or [])))
        rows.append(
            [
                1.0 if src == "NISP" else 0.0,
                1.0 if src == "TIMIT" else 0.0,
                1.0 if src == "CELEB" else 0.0,
                float(g),
                np.clip(age / 80.0, 0.0, 1.4) if math.isfinite(age) and age > 0 else 0.0,
                math.log1p(max(n_clips, 0.0)) / math.log(80.0),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def nearest_stats(
    train_x: np.ndarray,
    query_x: np.ndarray,
    *,
    device: torch.device,
    self_match_count: int = 0,
    batch_size: int = 512,
) -> np.ndarray:
    train, query = p13.robust_standardize(train_x, query_x)
    xt = torch.tensor(train, dtype=torch.float32, device=device)
    xq = torch.tensor(query, dtype=torch.float32, device=device)
    xt = torch.nn.functional.normalize(xt, dim=1)
    xq = torch.nn.functional.normalize(xq, dim=1)
    stats: List[np.ndarray] = []
    for start in range(0, xq.shape[0], int(batch_size)):
        end = min(start + int(batch_size), xq.shape[0])
        sim = xq[start:end] @ xt.T
        if self_match_count > 0:
            local = torch.arange(start, end, device=device)
            valid = local < int(self_match_count)
            if bool(valid.any()):
                sim[torch.arange(end - start, device=device)[valid], local[valid]] = -1.0
        top = torch.topk(sim, k=min(8, sim.shape[1]), dim=1).values
        arr = torch.stack(
            [
                1.0 - top[:, 0],
                1.0 - top[:, :3].mean(dim=1),
                top[:, 0],
                top[:, :8].std(dim=1),
            ],
            dim=1,
        )
        stats.append(arr.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(stats, axis=0)


def confidence_features(
    pred_mat: np.ndarray,
    selected_pred: np.ndarray,
    meta: Sequence[Mapping[str, Any]],
    nn_stats: np.ndarray,
) -> np.ndarray:
    pred_mat = np.asarray(pred_mat, dtype=np.float32)
    selected = np.asarray(selected_pred, dtype=np.float32).reshape(-1, 1)
    prior = prior_array(meta).reshape(-1, 1)
    q = np.quantile(pred_mat, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T.astype(np.float32)
    mean = pred_mat.mean(axis=1, keepdims=True)
    std = pred_mat.std(axis=1, keepdims=True)
    minv = pred_mat.min(axis=1, keepdims=True)
    maxv = pred_mat.max(axis=1, keepdims=True)
    top = pred_mat[:, : min(12, pred_mat.shape[1])]
    top_std = top.std(axis=1, keepdims=True)
    top_range = (top.max(axis=1) - top.min(axis=1)).reshape(-1, 1)
    engineered = np.concatenate(
        [
            selected / 200.0,
            (selected - mean) / 10.0,
            (selected - prior) / 10.0,
            (mean - prior) / 10.0,
            std / 10.0,
            (maxv - minv) / 10.0,
            top_std / 10.0,
            top_range / 10.0,
            q / 200.0,
            np.abs(selected - q[:, 2:3]) / 10.0,
            source_gender_matrix(meta),
            np.asarray(nn_stats, dtype=np.float32),
        ],
        axis=1,
    )
    bad = ~np.isfinite(engineered)
    if bool(bad.any()):
        engineered[bad] = 0.0
    return engineered.astype(np.float32)


class ErrorMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_error_mlp_once(
    train_x: np.ndarray,
    train_err: np.ndarray,
    query_x: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(train_err))
    rng.shuffle(indices)
    hold = max(32, int(0.18 * len(indices)))
    val_idx = indices[:hold]
    tr_idx = indices[hold:]
    xtr, xval = p13.robust_standardize(train_x[tr_idx], train_x[val_idx])
    _, xq = p13.robust_standardize(train_x[tr_idx], query_x)
    ytr = np.log1p(np.clip(train_err[tr_idx], 0.0, 40.0)).astype(np.float32)
    yval = np.log1p(np.clip(train_err[val_idx], 0.0, 40.0)).astype(np.float32)
    xt = torch.tensor(xtr, dtype=torch.float32, device=device)
    xv = torch.tensor(xval, dtype=torch.float32, device=device)
    xquery = torch.tensor(xq, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    yv = torch.tensor(yval, dtype=torch.float32, device=device)
    model = ErrorMLP(xt.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0.015)
    best = float("inf")
    best_state = None
    stale = 0
    for _epoch in range(360):
        model.train()
        perm = torch.randperm(xt.shape[0], device=device)
        for start in range(0, xt.shape[0], 128):
            batch = perm[start : start + 128]
            loss = torch.nn.functional.smooth_l1_loss(model(xt[batch]), yt[batch], beta=0.35)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.l1_loss(model(xv), yv).item()
        if val_loss + 1e-4 < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 45:
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        pred = torch.expm1(model(xquery)).detach().cpu().numpy().astype(np.float32)
    return np.clip(pred, 0.0, 40.0).astype(np.float32)


def cv_error_mlp(
    x_dev: np.ndarray,
    err_dev: np.ndarray,
    x_test: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray, str]],
    *,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    oof_sum = np.zeros(len(err_dev), dtype=np.float32)
    oof_count = np.zeros(len(err_dev), dtype=np.float32)
    for fold_no, (train_idx, val_idx, _) in enumerate(folds):
        pred = train_error_mlp_once(
            x_dev[train_idx],
            err_dev[train_idx],
            x_dev[val_idx],
            seed=int(seed) + 131 * fold_no,
            device=device,
        )
        oof_sum[val_idx] += pred
        oof_count[val_idx] += 1.0
    test_pred = train_error_mlp_once(x_dev, err_dev, x_test, seed=int(seed) + 999, device=device)
    return oof_sum / np.maximum(oof_count, 1.0), test_pred


def error_model_score(pred_err: np.ndarray, true_err: np.ndarray) -> float:
    pred_err = np.asarray(pred_err, dtype=np.float32)
    true_err = np.asarray(true_err, dtype=np.float32)
    coverages = (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)
    vals = []
    for coverage in coverages:
        threshold = float(np.quantile(pred_err, coverage))
        mask = pred_err <= threshold
        if int(mask.sum()) >= 12:
            vals.append(float(true_err[mask].mean()))
    rank_penalty = float(np.mean(np.abs(pred_err - true_err))) * 0.12
    return float(np.mean(vals)) + rank_penalty


def fit_error_models(
    x_dev: np.ndarray,
    true_err: np.ndarray,
    x_test: np.ndarray,
    error_folds: Sequence[Tuple[np.ndarray, np.ndarray, str]],
    *,
    seed: int,
    device: torch.device,
    skip_error_mlp: bool,
) -> Tuple[str, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    candidates: List[Tuple[str, np.ndarray, np.ndarray]] = []
    try:
        from sklearn.base import clone
        from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import HuberRegressor, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        estimators = [
            ("ridge", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=4.0))])),
            ("huber", Pipeline([("s", StandardScaler()), ("m", HuberRegressor(alpha=0.04, epsilon=1.35, max_iter=600))])),
            ("extra_trees", ExtraTreesRegressor(n_estimators=700, max_depth=5, min_samples_leaf=7, random_state=int(seed), n_jobs=-1)),
            ("random_forest", RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=8, random_state=int(seed), n_jobs=-1)),
            ("hist_gbdt", HistGradientBoostingRegressor(loss="absolute_error", max_iter=360, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=0.25, random_state=int(seed))),
            ("gbr_abs", GradientBoostingRegressor(loss="absolute_error", n_estimators=360, max_depth=2, learning_rate=0.025, random_state=int(seed))),
        ]
        for name, estimator in estimators:
            oof_sum = np.zeros(len(true_err), dtype=np.float32)
            oof_count = np.zeros(len(true_err), dtype=np.float32)
            for train_idx, val_idx, _ in error_folds:
                model = clone(estimator)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x_dev[train_idx], true_err[train_idx])
                oof_sum[val_idx] += np.asarray(model.predict(x_dev[val_idx]), dtype=np.float32)
                oof_count[val_idx] += 1.0
            model = clone(estimator)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_dev, true_err)
            test_pred = np.asarray(model.predict(x_test), dtype=np.float32)
            candidates.append((name, np.clip(oof_sum / np.maximum(oof_count, 1.0), 0.0, 40.0), np.clip(test_pred, 0.0, 40.0)))
    except Exception as exc:
        print(f"[phase15] sklearn error models skipped: {exc}", flush=True)

    if not bool(skip_error_mlp):
        oof, test = cv_error_mlp(x_dev, true_err, x_test, error_folds, seed=int(seed) + 77, device=device)
        candidates.append(("cuda_error_mlp", np.clip(oof, 0.0, 40.0), np.clip(test, 0.0, 40.0)))

    if len(candidates) >= 2:
        oof_stack = np.stack([c[1] for c in candidates], axis=1)
        test_stack = np.stack([c[2] for c in candidates], axis=1)
        candidates.append(("mean_error_ensemble", oof_stack.mean(axis=1), test_stack.mean(axis=1)))
        candidates.append(("median_error_ensemble", np.median(oof_stack, axis=1), np.median(test_stack, axis=1)))

    if not candidates:
        fallback = np.full(len(true_err), float(np.mean(true_err)), dtype=np.float32)
        return "constant", fallback, np.full(len(x_test), float(np.mean(true_err)), dtype=np.float32), []

    rows = []
    for name, oof, _test in candidates:
        rows.append(
            {
                "name": name,
                "score": error_model_score(oof, true_err),
                "error_mae": float(np.mean(np.abs(oof - true_err))),
                "mean_predicted_error": float(np.mean(oof)),
            }
        )
    rows.sort(key=lambda row: row["score"])
    best_name = str(rows[0]["name"])
    best = next(c for c in candidates if c[0] == best_name)
    return best_name, best[1], best[2], rows


def coverage_table(
    y_dev: np.ndarray,
    pred_dev: np.ndarray,
    err_dev_pred: np.ndarray,
    dev_meta: Sequence[Mapping[str, Any]],
    y_test: np.ndarray,
    pred_test: np.ndarray,
    err_test_pred: np.ndarray,
    test_meta: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for target_cov in (1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20):
        threshold = float("inf") if target_cov >= 0.999 else float(np.quantile(err_dev_pred, target_cov))
        dev_mask = np.asarray(err_dev_pred <= threshold, dtype=bool)
        test_mask = np.asarray(err_test_pred <= threshold, dtype=bool)
        dev_m = safe_metrics(y_dev, pred_dev, dev_meta, dev_mask)
        test_m = safe_metrics(y_test, pred_test, test_meta, test_mask)
        rows.append(
            {
                "target_coverage": float(target_cov),
                "threshold": threshold,
                "dev": dev_m,
                "test": test_m,
            }
        )
    return rows


def threshold_for_target(
    pred_err: np.ndarray,
    actual_err: np.ndarray,
    target_mae: float,
    *,
    min_count: int,
) -> Optional[Dict[str, float]]:
    order = np.argsort(np.asarray(pred_err, dtype=np.float32))
    sorted_pred_err = np.asarray(pred_err, dtype=np.float32)[order]
    sorted_actual = np.asarray(actual_err, dtype=np.float32)[order]
    cumsum = np.cumsum(sorted_actual)
    best: Optional[Dict[str, float]] = None
    for idx in range(int(min_count) - 1, len(order)):
        count = idx + 1
        mae = float(cumsum[idx] / float(count))
        if mae <= float(target_mae):
            best = {
                "threshold": float(sorted_pred_err[idx]),
                "dev_mae": mae,
                "dev_count": float(count),
                "dev_coverage": float(count / len(order)),
            }
    return best


def write_predictions(
    path: Path,
    y: np.ndarray,
    pred: np.ndarray,
    pred_err: np.ndarray,
    meta: Sequence[Mapping[str, Any]],
    phase12: Optional[np.ndarray],
    thresholds: Mapping[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "speaker_id",
        "source",
        "gender",
        "height_cm",
        "phase15_pred_cm",
        "phase15_abs_error_cm",
        "predicted_abs_error_cm",
        "confidence_score",
    ]
    if phase12 is not None:
        fields.extend(["phase12_pred_cm", "phase12_abs_error_cm"])
    for name in thresholds:
        fields.append(f"accept_{name}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(meta):
            err_hat = float(pred_err[idx])
            item = {
                "speaker_id": sid(row),
                "source": str(row.get("source", "")),
                "gender": int(row.get("gender", 0)),
                "height_cm": f"{float(y[idx]):.6f}",
                "phase15_pred_cm": f"{float(pred[idx]):.6f}",
                "phase15_abs_error_cm": f"{abs(float(pred[idx]) - float(y[idx])):.6f}",
                "predicted_abs_error_cm": f"{err_hat:.6f}",
                "confidence_score": f"{1.0 / (1.0 + max(err_hat, 0.0) / 5.0):.6f}",
            }
            if phase12 is not None:
                item["phase12_pred_cm"] = f"{float(phase12[idx]):.6f}"
                item["phase12_abs_error_cm"] = f"{abs(float(phase12[idx]) - float(y[idx])):.6f}"
            for name, threshold in thresholds.items():
                item[f"accept_{name}"] = int(err_hat <= float(threshold))
            writer.writerow(item)


def write_report(
    output_dir: Path,
    selected: Mapping[str, Any],
    phase12_metrics: Optional[Mapping[str, float]],
    error_model_name: str,
    error_rows: Sequence[Mapping[str, float]],
    table: Sequence[Mapping[str, Any]],
    target_rows: Mapping[str, Mapping[str, Any]],
    candidate_count: int,
    pred_names: Sequence[str],
) -> None:
    lines = [
        "# Phase 15 Selective Confidence Report",
        "",
        "## Result",
        f"- Height predictor: `{selected['name']}`",
        f"- Always-on OOF MAE: `{selected['oof']['mae']:.3f}cm`",
        f"- Always-on sealed test MAE: `{selected['test']['mae']:.3f}cm`",
        f"- Always-on sealed test short MAE: `{selected['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Error model: `{error_model_name}`",
        f"- Candidates rebuilt: `{candidate_count}`",
        f"- Confidence candidates used: `{len(pred_names)}`",
        "",
        "## Baseline",
    ]
    if phase12_metrics is not None:
        lines.append(
            f"- Phase12 always-on frontier: `{phase12_metrics['mae']:.3f}cm`, "
            f"short `{phase12_metrics.get('short_mae', float('nan')):.3f}cm`"
        )
    lines.extend(["", "## Selective Gates"])
    for row in table:
        dev = row["dev"]
        test = row["test"]
        lines.append(
            f"- dev target coverage `{row['target_coverage']:.0%}` | threshold `{row['threshold']:.3f}` | "
            f"dev `{dev.get('mae', float('nan')):.3f}cm` at `{dev.get('coverage', 0.0):.0%}` | "
            f"test `{test.get('mae', float('nan')):.3f}cm` at `{test.get('coverage', 0.0):.0%}` "
            f"(n={int(test.get('count', 0.0))})"
        )
    lines.extend(["", "## Target-Calibrated Gates"])
    if target_rows:
        for name, row in target_rows.items():
            test = row["test"]
            lines.append(
                f"- `{name}` dev-calibrated threshold `{row['threshold']:.3f}`: "
                f"dev `{row['dev_mae']:.3f}cm` at `{row['dev_coverage']:.0%}`, "
                f"test `{test.get('mae', float('nan')):.3f}cm` at `{test.get('coverage', 0.0):.0%}` "
                f"(n={int(test.get('count', 0.0))})"
            )
    else:
        lines.append("- No development-calibrated threshold reached the requested targets with enough speakers.")
    lines.extend(["", "## Error Models"])
    for row in error_rows[:10]:
        lines.append(
            f"- `{row['name']}`: score `{row['score']:.3f}`, "
            f"error-MAE `{row['error_mae']:.3f}`, mean predicted error `{row['mean_predicted_error']:.3f}`"
        )
    lines.extend(["", "## Research Verdict"])
    full_test = float(selected["test"]["mae"])
    best_selective = min(float(row["test"].get("mae", 999.0)) for row in table if float(row["test"].get("count", 0.0)) > 0)
    if full_test <= 3.0:
        lines.append("The always-on sealed-test 3cm gate is met.")
    elif best_selective <= 3.0:
        lines.append("A selective 3cm operating point exists, but it is not the same as always-on 3cm MAE.")
    elif full_test <= 4.5:
        lines.append("The always-on 4.5cm gate is met, but 3cm remains blocked.")
    elif best_selective <= 4.5:
        lines.append("The model can operate below 4.5cm only with confidence gating. Always-on 3cm is still not supported by current signal.")
    else:
        lines.append("Even confidence gating does not create a defensible 4.5cm/3cm operating point. The blocker is signal/label quality, not only training compute.")
    (output_dir / "PHASE15_SELECTIVE_CONFIDENCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = torch.device(str(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Phase15 is CUDA-only. Fix CUDA/PyTorch before running this.")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase15] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    data = p14.load_data(args)
    dev_y = np.asarray(data["dev_y"], dtype=np.float32)
    test_y = np.asarray(data["test_y"], dtype=np.float32)
    dev_meta = data["dev_meta"]
    test_meta = data["test_meta"]
    folds = p13.make_folds(dev_meta, dev_y, int(args.folds), int(args.repeats), int(args.seed))
    print(f"[phase15] dev={len(dev_y)} test={len(test_y)} folds={len(folds)}", flush=True)

    candidates, rows = build_candidate_pool(data, folds, device, args)
    selected = rows[0]
    selected_cand = next(c for c in candidates if str(c["name"]) == str(selected["name"]))
    selected_dev = clip_height(np.asarray(selected_cand["oof_pred"], dtype=np.float32))
    selected_test = clip_height(np.asarray(selected_cand["test_pred"], dtype=np.float32))

    dev_mat, test_mat, pred_names = prediction_matrix(candidates, rows, int(args.top_confidence_candidates))
    feat_name = canonical_feature_name(data["feature_sets"])
    dev_x, support_x, test_x = data["feature_sets"][feat_name]
    train_pool = np.concatenate([dev_x, support_x], axis=0) if len(support_x) else dev_x
    dev_nn = nearest_stats(train_pool, dev_x, device=device, self_match_count=len(dev_x))
    test_nn = nearest_stats(train_pool, test_x, device=device, self_match_count=0)
    x_conf_dev = confidence_features(dev_mat, selected_dev, dev_meta, dev_nn)
    x_conf_test = confidence_features(test_mat, selected_test, test_meta, test_nn)
    true_err_dev = np.abs(selected_dev - dev_y).astype(np.float32)
    error_folds = p13.make_folds(dev_meta, dev_y, int(args.folds), int(args.repeats), int(args.seed) + 503)
    print(f"[phase15] training confidence/error models features={x_conf_dev.shape[1]}", flush=True)
    error_name, err_dev_pred, err_test_pred, error_rows = fit_error_models(
        x_conf_dev,
        true_err_dev,
        x_conf_test,
        error_folds,
        seed=int(args.seed),
        device=device,
        skip_error_mlp=bool(args.skip_error_mlp),
    )
    print(f"[phase15] best error model={error_name}", flush=True)

    table = coverage_table(
        dev_y,
        selected_dev,
        err_dev_pred,
        data["dev_meta_metrics"],
        test_y,
        selected_test,
        err_test_pred,
        data["test_meta_metrics"],
    )
    target_rows: Dict[str, Mapping[str, Any]] = {}
    for target in (4.5, 3.0):
        gate = threshold_for_target(err_dev_pred, true_err_dev, target, min_count=max(20, int(0.12 * len(dev_y))))
        if gate is not None:
            mask_test = np.asarray(err_test_pred <= gate["threshold"], dtype=bool)
            target_rows[f"{target:g}cm"] = {
                **gate,
                "test": safe_metrics(test_y, selected_test, data["test_meta_metrics"], mask_test),
            }

    thresholds = {f"cov{int(row['target_coverage'] * 100)}": float(row["threshold"]) for row in table if math.isfinite(float(row["threshold"]))}
    for name, row in target_rows.items():
        thresholds[f"dev{name}"] = float(row["threshold"])
    write_predictions(
        output_dir / "phase15_predictions_test.csv",
        test_y,
        selected_test,
        err_test_pred,
        test_meta,
        data["phase12_pred"],
        thresholds,
    )

    payload = {
        "selected": selected,
        "phase12_reference": data["phase12_metrics"],
        "error_model": error_name,
        "error_models": error_rows,
        "coverage_table": table,
        "target_gates": target_rows,
        "candidate_count": len(candidates),
        "confidence_candidate_names": pred_names,
        "args": vars(args),
    }
    (output_dir / "phase15_report.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    write_report(output_dir, selected, data["phase12_metrics"], error_name, error_rows, table, target_rows, len(candidates), pred_names)
    print(f"[phase15] wrote {output_dir / 'PHASE15_SELECTIVE_CONFIDENCE_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
