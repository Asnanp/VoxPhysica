#!/usr/bin/env python
"""Phase 14 full research gauntlet.

This is the "do the science" phase:

- repeated target-domain CV on train+validation speakers
- sealed test used only for final reporting
- ECAPA, biomarker, metadata, external-prior and optional second ECAPA cache
- model zoo: ridge, random-projection ridge, GPU KNN, sklearn trees,
  optional XGBoost, and CUDA MLP residual learners
- residual calibrators and convex stacked ensembles selected from OOF evidence

The script is deliberately honest. If it does not hit 3cm, the report says so.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import warnings
from collections import defaultdict
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
import phase11_metadata_tail_calibrator as p11  # noqa: E402
import phase13_target_cv_residual_stack as p13  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 14 full ML/DL research gauntlet.")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--phase9-cache", default="outputs/phase9_ecapa_prior_stack/ecapa_m6_s6p0_limit0_celeb.npz")
    parser.add_argument("--extra-ecapa-cache", default="outputs/phase9_ecapa_prior_stack_m12s8/ecapa_m12_s8p0_limit0_celeb.npz")
    parser.add_argument("--biomarker-cache", default="outputs/phase10_super_stack/phase10_biomarker_cache.npz")
    parser.add_argument("--phase12-test-pred", default="outputs/phase12_residual_guard/phase12_predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase14_full_research_gauntlet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mlp-epochs", type=int, default=420)
    parser.add_argument("--mlp-patience", type=int, default=55)
    parser.add_argument("--skip-sklearn", action="store_true", help="Skip CPU sklearn tree/linear zoo; keeps CUDA ridge/KNN, XGBoost, and PyTorch MLP.")
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
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


def metric_score(metrics: Mapping[str, float], groups: int = 1, penalty: float = 0.002) -> float:
    mae = float(metrics.get("mae", 999.0))
    short = float(metrics.get("short_mae", mae))
    medium = float(metrics.get("medium_mae", mae))
    tall = float(metrics.get("tall_mae", mae))
    p90 = float(metrics.get("p90_ae", mae))
    bias = abs(float(metrics.get("bias", 0.0)))
    return (
        0.56 * mae
        + 0.23 * float(np.mean([short, medium, tall]))
        + 0.11 * p90
        + 0.08 * max(0.0, short - mae)
        + 0.02 * bias
        + float(groups) * float(penalty)
    )


def phase14_rows(
    candidates: Sequence[Mapping[str, Any]],
    dev_y: np.ndarray,
    test_y: np.ndarray,
    dev_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        oof_m = p9.metrics_np(dev_y, np.asarray(cand["oof_pred"], dtype=np.float32), dev_meta)
        test_m = p9.metrics_np(test_y, np.asarray(cand["test_pred"], dtype=np.float32), test_meta)
        row = {k: v for k, v in cand.items() if k not in {"oof_pred", "test_pred"}}
        row["oof"] = oof_m
        row["test"] = test_m
        row["selection_score"] = metric_score(oof_m, int(row.get("groups", 1)))
        rows.append(row)
    rows.sort(key=lambda x: float(x["selection_score"]))
    return rows


def combined_cache_features(cache: Mapping[str, Mapping[str, Any]]) -> Tuple[Dict[str, np.ndarray], int]:
    by_sid: Dict[str, np.ndarray] = {}
    dim = 0
    for split in ("train", "val", "test"):
        x = np.asarray(cache[split]["x"], dtype=np.float32)
        dim = int(x.shape[1])
        for idx, row in enumerate(cache[split]["meta"]):
            by_sid[sid(row)] = x[idx]
    return by_sid, dim


def align_from_cache(
    ref_meta: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    by_sid, dim = combined_cache_features(cache)
    x = np.zeros((len(ref_meta), dim), dtype=np.float32)
    present = np.zeros((len(ref_meta), 1), dtype=np.float32)
    for idx, row in enumerate(ref_meta):
        key = sid(row)
        if key in by_sid:
            x[idx] = by_sid[key]
            present[idx, 0] = 1.0
    return x, present


def load_data(args: argparse.Namespace) -> Dict[str, Any]:
    cache = p9.load_cache(resolve(args.phase9_cache))
    row_by_sid = p13.split_lookup(resolve(args.splits_dir))
    train_meta_all = p13.enrich_meta(cache["train"]["meta"], row_by_sid)
    val_meta = p13.enrich_meta(cache["val"]["meta"], row_by_sid)
    test_meta = p13.enrich_meta(cache["test"]["meta"], row_by_sid)
    target_idx = np.asarray([i for i, row in enumerate(train_meta_all) if p13.target_source(row)], dtype=np.int64)
    support_idx = np.asarray([i for i, row in enumerate(train_meta_all) if not p13.target_source(row)], dtype=np.int64)
    dev_y = np.concatenate([cache["train"]["y"][target_idx], cache["val"]["y"]], axis=0).astype(np.float32)
    test_y = np.asarray(cache["test"]["y"], dtype=np.float32)
    dev_meta = [train_meta_all[int(i)] for i in target_idx.tolist()] + list(val_meta)
    support_meta = [train_meta_all[int(i)] for i in support_idx.tolist()]
    test_meta_metrics = p13.meta_for_metrics(test_meta)
    dev_meta_metrics = p13.meta_for_metrics(dev_meta)

    dev_ecapa = np.concatenate([cache["train"]["x"][target_idx], cache["val"]["x"]], axis=0).astype(np.float32)
    support_ecapa = cache["train"]["x"][support_idx].astype(np.float32)
    test_ecapa = cache["test"]["x"].astype(np.float32)
    support_y = cache["train"]["y"][support_idx].astype(np.float32)

    feature_sets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "ecapa_meta": (
            p13.append_features(dev_ecapa, p13.metadata_matrix(dev_meta)),
            p13.append_features(support_ecapa, p13.metadata_matrix(support_meta)),
            p13.append_features(test_ecapa, p13.metadata_matrix(test_meta)),
        )
    }

    biomarker = p13.load_biomarker_cache(resolve(args.biomarker_cache))
    if biomarker is not None:
        bio_train_all, bio_train_present = p13.align_features(train_meta_all, biomarker["train"]["x"], biomarker["train"]["meta"])
        bio_val, bio_val_present = p13.align_features(val_meta, biomarker["val"]["x"], biomarker["val"]["meta"])
        bio_test, bio_test_present = p13.align_features(test_meta, biomarker["test"]["x"], biomarker["test"]["meta"])
        dev_bio = np.concatenate([bio_train_all[target_idx], bio_val], axis=0)
        dev_bio_present = np.concatenate([bio_train_present[target_idx], bio_val_present], axis=0)
        support_bio = bio_train_all[support_idx]
        support_bio_present = bio_train_present[support_idx]
        feature_sets["ecapa_bio_meta"] = (
            p13.append_features(dev_ecapa, dev_bio, dev_bio_present, p13.metadata_matrix(dev_meta)),
            p13.append_features(support_ecapa, support_bio, support_bio_present, p13.metadata_matrix(support_meta)),
            p13.append_features(test_ecapa, bio_test, bio_test_present, p13.metadata_matrix(test_meta)),
        )

    extra_path = resolve(args.extra_ecapa_cache)
    if extra_path.exists():
        extra_cache = p9.load_cache(extra_path)
        dev_extra, dev_extra_present = align_from_cache(dev_meta, extra_cache)
        support_extra, support_extra_present = align_from_cache(support_meta, extra_cache)
        test_extra, test_extra_present = align_from_cache(test_meta, extra_cache)
        base_dev, base_support, base_test = feature_sets.get("ecapa_bio_meta", feature_sets["ecapa_meta"])
        feature_sets["ecapa_dual_bio_meta"] = (
            p13.append_features(base_dev, dev_extra, dev_extra_present),
            p13.append_features(base_support, support_extra, support_extra_present),
            p13.append_features(base_test, test_extra, test_extra_present),
        )

    phase12_pred = p13.read_phase12_predictions(resolve(args.phase12_test_pred), test_meta)
    phase12_metrics = p9.metrics_np(test_y, phase12_pred, test_meta_metrics) if phase12_pred is not None else None
    return {
        "cache": cache,
        "feature_sets": feature_sets,
        "dev_y": dev_y,
        "test_y": test_y,
        "support_y": support_y,
        "dev_meta": dev_meta,
        "support_meta": support_meta,
        "test_meta": test_meta,
        "dev_meta_metrics": dev_meta_metrics,
        "test_meta_metrics": test_meta_metrics,
        "phase12_pred": phase12_pred,
        "phase12_metrics": phase12_metrics,
    }


def gpu_knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    k: int,
    temp: float,
    device: torch.device,
) -> np.ndarray:
    tr, qu = p13.robust_standardize(train_x, query_x)
    xt = torch.tensor(tr, dtype=torch.float32, device=device)
    xq = torch.tensor(qu, dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.float32, device=device)
    xt = torch.nn.functional.normalize(xt, dim=1)
    xq = torch.nn.functional.normalize(xq, dim=1)
    sim = xq @ xt.T
    top_sim, top_idx = torch.topk(sim, k=min(int(k), xt.shape[0]), dim=1)
    w = torch.softmax(top_sim / float(temp), dim=1)
    return (y[top_idx] * w).sum(dim=1).detach().cpu().numpy().astype(np.float32)


def cv_knn_candidate(
    name: str,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    support_x: np.ndarray,
    support_y: np.ndarray,
    test_x: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray, str]],
    *,
    k: int,
    temp: float,
    use_support: bool,
    device: torch.device,
) -> Dict[str, Any]:
    oof_sum = np.zeros(len(dev_y), dtype=np.float32)
    oof_count = np.zeros(len(dev_y), dtype=np.float32)
    for train_idx, val_idx, _ in folds:
        xtr = dev_x[train_idx]
        ytr = dev_y[train_idx]
        if use_support and len(support_y):
            xtr = np.concatenate([xtr, support_x], axis=0)
            ytr = np.concatenate([ytr, support_y], axis=0)
        oof_sum[val_idx] += gpu_knn_predict(xtr, ytr, dev_x[val_idx], k=k, temp=temp, device=device)
        oof_count[val_idx] += 1.0
    full_x = dev_x
    full_y = dev_y
    if use_support and len(support_y):
        full_x = np.concatenate([full_x, support_x], axis=0)
        full_y = np.concatenate([full_y, support_y], axis=0)
    test_pred = gpu_knn_predict(full_x, full_y, test_x, k=k, temp=temp, device=device)
    return {
        "name": name,
        "oof_pred": (oof_sum / np.maximum(oof_count, 1.0)).astype(np.float32),
        "test_pred": test_pred,
        "kind": "gpu_knn",
        "k": int(k),
        "temp": float(temp),
        "use_support": bool(use_support),
        "groups": 1,
    }


def random_projection(x: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    proj = rng.normal(0.0, 1.0 / math.sqrt(float(out_dim)), size=(x.shape[1], int(out_dim))).astype(np.float32)
    return (np.asarray(x, dtype=np.float32) @ proj).astype(np.float32)


def cv_rp_ridge_candidate(
    name: str,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    dev_meta: Sequence[Mapping[str, Any]],
    support_x: np.ndarray,
    support_y: np.ndarray,
    support_meta: Sequence[Mapping[str, Any]],
    test_x: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray, str]],
    *,
    dim: int,
    lam: float,
    seed: int,
    celeb_weight: float,
    short_boost: float,
    device: torch.device,
) -> Dict[str, Any]:
    rp_dev = random_projection(dev_x, int(dim), int(seed))
    rp_support = random_projection(support_x, int(dim), int(seed))
    rp_test = random_projection(test_x, int(dim), int(seed))
    return p13.cv_ridge_candidate(
        name=name,
        dev_x=rp_dev,
        dev_y=dev_y,
        dev_meta=dev_meta,
        support_x=rp_support,
        support_y=support_y,
        support_meta=support_meta,
        test_x=rp_test,
        folds=folds,
        lam=float(lam),
        celeb_weight=float(celeb_weight),
        short_boost=float(short_boost),
        device=device,
    ) | {"kind": "random_projection_ridge", "rp_dim": int(dim), "rp_seed": int(seed)}


def sklearn_candidate(
    name: str,
    estimator: Any,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    test_x: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray, str]],
    *,
    sample_weight_mode: str = "none",
) -> Dict[str, Any]:
    from sklearn.base import clone

    oof_sum = np.zeros(len(dev_y), dtype=np.float32)
    oof_count = np.zeros(len(dev_y), dtype=np.float32)
    def fit_model(model_obj: Any, x: np.ndarray, y: np.ndarray, kwargs: Mapping[str, Any]) -> None:
        if not kwargs:
            model_obj.fit(x, y)
            return
        try:
            model_obj.fit(x, y, **kwargs)
            return
        except Exception:
            if "sample_weight" in kwargs and hasattr(model_obj, "named_steps") and "m" in model_obj.named_steps:
                model_obj.fit(x, y, m__sample_weight=kwargs["sample_weight"])
                return
            raise

    for train_idx, val_idx, _ in folds:
        model = clone(estimator)
        xtr, xva = p13.robust_standardize(dev_x[train_idx], dev_x[val_idx])
        fit_kwargs: Dict[str, Any] = {}
        if sample_weight_mode == "short":
            w = np.where(dev_y[train_idx] < 160.0, 1.4, 1.0).astype(np.float32)
            fit_kwargs["sample_weight"] = w / max(float(w.mean()), 1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_model(model, xtr, dev_y[train_idx], fit_kwargs)
        oof_sum[val_idx] += model.predict(xva).astype(np.float32)
        oof_count[val_idx] += 1.0
    model = copy.deepcopy(estimator)
    xfull, xtest = p13.robust_standardize(dev_x, test_x)
    fit_kwargs = {}
    if sample_weight_mode == "short":
        w = np.where(dev_y < 160.0, 1.4, 1.0).astype(np.float32)
        fit_kwargs["sample_weight"] = w / max(float(w.mean()), 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_model(model, xfull, dev_y, fit_kwargs)
    return {
        "name": name,
        "oof_pred": (oof_sum / np.maximum(oof_count, 1.0)).astype(np.float32),
        "test_pred": model.predict(xtest).astype(np.float32),
        "kind": "sklearn",
        "estimator": estimator.__class__.__name__,
        "sample_weight_mode": sample_weight_mode,
        "groups": 1,
    }


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp_once(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    seed: int,
    epochs: int,
    patience: int,
    width: int,
    dropout: float,
    lr: float,
    device: torch.device,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(train_y))
    rng.shuffle(indices)
    hold = max(32, int(0.14 * len(indices)))
    val_idx = indices[:hold]
    tr_idx = indices[hold:]
    xtr, xval = p13.robust_standardize(train_x[tr_idx], train_x[val_idx])
    _, xq = p13.robust_standardize(train_x[tr_idx], query_x)
    y_mean = float(np.mean(train_y[tr_idx]))
    y_std = float(np.std(train_y[tr_idx]) + 1e-6)
    xt = torch.tensor(xtr, dtype=torch.float32, device=device)
    xv = torch.tensor(xval, dtype=torch.float32, device=device)
    xquery = torch.tensor(xq, dtype=torch.float32, device=device)
    yt = torch.tensor((train_y[tr_idx] - y_mean) / y_std, dtype=torch.float32, device=device)
    yv = torch.tensor((train_y[val_idx] - y_mean) / y_std, dtype=torch.float32, device=device)
    model = ResidualMLP(xt.shape[1], int(width), float(dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.025)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.45, patience=max(6, int(patience) // 4))
    best_state = None
    best = float("inf")
    stale = 0
    for epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(xt.shape[0], device=device)
        for start in range(0, xt.shape[0], 128):
            batch = perm[start : start + 128]
            pred = model(xt[batch])
            loss = torch.nn.functional.smooth_l1_loss(pred, yt[batch], beta=0.45)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.l1_loss(model(xv), yv).item()
        scheduler.step(val_loss)
        if val_loss + 1e-4 < best:
            best = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        pred = model(xquery).detach().cpu().numpy().astype(np.float32) * y_std + y_mean
    return pred.astype(np.float32)


def mlp_candidate(
    name: str,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    test_x: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray, str]],
    *,
    seed: int,
    epochs: int,
    patience: int,
    width: int,
    dropout: float,
    lr: float,
    device: torch.device,
) -> Dict[str, Any]:
    oof_sum = np.zeros(len(dev_y), dtype=np.float32)
    oof_count = np.zeros(len(dev_y), dtype=np.float32)
    for fold_no, (train_idx, val_idx, _) in enumerate(folds):
        pred = train_mlp_once(
            dev_x[train_idx],
            dev_y[train_idx],
            dev_x[val_idx],
            seed=int(seed) + fold_no * 17,
            epochs=int(epochs),
            patience=int(patience),
            width=int(width),
            dropout=float(dropout),
            lr=float(lr),
            device=device,
        )
        oof_sum[val_idx] += pred
        oof_count[val_idx] += 1.0
    test_pred = train_mlp_once(
        dev_x,
        dev_y,
        test_x,
        seed=int(seed) + 999,
        epochs=int(epochs),
        patience=int(patience),
        width=int(width),
        dropout=float(dropout),
        lr=float(lr),
        device=device,
    )
    return {
        "name": name,
        "oof_pred": (oof_sum / np.maximum(oof_count, 1.0)).astype(np.float32),
        "test_pred": test_pred.astype(np.float32),
        "kind": "cuda_mlp",
        "width": int(width),
        "dropout": float(dropout),
        "lr": float(lr),
        "groups": 1,
    }


def residual_family(
    candidates: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    dev_y: np.ndarray,
    dev_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    by_name = {str(c["name"]): c for c in candidates}
    out: List[Dict[str, Any]] = []
    fields_list = [("age",), ("source",), ("gender",), ("src_gender",), ("pred_bin",), ("prior_diff",), ("src_pred",), ("src_gender_prior",)]
    for row in rows[:top_n]:
        base = by_name[str(row["name"])]
        out.append(p13.affine_candidate(base, dev_y))
        for fields in fields_list:
            for shrinkage in (8.0, 20.0, 60.0):
                for scale in (0.25, 0.50, 0.75):
                    out.append(
                        p13.residual_offset_candidate(
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


def convex_stack(
    candidates: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    dev_y: np.ndarray,
    *,
    top_n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    by_name = {str(c["name"]): c for c in candidates}
    chosen = [by_name[str(row["name"])] for row in rows[: int(top_n)] if str(row["name"]) in by_name]
    if len(chosen) < 2:
        return []
    v = np.stack([np.asarray(c["oof_pred"], dtype=np.float32) for c in chosen], axis=1)
    t = np.stack([np.asarray(c["test_pred"], dtype=np.float32) for c in chosen], axis=1)
    y = np.asarray(dev_y, dtype=np.float32)
    out: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    for temp in (0.6, 1.0, 1.7):
        w = np.ones(v.shape[1], dtype=np.float32) / float(v.shape[1])
        logits = torch.tensor(np.log(w + 1e-8), dtype=torch.float32, requires_grad=True)
        vt = torch.tensor(v, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.Adam([logits], lr=0.04)
        for _ in range(1600):
            ww = torch.softmax(logits / float(temp), dim=0)
            pred = vt @ ww
            loss = torch.nn.functional.smooth_l1_loss(pred, yt, beta=1.5) + 0.010 * torch.sum(ww * torch.log(ww + 1e-8))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            ww = torch.softmax(logits / float(temp), dim=0).detach().cpu().numpy().astype(np.float32)
        out.append(
            {
                "name": f"convex_stack_top{top_n}_temp{temp:g}",
                "oof_pred": (v @ ww).astype(np.float32),
                "test_pred": (t @ ww).astype(np.float32),
                "kind": "convex_stack",
                "members": [str(c["name"]) for c in chosen],
                "weights": {str(c["name"]): float(ww[i]) for i, c in enumerate(chosen) if float(ww[i]) > 0.01},
                "groups": int(len(chosen)),
            }
        )
    for draws in (2000, 8000):
        best: Optional[Tuple[float, np.ndarray]] = None
        for _ in range(int(draws)):
            alpha = np.full(v.shape[1], 0.65, dtype=np.float32)
            w = rng.dirichlet(alpha).astype(np.float32)
            pred = v @ w
            mae = float(np.mean(np.abs(pred - y)))
            p90 = float(np.quantile(np.abs(pred - y), 0.90))
            score = mae + 0.03 * p90
            if best is None or score < best[0]:
                best = (score, w)
        if best is not None:
            ww = best[1]
            out.append(
                {
                    "name": f"random_convex_top{top_n}_draws{draws}",
                    "oof_pred": (v @ ww).astype(np.float32),
                    "test_pred": (t @ ww).astype(np.float32),
                    "kind": "random_convex_stack",
                    "members": [str(c["name"]) for c in chosen],
                    "weights": {str(c["name"]): float(ww[i]) for i, c in enumerate(chosen) if float(ww[i]) > 0.01},
                    "groups": int(len(chosen)),
                }
            )
    return out


def write_predictions(path: Path, y: np.ndarray, pred: np.ndarray, meta: Sequence[Mapping[str, Any]], phase12: Optional[np.ndarray]) -> None:
    fields = ["speaker_id", "source", "gender", "height_cm", "phase14_pred_cm", "phase14_abs_error_cm"]
    if phase12 is not None:
        fields.append("phase12_pred_cm")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(meta):
            item = {
                "speaker_id": sid(row),
                "source": str(row.get("source", "")),
                "gender": int(row.get("gender", 0)),
                "height_cm": f"{float(y[i]):.6f}",
                "phase14_pred_cm": f"{float(pred[i]):.6f}",
                "phase14_abs_error_cm": f"{abs(float(pred[i]) - float(y[i])):.6f}",
            }
            if phase12 is not None:
                item["phase12_pred_cm"] = f"{float(phase12[i]):.6f}"
            writer.writerow(item)


def write_report(output_dir: Path, selected: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], phase12_metrics: Optional[Mapping[str, float]], args: argparse.Namespace) -> None:
    lines = [
        "# Phase 14 Full Research Gauntlet Report",
        "",
        "## Result",
        f"- Selected method: `{selected['name']}`",
        f"- OOF MAE: `{selected['oof']['mae']:.3f}cm`",
        f"- Test MAE: `{selected['test']['mae']:.3f}cm`",
        f"- Test short MAE: `{selected['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Test within 3cm: `{selected['test'].get('within_3cm', float('nan')):.3f}`",
        f"- 3cm target met: `{selected['test']['mae'] <= 3.0}`",
        f"- 4.5cm target met: `{selected['test']['mae'] <= 4.5}`",
        "",
        "## References",
    ]
    if phase12_metrics is not None:
        lines.append(
            f"- Phase12 frontier: `{phase12_metrics['mae']:.3f}cm`, short `{phase12_metrics.get('short_mae', float('nan')):.3f}cm`, "
            f"delta `{float(selected['test']['mae']) - float(phase12_metrics['mae']):+.3f}cm`"
        )
    lines.extend(
        [
            f"- Folds: `{int(args.folds)}`",
            f"- Repeats: `{int(args.repeats)}`",
            f"- Candidates searched: `{len(rows)}`",
            "",
            "## Top Candidates",
        ]
    )
    for row in rows[:40]:
        lines.append(
            f"- `{row['name']}`: oof `{row['oof']['mae']:.3f}cm`, test `{row['test']['mae']:.3f}cm`, "
            f"short `{row['test'].get('short_mae', float('nan')):.3f}cm`, score `{row['selection_score']:.3f}`"
        )
    lines.extend(["", "## Research Verdict"])
    if selected["test"]["mae"] <= 3.0:
        lines.append("The sealed-test 3cm gate is met under this gauntlet.")
    elif selected["test"]["mae"] <= 4.5:
        lines.append("The gauntlet reaches the 4.5cm gate but not the 3cm gate.")
    else:
        lines.append("The gauntlet does not reach 4.5cm or 3cm. Any lower number would require either new reliable signal, better labels, or leakage.")
    (output_dir / "PHASE14_FULL_RESEARCH_GAUNTLET_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = torch.device(str(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Phase14 is CUDA-only for this research run.")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase14] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    data = load_data(args)
    dev_y = data["dev_y"]
    test_y = data["test_y"]
    support_y = data["support_y"]
    dev_meta = data["dev_meta"]
    support_meta = data["support_meta"]
    test_meta = data["test_meta"]
    dev_meta_metrics = data["dev_meta_metrics"]
    test_meta_metrics = data["test_meta_metrics"]
    folds = p13.make_folds(dev_meta, dev_y, int(args.folds), int(args.repeats), int(args.seed))
    print(f"[phase14] dev={len(dev_y)} support={len(support_y)} test={len(test_y)} folds={len(folds)}", flush=True)
    print(f"[phase14] feature sets: {', '.join(data['feature_sets'].keys())}", flush=True)

    candidates: List[Dict[str, Any]] = []
    for feat_name, (dev_x, support_x, test_x) in data["feature_sets"].items():
        print(f"[phase14] running feature set {feat_name} dim={dev_x.shape[1]}", flush=True)
        for lam in (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0):
            for cw in (0.0, 0.08, 0.18, 0.32):
                for sb in (1.0, 1.25, 1.55):
                    label = "target" if cw == 0.0 else "celeb"
                    candidates.append(
                        p13.cv_ridge_candidate(
                            name=f"{feat_name}_{label}_ridge_lam{lam:g}_cw{cw:g}_sb{sb:g}",
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
        for dim in (128, 256, 512):
            for lam in (30.0, 300.0, 3000.0):
                candidates.append(
                    cv_rp_ridge_candidate(
                        f"{feat_name}_rp{dim}_ridge_lam{lam:g}",
                        dev_x,
                        dev_y,
                        dev_meta,
                        support_x,
                        support_y,
                        support_meta,
                        test_x,
                        folds,
                        dim=dim,
                        lam=lam,
                        seed=int(args.seed) + dim,
                        celeb_weight=0.18,
                        short_boost=1.25,
                        device=device,
                    )
                )
        for k in (15, 31, 55, 89):
            for temp in (0.035, 0.060, 0.100):
                candidates.append(cv_knn_candidate(f"{feat_name}_knn_k{k}_t{temp:g}", dev_x, dev_y, support_x, support_y, test_x, folds, k=k, temp=temp, use_support=False, device=device))
                candidates.append(cv_knn_candidate(f"{feat_name}_knn_support_k{k}_t{temp:g}", dev_x, dev_y, support_x, support_y, test_x, folds, k=k, temp=temp, use_support=True, device=device))

    print(f"[phase14] after GPU linear/KNN candidates={len(candidates)}", flush=True)

    if bool(args.skip_sklearn):
        print("[phase14] sklearn CPU zoo skipped by --skip-sklearn", flush=True)
    else:
        try:
            from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
            from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            sklearn_features = data["feature_sets"].get("ecapa_bio_meta", next(iter(data["feature_sets"].values())))[0]
            sklearn_test = data["feature_sets"].get("ecapa_bio_meta", next(iter(data["feature_sets"].values())))[2]
            estimators = [
                ("ridge_a10", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=10.0))])),
                ("ridge_a100", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=100.0))])),
                ("elastic_a001", Pipeline([("s", StandardScaler()), ("m", ElasticNet(alpha=0.01, l1_ratio=0.08, max_iter=5000, random_state=int(args.seed)))])),
                ("huber", Pipeline([("s", StandardScaler()), ("m", HuberRegressor(alpha=0.03, epsilon=1.45, max_iter=700))])),
                ("extra_d2", ExtraTreesRegressor(n_estimators=360, max_depth=2, min_samples_leaf=8, random_state=int(args.seed), n_jobs=-1)),
                ("extra_d4", ExtraTreesRegressor(n_estimators=420, max_depth=4, min_samples_leaf=6, random_state=int(args.seed), n_jobs=-1)),
                ("rf_d4", RandomForestRegressor(n_estimators=360, max_depth=4, min_samples_leaf=6, random_state=int(args.seed), n_jobs=-1)),
                ("gbr_d2", GradientBoostingRegressor(loss="absolute_error", n_estimators=260, max_depth=2, learning_rate=0.025, random_state=int(args.seed))),
                ("hist_l2", HistGradientBoostingRegressor(loss="absolute_error", max_iter=260, learning_rate=0.035, l2_regularization=0.4, max_leaf_nodes=15, random_state=int(args.seed))),
            ]
            for name, est in estimators:
                for mode in ("none", "short"):
                    candidates.append(sklearn_candidate(f"sk_{name}_{mode}", est, sklearn_features, dev_y, sklearn_test, folds, sample_weight_mode=mode))
        except Exception as exc:
            print(f"[phase14] sklearn zoo skipped: {exc}", flush=True)

    if not bool(args.skip_xgb):
        try:
            import xgboost as xgb

            xgb_features = data["feature_sets"].get("ecapa_bio_meta", next(iter(data["feature_sets"].values())))[0]
            xgb_test = data["feature_sets"].get("ecapa_bio_meta", next(iter(data["feature_sets"].values())))[2]
            configs = [
                {"max_depth": 1, "learning_rate": 0.030, "n_estimators": 360, "subsample": 0.9, "colsample_bytree": 0.7, "reg_lambda": 8.0, "min_child_weight": 6.0},
                {"max_depth": 2, "learning_rate": 0.020, "n_estimators": 520, "subsample": 0.82, "colsample_bytree": 0.45, "reg_lambda": 24.0, "min_child_weight": 12.0},
                {"max_depth": 3, "learning_rate": 0.015, "n_estimators": 620, "subsample": 0.78, "colsample_bytree": 0.35, "reg_lambda": 45.0, "min_child_weight": 18.0},
            ]
            for i, cfg in enumerate(configs):
                model = xgb.XGBRegressor(objective="reg:absoluteerror", tree_method="hist", device="cuda", random_state=int(args.seed) + i, n_jobs=1, **cfg)
                candidates.append(sklearn_candidate(f"xgb_gpu_{i}", model, xgb_features, dev_y, xgb_test, folds, sample_weight_mode="short"))
        except Exception as exc:
            print(f"[phase14] xgboost unavailable/skipped: {exc}", flush=True)

    if not bool(args.skip_mlp):
        mlp_features = data["feature_sets"].get("ecapa_bio_meta", next(iter(data["feature_sets"].values())))[0]
        mlp_test = data["feature_sets"].get("ecapa_bio_meta", next(iter(data["feature_sets"].values())))[2]
        for width, dropout, lr in ((256, 0.08, 8e-4), (384, 0.12, 5e-4)):
            candidates.append(
                mlp_candidate(
                    f"cuda_mlp_w{width}_d{dropout:g}_lr{lr:g}",
                    mlp_features,
                    dev_y,
                    mlp_test,
                    folds,
                    seed=int(args.seed),
                    epochs=int(args.mlp_epochs),
                    patience=int(args.mlp_patience),
                    width=width,
                    dropout=dropout,
                    lr=lr,
                    device=device,
                )
            )

    rows = phase14_rows(candidates, dev_y, test_y, dev_meta_metrics, test_meta_metrics)
    print(f"[phase14] best raw {rows[0]['name']} oof={rows[0]['oof']['mae']:.3f} test={rows[0]['test']['mae']:.3f}", flush=True)
    candidates.extend(residual_family(candidates, rows, dev_y, dev_meta, test_meta, top_n=10))
    rows = phase14_rows(candidates, dev_y, test_y, dev_meta_metrics, test_meta_metrics)
    candidates.extend(convex_stack(candidates, rows, dev_y, top_n=12, seed=int(args.seed)))
    candidates.extend(convex_stack(candidates, rows, dev_y, top_n=24, seed=int(args.seed) + 7))
    rows = phase14_rows(candidates, dev_y, test_y, dev_meta_metrics, test_meta_metrics)
    selected = rows[0]
    selected_cand = next(c for c in candidates if str(c["name"]) == str(selected["name"]))
    selected_pred = np.asarray(selected_cand["test_pred"], dtype=np.float32)
    val_count = int(len(data["cache"]["val"]["y"]))
    selected_val_pred = np.asarray(selected_cand["oof_pred"], dtype=np.float32)[-val_count:]
    val_y = np.asarray(data["cache"]["val"]["y"], dtype=np.float32)
    val_meta = data["dev_meta"][-val_count:]
    write_predictions(output_dir / "phase14_predictions_val.csv", val_y, selected_val_pred, val_meta, None)
    write_predictions(output_dir / "phase14_predictions_test.csv", test_y, selected_pred, test_meta, data["phase12_pred"])
    report = {
        "selected": selected,
        "phase12_reference": data["phase12_metrics"],
        "candidate_count": len(candidates),
        "top_candidates": rows[:160],
        "args": vars(args),
    }
    (output_dir / "phase14_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_report(output_dir, selected, rows, data["phase12_metrics"], args)
    print(
        f"[phase14] selected={selected['name']} test_mae={selected['test']['mae']:.3f} "
        f"short={selected['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    if data["phase12_metrics"] is not None:
        print(f"[phase14] phase12_reference={data['phase12_metrics']['mae']:.3f}", flush=True)
    print(f"[phase14] wrote {output_dir / 'PHASE14_FULL_RESEARCH_GAUNTLET_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
