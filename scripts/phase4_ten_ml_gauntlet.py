#!/usr/bin/env python
"""Phase 4 ten-technique CUDA ML gauntlet.

This script is intentionally research-style, not hype-style. It tests a broad
set of defensible speaker-level learners and calibrators on CUDA, selects by
validation only, and reports sealed-test performance once.

The goal is to use more ML skill without cheating:

1. robust feature scaling
2. target-domain filtering
3. metadata augmentation
4. all-domain KNN support
5. target-domain KNN support
6. cosine/Laplacian kernel smoothing
7. random-projection ridge regression
8. bagged random-projection ridge
9. compact neural residual regressor
10. validation-safe convex super learner

It also reports an oracle diagnostic separately. Oracle numbers are useful to
understand whether any hidden model mixture exists, but they are not deployable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_SOURCES: Tuple[Tuple[str, str, str, str], ...] = (
    ("combo", "outputs/speaker_gpu_combo_full_ssl_cuda/predictions_val.csv", "outputs/speaker_gpu_combo_full_ssl_cuda/predictions_test.csv", "pred_cm"),
    ("target", "outputs/speaker_gpu_target_ssl_cuda/predictions_val.csv", "outputs/speaker_gpu_target_ssl_cuda/predictions_test.csv", "pred_cm"),
    ("phase1_raw", "outputs/speaker_gpu_phase1_fullpower/predictions_val.csv", "outputs/speaker_gpu_phase1_fullpower/predictions_test.csv", "pred_cm"),
    ("phase1_cal", "outputs/speaker_gpu_phase1_fullpower/predictions_val.csv", "outputs/speaker_gpu_phase1_fullpower/predictions_test.csv", "pred_calibrated_cm"),
    ("phase2_blend", "outputs/phase2_data_forensics/phase2_predictions_val.csv", "outputs/phase2_data_forensics/phase2_predictions_test.csv", "phase2_pred_cm"),
    ("phase2_knn", "outputs/phase2_data_forensics/phase2_predictions_val.csv", "outputs/phase2_data_forensics/phase2_predictions_test.csv", "knn_pred_cm"),
    ("phase3_final", "outputs/phase3_target_domain_rescue/phase3_predictions_val.csv", "outputs/phase3_target_domain_rescue/phase3_predictions_test.csv", "final_pred_cm"),
    ("epoch_topk_ensemble", "outputs/epoch_ensemble_gpu/topk_ensemble_eval/checkpoint_ensemble_predictions_val.csv", "outputs/epoch_ensemble_gpu/topk_ensemble_eval/checkpoint_ensemble_predictions_test.csv", "ensemble_pred_cm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 ten-technique CUDA ML gauntlet.")
    parser.add_argument("--speaker-cache", default="outputs/speaker_gpu_combo_full_ssl_cuda/speaker_gpu_cache.pt")
    parser.add_argument("--output-dir", default="outputs/phase4_ten_ml_gauntlet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mae-cm", type=float, default=3.0)
    parser.add_argument("--short-cm", type=float, default=160.0)
    parser.add_argument("--tall-cm", type=float, default=175.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--ridge-dims", default="96,192,384")
    parser.add_argument("--ridge-lambdas", default="0.1,0.3,1,3,10,30,100,300,1000,3000")
    parser.add_argument("--mlp-epochs", type=int, default=500)
    parser.add_argument("--mlp-patience", type=int, default=90)
    parser.add_argument("--random-blends", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def source_id(text: str) -> int:
    value = str(text or "").upper()
    if value == "TIMIT":
        return 0
    if value == "NISP":
        return 1
    if value in {"CELEB", "VOXCELEB"}:
        return 2
    return 3


def is_target_source(row: Mapping[str, Any]) -> bool:
    return str(row.get("source", "")).upper() in {"NISP", "TIMIT"}


def height_bin(y: torch.Tensor, short_cm: float, tall_cm: float) -> torch.Tensor:
    return torch.where(y < short_cm, torch.zeros_like(y, dtype=torch.long), torch.where(y < tall_cm, torch.ones_like(y, dtype=torch.long), torch.full_like(y, 2, dtype=torch.long)))


def metrics(y_true: torch.Tensor, pred: torch.Tensor, meta: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, float]:
    err = pred - y_true
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
    bins = height_bin(y_true, float(args.short_cm), float(args.tall_cm))
    for label, idx in (("short", 0), ("medium", 1), ("tall", 2)):
        mask = bins == idx
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


def selection_score(row: Mapping[str, float]) -> float:
    short = float(row.get("short_mae", row["mae"]))
    return float(row["mae"]) + 0.25 * max(0.0, short - float(row["mae"])) + 0.04 * float(row["p90_ae"])


def robust_standardize(train_x: torch.Tensor, *others: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    center = torch.quantile(train_x, 0.50, dim=0)
    q25 = torch.quantile(train_x, 0.25, dim=0)
    q75 = torch.quantile(train_x, 0.75, dim=0)
    scale = (q75 - q25).clamp_min(1e-3)
    return tuple(torch.nan_to_num((x - center) / scale, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0) for x in (train_x, *others))


def meta_tensor(meta: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    rows = []
    for row in meta:
        src = source_id(str(row.get("source", "")))
        gender = float(row.get("gender", 0))
        n_clips = math.log1p(float(row.get("n_clips", 0)))
        quality = float(row.get("quality_mean", 0.0))
        rows.append([gender, n_clips, quality, *[1.0 if src == idx else 0.0 for idx in range(4)]])
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    center = torch.nan_to_num(x.mean(dim=0), nan=0.0)
    scale = torch.nan_to_num(x.std(dim=0, unbiased=False), nan=1.0).clamp_min(1.0e-3)
    return (x - center) / scale


def load_payload(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(resolve(args.speaker_cache), map_location=device, weights_only=False)
    for split in ("train", "val", "test"):
        payload[split]["x"] = payload[split]["x"].to(device).float()
        payload[split]["y"] = payload[split]["y"].to(device).float()
    return payload


def read_prediction_csv(path: Path, column: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = str(row.get("speaker_id", "")).strip()
            if sid and column in row:
                out[sid] = float(row[column])
    return out


def attach_prediction(rows: Mapping[str, float], meta: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    missing = [str(row["speaker_id"]) for row in meta if str(row["speaker_id"]) not in rows]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} prediction rows, examples={missing[:5]}")
    return torch.tensor([float(rows[str(row["speaker_id"])]) for row in meta], dtype=torch.float32, device=device)


def load_prediction_candidates(payload: Mapping[str, Any], device: torch.device) -> List[Dict[str, Any]]:
    candidates = []
    for name, val_path, test_path, column in PREDICTION_SOURCES:
        rv = resolve(val_path)
        rt = resolve(test_path)
        if not rv.exists() or not rt.exists():
            continue
        val = attach_prediction(read_prediction_csv(rv, column), payload["val"]["metadata"], device)
        test = attach_prediction(read_prediction_csv(rt, column), payload["test"]["metadata"], device)
        candidates.append({"name": name, "family": "previous_prediction", "val_pred": val, "test_pred": test, "params": {"column": column}})
    return candidates


def group_median_prior(
    train_y: torch.Tensor,
    train_meta: Sequence[Mapping[str, Any]],
    query_meta: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    global_med = float(torch.median(train_y).item())
    buckets: Dict[Tuple[str, int], List[float]] = {}
    for y, row in zip(train_y.detach().cpu().tolist(), train_meta):
        if not is_target_source(row):
            continue
        key = (str(row.get("source", "UNKNOWN")), int(row.get("gender", 0)))
        buckets.setdefault(key, []).append(float(y))
    medians = {key: float(np.median(values)) for key, values in buckets.items() if values}
    values = []
    for row in query_meta:
        key = (str(row.get("source", "UNKNOWN")), int(row.get("gender", 0)))
        values.append(medians.get(key, global_med))
    return torch.tensor(values, dtype=torch.float32, device=device)


def knn_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_meta: Sequence[Mapping[str, Any]],
    query_x: torch.Tensor,
    query_meta: Sequence[Mapping[str, Any]],
    *,
    k: int,
    temperature: float,
    same_source_boost: float,
    same_gender_boost: float,
) -> torch.Tensor:
    tx = F.normalize(train_x, dim=1)
    qx = F.normalize(query_x, dim=1)
    sim = qx @ tx.T
    top_sim, top_idx = torch.topk(sim, k=min(k, train_x.shape[0]), dim=1)
    weights = torch.softmax(top_sim / float(temperature), dim=1)
    train_sources = torch.tensor([source_id(str(row.get("source", ""))) for row in train_meta], dtype=torch.long, device=train_x.device)
    train_genders = torch.tensor([int(row.get("gender", 0)) for row in train_meta], dtype=torch.long, device=train_x.device)
    query_sources = torch.tensor([source_id(str(row.get("source", ""))) for row in query_meta], dtype=torch.long, device=train_x.device).unsqueeze(1)
    query_genders = torch.tensor([int(row.get("gender", 0)) for row in query_meta], dtype=torch.long, device=train_x.device).unsqueeze(1)
    neighbor_sources = train_sources[top_idx]
    neighbor_genders = train_genders[top_idx]
    weights = weights * torch.where(neighbor_sources == query_sources, float(same_source_boost), 1.0)
    weights = weights * torch.where(neighbor_genders == query_genders, float(same_gender_boost), 1.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (train_y[top_idx] * weights).sum(dim=1)


def search_knn(
    name: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_meta: Sequence[Mapping[str, Any]],
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    test_x: torch.Tensor,
    test_meta: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    configs = []
    for k in (3, 5, 9, 15, 25, 35, 55, 75):
        for temperature in (0.03, 0.05, 0.08, 0.12):
            for source_boost in (1.0, 1.35):
                for gender_boost in (1.0, 1.15):
                    configs.append((k, temperature, source_boost, gender_boost))
    for k, temperature, source_boost, gender_boost in configs:
        pv = knn_predict(train_x, train_y, train_meta, val_x, val_meta, k=k, temperature=temperature, same_source_boost=source_boost, same_gender_boost=gender_boost)
        mv = metrics(val_y, pv, val_meta, args)
        score = selection_score(mv)
        if best is None or score < best["score"]:
            pt = knn_predict(train_x, train_y, train_meta, test_x, test_meta, k=k, temperature=temperature, same_source_boost=source_boost, same_gender_boost=gender_boost)
            best = {
                "name": name,
                "family": "knn_kernel_support",
                "score": score,
                "val_pred": pv,
                "test_pred": pt,
                "params": {"k": k, "temperature": temperature, "same_source_boost": source_boost, "same_gender_boost": gender_boost},
            }
    assert best is not None
    return best


def ridge_solve(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    eye = torch.eye(x.shape[1], dtype=torch.float32, device=x.device)
    eye[0, 0] = 0.0
    return torch.linalg.solve(x.T @ x + float(lam) * eye, x.T @ y)


def add_intercept(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device), x], dim=1)


def random_projection_features(x: torch.Tensor, meta: torch.Tensor, dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    proj = torch.randn((x.shape[1], int(dim)), dtype=torch.float32, device=x.device, generator=gen) / math.sqrt(float(dim))
    z = x @ proj
    z = torch.cat([z, meta], dim=1)
    z = torch.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
    return add_intercept(z)


def search_random_ridge(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_meta_x: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_meta_x: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    test_x: torch.Tensor,
    test_meta_x: torch.Tensor,
    args: argparse.Namespace,
    *,
    name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dims = [int(x.strip()) for x in str(args.ridge_dims).split(",") if x.strip()]
    lambdas = [float(x.strip()) for x in str(args.ridge_lambdas).split(",") if x.strip()]
    rows = []
    for dim in dims:
        for seed in (int(args.seed), int(args.seed) + 17, int(args.seed) + 41):
            xtr = random_projection_features(train_x, train_meta_x, dim, seed)
            xv = random_projection_features(val_x, val_meta_x, dim, seed)
            xt = random_projection_features(test_x, test_meta_x, dim, seed)
            for lam in lambdas:
                coef = ridge_solve(xtr, train_y, lam)
                pv = xv @ coef
                mv = metrics(val_y, pv, val_meta, args)
                rows.append(
                    {
                        "name": f"{name}_ridge_d{dim}_s{seed}_l{lam:g}",
                        "family": "random_projection_ridge",
                        "score": selection_score(mv),
                        "val_pred": pv,
                        "test_pred": xt @ coef,
                        "params": {"dim": dim, "seed": seed, "lambda": lam},
                    }
                )
    best = min(rows, key=lambda row: row["score"])
    top = sorted(rows, key=lambda row: row["score"])[: min(7, len(rows))]
    bag_val = torch.stack([row["val_pred"] for row in top], dim=0).mean(dim=0)
    bag_test = torch.stack([row["test_pred"] for row in top], dim=0).mean(dim=0)
    bagged = {
        "name": f"{name}_bagged_ridge_top{len(top)}",
        "family": "bagged_random_projection_ridge",
        "score": selection_score(metrics(val_y, bag_val, val_meta, args)),
        "val_pred": bag_val,
        "test_pred": bag_test,
        "params": {"members": [row["name"] for row in top]},
    }
    return best, bagged


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp_candidate(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_meta_x: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_meta_x: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    test_x: torch.Tensor,
    test_meta_x: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    seed = int(args.seed) + 101
    dim = 384
    xtr = random_projection_features(train_x, train_meta_x, dim, seed)[:, 1:]
    xv = random_projection_features(val_x, val_meta_x, dim, seed)[:, 1:]
    xt = random_projection_features(test_x, test_meta_x, dim, seed)[:, 1:]
    center = train_y.mean()
    scale = train_y.std(unbiased=False).clamp_min(1.0)
    y_scaled = (train_y - center) / scale
    weights = torch.ones_like(train_y)
    weights = weights * torch.where(train_y < float(args.short_cm), 1.9, 1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    model = ResidualMLP(xtr.shape[1]).to(xtr.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=2.0e-3)
    best: Optional[Dict[str, Any]] = None
    stale = 0
    for epoch in range(1, int(args.mlp_epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(xtr)
        loss = (F.smooth_l1_loss(pred, y_scaled, reduction="none", beta=0.35) * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pv = model(xv) * scale + center
                mv = metrics(val_y, pv, val_meta, args)
                score = selection_score(mv)
                if best is None or score < best["score"]:
                    best = {
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                        "score": score,
                        "epoch": epoch,
                    }
                    stale = 0
                else:
                    stale += 10
        if stale >= int(args.mlp_patience):
            break
    assert best is not None
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pv = model(xv) * scale + center
        pt = model(xt) * scale + center
    return {
        "name": "target_residual_mlp",
        "family": "compact_neural_residual",
        "score": selection_score(metrics(val_y, pv, val_meta, args)),
        "val_pred": pv,
        "test_pred": pt,
        "params": {"projection_dim": dim, "seed": seed, "best_epoch": int(best["epoch"])},
    }


def group_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return (str(row.get("source", "UNKNOWN")), int(row.get("gender", 0)))


def loo_group_calibrate(
    name: str,
    val_y: torch.Tensor,
    test_y: torch.Tensor,
    val_pred: torch.Tensor,
    test_pred: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    residuals = (val_y - val_pred).detach().cpu().numpy()
    keys = [group_key(row) for row in val_meta]
    global_res = float(np.median(residuals))
    val_adj = []
    bucket: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}
    for idx, (key, res) in enumerate(zip(keys, residuals)):
        bucket.setdefault(key, []).append((idx, float(res)))
    for idx, key in enumerate(keys):
        vals = [res for j, res in bucket.get(key, []) if j != idx]
        if not vals:
            vals = [float(residuals[j]) for j in range(len(residuals)) if j != idx]
        bias = float(np.median(vals)) if vals else global_res
        val_adj.append(float(val_pred[idx].item()) + 0.55 * bias)
    full_bias = {key: float(np.median([res for _, res in vals])) for key, vals in bucket.items()}
    test_adj = []
    for idx, row in enumerate(test_meta):
        bias = full_bias.get(group_key(row), global_res)
        test_adj.append(float(test_pred[idx].item()) + 0.55 * bias)
    pv = torch.tensor(val_adj, dtype=torch.float32, device=val_y.device)
    pt = torch.tensor(test_adj, dtype=torch.float32, device=test_y.device)
    return {
        "name": f"{name}_loo_group_cal",
        "family": "leave_one_out_group_calibration",
        "score": selection_score(metrics(val_y, pv, val_meta, args)),
        "val_pred": pv,
        "test_pred": pt,
        "params": {"shrink": 0.55},
    }


def random_convex_stack(
    candidates: Sequence[Dict[str, Any]],
    val_y: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    test_y: torch.Tensor,
    test_meta: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    *,
    oracle: bool,
) -> Dict[str, Any]:
    device = val_y.device
    names = [row["name"] for row in candidates]
    val_stack = torch.stack([row["val_pred"] for row in candidates], dim=1).to(device)
    test_stack = torch.stack([row["test_pred"] for row in candidates], dim=1).to(device)
    best: Optional[Dict[str, Any]] = None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + (909 if oracle else 707))
    probes = []
    for idx in range(len(candidates)):
        w = torch.zeros(len(candidates), dtype=torch.float32, device=device)
        w[idx] = 1.0
        probes.append(w)
    for _ in range(int(args.random_blends)):
        raw = torch.rand(len(candidates), dtype=torch.float32, device=device, generator=gen).pow(2.0)
        probes.append(raw / raw.sum().clamp_min(1e-6))
    for w in probes:
        pv = val_stack @ w
        pt = test_stack @ w
        mv = metrics(val_y, pv, val_meta, args)
        mt = metrics(test_y, pt, test_meta, args)
        score = float(mt["mae"]) if oracle else selection_score(mv)
        if best is None or score < best["score"]:
            best = {
                "name": "oracle_random_convex_stack" if oracle else "validation_random_convex_stack",
                "family": "convex_super_learner",
                "score": score,
                "val_pred": pv,
                "test_pred": pt,
                "params": {"oracle": oracle, "weights": {name: float(w[i].item()) for i, name in enumerate(names)}},
            }
    assert best is not None
    return best


def candidate_public(row: Mapping[str, Any], val_y: torch.Tensor, val_meta: Sequence[Mapping[str, Any]], test_y: torch.Tensor, test_meta: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "name": row["name"],
        "family": row["family"],
        "params": row.get("params", {}),
        "val": metrics(val_y, row["val_pred"], val_meta, args),
        "test": metrics(test_y, row["test_pred"], test_meta, args),
        "selection_score": selection_score(metrics(val_y, row["val_pred"], val_meta, args)),
    }


def write_predictions(path: Path, y: torch.Tensor, pred: torch.Tensor, meta: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for idx, row in enumerate(meta):
        true = float(y[idx].item())
        value = float(pred[idx].item())
        rows.append(
            {
                "speaker_id": str(row.get("speaker_id", "")),
                "source": str(row.get("source", "UNKNOWN")),
                "gender": int(row.get("gender", 0)),
                "height_cm": f"{true:.6f}",
                "phase4_pred_cm": f"{value:.6f}",
                "phase4_abs_error_cm": f"{abs(value - true):.6f}",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    sel = report["selected"]
    oracle = report["oracle"]
    lines = [
        "# Phase 4 Ten-ML Gauntlet Report",
        "",
        "## Result",
        f"- Selected deployable method: `{sel['name']}`",
        f"- Selected test MAE: `{sel['test']['mae']:.3f}cm`",
        f"- Selected short MAE: `{sel['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Target 3cm met: `{report['target_met']}`",
        "",
        "## Oracle Diagnostic",
        f"- Oracle method: `{oracle['name']}`",
        f"- Oracle test MAE: `{oracle['test']['mae']:.3f}cm`",
        "- Oracle uses sealed-test labels for selection and is not deployable.",
        "",
        "## Ten ML Techniques Used",
    ]
    for item in report["techniques"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Top Candidates By Validation Score"])
    for row in report["top_by_validation"][:12]:
        lines.append(f"- `{row['name']}` ({row['family']}): val `{row['val']['mae']:.3f}cm`, test `{row['test']['mae']:.3f}cm`, short `{row['test'].get('short_mae', float('nan')):.3f}cm`")
    lines.extend(["", "## Top Candidates By Test MAE Diagnostic"])
    for row in report["top_by_test"][:12]:
        lines.append(f"- `{row['name']}` ({row['family']}): test `{row['test']['mae']:.3f}cm`, val `{row['val']['mae']:.3f}cm`, short `{row['test'].get('short_mae', float('nan')):.3f}cm`")
    lines.extend(
        [
            "",
            "## Conclusion",
            "This gauntlet uses substantially more ML machinery than the earlier phases. If it still does not approach 3cm, the blocker is the available supervision/domain signal, not lack of model variety.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Phase 4. Refusing CPU.")
    seed_everything(int(args.seed))
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase4] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    payload = load_payload(args, device)
    train_x_raw, val_x_raw, test_x_raw = payload["train"]["x"], payload["val"]["x"], payload["test"]["x"]
    train_x, val_x, test_x = robust_standardize(train_x_raw, val_x_raw, test_x_raw)
    train_y = payload["train"]["y"]
    val_y = payload["val"]["y"]
    test_y = payload["test"]["y"]
    train_meta = payload["train"]["metadata"]
    val_meta = payload["val"]["metadata"]
    test_meta = payload["test"]["metadata"]
    train_meta_x = meta_tensor(train_meta, device)
    val_meta_x = meta_tensor(val_meta, device)
    test_meta_x = meta_tensor(test_meta, device)
    target_mask = torch.tensor([is_target_source(row) for row in train_meta], dtype=torch.bool, device=device)
    if target_mask.sum().item() < 32:
        raise RuntimeError("Not enough target-domain train speakers for Phase 4.")

    candidates: List[Dict[str, Any]] = load_prediction_candidates(payload, device)
    print(f"[phase4] loaded {len(candidates)} previous prediction candidates", flush=True)

    prior_val = group_median_prior(train_y, train_meta, val_meta, device=device)
    prior_test = group_median_prior(train_y, train_meta, test_meta, device=device)
    candidates.append({"name": "target_group_median_prior", "family": "metadata_prior", "val_pred": prior_val, "test_pred": prior_test, "params": {}})

    print("[phase4] searching KNN/kernel support models", flush=True)
    candidates.append(search_knn("all_domain_knn_kernel", train_x, train_y, train_meta, val_x, val_y, val_meta, test_x, test_meta, args))
    candidates.append(search_knn("target_domain_knn_kernel", train_x[target_mask], train_y[target_mask], [row for row in train_meta if is_target_source(row)], val_x, val_y, val_meta, test_x, test_meta, args))

    print("[phase4] searching random-projection ridge models", flush=True)
    best_all_ridge, bag_all_ridge = search_random_ridge(train_x, train_y, train_meta_x, val_x, val_y, val_meta_x, val_meta, test_x, test_meta_x, args, name="all_domain")
    candidates.extend([best_all_ridge, bag_all_ridge])
    best_target_ridge, bag_target_ridge = search_random_ridge(
        train_x[target_mask],
        train_y[target_mask],
        train_meta_x[target_mask],
        val_x,
        val_y,
        val_meta_x,
        val_meta,
        test_x,
        test_meta_x,
        args,
        name="target_domain",
    )
    candidates.extend([best_target_ridge, bag_target_ridge])

    print("[phase4] training compact neural residual model", flush=True)
    candidates.append(train_mlp_candidate(train_x[target_mask], train_y[target_mask], train_meta_x[target_mask], val_x, val_y, val_meta_x, val_meta, test_x, test_meta_x, args))

    print("[phase4] adding leave-one-out group calibration candidates", flush=True)
    base_for_cal = list(candidates)
    for row in base_for_cal:
        if row["family"] in {"previous_prediction", "convex_super_learner"} or row["name"] in {"target_domain_knn_kernel", "target_domain_bagged_ridge_top7"}:
            candidates.append(loo_group_calibrate(row["name"], val_y, test_y, row["val_pred"], row["test_pred"], val_meta, test_meta, args))

    print("[phase4] searching validation-safe convex super learner", flush=True)
    validation_stack = random_convex_stack(candidates, val_y, val_meta, test_y, test_meta, args, oracle=False)
    candidates.append(validation_stack)
    oracle_stack = random_convex_stack(candidates, val_y, val_meta, test_y, test_meta, args, oracle=True)

    public_rows = [candidate_public(row, val_y, val_meta, test_y, test_meta, args) for row in candidates]
    top_by_val = sorted(public_rows, key=lambda row: row["selection_score"])
    top_by_test = sorted(public_rows, key=lambda row: row["test"]["mae"])
    selected_name = validation_stack["name"]
    selected_public = candidate_public(validation_stack, val_y, val_meta, test_y, test_meta, args)
    oracle_public = candidate_public(oracle_stack, val_y, val_meta, test_y, test_meta, args)

    report = {
        "phase": "phase4_ten_ml_gauntlet",
        "device": torch.cuda.get_device_name(0),
        "target_mae_cm": float(args.target_mae_cm),
        "target_met": bool(selected_public["test"]["mae"] <= float(args.target_mae_cm)),
        "speaker_counts": {
            "train": len(train_meta),
            "target_train": int(target_mask.sum().item()),
            "val": len(val_meta),
            "test": len(test_meta),
        },
        "source_counts": {
            "train": dict(Counter(str(row.get("source", "UNKNOWN")) for row in train_meta)),
            "val": dict(Counter(str(row.get("source", "UNKNOWN")) for row in val_meta)),
            "test": dict(Counter(str(row.get("source", "UNKNOWN")) for row in test_meta)),
        },
        "techniques": [
            "robust median/IQR feature scaling",
            "target-domain train filtering",
            "speaker metadata augmentation",
            "all-domain KNN support regression",
            "target-domain KNN support regression",
            "cosine soft-kernel neighbor weighting",
            "random-projection ridge regression",
            "bagged random-projection ridge regression",
            "compact neural residual regressor",
            "validation-safe convex super learner",
            "leave-one-out group residual calibration",
        ],
        "selected": selected_public,
        "oracle": oracle_public,
        "top_by_validation": top_by_val,
        "top_by_test": top_by_test,
        "all_candidates": public_rows,
        "selected_name": selected_name,
    }
    (output_dir / "phase4_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_markdown(output_dir / "PHASE4_REPORT.md", report)
    write_predictions(output_dir / "phase4_predictions_val.csv", val_y, validation_stack["val_pred"], val_meta)
    write_predictions(output_dir / "phase4_predictions_test.csv", test_y, validation_stack["test_pred"], test_meta)

    print(
        f"[phase4] selected={selected_public['name']} val_mae={selected_public['val']['mae']:.3f} "
        f"test_mae={selected_public['test']['mae']:.3f} short={selected_public['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        f"[phase4] oracle_not_deployable test_mae={oracle_public['test']['mae']:.3f}",
        flush=True,
    )
    print(f"[phase4] wrote {output_dir / 'PHASE4_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
