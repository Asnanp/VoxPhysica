#!/usr/bin/env python
"""Phase 19 CUDA residual MoE bagger.

This phase tries a new route after the short-tail calibration plateau:

- fold-local KNN support predictions become the base estimate
- a compact mixture-of-experts neural residual model predicts corrections
- auxiliary height-bin and ranking losses shape the short/medium/tall ordering
- model selection uses target-domain out-of-fold speakers, then reports test once

It is intentionally CUDA-only and evidence-first. A sealed-test 3cm claim is only
valid if the report says the sealed-test metric actually crossed 3cm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase17_short_tail_rescue as p17  # noqa: E402
import phase18_oof_short_rescue as p18  # noqa: E402


@dataclass(frozen=True)
class Config:
    name: str
    dim: int
    hidden: int
    experts: int
    dropout: float
    lr: float
    weight_decay: float
    short_weight: float
    support_weight: float
    k: int
    temp: float
    short_boost: float
    rank_weight: float
    bin_weight: float
    residual_scale: float
    proj_seed_offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA residual MoE bagger.")
    parser.add_argument("--speaker-cache", default="outputs/speaker_gpu_combo_full_ssl_cuda/speaker_gpu_cache.pt")
    parser.add_argument("--output-dir", default="outputs/phase19_cuda_moe_residual_bagger")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--patience", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--short-cm", type=float, default=160.0)
    parser.add_argument("--tall-cm", type=float, default=175.0)
    parser.add_argument("--max-configs", type=int, default=0, help="Debug limit; 0 means all configs.")
    parser.add_argument("--complexity-penalty", type=float, default=0.010)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def sid(row: Mapping[str, Any]) -> str:
    return str(row.get("speaker_id", "")).strip()


def target_source(row: Mapping[str, Any]) -> bool:
    return p17.target_source(row)


def source_id(row: Mapping[str, Any]) -> int:
    return p17.source_id(row)


def clip_pred(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    return np.clip(arr.astype(np.float32), 145.0, 195.0).astype(np.float32)


def load_payload(path: Path, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    for split in ("train", "val", "test"):
        payload[split]["x"] = payload[split]["x"].to(device).float()
        payload[split]["y"] = payload[split]["y"].to(device).float()
    return payload


def make_data(payload: Mapping[str, Any]) -> Dict[str, Any]:
    train_meta = list(payload["train"]["metadata"])
    target_mask = torch.tensor([target_source(row) for row in train_meta], dtype=torch.bool, device=payload["train"]["x"].device)
    support_mask = ~target_mask
    dev_x = torch.cat([payload["train"]["x"][target_mask], payload["val"]["x"]], dim=0)
    dev_y = torch.cat([payload["train"]["y"][target_mask], payload["val"]["y"]], dim=0)
    dev_meta = [row for row in train_meta if target_source(row)] + list(payload["val"]["metadata"])
    support_meta = [row for row in train_meta if not target_source(row)]
    return {
        "dev_x": dev_x,
        "dev_y": dev_y,
        "dev_meta": dev_meta,
        "support_x": payload["train"]["x"][support_mask],
        "support_y": payload["train"]["y"][support_mask],
        "support_meta": support_meta,
        "test_x": payload["test"]["x"],
        "test_y": payload["test"]["y"],
        "test_meta": list(payload["test"]["metadata"]),
    }


def make_folds(
    meta: Sequence[Mapping[str, Any]],
    y: torch.Tensor,
    *,
    folds: int,
    repeats: int,
    seed: int,
    short_cm: float,
    tall_cm: float,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    y_np = y.detach().cpu().numpy().astype(np.float32)
    buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for idx, row in enumerate(meta):
        bin_id = 0 if y_np[idx] < short_cm else (1 if y_np[idx] < tall_cm else 2)
        buckets[(source_id(row), int(row.get("gender", 0)), bin_id)].append(idx)
    all_indices = set(range(len(meta)))
    out: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for rep in range(int(repeats)):
        rng = random.Random(int(seed) + 10007 * rep)
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


def take_meta(meta: Sequence[Mapping[str, Any]], indices: Iterable[int]) -> List[Mapping[str, Any]]:
    return [meta[int(i)] for i in indices]


def meta_matrix(meta: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    rows: List[List[float]] = []
    for row in meta:
        src = source_id(row)
        gender = float(int(row.get("gender", 0)))
        n_clips = math.log1p(float(row.get("n_clips", 0.0))) / math.log(80.0)
        quality = float(row.get("quality_mean", 0.0))
        rows.append(
            [
                gender,
                n_clips,
                quality,
                1.0 if src == 0 else 0.0,
                1.0 if src == 1 else 0.0,
                1.0 if src == 2 else 0.0,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device)


def meta_standardize(train_meta: Sequence[Mapping[str, Any]], query_meta: Sequence[Mapping[str, Any]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    train = meta_matrix(train_meta, device)
    query = meta_matrix(query_meta, device)
    center = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp_min(1e-3)
    return (train - center) / scale, (query - center) / scale


def projected_features(
    train_x: torch.Tensor,
    query_x: torch.Tensor,
    train_meta: Sequence[Mapping[str, Any]],
    query_meta: Sequence[Mapping[str, Any]],
    *,
    dim: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    train_z, query_z = p17.robust_standardize(train_x, query_x)
    gen = torch.Generator(device=train_x.device)
    gen.manual_seed(int(seed))
    proj = torch.randn((train_z.shape[1], int(dim)), dtype=torch.float32, device=train_x.device, generator=gen)
    proj = proj / math.sqrt(float(dim))
    train_p = train_z @ proj
    query_p = query_z @ proj
    meta_train, meta_query = meta_standardize(train_meta, query_meta, train_x.device)
    return torch.cat([train_p, meta_train], dim=1).clamp(-8.0, 8.0), torch.cat([query_p, meta_query], dim=1).clamp(-8.0, 8.0)


def knn_base(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_meta: Sequence[Mapping[str, Any]],
    query_x: torch.Tensor,
    query_meta: Sequence[Mapping[str, Any]],
    *,
    k: int,
    temp: float,
    short_boost: float,
    short_cm: float,
    leave_one_out: bool,
) -> torch.Tensor:
    train_z, query_z = p17.robust_standardize(train_x, query_x)
    tx = F.normalize(train_z, dim=1)
    qx = F.normalize(query_z, dim=1)
    sim = qx @ tx.T
    if leave_one_out and query_x.shape[0] == train_x.shape[0]:
        sim = sim.clone()
        sim.fill_diagonal_(-1.0e9)
    top_sim, top_idx = torch.topk(sim, k=min(int(k), train_x.shape[0] - (1 if leave_one_out else 0)), dim=1)
    weights = torch.softmax(top_sim / float(temp), dim=1)
    neighbor_y = train_y[top_idx]
    neighbor_short = (neighbor_y < float(short_cm)).float()
    weights = weights * torch.where(neighbor_short > 0, float(short_boost), 1.0)
    train_sources = torch.tensor([source_id(row) for row in train_meta], dtype=torch.long, device=train_x.device)
    query_sources = torch.tensor([source_id(row) for row in query_meta], dtype=torch.long, device=train_x.device).unsqueeze(1)
    train_genders = torch.tensor([int(row.get("gender", 0)) for row in train_meta], dtype=torch.long, device=train_x.device)
    query_genders = torch.tensor([int(row.get("gender", 0)) for row in query_meta], dtype=torch.long, device=train_x.device).unsqueeze(1)
    weights = weights * torch.where(train_sources[top_idx] == query_sources, 1.25, 1.0)
    weights = weights * torch.where(train_genders[top_idx] == query_genders, 1.10, 1.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (neighbor_y * weights).sum(dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualMoE(nn.Module):
    def __init__(self, input_dim: int, hidden: int, experts: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.stem = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(ResidualBlock(hidden, dropout), ResidualBlock(hidden, dropout))
        self.gate = nn.Linear(hidden, experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)) for _ in range(experts)])
        self.bin_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(x))
        gate = torch.softmax(self.gate(h), dim=1)
        expert_values = torch.cat([expert(h) for expert in self.experts], dim=1)
        residual = torch.sum(gate * expert_values, dim=1)
        residual = torch.tanh(residual / self.residual_scale) * self.residual_scale
        return residual, self.bin_head(h)


def bin_targets(y: torch.Tensor, short_cm: float, tall_cm: float) -> torch.Tensor:
    return torch.where(y < float(short_cm), torch.zeros_like(y, dtype=torch.long), torch.where(y < float(tall_cm), torch.ones_like(y, dtype=torch.long), torch.full_like(y, 2, dtype=torch.long)))


def sample_weights(y: torch.Tensor, *, short_weight: float, short_cm: float, tall_cm: float) -> torch.Tensor:
    return torch.where(y < float(short_cm), torch.tensor(float(short_weight), device=y.device), torch.where(y >= float(tall_cm), torch.tensor(1.25, device=y.device), torch.tensor(1.0, device=y.device)))


def ranking_loss(pred: torch.Tensor, y: torch.Tensor, count: int = 512) -> torch.Tensor:
    n = pred.shape[0]
    if n < 4:
        return pred.new_tensor(0.0)
    idx_a = torch.randint(0, n, (int(count),), device=pred.device)
    idx_b = torch.randint(0, n, (int(count),), device=pred.device)
    diff_y = y[idx_a] - y[idx_b]
    mask = diff_y.abs() >= 4.0
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    sign = diff_y[mask].sign()
    margin = (diff_y[mask].abs() / 12.0).clamp(0.25, 1.5)
    diff_p = pred[idx_a[mask]] - pred[idx_b[mask]]
    return F.softplus(-(sign * diff_p) / margin).mean()


@torch.no_grad()
def eval_model(model: ResidualMoE, x: torch.Tensor, base: torch.Tensor, y: torch.Tensor, meta: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Tuple[float, Dict[str, float], torch.Tensor]:
    model.eval()
    residual, _bin = model(x)
    pred = (base + residual).clamp(145.0, 195.0)
    metrics = p17.metrics_np(
        y.detach().cpu().numpy().astype(np.float32),
        pred.detach().cpu().numpy().astype(np.float32),
        meta,
        short_cm=float(args.short_cm),
        tall_cm=float(args.tall_cm),
    )
    score = float(metrics["mae"]) + 0.16 * float(metrics.get("short_mae", metrics["mae"])) + 0.08 * float(metrics.get("p90_ae", metrics["mae"]))
    return score, metrics, pred.detach()


def train_residual_model(
    train_x: torch.Tensor,
    train_base: torch.Tensor,
    train_y: torch.Tensor,
    train_meta: Sequence[Mapping[str, Any]],
    val_x: torch.Tensor,
    val_base: torch.Tensor,
    val_y: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    cfg: Config,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
    seed_everything(seed)
    device = train_x.device
    model = ResidualMoE(train_x.shape[1], int(cfg.hidden), int(cfg.experts), float(cfg.dropout), float(cfg.residual_scale)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(10, int(args.epochs)))
    weights = sample_weights(train_y, short_weight=float(cfg.short_weight), short_cm=float(args.short_cm), tall_cm=float(args.tall_cm))
    bins = bin_targets(train_y, float(args.short_cm), float(args.tall_cm))
    best_score = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Dict[str, float] = {}
    best_pred: Optional[torch.Tensor] = None
    bad = 0
    n = int(train_x.shape[0])
    batch_size = min(int(args.batch_size), n)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = train_x[idx]
            yb = train_y[idx]
            base_b = train_base[idx]
            wb = weights[idx]
            residual, logits = model(xb)
            pred = (base_b + residual).clamp(145.0, 195.0)
            err = pred - yb
            huber = torch.sqrt(err * err + 1.0) - 1.0
            reg_loss = (huber * wb).sum() / wb.sum().clamp_min(1e-6)
            bin_loss = F.cross_entropy(logits, bins[idx], weight=torch.tensor([1.8, 1.0, 1.25], dtype=torch.float32, device=device))
            rank = ranking_loss(pred, yb)
            base_anchor = 0.008 * torch.mean(residual * residual)
            loss = reg_loss + float(cfg.bin_weight) * bin_loss + float(cfg.rank_weight) * rank + base_anchor
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        scheduler.step()
        if epoch < 10 or epoch % 5 == 0:
            score, metrics, pred_val = eval_model(model, val_x, val_base, val_y, val_meta, args)
            if score < best_score - 1e-4:
                best_score = score
                best_metrics = metrics
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_pred = pred_val.detach().clone()
                bad = 0
            else:
                bad += 1
            if bad >= int(args.patience):
                break
    if best_state is None or best_pred is None:
        raise RuntimeError("ResidualMoE failed to produce a validation checkpoint.")
    return best_pred, {"score": best_score, "metrics": best_metrics, "epochs": epoch}, best_state


def support_for_fold(data: Mapping[str, Any], train_idx: np.ndarray, cfg: Config, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, List[Mapping[str, Any]]]:
    device = data["dev_x"].device
    train_index = torch.tensor(train_idx, dtype=torch.long, device=device)
    x = data["dev_x"][train_index]
    y = data["dev_y"][train_index]
    meta = take_meta(data["dev_meta"], train_idx)
    if float(cfg.support_weight) > 0.0 and data["support_x"].shape[0] > 0:
        support_n = int(data["support_x"].shape[0])
        weight = max(0.0, min(1.0, float(cfg.support_weight)))
        keep = max(1, int(round(support_n * weight)))
        # Deterministic spread across CELEB/height support rather than random sampling.
        idx = torch.linspace(0, support_n - 1, steps=keep, device=device).round().long().unique()
        x = torch.cat([x, data["support_x"][idx]], dim=0)
        y = torch.cat([y, data["support_y"][idx]], dim=0)
        meta = meta + [data["support_meta"][int(i)] for i in idx.detach().cpu().tolist()]
    return x, y, meta


def run_config(cfg: Config, data: Mapping[str, Any], folds: Sequence[Tuple[np.ndarray, np.ndarray, str]], args: argparse.Namespace) -> Dict[str, Any]:
    device = data["dev_x"].device
    dev_y = data["dev_y"]
    oof_sum = torch.zeros_like(dev_y)
    oof_count = torch.zeros_like(dev_y)
    fold_reports: List[Dict[str, Any]] = []
    for fold_no, (train_idx, val_idx, fold_name) in enumerate(folds, start=1):
        train_support_x, train_support_y, train_support_meta = support_for_fold(data, train_idx, cfg, args)
        train_target_index = torch.tensor(train_idx, dtype=torch.long, device=device)
        val_index = torch.tensor(val_idx, dtype=torch.long, device=device)
        train_target_x = data["dev_x"][train_target_index]
        train_target_y = data["dev_y"][train_target_index]
        train_target_meta = take_meta(data["dev_meta"], train_idx)
        val_x_raw = data["dev_x"][val_index]
        val_y = data["dev_y"][val_index]
        val_meta = take_meta(data["dev_meta"], val_idx)
        base_train = knn_base(
            train_target_x,
            train_target_y,
            train_target_meta,
            train_target_x,
            train_target_meta,
            k=int(cfg.k),
            temp=float(cfg.temp),
            short_boost=float(cfg.short_boost),
            short_cm=float(args.short_cm),
            leave_one_out=True,
        )
        base_val = knn_base(
            train_support_x,
            train_support_y,
            train_support_meta,
            val_x_raw,
            val_meta,
            k=int(cfg.k),
            temp=float(cfg.temp),
            short_boost=float(cfg.short_boost),
            short_cm=float(args.short_cm),
            leave_one_out=False,
        )
        x_train, x_val = projected_features(
            train_target_x,
            val_x_raw,
            train_target_meta,
            val_meta,
            dim=int(cfg.dim),
            seed=int(args.seed) + int(cfg.proj_seed_offset) + 101 * fold_no,
        )
        base_train_feature = ((base_train - 170.0) / 18.0).unsqueeze(1)
        base_val_feature = ((base_val - 170.0) / 18.0).unsqueeze(1)
        x_train = torch.cat([x_train, base_train_feature], dim=1)
        x_val = torch.cat([x_val, base_val_feature], dim=1)
        pred_val, fold_info, _state = train_residual_model(
            x_train,
            base_train,
            train_target_y,
            train_target_meta,
            x_val,
            base_val,
            val_y,
            val_meta,
            cfg,
            args,
            seed=int(args.seed) + int(cfg.proj_seed_offset) + 17 * fold_no,
        )
        oof_sum[val_index] += pred_val.to(device)
        oof_count[val_index] += 1.0
        fold_reports.append({"fold": fold_name, **fold_info})
    if torch.any(oof_count <= 0):
        raise RuntimeError(f"OOF coverage failed for {cfg.name}")
    oof_pred = (oof_sum / oof_count.clamp_min(1.0)).clamp(145.0, 195.0)

    full_support_x, full_support_y, full_support_meta = support_for_fold(data, np.arange(len(data["dev_meta"]), dtype=np.int64), cfg, args)
    base_full = knn_base(
        data["dev_x"],
        data["dev_y"],
        data["dev_meta"],
        data["dev_x"],
        data["dev_meta"],
        k=int(cfg.k),
        temp=float(cfg.temp),
        short_boost=float(cfg.short_boost),
        short_cm=float(args.short_cm),
        leave_one_out=True,
    )
    base_test = knn_base(
        full_support_x,
        full_support_y,
        full_support_meta,
        data["test_x"],
        data["test_meta"],
        k=int(cfg.k),
        temp=float(cfg.temp),
        short_boost=float(cfg.short_boost),
        short_cm=float(args.short_cm),
        leave_one_out=False,
    )
    x_full, x_test = projected_features(
        data["dev_x"],
        data["test_x"],
        data["dev_meta"],
        data["test_meta"],
        dim=int(cfg.dim),
        seed=int(args.seed) + int(cfg.proj_seed_offset),
    )
    x_full = torch.cat([x_full, ((base_full - 170.0) / 18.0).unsqueeze(1)], dim=1)
    x_test = torch.cat([x_test, ((base_test - 170.0) / 18.0).unsqueeze(1)], dim=1)
    test_pred, full_info, state = train_residual_model(
        x_full,
        base_full,
        data["dev_y"],
        data["dev_meta"],
        x_test,
        base_test,
        data["test_y"],
        data["test_meta"],
        cfg,
        args,
        seed=int(args.seed) + int(cfg.proj_seed_offset) + 911,
    )
    oof_np = clip_pred(oof_pred)
    test_np = clip_pred(test_pred)
    dev_y_np = data["dev_y"].detach().cpu().numpy().astype(np.float32)
    test_y_np = data["test_y"].detach().cpu().numpy().astype(np.float32)
    oof_m = p17.metrics_np(dev_y_np, oof_np, data["dev_meta"], short_cm=float(args.short_cm), tall_cm=float(args.tall_cm))
    test_m = p17.metrics_np(test_y_np, test_np, data["test_meta"], short_cm=float(args.short_cm), tall_cm=float(args.tall_cm))
    return {
        "name": cfg.name,
        "family": "cuda_residual_moe",
        "config": cfg.__dict__,
        "oof_pred": oof_np,
        "test_pred": test_np,
        "oof": oof_m,
        "test": test_m,
        "fold_reports": fold_reports,
        "full_train_info": full_info,
        "state_dict": state,
        "groups": 8,
    }


def configs(seed: int) -> List[Config]:
    return [
        Config("moe_balanced_d384", 384, 384, 4, 0.18, 8e-4, 3e-3, 2.6, 0.00, 45, 0.055, 2.8, 0.055, 0.12, 8.0, 0),
        Config("moe_short_d384", 384, 384, 4, 0.20, 7e-4, 4e-3, 4.0, 0.00, 75, 0.080, 4.0, 0.075, 0.16, 10.0, 19),
        Config("moe_support_d384", 384, 384, 4, 0.20, 7e-4, 4e-3, 3.2, 0.20, 45, 0.055, 3.4, 0.065, 0.14, 9.0, 37),
        Config("moe_wide_d768", 768, 512, 5, 0.22, 6e-4, 5e-3, 3.4, 0.08, 75, 0.080, 3.4, 0.065, 0.16, 9.0, 53),
        Config("moe_lowvar_d256", 256, 320, 3, 0.16, 9e-4, 3e-3, 2.8, 0.00, 45, 0.080, 2.8, 0.045, 0.10, 7.0, 71),
        Config("moe_short_smooth_d512", 512, 448, 4, 0.24, 5e-4, 6e-3, 4.6, 0.12, 75, 0.120, 4.2, 0.080, 0.18, 11.0, 89),
    ]


def selection_score(row: Mapping[str, Any], complexity_penalty: float) -> float:
    m = row["oof"]
    mae = float(m.get("mae", 999.0))
    short = float(m.get("short_mae", mae))
    medium = float(m.get("medium_mae", mae))
    tall = float(m.get("tall_mae", mae))
    p90 = float(m.get("p90_ae", mae))
    short_bias = abs(float(m.get("short_bias", m.get("bias", 0.0))))
    bin_mean = (short + medium + tall) / 3.0
    return 0.48 * mae + 0.22 * bin_mean + 0.16 * short + 0.10 * p90 + 0.04 * short_bias + float(complexity_penalty) * float(row.get("groups", 1))


def short_score(row: Mapping[str, Any]) -> float:
    m = row["oof"]
    mae = float(m.get("mae", 999.0))
    short = float(m.get("short_mae", mae))
    p90 = float(m.get("p90_ae", mae))
    short_bias = abs(float(m.get("short_bias", m.get("bias", 0.0))))
    return short + 0.22 * mae + 0.05 * p90 + 0.08 * short_bias + 1.8 * max(0.0, mae - 6.35)


def write_predictions(path: Path, y: np.ndarray, pred: np.ndarray, meta: Sequence[Mapping[str, Any]], column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["speaker_id", "source", "gender", "height_cm", column, "abs_error_cm"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(meta):
            true = float(y[idx])
            value = float(pred[idx])
            writer.writerow(
                {
                    "speaker_id": sid(row),
                    "source": str(row.get("source", "")),
                    "gender": int(row.get("gender", 0)),
                    "height_cm": f"{true:.6f}",
                    column: f"{value:.6f}",
                    "abs_error_cm": f"{abs(value - true):.6f}",
                }
            )


def public_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"oof_pred", "test_pred", "state_dict"}}


def write_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    selected = report["selected"]
    short_primary = report["short_primary"]
    lines = [
        "# Phase 19 CUDA Residual MoE Bagger Report",
        "",
        "## Result",
        f"- Selected method: `{selected['name']}`",
        f"- Development OOF MAE: `{selected['oof']['mae']:.3f}cm`",
        f"- Development OOF short MAE: `{selected['oof'].get('short_mae', float('nan')):.3f}cm`",
        f"- Test MAE: `{selected['test']['mae']:.3f}cm`",
        f"- Test short MAE: `{selected['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Test within 3cm: `{selected['test'].get('within_3cm', float('nan')):.3f}`",
        "",
        "## Short-Primary Candidate",
        f"- Method: `{short_primary['name']}`",
        f"- Development OOF MAE: `{short_primary['oof']['mae']:.3f}cm`",
        f"- Development OOF short MAE: `{short_primary['oof'].get('short_mae', float('nan')):.3f}cm`",
        f"- Test MAE: `{short_primary['test']['mae']:.3f}cm`",
        f"- Test short MAE: `{short_primary['test'].get('short_mae', float('nan')):.3f}cm`",
        "",
        "## References",
    ]
    for row in report["references"][:10]:
        lines.append(
            f"- `{row['name']}`: test `{row['test']['mae']:.3f}cm`, short `{row['test'].get('short_mae', float('nan')):.3f}cm`"
        )
    lines.extend(["", "## Top Candidates"])
    for row in report["top_candidates"]:
        lines.append(
            f"- `{row['name']}`: oof `{row['oof']['mae']:.3f}cm`, oof_short `{row['oof'].get('short_mae', float('nan')):.3f}cm`, "
            f"test `{row['test']['mae']:.3f}cm`, test_short `{row['test'].get('short_mae', float('nan')):.3f}cm`, score `{row['selection_score']:.3f}`"
        )
    lines.extend(["", "## Data Counts"])
    for key, value in report["counts"].items():
        lines.append(f"- {key}: `{value}`")
    output_dir.joinpath("PHASE19_CUDA_MOE_RESIDUAL_BAGGER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Phase19 is CUDA-only. Refusing CPU.")
    seed_everything(int(args.seed))
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase19] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    payload = load_payload(resolve(args.speaker_cache), device)
    data = make_data(payload)
    folds = make_folds(data["dev_meta"], data["dev_y"], folds=int(args.folds), repeats=int(args.repeats), seed=int(args.seed), short_cm=float(args.short_cm), tall_cm=float(args.tall_cm))
    dev_y_np = data["dev_y"].detach().cpu().numpy().astype(np.float32)
    test_y_np = data["test_y"].detach().cpu().numpy().astype(np.float32)
    cfgs = configs(int(args.seed))
    if int(args.max_configs) > 0:
        cfgs = cfgs[: int(args.max_configs)]
    print(
        f"[phase19] dev={len(dev_y_np)} dev_short={int((dev_y_np < float(args.short_cm)).sum())} "
        f"support={len(data['support_meta'])} test={len(test_y_np)} configs={len(cfgs)} folds={len(folds)}",
        flush=True,
    )
    results: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(cfgs, start=1):
        print(f"[phase19] config {idx}/{len(cfgs)} {cfg.name}", flush=True)
        try:
            row = run_config(cfg, data, folds, args)
            row["selection_score"] = selection_score(row, float(args.complexity_penalty))
            row["short_score"] = short_score(row)
            results.append(row)
            print(
                f"[phase19] {cfg.name}: oof={row['oof']['mae']:.3f} short={row['oof'].get('short_mae', float('nan')):.3f} "
                f"test={row['test']['mae']:.3f} test_short={row['test'].get('short_mae', float('nan')):.3f}",
                flush=True,
            )
        except RuntimeError as exc:
            print(f"[phase19] config failed {cfg.name}: {exc}", flush=True)
            torch.cuda.empty_cache()
    if not results:
        raise RuntimeError("No Phase19 configs completed.")
    results.sort(key=lambda row: float(row["selection_score"]))
    selected = results[0]
    short_primary = min(results, key=lambda row: float(row["short_score"]))
    references = p18.load_reference_rows(payload, short_cm=float(args.short_cm), tall_cm=float(args.tall_cm))
    phase18_report = resolve("outputs/phase18_oof_short_rescue/phase18_report.json")
    if phase18_report.exists():
        ref = json.loads(phase18_report.read_text(encoding="utf-8"))
        references.insert(0, {"name": "phase18_short_primary", "val": {}, "test": ref["short_primary"]["test"]})
        references.insert(0, {"name": "phase18_balanced", "val": {}, "test": ref["selected"]["test"]})
    write_predictions(output_dir / "phase19_predictions_oof_dev.csv", dev_y_np, np.asarray(selected["oof_pred"], dtype=np.float32), data["dev_meta"], "phase19_pred_cm")
    write_predictions(output_dir / "phase19_predictions_test.csv", test_y_np, np.asarray(selected["test_pred"], dtype=np.float32), data["test_meta"], "phase19_pred_cm")
    write_predictions(output_dir / "phase19_short_primary_predictions_test.csv", test_y_np, np.asarray(short_primary["test_pred"], dtype=np.float32), data["test_meta"], "phase19_short_primary_pred_cm")
    report = {
        "phase": "phase19_cuda_moe_residual_bagger",
        "selected": public_row(selected),
        "short_primary": public_row(short_primary),
        "references": references,
        "top_candidates": [public_row(row) for row in results[:20]],
        "counts": {
            "dev": int(len(dev_y_np)),
            "dev_short": int((dev_y_np < float(args.short_cm)).sum()),
            "support": int(len(data["support_meta"])),
            "test": int(len(test_y_np)),
            "test_short": int((test_y_np < float(args.short_cm)).sum()),
            "support_source_counts": dict(Counter(str(row.get("source", "UNKNOWN")) for row in data["support_meta"])),
        },
        "args": vars(args),
    }
    (output_dir / "phase19_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_report(output_dir, report)
    torch.save(
        {
            "selected": {k: v for k, v in selected.items() if k in {"name", "config", "oof", "test", "selection_score"}},
            "short_primary": {k: v for k, v in short_primary.items() if k in {"name", "config", "oof", "test", "short_score"}},
        },
        output_dir / "phase19_selected_summary.pt",
    )
    print(
        f"[phase19] selected={selected['name']} test_mae={selected['test']['mae']:.3f} "
        f"test_short={selected['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        f"[phase19] short_primary={short_primary['name']} test_mae={short_primary['test']['mae']:.3f} "
        f"test_short={short_primary['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(f"[phase19] wrote {output_dir / 'PHASE19_CUDA_MOE_RESIDUAL_BAGGER_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
