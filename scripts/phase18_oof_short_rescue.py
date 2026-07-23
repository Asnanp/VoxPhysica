#!/usr/bin/env python
"""Phase 18 robust OOF short-tail rescue.

Phase 17 proved that selecting on the 10 validation short speakers is too
fragile: a candidate can get excellent val-short MAE and fail on sealed
test-short speakers. Phase 18 moves selection to repeated out-of-fold target
domain scoring over train+val target speakers, then reports test once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase17_short_tail_rescue as p17  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust OOF short-tail rescue.")
    parser.add_argument("--speaker-cache", default="outputs/speaker_gpu_combo_full_ssl_cuda/speaker_gpu_cache.pt")
    parser.add_argument("--output-dir", default="outputs/phase18_oof_short_rescue")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--short-cm", type=float, default=160.0)
    parser.add_argument("--tall-cm", type=float, default=175.0)
    parser.add_argument("--top-base-for-gates", type=int, default=80)
    parser.add_argument("--complexity-penalty", type=float, default=0.004)
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
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def sid(row: Mapping[str, Any]) -> str:
    return str(row.get("speaker_id", "")).strip()


def source_id(row: Mapping[str, Any]) -> int:
    return p17.source_id(row)


def target_source(row: Mapping[str, Any]) -> bool:
    return p17.target_source(row)


def height_bin_value(height: float, short_cm: float, tall_cm: float) -> int:
    if float(height) < float(short_cm):
        return 0
    if float(height) < float(tall_cm):
        return 1
    return 2


def clip_pred(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), 145.0, 195.0).astype(np.float32)


def load_payload(path: Path, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    for split in ("train", "val", "test"):
        payload[split]["x"] = payload[split]["x"].to(device).float()
        payload[split]["y"] = payload[split]["y"].to(device).float()
    return payload


def make_dev_splits(payload: Mapping[str, Any]) -> Dict[str, Any]:
    train_meta = list(payload["train"]["metadata"])
    target_idx = torch.tensor([target_source(row) for row in train_meta], dtype=torch.bool, device=payload["train"]["x"].device)
    support_idx = ~target_idx
    dev_x = torch.cat([payload["train"]["x"][target_idx], payload["val"]["x"]], dim=0)
    dev_y = torch.cat([payload["train"]["y"][target_idx], payload["val"]["y"]], dim=0)
    dev_meta = [row for row in train_meta if target_source(row)] + list(payload["val"]["metadata"])
    support_meta = [row for row in train_meta if not target_source(row)]
    return {
        "dev_x": dev_x,
        "dev_y": dev_y,
        "dev_meta": dev_meta,
        "support_x": payload["train"]["x"][support_idx],
        "support_y": payload["train"]["y"][support_idx],
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
        key = (
            source_id(row),
            int(row.get("gender", 0)),
            height_bin_value(float(y_np[idx]), float(short_cm), float(tall_cm)),
        )
        buckets[key].append(idx)
    all_indices = set(range(len(meta)))
    out: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for rep in range(int(repeats)):
        rng = random.Random(int(seed) + 7919 * rep)
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


def meta_raw(meta: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
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


def standardize_meta(train_meta: Sequence[Mapping[str, Any]], query_meta: Sequence[Mapping[str, Any]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    train = meta_raw(train_meta, device)
    query = meta_raw(query_meta, device)
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
    device = train_x.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    proj = torch.randn((train_z.shape[1], int(dim)), dtype=torch.float32, device=device, generator=gen) / math.sqrt(float(dim))
    train_proj = train_z @ proj
    query_proj = query_z @ proj
    meta_train, meta_query = standardize_meta(train_meta, query_meta, device)
    return torch.cat([train_proj, meta_train], dim=1).clamp(-8.0, 8.0), torch.cat([query_proj, meta_query], dim=1).clamp(-8.0, 8.0)


def sample_weights(meta: Sequence[Mapping[str, Any]], *, support_weight: float, short_boost: float, short_cm: float, device: torch.device) -> torch.Tensor:
    vals = []
    for row in meta:
        src = str(row.get("source", "")).upper()
        w = float(support_weight) if src in {"CELEB", "VOXCELEB", "HEIGHTCELEB"} else 1.0
        if float(row.get("height_cm", 0.0)) < float(short_cm):
            w *= float(short_boost)
        vals.append(w)
    weights = torch.tensor(vals, dtype=torch.float32, device=device).clamp_min(1e-4)
    return weights / weights.mean().clamp_min(1e-6)


def take_meta(meta: Sequence[Mapping[str, Any]], indices: Iterable[int]) -> List[Mapping[str, Any]]:
    return [meta[int(i)] for i in indices]


def cv_ridge_candidate(
    *,
    name: str,
    data: Mapping[str, Any],
    folds: Sequence[Tuple[np.ndarray, np.ndarray, str]],
    dim: int,
    proj_seed: int,
    lam: float,
    short_boost: float,
    support_weight: float,
    short_cm: float,
) -> Dict[str, Any]:
    device = data["dev_x"].device
    dev_x = data["dev_x"]
    dev_y = data["dev_y"]
    dev_meta = data["dev_meta"]
    support_x = data["support_x"]
    support_y = data["support_y"]
    support_meta = data["support_meta"]
    use_support = support_x.shape[0] > 0 and float(support_weight) > 0.0
    oof_sum = torch.zeros_like(dev_y)
    oof_count = torch.zeros_like(dev_y)
    for train_idx, val_idx, _fold_name in folds:
        train_index = torch.tensor(train_idx, dtype=torch.long, device=device)
        val_index = torch.tensor(val_idx, dtype=torch.long, device=device)
        x_train = dev_x[train_index]
        y_train = dev_y[train_index]
        meta_train = take_meta(dev_meta, train_idx)
        if use_support:
            x_train = torch.cat([x_train, support_x], dim=0)
            y_train = torch.cat([y_train, support_y], dim=0)
            meta_train = meta_train + list(support_meta)
        xtr, xv = projected_features(x_train, dev_x[val_index], meta_train, take_meta(dev_meta, val_idx), dim=dim, seed=proj_seed)
        weights = sample_weights(meta_train, support_weight=support_weight, short_boost=short_boost, short_cm=short_cm, device=device)
        pred = p17.ridge_predict(xtr, y_train, xv, weights, lam)
        oof_sum[val_index] += pred
        oof_count[val_index] += 1.0
    if torch.any(oof_count <= 0):
        raise RuntimeError(f"OOF coverage failed for {name}")
    oof = (oof_sum / oof_count.clamp_min(1.0)).detach().cpu().numpy().astype(np.float32)

    full_x = dev_x
    full_y = dev_y
    full_meta = list(dev_meta)
    if use_support:
        full_x = torch.cat([full_x, support_x], dim=0)
        full_y = torch.cat([full_y, support_y], dim=0)
        full_meta = full_meta + list(support_meta)
    xfull, xtest = projected_features(full_x, data["test_x"], full_meta, data["test_meta"], dim=dim, seed=proj_seed)
    weights = sample_weights(full_meta, support_weight=support_weight, short_boost=short_boost, short_cm=short_cm, device=device)
    test = p17.ridge_predict(xfull, full_y, xtest, weights, lam).detach().cpu().numpy().astype(np.float32)
    return {
        "name": name,
        "family": "oof_short_weighted_ridge",
        "oof_pred": clip_pred(oof),
        "test_pred": clip_pred(test),
        "groups": 1,
        "params": {
            "dim": int(dim),
            "proj_seed": int(proj_seed),
            "lambda": float(lam),
            "short_boost": float(short_boost),
            "support_weight": float(support_weight),
        },
    }


def knn_predict(
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
    return_prob: bool,
) -> torch.Tensor:
    train_z, query_z = p17.robust_standardize(train_x, query_x)
    return p17.knn_height_or_prob(
        train_z,
        train_y,
        train_meta,
        query_z,
        query_meta,
        k=int(k),
        temperature=float(temp),
        short_boost=float(short_boost),
        same_source_boost=1.25,
        same_gender_boost=1.10,
        short_cm=float(short_cm),
        return_prob=bool(return_prob),
    )


def cv_knn_candidate(
    *,
    name: str,
    data: Mapping[str, Any],
    folds: Sequence[Tuple[np.ndarray, np.ndarray, str]],
    k: int,
    temp: float,
    short_boost: float,
    include_support: bool,
    short_cm: float,
    return_prob: bool,
) -> Dict[str, Any]:
    device = data["dev_x"].device
    dev_x = data["dev_x"]
    dev_y = data["dev_y"]
    dev_meta = data["dev_meta"]
    support_x = data["support_x"]
    support_y = data["support_y"]
    support_meta = data["support_meta"]
    oof_sum = torch.zeros_like(dev_y)
    oof_count = torch.zeros_like(dev_y)
    for train_idx, val_idx, _fold_name in folds:
        train_index = torch.tensor(train_idx, dtype=torch.long, device=device)
        val_index = torch.tensor(val_idx, dtype=torch.long, device=device)
        x_train = dev_x[train_index]
        y_train = dev_y[train_index]
        meta_train = take_meta(dev_meta, train_idx)
        if include_support and support_x.shape[0] > 0:
            x_train = torch.cat([x_train, support_x], dim=0)
            y_train = torch.cat([y_train, support_y], dim=0)
            meta_train = meta_train + list(support_meta)
        pred = knn_predict(
            x_train,
            y_train,
            meta_train,
            dev_x[val_index],
            take_meta(dev_meta, val_idx),
            k=k,
            temp=temp,
            short_boost=short_boost,
            short_cm=short_cm,
            return_prob=return_prob,
        )
        oof_sum[val_index] += pred
        oof_count[val_index] += 1.0
    if torch.any(oof_count <= 0):
        raise RuntimeError(f"OOF coverage failed for {name}")
    oof = (oof_sum / oof_count.clamp_min(1.0)).detach().cpu().numpy().astype(np.float32)

    full_x = dev_x
    full_y = dev_y
    full_meta = list(dev_meta)
    if include_support and support_x.shape[0] > 0:
        full_x = torch.cat([full_x, support_x], dim=0)
        full_y = torch.cat([full_y, support_y], dim=0)
        full_meta = full_meta + list(support_meta)
    test = knn_predict(
        full_x,
        full_y,
        full_meta,
        data["test_x"],
        data["test_meta"],
        k=k,
        temp=temp,
        short_boost=short_boost,
        short_cm=short_cm,
        return_prob=return_prob,
    ).detach().cpu().numpy().astype(np.float32)
    return {
        "name": name,
        "family": "oof_knn_short_prob" if return_prob else "oof_knn_height",
        "oof_pred": oof.astype(np.float32) if return_prob else clip_pred(oof),
        "test_pred": test.astype(np.float32) if return_prob else clip_pred(test),
        "groups": 1,
        "params": {
            "k": int(k),
            "temperature": float(temp),
            "short_boost": float(short_boost),
            "include_support": bool(include_support),
            "return_prob": bool(return_prob),
        },
    }


def selection_score(metrics: Mapping[str, float], groups: int, complexity_penalty: float) -> float:
    mae = float(metrics.get("mae", 999.0))
    short = float(metrics.get("short_mae", mae))
    medium = float(metrics.get("medium_mae", mae))
    tall = float(metrics.get("tall_mae", mae))
    p90 = float(metrics.get("p90_ae", mae))
    short_bias = abs(float(metrics.get("short_bias", metrics.get("bias", 0.0))))
    bias = abs(float(metrics.get("bias", 0.0)))
    bin_mean = (short + medium + tall) / 3.0
    tall_guard = max(0.0, tall - (mae + 1.6))
    global_guard = max(0.0, mae - 6.2)
    return (
        0.36 * mae
        + 0.30 * short
        + 0.16 * bin_mean
        + 0.10 * p90
        + 0.04 * short_bias
        + 0.02 * bias
        + 0.08 * tall_guard
        + 0.10 * global_guard
        + float(complexity_penalty) * float(groups)
    )


def short_primary_score(row: Mapping[str, Any]) -> float:
    name = str(row.get("name", ""))
    metrics = row["oof"]
    mae = float(metrics.get("mae", 999.0))
    short = float(metrics.get("short_mae", mae))
    tall = float(metrics.get("tall_mae", mae))
    p90 = float(metrics.get("p90_ae", mae))
    short_bias = abs(float(metrics.get("short_bias", metrics.get("bias", 0.0))))
    global_guard = max(0.0, mae - 6.25)
    tall_guard = max(0.0, tall - 7.0)
    stability_penalty = 0.0
    for marker, penalty in (("knn_height_k5_", 0.60), ("knn_height_k9_", 0.50), ("knn_height_k15_", 0.38), ("knn_height_k25_", 0.26)):
        if marker in name:
            stability_penalty += penalty
            break
    if "_t2_" in name or "_t2_d" in name:
        stability_penalty += 0.28
    return (
        short
        + 0.18 * mae
        + 0.05 * p90
        + 0.08 * short_bias
        + 2.0 * global_guard
        + 0.60 * tall_guard
        + stability_penalty
        + 0.002 * float(row.get("groups", 1))
    )


def candidate_rows(
    candidates: Sequence[Mapping[str, Any]],
    dev_y: np.ndarray,
    test_y: np.ndarray,
    dev_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    *,
    short_cm: float,
    tall_cm: float,
    complexity_penalty: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        oof_m = p17.metrics_np(dev_y, np.asarray(cand["oof_pred"], dtype=np.float32), dev_meta, short_cm=short_cm, tall_cm=tall_cm)
        test_m = p17.metrics_np(test_y, np.asarray(cand["test_pred"], dtype=np.float32), test_meta, short_cm=short_cm, tall_cm=tall_cm)
        row = {k: v for k, v in cand.items() if k not in {"oof_pred", "test_pred"}}
        row["oof"] = oof_m
        row["test"] = test_m
        row["selection_score"] = selection_score(oof_m, int(row.get("groups", 1)), float(complexity_penalty))
        rows.append(row)
    rows.sort(key=lambda item: float(item["selection_score"]))
    return rows


def sigmoid_gate(pred: np.ndarray, cutoff: float, temp: float) -> np.ndarray:
    z = (float(cutoff) - np.asarray(pred, dtype=np.float32)) / max(float(temp), 1e-3)
    return (1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))).astype(np.float32)


def add_gated_candidates(
    candidates: List[Dict[str, Any]],
    base_candidates: Sequence[Mapping[str, Any]],
    prob_oof: np.ndarray,
    prob_test: np.ndarray,
) -> None:
    for base in base_candidates:
        oof_base = np.asarray(base["oof_pred"], dtype=np.float32)
        test_base = np.asarray(base["test_pred"], dtype=np.float32)
        for cutoff in (158.0, 160.0, 162.0, 164.0, 166.0):
            for temp in (2.0, 4.0, 6.0):
                pred_gate_oof = sigmoid_gate(oof_base, cutoff, temp)
                pred_gate_test = sigmoid_gate(test_base, cutoff, temp)
                gate_specs = (
                    ("predgate", pred_gate_oof, pred_gate_test),
                    ("probgate", prob_oof, prob_test),
                    ("maxgate", np.maximum(pred_gate_oof, prob_oof), np.maximum(pred_gate_test, prob_test)),
                    ("prodgate", pred_gate_oof * prob_oof, pred_gate_test * prob_test),
                )
                for gate_name, goof, gtest in gate_specs:
                    for delta in (0.75, 1.25, 1.75, 2.5, 3.5, 4.5, 6.0, 8.0):
                        candidates.append(
                            {
                                "name": f"{base['name']}__lower_{gate_name}_c{cutoff:g}_t{temp:g}_d{delta:g}",
                                "family": "oof_gated_short_lowering",
                                "oof_pred": clip_pred(oof_base - float(delta) * goof),
                                "test_pred": clip_pred(test_base - float(delta) * gtest),
                                "groups": int(base.get("groups", 1)) + 2,
                                "params": {
                                    "base": str(base["name"]),
                                    "cutoff": float(cutoff),
                                    "temperature": float(temp),
                                    "gate": gate_name,
                                    "delta": float(delta),
                                },
                            }
                        )


def load_reference_rows(
    payload: Mapping[str, Any],
    *,
    short_cm: float,
    tall_cm: float,
) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    val_y = payload["val"]["y"].detach().cpu().numpy().astype(np.float32)
    test_y = payload["test"]["y"].detach().cpu().numpy().astype(np.float32)
    val_meta = list(payload["val"]["metadata"])
    test_meta = list(payload["test"]["metadata"])
    for cand in p17.load_prediction_candidates(payload):
        val_m = p17.metrics_np(val_y, np.asarray(cand["val_pred"], dtype=np.float32), val_meta, short_cm=short_cm, tall_cm=tall_cm)
        test_m = p17.metrics_np(test_y, np.asarray(cand["test_pred"], dtype=np.float32), test_meta, short_cm=short_cm, tall_cm=tall_cm)
        refs.append({"name": cand["name"], "val": val_m, "test": test_m})
    refs.sort(key=lambda row: float(row["test"]["mae"]))
    return refs


def write_predictions(path: Path, y: np.ndarray, pred: np.ndarray, meta: Sequence[Mapping[str, Any]], prob: Optional[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["speaker_id", "source", "gender", "height_cm", "phase18_pred_cm", "phase18_abs_error_cm"]
    if prob is not None:
        fields.append("oof_short_probability")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(meta):
            value = float(pred[idx])
            true = float(y[idx])
            item = {
                "speaker_id": sid(row),
                "source": str(row.get("source", "")),
                "gender": int(row.get("gender", 0)),
                "height_cm": f"{true:.6f}",
                "phase18_pred_cm": f"{value:.6f}",
                "phase18_abs_error_cm": f"{abs(value - true):.6f}",
            }
            if prob is not None:
                item["oof_short_probability"] = f"{float(prob[idx]):.6f}"
            writer.writerow(item)


def public_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"params"}}


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    selected = report["selected"]
    short_primary = report["short_primary"]
    best_short = report["best_test_short_diagnostic"]
    lines = [
        "# Phase 18 OOF Short-Tail Rescue Report",
        "",
        "## Result",
        f"- Selected method: `{selected['name']}`",
        f"- Development OOF MAE: `{selected['oof']['mae']:.3f}cm`",
        f"- Development OOF short MAE: `{selected['oof'].get('short_mae', float('nan')):.3f}cm`",
        f"- Test MAE: `{selected['test']['mae']:.3f}cm`",
        f"- Test short MAE: `{selected['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Test short bias: `{selected['test'].get('short_bias', float('nan')):.3f}cm`",
        f"- Within 5cm: `{selected['test'].get('within_5cm', float('nan')):.3f}`",
        "",
        "## Short-Primary Deploy Candidate",
        f"- Method: `{short_primary['name']}`",
        f"- Development OOF MAE: `{short_primary['oof']['mae']:.3f}cm`",
        f"- Development OOF short MAE: `{short_primary['oof'].get('short_mae', float('nan')):.3f}cm`",
        f"- Test MAE: `{short_primary['test']['mae']:.3f}cm`",
        f"- Test short MAE: `{short_primary['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Test short bias: `{short_primary['test'].get('short_bias', float('nan')):.3f}cm`",
        "- Use this when short-speaker error matters more than the balanced global score.",
        "",
        "## Best Test Short Diagnostic",
        f"- Method: `{best_short['name']}`",
        f"- Test MAE: `{best_short['test']['mae']:.3f}cm`",
        f"- Test short MAE: `{best_short['test'].get('short_mae', float('nan')):.3f}cm`",
        "- Diagnostic means selected by sealed test short MAE; not deployable.",
        "",
        "## References",
    ]
    for row in report["references"][:8]:
        lines.append(
            f"- `{row['name']}`: test `{row['test']['mae']:.3f}cm`, short `{row['test'].get('short_mae', float('nan')):.3f}cm`, "
            f"val `{row['val']['mae']:.3f}cm`"
        )
    lines.extend(["", "## Top OOF Candidates"])
    for row in report["top_by_selection"][:30]:
        lines.append(
            f"- `{row['name']}` ({row['family']}): oof `{row['oof']['mae']:.3f}cm`, "
            f"oof_short `{row['oof'].get('short_mae', float('nan')):.3f}cm`, "
            f"test `{row['test']['mae']:.3f}cm`, test_short `{row['test'].get('short_mae', float('nan')):.3f}cm`, "
            f"score `{row['selection_score']:.3f}`"
        )
    lines.extend(["", "## Top Test Short Diagnostic"])
    for row in report["top_by_test_short"][:20]:
        lines.append(
            f"- `{row['name']}` ({row['family']}): test_short `{row['test'].get('short_mae', float('nan')):.3f}cm`, "
            f"test `{row['test']['mae']:.3f}cm`, oof_short `{row['oof'].get('short_mae', float('nan')):.3f}cm`"
        )
    lines.extend(["", "## Data Counts"])
    counts = report["counts"]
    for key in ("dev", "dev_short", "support", "test", "test_short"):
        lines.append(f"- {key}: `{int(counts[key])}`")
    lines.append(f"- Candidates searched: `{int(report['candidate_count'])}`")
    lines.append("")
    lines.append("## Read")
    lines.append(
        "This phase is stricter than Phase 17: selection is repeated out-of-fold on target-domain train+val speakers, not the tiny validation-short slice alone."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Phase18 is CUDA-only for this run. Refusing CPU.")
    seed_everything(int(args.seed))
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase18] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    payload = load_payload(resolve(args.speaker_cache), device)
    data = make_dev_splits(payload)
    folds = make_folds(
        data["dev_meta"],
        data["dev_y"],
        folds=int(args.folds),
        repeats=int(args.repeats),
        seed=int(args.seed),
        short_cm=float(args.short_cm),
        tall_cm=float(args.tall_cm),
    )
    dev_y_np = data["dev_y"].detach().cpu().numpy().astype(np.float32)
    test_y_np = data["test_y"].detach().cpu().numpy().astype(np.float32)
    print(
        f"[phase18] dev target speakers={len(dev_y_np)} support={len(data['support_meta'])} "
        f"test={len(test_y_np)} folds={len(folds)}",
        flush=True,
    )
    print(
        f"[phase18] dev short={(dev_y_np < float(args.short_cm)).sum()} test short={(test_y_np < float(args.short_cm)).sum()}",
        flush=True,
    )

    prob_candidates: List[Dict[str, Any]] = []
    for include_support in (False, True):
        for k in (5, 9, 15, 25, 45, 75):
            for temp in (0.035, 0.055, 0.08):
                for short_boost in (1.0, 2.0, 3.5):
                    name = f"prob_knn_k{k}_t{temp:g}_sb{short_boost:g}_{'support' if include_support else 'target'}"
                    prob_candidates.append(
                        cv_knn_candidate(
                            name=name,
                            data=data,
                            folds=folds,
                            k=k,
                            temp=temp,
                            short_boost=short_boost,
                            include_support=include_support,
                            short_cm=float(args.short_cm),
                            return_prob=True,
                        )
                    )
    truth_short = (dev_y_np < float(args.short_cm)).astype(np.float32)
    prob_ranked = sorted(
        prob_candidates,
        key=lambda cand: float(np.mean((np.asarray(cand["oof_pred"], dtype=np.float32) - truth_short) ** 2)),
    )
    prob_oof = np.asarray(prob_ranked[0]["oof_pred"], dtype=np.float32)
    prob_test = np.asarray(prob_ranked[0]["test_pred"], dtype=np.float32)
    print(f"[phase18] best short-prob gate={prob_ranked[0]['name']}", flush=True)

    candidates: List[Dict[str, Any]] = []
    for cand in prob_candidates:
        if cand["family"] == "oof_knn_short_prob":
            continue
    for include_support in (False, True):
        for k in (5, 9, 15, 25, 45, 75):
            for temp in (0.035, 0.055, 0.08):
                for short_boost in (1.0, 1.8, 2.8, 4.0):
                    name = f"knn_height_k{k}_t{temp:g}_sb{short_boost:g}_{'support' if include_support else 'target'}"
                    candidates.append(
                        cv_knn_candidate(
                            name=name,
                            data=data,
                            folds=folds,
                            k=k,
                            temp=temp,
                            short_boost=short_boost,
                            include_support=include_support,
                            short_cm=float(args.short_cm),
                            return_prob=False,
                        )
                    )
    print(f"[phase18] KNN height candidates={len(candidates)}", flush=True)

    dims = (96, 192, 384)
    lambdas = (30.0, 100.0, 300.0, 1000.0, 3000.0)
    short_boosts = (1.0, 1.8, 2.8, 4.2)
    support_weights = (0.0, 0.08, 0.20)
    seeds = (int(args.seed), int(args.seed) + 37)
    for dim in dims:
        for proj_seed in seeds:
            for lam in lambdas:
                for short_boost in short_boosts:
                    for support_weight in support_weights:
                        name = f"ridge_d{dim}_s{proj_seed}_l{lam:g}_sb{short_boost:g}_cw{support_weight:g}"
                        candidates.append(
                            cv_ridge_candidate(
                                name=name,
                                data=data,
                                folds=folds,
                                dim=dim,
                                proj_seed=proj_seed,
                                lam=lam,
                                short_boost=short_boost,
                                support_weight=support_weight,
                                short_cm=float(args.short_cm),
                            )
                        )
                        if len(candidates) % 120 == 0:
                            print(f"[phase18] base candidates={len(candidates)}", flush=True)

    rows = candidate_rows(
        candidates,
        dev_y_np,
        test_y_np,
        data["dev_meta"],
        data["test_meta"],
        short_cm=float(args.short_cm),
        tall_cm=float(args.tall_cm),
        complexity_penalty=float(args.complexity_penalty),
    )
    print(
        f"[phase18] best base oof={rows[0]['oof']['mae']:.3f} short={rows[0]['oof'].get('short_mae', float('nan')):.3f} "
        f"test={rows[0]['test']['mae']:.3f}",
        flush=True,
    )
    by_name = {str(c["name"]): c for c in candidates}
    top_bases = [by_name[str(row["name"])] for row in rows[: int(args.top_base_for_gates)]]
    add_gated_candidates(candidates, top_bases, prob_oof, prob_test)
    print(f"[phase18] total candidates after gates={len(candidates)}", flush=True)

    rows = candidate_rows(
        candidates,
        dev_y_np,
        test_y_np,
        data["dev_meta"],
        data["test_meta"],
        short_cm=float(args.short_cm),
        tall_cm=float(args.tall_cm),
        complexity_penalty=float(args.complexity_penalty),
    )
    selected = rows[0]
    selected_cand = next(c for c in candidates if str(c["name"]) == str(selected["name"]))
    short_primary = min(rows, key=short_primary_score)
    short_primary_cand = next(c for c in candidates if str(c["name"]) == str(short_primary["name"]))
    selected_oof = np.asarray(selected_cand["oof_pred"], dtype=np.float32)
    selected_test = np.asarray(selected_cand["test_pred"], dtype=np.float32)
    short_primary_oof = np.asarray(short_primary_cand["oof_pred"], dtype=np.float32)
    short_primary_test = np.asarray(short_primary_cand["test_pred"], dtype=np.float32)
    top_by_test_short = sorted(rows, key=lambda row: float(row["test"].get("short_mae", 999.0)))
    references = load_reference_rows(payload, short_cm=float(args.short_cm), tall_cm=float(args.tall_cm))

    write_predictions(output_dir / "phase18_predictions_oof_dev.csv", dev_y_np, selected_oof, data["dev_meta"], prob_oof)
    write_predictions(output_dir / "phase18_predictions_test.csv", test_y_np, selected_test, data["test_meta"], prob_test)
    write_predictions(output_dir / "phase18_short_primary_predictions_oof_dev.csv", dev_y_np, short_primary_oof, data["dev_meta"], prob_oof)
    write_predictions(output_dir / "phase18_short_primary_predictions_test.csv", test_y_np, short_primary_test, data["test_meta"], prob_test)

    report = {
        "phase": "phase18_oof_short_rescue",
        "selected": selected,
        "short_primary": short_primary,
        "best_test_short_diagnostic": top_by_test_short[0],
        "top_by_selection": [public_row(row) for row in rows[:160]],
        "top_by_test_short": [public_row(row) for row in top_by_test_short[:80]],
        "references": references,
        "candidate_count": len(candidates),
        "probability_gate": {
            "name": prob_ranked[0]["name"],
            "oof_brier": float(np.mean((prob_oof - truth_short) ** 2)),
        },
        "counts": {
            "dev": len(dev_y_np),
            "dev_short": int((dev_y_np < float(args.short_cm)).sum()),
            "support": len(data["support_meta"]),
            "test": len(test_y_np),
            "test_short": int((test_y_np < float(args.short_cm)).sum()),
            "support_source_counts": dict(Counter(str(row.get("source", "UNKNOWN")) for row in data["support_meta"])),
        },
        "args": vars(args),
    }
    (output_dir / "phase18_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_report(output_dir / "PHASE18_OOF_SHORT_TAIL_REPORT.md", report)
    print(
        f"[phase18] selected={selected['name']} oof_mae={selected['oof']['mae']:.3f} "
        f"oof_short={selected['oof'].get('short_mae', float('nan')):.3f} "
        f"test_mae={selected['test']['mae']:.3f} test_short={selected['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        f"[phase18] short_primary={short_primary['name']} oof_mae={short_primary['oof']['mae']:.3f} "
        f"oof_short={short_primary['oof'].get('short_mae', float('nan')):.3f} "
        f"test_mae={short_primary['test']['mae']:.3f} test_short={short_primary['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        f"[phase18] diagnostic_best_test_short={top_by_test_short[0]['name']} "
        f"test_short={top_by_test_short[0]['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(f"[phase18] wrote {output_dir / 'PHASE18_OOF_SHORT_TAIL_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
