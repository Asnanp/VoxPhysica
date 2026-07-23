#!/usr/bin/env python
"""Phase 17 short-tail rescue.

This is a validation-selected, short-speaker-focused layer. It does not retrain
the audio model. Instead it uses the speaker cache, prior prediction files, a
GPU KNN short detector, and gated tail corrections to attack the failure slice
that keeps showing up: speakers below 160cm.

Selection is done on validation labels only. Test labels are reported once for
diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PREDICTION_SOURCES: Tuple[Tuple[str, str, str, str], ...] = (
    ("phase16_stack", "outputs/phase16_topk_ml_gauntlet/phase4_predictions_val.csv", "outputs/phase16_topk_ml_gauntlet/phase4_predictions_test.csv", "phase4_pred_cm"),
    ("topk_epoch_ensemble", "outputs/epoch_ensemble_gpu/topk_ensemble_eval/checkpoint_ensemble_predictions_val.csv", "outputs/epoch_ensemble_gpu/topk_ensemble_eval/checkpoint_ensemble_predictions_test.csv", "ensemble_pred_cm"),
    ("best_epoch_ensemble", "outputs/epoch_ensemble_gpu/ensemble_eval/checkpoint_ensemble_predictions_val.csv", "outputs/epoch_ensemble_gpu/ensemble_eval/checkpoint_ensemble_predictions_test.csv", "ensemble_pred_cm"),
    ("phase3_final", "outputs/phase3_target_domain_rescue/phase3_predictions_val.csv", "outputs/phase3_target_domain_rescue/phase3_predictions_test.csv", "final_pred_cm"),
    ("phase2_blend", "outputs/phase2_data_forensics/phase2_predictions_val.csv", "outputs/phase2_data_forensics/phase2_predictions_test.csv", "phase2_pred_cm"),
    ("phase2_knn", "outputs/phase2_data_forensics/phase2_predictions_val.csv", "outputs/phase2_data_forensics/phase2_predictions_test.csv", "knn_pred_cm"),
    ("combo", "outputs/speaker_gpu_combo_full_ssl_cuda/predictions_val.csv", "outputs/speaker_gpu_combo_full_ssl_cuda/predictions_test.csv", "pred_cm"),
    ("target", "outputs/speaker_gpu_target_ssl_cuda/predictions_val.csv", "outputs/speaker_gpu_target_ssl_cuda/predictions_test.csv", "pred_cm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short-tail rescue layer.")
    parser.add_argument("--speaker-cache", default="outputs/speaker_gpu_combo_full_ssl_cuda/speaker_gpu_cache.pt")
    parser.add_argument("--output-dir", default="outputs/phase17_short_tail_rescue")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--short-cm", type=float, default=160.0)
    parser.add_argument("--tall-cm", type=float, default=175.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=5000)
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
    value = str(row.get("source", "")).upper()
    if value == "TIMIT":
        return 0
    if value == "NISP":
        return 1
    if value in {"CELEB", "VOXCELEB"}:
        return 2
    return 3


def target_source(row: Mapping[str, Any]) -> bool:
    return str(row.get("source", "")).upper() in {"NISP", "TIMIT"}


def load_payload(path: Path, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    for split in ("train", "val", "test"):
        payload[split]["x"] = payload[split]["x"].to(device).float()
        payload[split]["y"] = payload[split]["y"].to(device).float()
    return payload


def read_prediction_csv(path: Path, column: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = str(row.get("speaker_id", "")).strip()
            value = row.get(column)
            if not key or value is None or value == "":
                continue
            out[key] = float(value)
    return out


def align_prediction(preds: Mapping[str, float], meta: Sequence[Mapping[str, Any]]) -> Optional[np.ndarray]:
    values: List[float] = []
    for row in meta:
        key = sid(row)
        if key not in preds:
            return None
        values.append(float(preds[key]))
    return np.asarray(values, dtype=np.float32)


def load_prediction_candidates(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    val_meta = payload["val"]["metadata"]
    test_meta = payload["test"]["metadata"]
    candidates: List[Dict[str, Any]] = []
    for name, val_path, test_path, column in PREDICTION_SOURCES:
        vp = resolve(val_path)
        tp = resolve(test_path)
        if not vp.exists() or not tp.exists():
            continue
        val = align_prediction(read_prediction_csv(vp, column), val_meta)
        test = align_prediction(read_prediction_csv(tp, column), test_meta)
        if val is None or test is None:
            continue
        candidates.append(
            {
                "name": name,
                "family": "previous_prediction",
                "val_pred": val,
                "test_pred": test,
                "groups": 1,
                "params": {"column": column},
            }
        )
    return candidates


def height_bin(y: np.ndarray, short_cm: float, tall_cm: float) -> np.ndarray:
    return np.where(y < float(short_cm), 0, np.where(y < float(tall_cm), 1, 2))


def metrics_np(
    y_true: np.ndarray,
    pred: np.ndarray,
    meta: Sequence[Mapping[str, Any]],
    *,
    short_cm: float,
    tall_cm: float,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.float32)
    p = np.asarray(pred, dtype=np.float32)
    err = p - y
    ae = np.abs(err)
    out: Dict[str, float] = {
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90.0)),
        "bias": float(np.mean(err)),
        "within_3cm": float(np.mean(ae <= 3.0)),
        "within_5cm": float(np.mean(ae <= 5.0)),
        "count": float(len(y)),
    }
    bins = height_bin(y, float(short_cm), float(tall_cm))
    for label, idx in (("short", 0), ("medium", 1), ("tall", 2)):
        mask = bins == idx
        if mask.any():
            out[f"{label}_mae"] = float(np.mean(ae[mask]))
            out[f"{label}_bias"] = float(np.mean(err[mask]))
            out[f"{label}_n"] = float(mask.sum())
    for source in sorted({str(row.get("source", "UNKNOWN")).upper() for row in meta}):
        mask = np.asarray([str(row.get("source", "UNKNOWN")).upper() == source for row in meta], dtype=bool)
        if mask.any():
            out[f"source_{source.lower()}_mae"] = float(np.mean(ae[mask]))
    return out


def selection_score(metrics: Mapping[str, float], groups: int) -> float:
    mae = float(metrics.get("mae", 999.0))
    short = float(metrics.get("short_mae", mae))
    medium = float(metrics.get("medium_mae", mae))
    tall = float(metrics.get("tall_mae", mae))
    p90 = float(metrics.get("p90_ae", mae))
    short_bias = abs(float(metrics.get("short_bias", metrics.get("bias", 0.0))))
    # This intentionally cares more about short speakers than global MAE.
    return (
        0.50 * short
        + 0.20 * mae
        + 0.12 * ((short + medium + tall) / 3.0)
        + 0.10 * p90
        + 0.05 * short_bias
        + 0.003 * float(groups)
    )


def robust_standardize(train_x: torch.Tensor, *others: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    center = torch.quantile(train_x, 0.50, dim=0)
    q25 = torch.quantile(train_x, 0.25, dim=0)
    q75 = torch.quantile(train_x, 0.75, dim=0)
    scale = (q75 - q25).clamp_min(1e-3)
    return tuple(torch.nan_to_num((x - center) / scale, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0) for x in (train_x, *others))


def meta_tensor(meta: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    rows = []
    for row in meta:
        src = source_id(row)
        gender = float(row.get("gender", 0))
        n_clips = math.log1p(float(row.get("n_clips", 0)))
        rows.append([gender, n_clips, *[1.0 if src == idx else 0.0 for idx in range(4)]])
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    return (x - x.mean(dim=0)) / x.std(dim=0, unbiased=False).clamp_min(1e-3)


def random_project(x: torch.Tensor, meta: torch.Tensor, dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    proj = torch.randn((x.shape[1], int(dim)), dtype=torch.float32, device=x.device, generator=gen) / math.sqrt(float(dim))
    z = x @ proj
    return torch.cat([z, meta], dim=1).clamp(-8.0, 8.0)


def ridge_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    weights: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    ones_train = torch.ones((train_x.shape[0], 1), dtype=torch.float32, device=train_x.device)
    ones_query = torch.ones((query_x.shape[0], 1), dtype=torch.float32, device=query_x.device)
    x = torch.cat([ones_train, train_x], dim=1)
    q = torch.cat([ones_query, query_x], dim=1)
    w = weights.float().clamp_min(1e-4)
    y_mean = (train_y * w).sum() / w.sum().clamp_min(1e-6)
    yc = train_y - y_mean
    sqrt_w = torch.sqrt(w).unsqueeze(1)
    xw = x * sqrt_w
    yw = yc * sqrt_w.squeeze(1)
    eye = torch.eye(x.shape[1], dtype=torch.float32, device=x.device)
    eye[0, 0] = 0.0
    beta = torch.linalg.solve(xw.T @ xw + float(lam) * eye, xw.T @ yw)
    return q @ beta + y_mean


def knn_height_or_prob(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_meta: Sequence[Mapping[str, Any]],
    query_x: torch.Tensor,
    query_meta: Sequence[Mapping[str, Any]],
    *,
    k: int,
    temperature: float,
    short_boost: float,
    same_source_boost: float,
    same_gender_boost: float,
    short_cm: float,
    return_prob: bool,
) -> torch.Tensor:
    tx = F.normalize(train_x, dim=1)
    qx = F.normalize(query_x, dim=1)
    sim = qx @ tx.T
    top_sim, top_idx = torch.topk(sim, k=min(int(k), train_x.shape[0]), dim=1)
    weights = torch.softmax(top_sim / float(temperature), dim=1)
    train_sources = torch.tensor([source_id(row) for row in train_meta], dtype=torch.long, device=train_x.device)
    query_sources = torch.tensor([source_id(row) for row in query_meta], dtype=torch.long, device=train_x.device).unsqueeze(1)
    train_genders = torch.tensor([int(row.get("gender", 0)) for row in train_meta], dtype=torch.long, device=train_x.device)
    query_genders = torch.tensor([int(row.get("gender", 0)) for row in query_meta], dtype=torch.long, device=train_x.device).unsqueeze(1)
    neighbor_y = train_y[top_idx]
    neighbor_short = (neighbor_y < float(short_cm)).float()
    weights = weights * torch.where(neighbor_short > 0, float(short_boost), 1.0)
    weights = weights * torch.where(train_sources[top_idx] == query_sources, float(same_source_boost), 1.0)
    weights = weights * torch.where(train_genders[top_idx] == query_genders, float(same_gender_boost), 1.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    if return_prob:
        return (neighbor_short * weights).sum(dim=1)
    return (neighbor_y * weights).sum(dim=1)


def append_gpu_support_candidates(
    candidates: List[Dict[str, Any]],
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    train_meta = list(payload["train"]["metadata"])
    val_meta = list(payload["val"]["metadata"])
    test_meta = list(payload["test"]["metadata"])
    train_x_raw, val_x_raw, test_x_raw = payload["train"]["x"], payload["val"]["x"], payload["test"]["x"]
    train_x, val_x, test_x = robust_standardize(train_x_raw, val_x_raw, test_x_raw)
    train_y = payload["train"]["y"]
    val_y = payload["val"]["y"].detach().cpu().numpy().astype(np.float32)
    target_mask = torch.tensor([target_source(row) for row in train_meta], dtype=torch.bool, device=device)
    target_meta = [row for row in train_meta if target_source(row)]
    if target_mask.sum().item() < 32:
        raise RuntimeError("Not enough target-domain speakers for short-tail support models.")
    train_x_t = train_x[target_mask]
    train_y_t = train_y[target_mask]

    prob_val_best: Optional[torch.Tensor] = None
    prob_test_best: Optional[torch.Tensor] = None
    best_prob_score = float("inf")
    for k in (5, 9, 15, 25, 45, 75, 115):
        for temp in (0.035, 0.055, 0.08, 0.12):
            for short_boost in (1.0, 1.5, 2.2, 3.0):
                pv = knn_height_or_prob(
                    train_x_t,
                    train_y_t,
                    target_meta,
                    val_x,
                    val_meta,
                    k=k,
                    temperature=temp,
                    short_boost=short_boost,
                    same_source_boost=1.20,
                    same_gender_boost=1.10,
                    short_cm=float(args.short_cm),
                    return_prob=True,
                )
                pt = knn_height_or_prob(
                    train_x_t,
                    train_y_t,
                    target_meta,
                    test_x,
                    test_meta,
                    k=k,
                    temperature=temp,
                    short_boost=short_boost,
                    same_source_boost=1.20,
                    same_gender_boost=1.10,
                    short_cm=float(args.short_cm),
                    return_prob=True,
                )
                pred_bin = (pv.detach().cpu().numpy() >= 0.35).astype(np.float32)
                truth_bin = (val_y < float(args.short_cm)).astype(np.float32)
                brier = float(np.mean((pred_bin - truth_bin) ** 2))
                if brier < best_prob_score:
                    best_prob_score = brier
                    prob_val_best = pv.detach().clone()
                    prob_test_best = pt.detach().clone()

    assert prob_val_best is not None and prob_test_best is not None
    prob_val = prob_val_best.detach().cpu().numpy().astype(np.float32)
    prob_test = prob_test_best.detach().cpu().numpy().astype(np.float32)

    for k in (5, 9, 15, 25, 45, 75):
        for temp in (0.035, 0.055, 0.08, 0.12):
            for short_boost in (1.0, 1.6, 2.4, 3.4):
                pv = knn_height_or_prob(
                    train_x_t,
                    train_y_t,
                    target_meta,
                    val_x,
                    val_meta,
                    k=k,
                    temperature=temp,
                    short_boost=short_boost,
                    same_source_boost=1.20,
                    same_gender_boost=1.10,
                    short_cm=float(args.short_cm),
                    return_prob=False,
                ).detach().cpu().numpy().astype(np.float32)
                pt = knn_height_or_prob(
                    train_x_t,
                    train_y_t,
                    target_meta,
                    test_x,
                    test_meta,
                    k=k,
                    temperature=temp,
                    short_boost=short_boost,
                    same_source_boost=1.20,
                    same_gender_boost=1.10,
                    short_cm=float(args.short_cm),
                    return_prob=False,
                ).detach().cpu().numpy().astype(np.float32)
                candidates.append(
                    {
                        "name": f"target_knn_height_k{k}_t{temp:g}_sb{short_boost:g}",
                        "family": "target_knn_height",
                        "val_pred": pv,
                        "test_pred": pt,
                        "groups": 1,
                        "params": {"k": k, "temperature": temp, "short_boost": short_boost},
                    }
                )

    meta_train = meta_tensor(target_meta, device)
    meta_val = meta_tensor(val_meta, device)
    meta_test = meta_tensor(test_meta, device)
    for dim in (192, 384, 768):
        for seed in (int(args.seed), int(args.seed) + 23):
            xtr = random_project(train_x_t, meta_train, dim, seed)
            xv = random_project(val_x, meta_val, dim, seed)
            xt = random_project(test_x, meta_test, dim, seed)
            for lam in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
                for short_weight in (1.0, 1.8, 2.8, 4.0):
                    weights = torch.where(train_y_t < float(args.short_cm), torch.tensor(float(short_weight), device=device), torch.tensor(1.0, device=device))
                    pv = ridge_predict(xtr, train_y_t, xv, weights, lam).detach().cpu().numpy().astype(np.float32)
                    pt = ridge_predict(xtr, train_y_t, xt, weights, lam).detach().cpu().numpy().astype(np.float32)
                    candidates.append(
                        {
                            "name": f"short_weighted_ridge_d{dim}_l{lam:g}_sw{short_weight:g}",
                            "family": "short_weighted_ridge",
                            "val_pred": pv,
                            "test_pred": pt,
                            "groups": 1,
                            "params": {"dim": dim, "lambda": lam, "short_weight": short_weight},
                        }
                    )

    return prob_val, prob_test


def sigmoid_gate(pred: np.ndarray, cutoff: float, temp: float) -> np.ndarray:
    z = (float(cutoff) - np.asarray(pred, dtype=np.float32)) / max(float(temp), 1e-3)
    return (1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))).astype(np.float32)


def append_gated_tail_candidates(
    candidates: List[Dict[str, Any]],
    bases: Sequence[Mapping[str, Any]],
    prob_val: np.ndarray,
    prob_test: np.ndarray,
    args: argparse.Namespace,
) -> None:
    for base in bases:
        val_base = np.asarray(base["val_pred"], dtype=np.float32)
        test_base = np.asarray(base["test_pred"], dtype=np.float32)
        for cutoff in (158.0, 160.0, 162.0, 164.0, 166.0):
            for temp in (2.0, 4.0, 6.0):
                gv_pred = sigmoid_gate(val_base, cutoff, temp)
                gt_pred = sigmoid_gate(test_base, cutoff, temp)
                gate_specs = [
                    ("predgate", gv_pred, gt_pred),
                    ("probgate", prob_val, prob_test),
                    ("maxgate", np.maximum(gv_pred, prob_val), np.maximum(gt_pred, prob_test)),
                    ("prodgate", gv_pred * prob_val, gt_pred * prob_test),
                ]
                for gate_name, gv, gt in gate_specs:
                    for delta in (0.75, 1.25, 1.75, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0):
                        candidates.append(
                            {
                                "name": f"{base['name']}__lower_{gate_name}_c{cutoff:g}_t{temp:g}_d{delta:g}",
                                "family": "gated_short_lowering",
                                "val_pred": (val_base - float(delta) * gv).astype(np.float32),
                                "test_pred": (test_base - float(delta) * gt).astype(np.float32),
                                "groups": 4,
                                "params": {"base": base["name"], "gate": gate_name, "cutoff": cutoff, "temp": temp, "delta": delta},
                            }
                        )

    knn_bases = [c for c in candidates if c.get("family") == "target_knn_height"]
    previous = [c for c in bases if c.get("family") == "previous_prediction"]
    for base in previous[:6]:
        val_base = np.asarray(base["val_pred"], dtype=np.float32)
        test_base = np.asarray(base["test_pred"], dtype=np.float32)
        for knn in knn_bases[:24]:
            val_knn = np.asarray(knn["val_pred"], dtype=np.float32)
            test_knn = np.asarray(knn["test_pred"], dtype=np.float32)
            for gate_scale in (0.20, 0.35, 0.50, 0.70):
                gv = np.clip(prob_val * float(gate_scale), 0.0, 0.85)
                gt = np.clip(prob_test * float(gate_scale), 0.0, 0.85)
                candidates.append(
                    {
                        "name": f"{base['name']}__shortblend_{knn['name']}_g{gate_scale:g}",
                        "family": "prob_gated_knn_blend",
                        "val_pred": ((1.0 - gv) * val_base + gv * val_knn).astype(np.float32),
                        "test_pred": ((1.0 - gt) * test_base + gt * test_knn).astype(np.float32),
                        "groups": 3,
                        "params": {"base": base["name"], "knn": knn["name"], "gate_scale": gate_scale},
                    }
                )


def ranked_rows(
    candidates: Sequence[Mapping[str, Any]],
    val_y: np.ndarray,
    test_y: np.ndarray,
    val_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        val_m = metrics_np(val_y, np.asarray(cand["val_pred"], dtype=np.float32), val_meta, short_cm=float(args.short_cm), tall_cm=float(args.tall_cm))
        test_m = metrics_np(test_y, np.asarray(cand["test_pred"], dtype=np.float32), test_meta, short_cm=float(args.short_cm), tall_cm=float(args.tall_cm))
        row = {k: v for k, v in cand.items() if k not in {"val_pred", "test_pred"}}
        row["val"] = val_m
        row["test"] = test_m
        row["selection_score"] = selection_score(val_m, int(row.get("groups", 1)))
        rows.append(row)
    rows.sort(key=lambda item: float(item["selection_score"]))
    return rows


def write_predictions(
    path: Path,
    y: np.ndarray,
    pred: np.ndarray,
    meta: Sequence[Mapping[str, Any]],
    prob: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["speaker_id", "source", "gender", "height_cm", "phase17_pred_cm", "phase17_abs_error_cm", "short_probability"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(meta):
            value = float(pred[idx])
            true = float(y[idx])
            writer.writerow(
                {
                    "speaker_id": sid(row),
                    "source": str(row.get("source", "")),
                    "gender": int(row.get("gender", 0)),
                    "height_cm": f"{true:.6f}",
                    "phase17_pred_cm": f"{value:.6f}",
                    "phase17_abs_error_cm": f"{abs(value - true):.6f}",
                    "short_probability": f"{float(prob[idx]):.6f}",
                }
            )


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    selected = report["selected"]
    best_short = report["best_short_diagnostic"]
    lines = [
        "# Phase 17 Short-Tail Rescue Report",
        "",
        "## Result",
        f"- Selected method: `{selected['name']}`",
        f"- Validation short MAE: `{selected['val'].get('short_mae', float('nan')):.3f}cm`",
        f"- Test MAE: `{selected['test']['mae']:.3f}cm`",
        f"- Test short MAE: `{selected['test'].get('short_mae', float('nan')):.3f}cm`",
        f"- Test short bias: `{selected['test'].get('short_bias', float('nan')):.3f}cm`",
        f"- Within 5cm: `{selected['test'].get('within_5cm', float('nan')):.3f}`",
        "",
        "## Best Short Diagnostic",
        f"- Method: `{best_short['name']}`",
        f"- Test MAE: `{best_short['test']['mae']:.3f}cm`",
        f"- Test short MAE: `{best_short['test'].get('short_mae', float('nan')):.3f}cm`",
        "- Diagnostic means selected by test short MAE; not deployable.",
        "",
        "## Top Validation Short-Tail Candidates",
    ]
    for row in report["top_by_selection"][:25]:
        lines.append(
            f"- `{row['name']}` ({row['family']}): val_short `{row['val'].get('short_mae', float('nan')):.3f}`, "
            f"val_mae `{row['val']['mae']:.3f}`, test_short `{row['test'].get('short_mae', float('nan')):.3f}`, "
            f"test_mae `{row['test']['mae']:.3f}`, score `{row['selection_score']:.3f}`"
        )
    lines.extend(["", "## Top Test Short Diagnostic"])
    for row in report["top_by_test_short"][:15]:
        lines.append(
            f"- `{row['name']}` ({row['family']}): test_short `{row['test'].get('short_mae', float('nan')):.3f}`, "
            f"test_mae `{row['test']['mae']:.3f}`, val_short `{row['val'].get('short_mae', float('nan')):.3f}`"
        )
    lines.extend(["", "## Data Counts"])
    lines.append(f"- Val short speakers: `{int(report['counts']['val_short'])}`")
    lines.append(f"- Test short speakers: `{int(report['counts']['test_short'])}`")
    lines.append(f"- Candidates searched: `{int(report['candidate_count'])}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Phase 17. Refusing CPU.")
    seed_everything(int(args.seed))
    device = torch.device("cuda")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase17] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    payload = load_payload(resolve(args.speaker_cache), device)
    val_y = payload["val"]["y"].detach().cpu().numpy().astype(np.float32)
    test_y = payload["test"]["y"].detach().cpu().numpy().astype(np.float32)
    val_meta = list(payload["val"]["metadata"])
    test_meta = list(payload["test"]["metadata"])

    candidates = load_prediction_candidates(payload)
    print(f"[phase17] loaded {len(candidates)} prediction bases", flush=True)
    prob_val, prob_test = append_gpu_support_candidates(candidates, payload, args, device)
    bases = list(candidates)
    append_gated_tail_candidates(candidates, bases, prob_val, prob_test, args)
    if len(candidates) > int(args.max_candidates):
        # Cheap pre-filter by validation metrics before the more expensive report JSON.
        rows_prefilter = ranked_rows(candidates, val_y, test_y, val_meta, test_meta, args)
        keep_names = {str(row["name"]) for row in rows_prefilter[: int(args.max_candidates)]}
        candidates = [cand for cand in candidates if str(cand["name"]) in keep_names]
    print(f"[phase17] candidates={len(candidates)}", flush=True)

    rows = ranked_rows(candidates, val_y, test_y, val_meta, test_meta, args)
    selected = rows[0]
    selected_cand = next(c for c in candidates if c["name"] == selected["name"])
    best_short = min(rows, key=lambda row: float(row["test"].get("short_mae", 999.0)))
    top_by_test_short = sorted(rows, key=lambda row: float(row["test"].get("short_mae", 999.0)))
    top_by_test_mae = sorted(rows, key=lambda row: float(row["test"].get("mae", 999.0)))

    selected_val = np.asarray(selected_cand["val_pred"], dtype=np.float32)
    selected_test = np.asarray(selected_cand["test_pred"], dtype=np.float32)
    write_predictions(output_dir / "phase17_predictions_val.csv", val_y, selected_val, val_meta, prob_val)
    write_predictions(output_dir / "phase17_predictions_test.csv", test_y, selected_test, test_meta, prob_test)

    public_rows = [
        {k: v for k, v in row.items() if k not in {"params"}}
        for row in rows
    ]
    report = {
        "phase": "phase17_short_tail_rescue",
        "selected": selected,
        "best_short_diagnostic": best_short,
        "top_by_selection": public_rows[:100],
        "top_by_test_short": top_by_test_short[:50],
        "top_by_test_mae": top_by_test_mae[:50],
        "candidate_count": len(candidates),
        "counts": {
            "val": len(val_y),
            "test": len(test_y),
            "val_short": int((val_y < float(args.short_cm)).sum()),
            "test_short": int((test_y < float(args.short_cm)).sum()),
            "train_source_counts": dict(Counter(str(row.get("source", "UNKNOWN")) for row in payload["train"]["metadata"])),
        },
        "args": vars(args),
    }
    (output_dir / "phase17_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_report(output_dir / "PHASE17_SHORT_TAIL_REPORT.md", report)
    print(
        f"[phase17] selected={selected['name']} val_short={selected['val'].get('short_mae', float('nan')):.3f} "
        f"test_mae={selected['test']['mae']:.3f} test_short={selected['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        f"[phase17] diagnostic_best_short={best_short['name']} test_short={best_short['test'].get('short_mae', float('nan')):.3f}",
        flush=True,
    )
    print(f"[phase17] wrote {output_dir / 'PHASE17_SHORT_TAIL_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
