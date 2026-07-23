#!/usr/bin/env python
"""Phase 22: 3cm reality gauntlet.

This phase is deliberately different from another architecture knob turn.
It asks three research questions:

1. Which historical prediction candidates are actually useful on the sealed
   speaker test set?
2. If a perfect selector could choose among all existing candidates per
   speaker, is 3cm even present in the current signal pool?
3. What validation-safe ensemble can be selected without looking at test
   labels, and which speakers still block the 3cm target?

The script keeps the deployable path and the diagnostic oracle path separate.
The oracle is not a model; it is a lower-bound audit that tells us whether the
current features/predictions contain enough information to justify deeper work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Candidate:
    name: str
    test_pred: np.ndarray
    val_pred: Optional[np.ndarray]
    source_path: str
    column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA-only 3cm prediction-pool gauntlet.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/phase22_3cm_reality_gauntlet")
    parser.add_argument("--phase3-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--blend-probes", type=int, default=120000)
    parser.add_argument("--blend-batch", type=int, default=4096)
    parser.add_argument("--target-mae", type=float, default=3.0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"speaker_id", "height_cm"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing required columns: {sorted(missing)}")
    keep = ["speaker_id", "height_cm"]
    for col in ("source", "gender", "age"):
        if col in df.columns:
            keep.append(col)
    return df[keep].copy()


def is_prediction_column(df: pd.DataFrame, col: str) -> bool:
    lower = col.lower()
    if col in {"height_cm", "gender", "age"}:
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    if any(token in lower for token in ("error", "mae", "rmse", "count", "probability", "std", "uncert")):
        return False
    return lower.endswith("_cm") or "pred" in lower


def align_prediction(base: pd.DataFrame, pred_df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
    if "speaker_id" not in pred_df.columns or col not in pred_df.columns:
        return None
    pred_small = pred_df[["speaker_id", col]].copy()
    if pred_small["speaker_id"].duplicated().any():
        pred_small = (
            pred_small.groupby("speaker_id", as_index=False)[col]
            .mean(numeric_only=True)
        )
    merged = base[["speaker_id"]].merge(pred_small, on="speaker_id", how="left")
    if merged[col].isna().any():
        return None
    values = merged[col].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        return None
    if float(np.max(np.abs(values))) > 300.0:
        return None
    return values


def candidate_val_paths(test_path: Path) -> List[Path]:
    names = []
    name = test_path.name
    replacements = [
        ("_predictions_test.csv", "_predictions_val.csv"),
        ("_predictions_test.csv", "_predictions_oof_dev.csv"),
        ("_test.csv", "_val.csv"),
        ("_test.csv", "_oof_dev.csv"),
        ("test_speaker_predictions.csv", "val_speaker_predictions.csv"),
        ("test_clip_predictions.csv", "val_clip_predictions.csv"),
        ("predictions_test.csv", "predictions_val.csv"),
        ("predictions_test.csv", "predictions_oof_dev.csv"),
    ]
    for old, new in replacements:
        if old in name:
            names.append(test_path.with_name(name.replace(old, new)))
    return list(dict.fromkeys(names))


def iter_prediction_csvs(outputs_root: Path, output_dir: Path) -> Iterable[Path]:
    skip_parts = {
        ".git",
        "__pycache__",
        "pytest-temp",
        output_dir.name.lower(),
    }
    for dirpath, dirnames, filenames in os.walk(outputs_root):
        dirnames[:] = [d for d in dirnames if d.lower() not in skip_parts]
        for filename in filenames:
            lower = filename.lower()
            if not lower.endswith(".csv"):
                continue
            if "prediction" not in lower or "test" not in lower:
                continue
            if "oracle" in lower:
                continue
            yield Path(dirpath) / filename


def load_candidates(outputs_root: Path, output_dir: Path, val_base: pd.DataFrame, test_base: pd.DataFrame) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen: set[bytes] = set()
    for test_path in iter_prediction_csvs(outputs_root, output_dir):
        try:
            test_df = pd.read_csv(test_path)
        except Exception:
            continue
        if "speaker_id" not in test_df.columns or "height_cm" not in test_df.columns:
            continue
        val_df: Optional[pd.DataFrame] = None
        for val_path in candidate_val_paths(test_path):
            if not val_path.exists():
                continue
            try:
                val_df = pd.read_csv(val_path)
                break
            except Exception:
                val_df = None
        for col in [c for c in test_df.columns if is_prediction_column(test_df, c)]:
            test_pred = align_prediction(test_base, test_df, col)
            if test_pred is None:
                continue
            val_pred = align_prediction(val_base, val_df, col) if val_df is not None and col in val_df.columns else None
            signature = np.round(test_pred, 4).tobytes()
            if signature in seen:
                continue
            seen.add(signature)
            rel = test_path.relative_to(outputs_root).as_posix()
            candidates.append(
                Candidate(
                    name=f"{rel}:{col}",
                    test_pred=test_pred,
                    val_pred=val_pred,
                    source_path=str(test_path),
                    column=col,
                )
            )
    return candidates


def split_masks(y: np.ndarray, meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    masks = {
        "short": y < 165.0,
        "medium": (y >= 165.0) & (y < 178.0),
        "tall": y >= 178.0,
    }
    if "source" in meta.columns:
        source = meta["source"].astype(str).str.upper().to_numpy()
        masks["source_nisp"] = source == "NISP"
        masks["source_timit"] = source == "TIMIT"
    if "gender" in meta.columns:
        gender = meta["gender"].to_numpy()
        masks["female"] = gender == 0
        masks["male"] = gender == 1
    return masks


def metrics(y: np.ndarray, pred: np.ndarray, meta: pd.DataFrame, masks: Dict[str, np.ndarray] = None) -> Dict[str, float]:
    err = np.asarray(pred, dtype=np.float32) - np.asarray(y, dtype=np.float32)
    ae = np.abs(err)
    out: Dict[str, float] = {
        "count": float(len(y)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "bias": float(np.mean(err)),
        "within_3cm": float(np.mean(ae <= 3.0)),
        "within_5cm": float(np.mean(ae <= 5.0)),
    }
    if masks is None:
        masks = split_masks(y, meta)
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        out[f"{name}_n"] = float(mask.sum())
        out[f"{name}_mae"] = float(np.mean(ae[mask]))
    return out


def selection_score(m: Mapping[str, float]) -> float:
    mae = float(m["mae"])
    p90 = float(m.get("p90_ae", mae))
    bias = abs(float(m.get("bias", 0.0)))
    short = float(m.get("short_mae", mae))
    # This is intentionally harsher than normal MAE because the short tail is
    # where every previous attempt bled error.
    return mae + 0.020 * p90 + 0.035 * bias + 0.120 * max(0.0, short - mae)


def candidate_rows(candidates: Sequence[Candidate], val_y: np.ndarray, test_y: np.ndarray, val_meta: pd.DataFrame, test_meta: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        test_m = metrics(test_y, cand.test_pred, test_meta)
        row: Dict[str, Any] = {
            "name": cand.name,
            "path": cand.source_path,
            "column": cand.column,
            "has_val": cand.val_pred is not None,
            "test": test_m,
        }
        if cand.val_pred is not None:
            val_m = metrics(val_y, cand.val_pred, val_meta)
            row["val"] = val_m
            row["score"] = selection_score(val_m)
        else:
            row["score"] = None
        rows.append(row)
    return rows


def torch_metrics(y: torch.Tensor, pred: torch.Tensor, short_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    err = pred - y
    ae = torch.abs(err)
    mae = ae.mean(dim=1)
    p90 = torch.quantile(ae, 0.90, dim=1)
    bias = torch.abs(err.mean(dim=1))
    if bool(short_mask.any()):
        short_mae = ae[:, short_mask].mean(dim=1)
    else:
        short_mae = mae
    return mae, p90, bias, short_mae


def gpu_convex_search(
    paired: Sequence[Candidate],
    val_y: np.ndarray,
    test_y: np.ndarray,
    val_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    device: torch.device,
    top_k: int,
    probes: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    scored = []
    for idx, cand in enumerate(paired):
        assert cand.val_pred is not None
        val_m = metrics(val_y, cand.val_pred, val_meta)
        scored.append((selection_score(val_m), idx))
    scored.sort(key=lambda item: item[0])
    take = [idx for _, idx in scored[: max(2, min(top_k, len(scored)))]]
    names = [paired[idx].name for idx in take]
    val_mat = torch.tensor(np.stack([paired[idx].val_pred for idx in take], axis=1), dtype=torch.float32, device=device)
    test_mat = torch.tensor(np.stack([paired[idx].test_pred for idx in take], axis=1), dtype=torch.float32, device=device)
    y_val = torch.tensor(val_y, dtype=torch.float32, device=device).view(1, -1)
    short_mask = torch.tensor((val_y < 165.0), dtype=torch.bool, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    best_score = float("inf")
    best_weights: Optional[torch.Tensor] = None
    best_kind = "random_convex"
    k = len(take)

    # Pairwise grid is low-variance and catches useful two-model switches.
    if k >= 2:
        grid = torch.linspace(0.0, 1.0, 101, device=device)
        for i in range(min(k, 24)):
            for j in range(i + 1, min(k, 24)):
                pred = grid[:, None] * val_mat[:, i].view(1, -1) + (1.0 - grid[:, None]) * val_mat[:, j].view(1, -1)
                mae, p90, bias, short_mae = torch_metrics(y_val, pred, short_mask)
                score = mae + 0.020 * p90 + 0.035 * bias + 0.120 * torch.clamp(short_mae - mae, min=0.0)
                value, arg = torch.min(score, dim=0)
                if float(value.item()) < best_score:
                    weights = torch.zeros(k, dtype=torch.float32, device=device)
                    w = float(grid[int(arg.item())].item())
                    weights[i] = w
                    weights[j] = 1.0 - w
                    best_score = float(value.item())
                    best_weights = weights
                    best_kind = "pairwise_convex"

    remaining = int(max(0, probes))
    while remaining > 0:
        b = min(int(batch_size), remaining)
        remaining -= b
        raw = torch.rand((b, k), device=device, generator=generator)
        # Power transform creates sparse-ish convex probes without changing the
        # optimization objective into an unrestricted stacker.
        weights = raw.pow(4.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pred = weights @ val_mat.T
        mae, p90, bias, short_mae = torch_metrics(y_val, pred, short_mask)
        score = mae + 0.020 * p90 + 0.035 * bias + 0.120 * torch.clamp(short_mae - mae, min=0.0)
        value, arg = torch.min(score, dim=0)
        if float(value.item()) < best_score:
            best_score = float(value.item())
            best_weights = weights[int(arg.item())].detach().clone()
            best_kind = "random_convex"

    if best_weights is None:
        raise RuntimeError("Convex search did not produce weights")
    test_pred = (best_weights @ test_mat.T).detach().cpu().numpy().astype(np.float32)
    val_pred = (best_weights @ val_mat.T).detach().cpu().numpy().astype(np.float32)
    nonzero = [
        {"name": names[idx], "weight": float(best_weights[idx].detach().cpu().item())}
        for idx in range(k)
        if float(best_weights[idx].detach().cpu().item()) > 1e-4
    ]
    nonzero.sort(key=lambda item: abs(float(item["weight"])), reverse=True)
    return {
        "name": f"phase22_{best_kind}",
        "kind": best_kind,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "weights": nonzero,
        "val": metrics(val_y, val_pred, val_meta),
        "test": metrics(test_y, test_pred, test_meta),
        "score": float(best_score),
    }


def gate_search(
    paired: Sequence[Candidate],
    val_y: np.ndarray,
    test_y: np.ndarray,
    val_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
) -> Dict[str, Any]:
    val_masks = split_masks(val_y, val_meta)
    test_masks = split_masks(test_y, test_meta)
    ranked = []
    for idx, cand in enumerate(paired):
        assert cand.val_pred is not None
        val_m = metrics(val_y, cand.val_pred, val_meta, val_masks)
        ranked.append((selection_score(val_m), idx))
    ranked.sort(key=lambda item: item[0])
    top = [idx for _, idx in ranked[: min(24, len(ranked))]]
    best: Optional[Dict[str, Any]] = None
    for base_idx in top:
        base = paired[base_idx]
        assert base.val_pred is not None
        for alt_idx in top:
            if alt_idx == base_idx:
                continue
            alt = paired[alt_idx]
            assert alt.val_pred is not None
            signals = {
                "base": base.val_pred,
                "alt": alt.val_pred,
                "mean": 0.5 * (base.val_pred + alt.val_pred),
                "min": np.minimum(base.val_pred, alt.val_pred),
                "spread": base.val_pred - alt.val_pred,
            }
            test_signals = {
                "base": base.test_pred,
                "alt": alt.test_pred,
                "mean": 0.5 * (base.test_pred + alt.test_pred),
                "min": np.minimum(base.test_pred, alt.test_pred),
                "spread": base.test_pred - alt.test_pred,
            }
            for signal_name, signal in signals.items():
                qs = np.unique(np.quantile(signal, np.linspace(0.05, 0.95, 37)))
                for threshold in qs:
                    for direction in ("lt", "gt"):
                        val_mask = signal < threshold if direction == "lt" else signal > threshold
                        n = int(val_mask.sum())
                        if n < 3 or n > len(val_mask) - 3:
                            continue
                        test_signal = test_signals[signal_name]
                        test_mask = test_signal < threshold if direction == "lt" else test_signal > threshold
                        val_pred = base.val_pred.copy()
                        test_pred = base.test_pred.copy()
                        val_pred[val_mask] = alt.val_pred[val_mask]
                        test_pred[test_mask] = alt.test_pred[test_mask]
                        val_m = metrics(val_y, val_pred, val_meta, val_masks)
                        test_m = metrics(test_y, test_pred, test_meta, test_masks)
                        score = selection_score(val_m) + 0.004 * n
                        if best is None or score < float(best["score"]):
                            best = {
                                "name": f"gate_{signal_name}_{direction}_{float(threshold):.3f}",
                                "kind": "validation_gate",
                                "base": base.name,
                                "alt": alt.name,
                                "signal": signal_name,
                                "direction": direction,
                                "threshold": float(threshold),
                                "val_switched": int(val_mask.sum()),
                                "test_switched": int(test_mask.sum()),
                                "val_pred": val_pred,
                                "test_pred": test_pred,
                                "val": val_m,
                                "test": test_m,
                                "score": float(score),
                            }
    if best is None:
        raise RuntimeError("Gate search produced no candidate")
    return best


def oracle_result(candidates: Sequence[Candidate], y: np.ndarray, meta: pd.DataFrame, require_val: bool = False) -> Dict[str, Any]:
    usable = [cand for cand in candidates if (cand.val_pred is not None or not require_val)]
    if not usable:
        raise RuntimeError("No usable candidates for oracle")
    pred_mat = np.stack([cand.test_pred for cand in usable], axis=1).astype(np.float32)
    err_mat = np.abs(pred_mat - y.reshape(-1, 1))
    index = np.argmin(err_mat, axis=1)
    pred = pred_mat[np.arange(len(y)), index]
    return {
        "pred": pred,
        "best_index": index,
        "candidate_names": [usable[int(i)].name for i in index],
        "candidate_pool_count": len(usable),
        "metrics": metrics(y, pred, meta),
    }


def error_budget(y: np.ndarray, pred: np.ndarray, target_mae: float) -> Dict[str, Any]:
    ae = np.abs(pred - y)
    total = float(ae.sum())
    target_total = float(target_mae * len(y))
    need = max(0.0, total - target_total)
    sorted_err = np.sort(ae)[::-1]
    cumulative = 0.0
    count = 0
    for value in sorted_err:
        if cumulative >= need:
            break
        cumulative += float(value)
        count += 1
    short_mask = y < 165.0
    short_perfect_total = total - float(ae[short_mask].sum())
    return {
        "total_abs_error_cm": total,
        "target_total_abs_error_cm": target_total,
        "needed_reduction_cm": need,
        "worst_speakers_if_perfect": int(count),
        "mae_if_short_perfect": float(short_perfect_total / len(y)),
    }


def write_prediction_csv(path: Path, base: pd.DataFrame, pred: np.ndarray, col: str, extras: Optional[Mapping[str, Sequence[Any]]] = None) -> None:
    fields = ["speaker_id"]
    for name in ("source", "gender"):
        if name in base.columns:
            fields.append(name)
    fields.extend(["height_cm", col, f"{col}_abs_error"])
    if extras:
        fields.extend(extras.keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        y = base["height_cm"].to_numpy(dtype=np.float32)
        for idx, row in base.iterrows():
            item: Dict[str, Any] = {
                "speaker_id": row["speaker_id"],
                "height_cm": f"{float(row['height_cm']):.6f}",
                col: f"{float(pred[idx]):.6f}",
                f"{col}_abs_error": f"{abs(float(pred[idx]) - float(y[idx])):.6f}",
            }
            for name in ("source", "gender"):
                if name in base.columns:
                    item[name] = row[name]
            if extras:
                for key, values in extras.items():
                    item[key] = values[idx]
            writer.writerow(item)


def blocker_rows(
    base: pd.DataFrame,
    selected_pred: np.ndarray,
    oracle: Mapping[str, Any],
    limit: int = 40,
) -> List[Dict[str, Any]]:
    y = base["height_cm"].to_numpy(dtype=np.float32)
    selected_err = np.abs(selected_pred - y)
    oracle_pred = np.asarray(oracle["pred"], dtype=np.float32)
    oracle_names = list(oracle["candidate_names"])
    order = np.argsort(selected_err)[::-1][:limit]
    rows = []
    for idx in order:
        row = base.iloc[int(idx)]
        item: Dict[str, Any] = {
            "speaker_id": row["speaker_id"],
            "height_cm": float(y[idx]),
            "selected_pred_cm": float(selected_pred[idx]),
            "selected_abs_error_cm": float(selected_err[idx]),
            "oracle_pred_cm": float(oracle_pred[idx]),
            "oracle_abs_error_cm": float(abs(float(oracle_pred[idx]) - float(y[idx]))),
            "oracle_candidate": oracle_names[int(idx)],
        }
        for name in ("source", "gender"):
            if name in base.columns:
                item[name] = row[name]
        rows.append(item)
    return rows


def write_blockers(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_metrics(m: Mapping[str, float]) -> str:
    parts = [
        f"mae `{m['mae']:.3f}cm`",
        f"short `{m.get('short_mae', float('nan')):.3f}cm`",
        f"p90 `{m.get('p90_ae', float('nan')):.3f}cm`",
        f"within3 `{100.0 * m.get('within_3cm', 0.0):.1f}%`",
    ]
    return ", ".join(parts)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value
    return value


def deploy_selection_score(item: Mapping[str, Any]) -> float:
    kind = str(item.get("kind", ""))
    score = float(item.get("score", selection_score(item["val"])))
    if kind == "validation_gate":
        score += 0.075
    elif "convex" in kind:
        score += 0.025
    return score


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Phase22 is CUDA-only. Start it with --device cuda on the RTX GPU.")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    outputs_root = resolve(args.outputs_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_base = read_base(resolve(args.phase3_val))
    test_base = read_base(resolve(args.phase3_test))
    val_y = val_base["height_cm"].to_numpy(dtype=np.float32)
    test_y = test_base["height_cm"].to_numpy(dtype=np.float32)

    print(f"[phase22] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("[phase22] crawling historical prediction candidates", flush=True)
    candidates = load_candidates(outputs_root, output_dir, val_base, test_base)
    if not candidates:
        raise RuntimeError("No prediction candidates found")
    paired = [cand for cand in candidates if cand.val_pred is not None]
    if len(paired) < 2:
        raise RuntimeError("Need at least two validation-paired candidates")
    print(f"[phase22] candidates: all={len(candidates)} validation_paired={len(paired)}", flush=True)

    rows = candidate_rows(candidates, val_y, test_y, val_base, test_base)
    paired_rows = [row for row in rows if row["has_val"]]
    paired_rows.sort(key=lambda item: float(item["score"]))
    all_rows_by_test = sorted(rows, key=lambda item: float(item["test"]["mae"]))

    print("[phase22] running CUDA convex super-learner", flush=True)
    convex = gpu_convex_search(
        paired,
        val_y,
        test_y,
        val_base,
        test_base,
        device,
        top_k=int(args.top_k),
        probes=int(args.blend_probes),
        batch_size=int(args.blend_batch),
        seed=int(args.seed),
    )
    print("[phase22] running validation-safe gate search", flush=True)
    gate = gate_search(paired, val_y, test_y, val_base, test_base)

    deploy_candidates = [
        {
            "name": paired_rows[0]["name"],
            "kind": "best_validation_individual",
            "test_pred": next(c.test_pred for c in paired if c.name == paired_rows[0]["name"]),
            "val": paired_rows[0]["val"],
            "test": paired_rows[0]["test"],
            "score": paired_rows[0]["score"],
        },
        convex,
        gate,
    ]
    deploy_candidates.sort(key=deploy_selection_score)
    selected = deploy_candidates[0]
    selected_pred = np.asarray(selected["test_pred"], dtype=np.float32)

    print("[phase22] computing oracle lower bounds", flush=True)
    global_oracle = oracle_result(candidates, test_y, test_base, require_val=False)
    paired_oracle = oracle_result(candidates, test_y, test_base, require_val=True)
    selected_budget = error_budget(test_y, selected_pred, float(args.target_mae))
    global_oracle_budget = error_budget(test_y, np.asarray(global_oracle["pred"], dtype=np.float32), float(args.target_mae))

    blockers = blocker_rows(test_base, selected_pred, global_oracle, limit=50)
    write_prediction_csv(output_dir / "phase22_predictions_test.csv", test_base, selected_pred, "phase22_pred_cm")
    write_prediction_csv(
        output_dir / "phase22_research_oracle_predictions_test.csv",
        test_base,
        np.asarray(global_oracle["pred"], dtype=np.float32),
        "phase22_oracle_pred_cm",
        extras={"oracle_candidate": global_oracle["candidate_names"]},
    )
    write_blockers(output_dir / "phase22_blockers_test.csv", blockers)

    report = {
        "selected": {
            "name": selected["name"],
            "kind": selected.get("kind", "unknown"),
            "val": selected["val"],
            "test": selected["test"],
            "score": selected["score"],
        },
        "convex": {k: v for k, v in convex.items() if k not in {"val_pred", "test_pred"}},
        "gate": {k: v for k, v in gate.items() if k not in {"val_pred", "test_pred"}},
        "global_oracle": {
            "metrics": global_oracle["metrics"],
            "candidate_pool_count": global_oracle["candidate_pool_count"],
            "budget": global_oracle_budget,
        },
        "paired_oracle": {
            "metrics": paired_oracle["metrics"],
            "candidate_pool_count": paired_oracle["candidate_pool_count"],
        },
        "selected_budget": selected_budget,
        "candidate_counts": {
            "all": len(candidates),
            "validation_paired": len(paired),
        },
        "top_validation_paired": paired_rows[:30],
        "top_test_candidates_diagnostic": all_rows_by_test[:30],
        "blockers": blockers,
        "args": vars(args),
    }
    (output_dir / "phase22_report.json").write_text(json.dumps(json_ready(report), indent=2, allow_nan=True), encoding="utf-8")

    lines = [
        "# Phase 22 3cm Reality Gauntlet Report",
        "",
        "## Result",
        f"- Selected validation-safe method: `{selected['name']}` ({selected.get('kind', 'unknown')})",
        f"- Selected validation: {format_metrics(selected['val'])}",
        f"- Selected sealed test: {format_metrics(selected['test'])}",
        f"- Global research oracle: {format_metrics(global_oracle['metrics'])}",
        f"- Validation-paired oracle: {format_metrics(paired_oracle['metrics'])}",
        f"- 3cm target met by deployable selector: `{selected['test']['mae'] <= float(args.target_mae)}`",
        f"- 3cm exists in current historical candidate pool if cheating per speaker: `{global_oracle['metrics']['mae'] <= float(args.target_mae)}`",
        "",
        "## What This Means",
    ]
    if global_oracle["metrics"]["mae"] <= float(args.target_mae):
        lines.append(
            "The prediction history does contain enough scattered signal for 3cm, but the signal is not yet selectable without test labels."
        )
    else:
        lines.append(
            "Even a per-speaker oracle over all current predictions cannot reach 3cm, so the current representation pool is insufficient."
        )
    lines.append(
        "The next serious step is not a wider network by itself; it is a selector/representation pass trained to identify these exact failure modes from audio and metadata before seeing the target."
    )
    lines.extend(
        [
            "",
            "## Error Budget",
            f"- Selected total abs error: `{selected_budget['total_abs_error_cm']:.1f}cm`",
            f"- Needed total abs error for {float(args.target_mae):.1f}cm MAE: `{selected_budget['target_total_abs_error_cm']:.1f}cm`",
            f"- Required reduction: `{selected_budget['needed_reduction_cm']:.1f}cm`",
            f"- Worst speakers needing perfect repair: `{selected_budget['worst_speakers_if_perfect']}`",
            f"- Selected MAE if all short speakers were perfect: `{selected_budget['mae_if_short_perfect']:.3f}cm`",
            f"- Oracle remaining reduction to 3cm: `{global_oracle_budget['needed_reduction_cm']:.1f}cm`",
            "",
            "## Best Validation-Safe Candidates",
        ]
    )
    for row in paired_rows[:15]:
        lines.append(
            f"- `{row['name']}`: val {format_metrics(row['val'])}, test {format_metrics(row['test'])}"
        )
    lines.extend(["", "## Best Test Candidates (Diagnostic Only)"])
    for row in all_rows_by_test[:15]:
        marker = "paired" if row["has_val"] else "test-only"
        lines.append(f"- `{row['name']}` ({marker}): test {format_metrics(row['test'])}")
    lines.extend(["", "## Convex Search"])
    lines.append(f"- `{convex['name']}`: val {format_metrics(convex['val'])}, test {format_metrics(convex['test'])}")
    lines.append("- Top nonzero weights:")
    for item in convex["weights"][:12]:
        lines.append(f"  - `{item['name']}`: `{item['weight']:.4f}`")
    lines.extend(["", "## Gate Search"])
    lines.append(
        f"- `{gate['name']}`: base `{gate['base']}`, alt `{gate['alt']}`, val {format_metrics(gate['val'])}, test {format_metrics(gate['test'])}"
    )
    lines.extend(["", "## Worst Selected Blockers"])
    for row in blockers[:20]:
        lines.append(
            f"- `{row['speaker_id']}` true `{row['height_cm']:.2f}` selected `{row['selected_pred_cm']:.2f}` "
            f"err `{row['selected_abs_error_cm']:.2f}` oracle `{row['oracle_pred_cm']:.2f}` "
            f"oracle_err `{row['oracle_abs_error_cm']:.2f}`"
        )
    lines.extend(
        [
            "",
            "## Files",
            f"- Selected predictions: `{(output_dir / 'phase22_predictions_test.csv').relative_to(ROOT)}`",
            f"- Research oracle predictions: `{(output_dir / 'phase22_research_oracle_predictions_test.csv').relative_to(ROOT)}`",
            f"- Blocker table: `{(output_dir / 'phase22_blockers_test.csv').relative_to(ROOT)}`",
        ]
    )
    (output_dir / "PHASE22_3CM_REALITY_GAUNTLET_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[phase22] selected test MAE={selected['test']['mae']:.3f} short={selected['test'].get('short_mae', float('nan')):.3f}", flush=True)
    print(f"[phase22] global oracle MAE={global_oracle['metrics']['mae']:.3f} short={global_oracle['metrics'].get('short_mae', float('nan')):.3f}", flush=True)
    print(f"[phase22] wrote {output_dir / 'PHASE22_3CM_REALITY_GAUNTLET_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
