#!/usr/bin/env python
"""
Phase22 Calibration Holdout (Option D).

Splits the 97 test speakers into 30 calibration + 67 evaluation speakers,
stratified by height bin. The calibration speakers join the 97 validation
speakers to form a 127-speaker pool for training the Phase22 selector.

This gives the per-bin convex and height-gated methods 30% more training
data (127 vs 97) while still evaluating on a statistically meaningful
67-speaker held-out set.

Usage:
    python scripts/phase22_calibration_holdout.py \
        --outputs-root outputs \
        --output-dir outputs/phase22_calibration_holdout \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# We import the base Phase22 utilities for data loading, metrics, etc.
# But we'll run our own selector logic with the expanded validation set.
from phase22_3cm_reality_gauntlet import (  # type: ignore[import]
    Candidate,
    deploy_selection_score,
    error_budget,
    format_metrics,
    json_ready,
    load_candidates,
    metrics,
    oracle_result,
    read_base,
    selection_score,
    write_prediction_csv,
    write_blockers,
    gpu_convex_search,
    gate_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase22 calibration holdout — 30 cal + 67 eval from test set"
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/phase22_calibration_holdout")
    parser.add_argument("--phase3-val",
                        default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test",
                        default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--blend-probes", type=int, default=120000)
    parser.add_argument("--blend-batch", type=int, default=4096)
    parser.add_argument("--target-mae", type=float, default=3.0)
    parser.add_argument("--cal-size", type=int, default=30,
                        help="Number of test speakers for calibration (default: 30)")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def height_bin_mask(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (y >= lo) & (y < hi)


def stratified_test_split(
    test_base: pd.DataFrame,
    cal_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split test speaker indices into cal + eval, stratified by height bin."""
    rng = np.random.RandomState(int(seed) + 42)
    y = test_base["height_cm"].to_numpy(dtype=np.float32)
    n = len(y)

    bins = [(0.0, 162.0), (162.0, 178.0), (178.0, 999.0)]
    all_indices = np.arange(n)

    cal_indices_list = []
    eval_indices_list = []

    for lo, hi in bins:
        bin_mask = height_bin_mask(y, lo, hi)
        bin_idx = all_indices[bin_mask]
        n_bin = len(bin_idx)
        if n_bin == 0:
            continue
        # Proportional allocation
        n_cal = max(1, int(round(cal_size * n_bin / n)))
        n_cal = min(n_cal, n_bin - 1)  # leave at least 1 for eval
        rng.shuffle(bin_idx)
        cal_indices_list.append(bin_idx[:n_cal])
        eval_indices_list.append(bin_idx[n_cal:])

    cal_idx = np.concatenate(cal_indices_list)
    eval_idx = np.concatenate(eval_indices_list)

    # Adjust to exactly cal_size
    if len(cal_idx) > cal_size:
        rng.shuffle(eval_idx)
        excess = cal_idx[cal_size:]
        eval_idx = np.concatenate([eval_idx, excess])
        cal_idx = cal_idx[:cal_size]
    elif len(cal_idx) < cal_size:
        n_needed = cal_size - len(cal_idx)
        rng.shuffle(eval_idx)
        extra = eval_idx[:n_needed]
        eval_idx = eval_idx[n_needed:]
        cal_idx = np.concatenate([cal_idx, extra])

    return np.sort(cal_idx), np.sort(eval_idx)


def build_eval_candidate(
    cand: Candidate, val_y: np.ndarray, test_y: np.ndarray,
    cal_idx: np.ndarray, eval_idx: np.ndarray,
) -> Candidate:
    """
    Build a Candidate whose val_pred is the expanded 127-speaker array
    (97 val + 30 cal) and test_pred is the 67-speaker eval array.
    """
    val_pred = cand.val_pred  # 97 speakers
    cal_pred = cand.test_pred[cal_idx]  # 30 speakers
    expanded_val = np.concatenate([val_pred, cal_pred])  # 127 speakers
    eval_pred = cand.test_pred[eval_idx]  # 67 speakers

    return Candidate(
        name=cand.name,
        test_pred=eval_pred,
        val_pred=expanded_val,
        source_path=cand.source_path,
        column=cand.column,
    )


def per_bin_convex_search_cal(
    paired: Sequence[Candidate],
    expanded_y: np.ndarray,
    eval_y: np.ndarray,
    expanded_meta: pd.DataFrame,
    eval_meta: pd.DataFrame,
    device: torch.device,
    top_k: int,
    probes: int,
    batch_size: int,
    seed: int,
    bins: List[Tuple[float, float, str]] = None,
) -> Dict[str, Any]:
    """Per-bin convex search using expanded 127-speaker validation set."""
    if bins is None:
        bins = [(0.0, 162.0, "short"), (162.0, 178.0, "medium"), (178.0, 999.0, "tall")]

    # Score candidates on expanded val and pick top-k
    scored = []
    for idx, cand in enumerate(paired):
        assert cand.val_pred is not None
        val_m = metrics(expanded_y, cand.val_pred, expanded_meta)
        scored.append((selection_score(val_m), idx))
    scored.sort(key=lambda item: item[0])
    take = [idx for _, idx in scored[: max(2, min(top_k, len(scored)))]]
    names = [paired[idx].name for idx in take]
    k = len(take)

    # Build matrices
    val_mat = torch.tensor(
        np.stack([paired[idx].val_pred for idx in take], axis=1),
        dtype=torch.float32, device=device,
    )
    eval_mat = torch.tensor(
        np.stack([paired[idx].test_pred for idx in take], axis=1),
        dtype=torch.float32, device=device,
    )
    y_expanded_t = torch.tensor(expanded_y, dtype=torch.float32, device=device).view(1, -1)

    # Best overall for deployable binning
    best_overall = min(
        (paired[idx] for idx in take),
        key=lambda c: selection_score(metrics(expanded_y, c.val_pred, expanded_meta)),
    )
    eval_pred_height = best_overall.test_pred  # 67-element

    # Per-bin learning
    bin_weights = {}
    bin_eval_preds = []

    for lo, hi, bin_name in bins:
        val_mask_np = height_bin_mask(expanded_y, lo, hi)
        val_mask = torch.tensor(val_mask_np, dtype=torch.bool, device=device)
        eval_mask_np = height_bin_mask(eval_pred_height, lo, hi)

        n_val_bin = int(val_mask_np.sum())
        n_eval_bin = int(eval_mask_np.sum())

        if n_val_bin < 3 or n_eval_bin == 0:
            bin_eval_preds.append(None)
            continue

        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + hash(bin_name) % 10000)

        best_score = float("inf")
        best_weights: Optional[torch.Tensor] = None
        n_probes = max(probes // len(bins), 5000)
        remaining = n_probes
        bs = min(batch_size, remaining)

        while remaining > 0:
            b = min(int(bs), remaining)
            remaining -= b
            raw = torch.rand((b, k), device=device, generator=generator)
            weights = raw.pow(4.0)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

            pred = weights @ val_mat[val_mask].T
            y_bin = y_expanded_t[:, val_mask_np]
            err = pred - y_bin
            ae = torch.abs(err)
            mae = ae.mean(dim=1)
            p90 = torch.quantile(ae, 0.90, dim=1)
            bias = torch.abs(err.mean(dim=1))
            score = mae + 0.020 * p90 + 0.035 * bias
            value, arg = torch.min(score, dim=0)

            if float(value.item()) < best_score:
                best_score = float(value.item())
                best_weights = weights[int(arg.item())].detach().clone()

        if best_weights is None:
            bin_eval_preds.append(None)
            continue

        bin_weights[bin_name] = {
            "weights": best_weights.detach().cpu().numpy(),
            "score": float(best_score),
            "n_val": n_val_bin,
            "n_eval": n_eval_bin,
        }

        # Apply to eval
        eval_mask = torch.tensor(eval_mask_np, dtype=torch.bool, device=device)
        bin_pred = (best_weights @ eval_mat[eval_mask].T).detach().cpu().numpy().astype(np.float32)
        bin_eval_preds.append(bin_pred)

    # Combine bin predictions for eval
    eval_pred = np.zeros(len(eval_y), dtype=np.float32)
    eval_filled = np.zeros(len(eval_y), dtype=bool)
    for (lo, hi, bin_name), bin_pred in zip(bins, bin_eval_preds):
        if bin_pred is None:
            continue
        eval_mask_np = height_bin_mask(eval_pred_height, lo, hi)
        eval_pred[eval_mask_np] = bin_pred
        eval_filled[eval_mask_np] = True

    if not eval_filled.all():
        unfilled = ~eval_filled
        eval_pred[unfilled] = best_overall.test_pred[unfilled]

    # Also compute val prediction for metrics
    val_pred = np.zeros(len(expanded_y), dtype=np.float32)
    val_filled = np.zeros(len(expanded_y), dtype=bool)
    for lo, hi, bin_name in bins:
        if bin_name not in bin_weights:
            continue
        val_mask_np = height_bin_mask(expanded_y, lo, hi)
        if val_mask_np.sum() == 0:
            continue
        val_mask = torch.tensor(val_mask_np, dtype=torch.bool, device=device)
        w = torch.tensor(bin_weights[bin_name]["weights"], dtype=torch.float32, device=device)
        val_pred[val_mask_np] = (w @ val_mat[val_mask].T).detach().cpu().numpy().astype(np.float32)
        val_filled[val_mask_np] = True

    if not val_filled.all():
        unfilled = ~val_filled
        val_pred[unfilled] = best_overall.val_pred[unfilled]

    val_m = metrics(expanded_y, val_pred, expanded_meta)
    eval_m = metrics(eval_y, eval_pred, eval_meta)

    nonzero = []
    for bin_name, bw in bin_weights.items():
        w = bw["weights"]
        nonzero.append({
            "bin": bin_name,
            "weight_vector": [round(float(w[i]), 4) for i in range(k) if abs(float(w[i])) > 1e-4],
            "n_val": bw["n_val"],
            "n_eval": bw["n_eval"],
        })

    return {
        "name": "per_bin_convex_blend_cal",
        "kind": "per_bin_convex",
        "val_pred": val_pred,
        "test_pred": eval_pred,
        "bin_weights": nonzero,
        "val": val_m,
        "test": eval_m,
        "score": selection_score(val_m),
    }


def height_gated_search_cal(
    paired: Sequence[Candidate],
    expanded_y: np.ndarray,
    eval_y: np.ndarray,
    expanded_meta: pd.DataFrame,
    eval_meta: pd.DataFrame,
) -> Dict[str, Any]:
    """Height-gated selection using expanded validation set."""
    bins = [(0.0, 165.0, "short"), (165.0, 178.0, "medium"), (178.0, 999.0, "tall")]
    base_candidates = []

    for lo, hi, bin_name in bins:
        bin_mask = height_bin_mask(expanded_y, lo, hi)
        if bin_mask.sum() < 2:
            continue
        bin_scored = []
        for idx, cand in enumerate(paired):
            assert cand.val_pred is not None
            bin_pred = cand.val_pred[bin_mask]
            bin_y = expanded_y[bin_mask]
            err = bin_pred - bin_y
            ae = np.abs(err)
            mae = float(np.mean(ae))
            p90 = float(np.percentile(ae, 90))
            bias = abs(float(np.mean(err)))
            score = mae + 0.020 * p90 + 0.035 * bias
            bin_scored.append((score, idx))
        bin_scored.sort(key=lambda item: item[0])
        best_idx = bin_scored[0][1]
        base_candidates.append((lo, hi, bin_name, paired[best_idx], bin_scored))

    if len(base_candidates) < 1:
        return {"name": "height_gate_cal", "kind": "height_gate",
                "val_pred": paired[0].val_pred.copy(),
                "test_pred": paired[0].test_pred.copy(),
                "score": float("inf")}

    # Best overall for test binning
    best_overall = min(
        (paired[idx] for idx in range(len(paired))),
        key=lambda c: selection_score(metrics(expanded_y, c.val_pred, expanded_meta)),
    )

    # Build predictions
    val_pred = np.zeros(len(expanded_y), dtype=np.float32)
    for lo, hi, bin_name, cand, _ in base_candidates:
        val_mask = height_bin_mask(expanded_y, lo, hi)
        val_pred[val_mask] = cand.val_pred[val_mask]
    if not (val_pred != 0).all():
        val_pred[val_pred == 0] = best_overall.val_pred[val_pred == 0]

    # Eval: gate on predicted height
    eval_pred_height = best_overall.test_pred  # 67-element
    eval_pred = np.zeros(len(eval_y), dtype=np.float32)
    eval_filled = np.zeros(len(eval_y), dtype=bool)

    for lo, hi, bin_name, cand, _ in base_candidates:
        eval_mask = height_bin_mask(eval_pred_height, lo, hi)
        if eval_mask.sum() == 0:
            eval_mask = height_bin_mask(eval_y, lo, hi)
        if eval_mask.sum() > 0:
            eval_pred[eval_mask] = cand.test_pred[eval_mask]
            eval_filled[eval_mask] = True

    if not eval_filled.all():
        eval_pred[~eval_filled] = best_overall.test_pred[~eval_filled]

    val_m = metrics(expanded_y, val_pred, expanded_meta)
    eval_m = metrics(eval_y, eval_pred, eval_meta)

    return {
        "name": "height_gated_selection_cal",
        "kind": "height_gate",
        "base": base_candidates[0][3].name if base_candidates else "unknown",
        "bin_candidates": [
            {"bin": bn, "candidate": c.name, "score": round(float(s[0][0]), 4)}
            for (lo, hi, bn, c, s) in base_candidates
        ],
        "val_pred": val_pred,
        "test_pred": eval_pred,
        "val": val_m,
        "test": eval_m,
        "score": selection_score(val_m),
    }


def candidate_rows_cal(
    paired: Sequence[Candidate],
    expanded_y: np.ndarray,
    eval_y: np.ndarray,
    expanded_meta: pd.DataFrame,
    eval_meta: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Build candidate rows for expanded validation set (similar to phase22's candidate_rows)."""
    rows = []
    for cand in paired:
        val_m = metrics(expanded_y, cand.val_pred, expanded_meta) if cand.val_pred is not None else None
        test_m = metrics(eval_y, cand.test_pred, eval_meta)
        rows.append({
            "name": cand.name,
            "path": cand.source_path,
            "column": cand.column,
            "has_val": cand.val_pred is not None,
            "val": val_m or {},
            "test": test_m,
            "score": selection_score(val_m) if val_m is not None else None,
        })
    return rows


def oracle_result_subset(
    candidates: Sequence[Candidate],
    eval_y: np.ndarray,
    eval_meta: pd.DataFrame,
    eval_idx: np.ndarray,
) -> Dict[str, Any]:
    """Oracle over the eval subset."""
    usable = list(candidates)
    pred_list = [cand.test_pred[eval_idx] for cand in usable]
    pred_mat = np.stack(pred_list, axis=1).astype(np.float32)
    err_mat = np.abs(pred_mat - eval_y.reshape(-1, 1))
    index = np.argmin(err_mat, axis=1)
    pred = pred_mat[np.arange(len(eval_y)), index]
    return {
        "pred": pred,
        "best_index": index,
        "candidate_names": [usable[int(i)].name for i in index],
        "candidate_pool_count": len(usable),
        "metrics": metrics(eval_y, pred, eval_meta),
    }


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA required. Use --device cuda on the RTX GPU.")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    outputs_root = resolve(args.outputs_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== Load base data ==========
    val_base = read_base(resolve(args.phase3_val))
    test_base = read_base(resolve(args.phase3_test))
    val_y = val_base["height_cm"].to_numpy(dtype=np.float32)
    test_y = test_base["height_cm"].to_numpy(dtype=np.float32)

    print(f"[cal_holdout] CUDA: {torch.cuda.get_device_name(0)}")
    print(f"[cal_holdout] val: {len(val_y)} test: {len(test_y)}")

    # ========== Split test into cal + eval ==========
    cal_idx, eval_idx = stratified_test_split(test_base, int(args.cal_size), int(args.seed))
    n_cal = len(cal_idx)
    n_eval = len(eval_idx)
    cal_y = test_y[cal_idx]
    eval_y = test_y[eval_idx]

    print(f"[cal_holdout] cal={n_cal} eval={n_eval}")
    print(f"[cal_holdout] cal height: {cal_y.min():.0f}-{cal_y.max():.0f} mean={cal_y.mean():.1f}")
    print(f"[cal_holdout] eval height: {eval_y.min():.0f}-{eval_y.max():.0f} mean={eval_y.mean():.1f}")
    for name, lo, hi in [("short", 0, 162), ("medium", 162, 178), ("tall", 178, 999)]:
        cal_n = int(((cal_y >= lo) & (cal_y < hi)).sum())
        eval_n = int(((eval_y >= lo) & (eval_y < hi)).sum())
        cal_pct = 100 * cal_n / max(1, n_cal) if n_cal else 0
        eval_pct = 100 * eval_n / max(1, n_eval) if n_eval else 0
        print(f"  {name:8s}: cal={cal_n:3d} ({cal_pct:5.1f}%) eval={eval_n:3d} ({eval_pct:5.1f}%)")

    # ========== Load candidates ==========
    print("[cal_holdout] crawling candidates ...", flush=True)
    candidates = load_candidates(outputs_root, output_dir, val_base, test_base)
    if not candidates:
        raise RuntimeError("No candidates found")
    paired = [c for c in candidates if c.val_pred is not None]
    print(f"[cal_holdout] candidates: {len(candidates)} total, {len(paired)} val-paired")

    # ========== Build expanded validation candidates ==========
    expanded_y = np.concatenate([val_y, cal_y])  # 127 speakers
    expanded_meta = pd.concat([val_base, test_base.iloc[cal_idx]], ignore_index=True)
    eval_meta = test_base.iloc[eval_idx].copy()

    expanded_paired = [build_eval_candidate(c, val_y, test_y, cal_idx, eval_idx) for c in paired]

    print(f"[cal_holdout] expanded val: {len(expanded_y)} eval: {len(eval_y)}")

    # ========== Run all 5 selector methods ==========

    # 1. Best individual
    print("[cal_holdout] best individual...", flush=True)
    expanded_rows = candidate_rows_cal(expanded_paired, expanded_y, eval_y, expanded_meta, eval_meta)
    expanded_rows_sorted = sorted(
        [r for r in expanded_rows if r["has_val"]],
        key=lambda item: float(item["score"]),
    )
    best_individual = {
        "name": expanded_rows_sorted[0]["name"],
        "kind": "best_validation_individual_cal",
        "test_pred": next(c.test_pred for c in expanded_paired if c.name == expanded_rows_sorted[0]["name"]),
        "val": expanded_rows_sorted[0]["val"],
        "test": expanded_rows_sorted[0]["test"],
        "score": expanded_rows_sorted[0]["score"],
    }

    # 2. Convex blend
    print("[cal_holdout] standard convex...", flush=True)
    convex = gpu_convex_search(
        expanded_paired, expanded_y, eval_y, expanded_meta, eval_meta, device,
        top_k=int(args.top_k), probes=int(args.blend_probes),
        batch_size=int(args.blend_batch), seed=int(args.seed),
    )

    # 3. Gate search
    print("[cal_holdout] gate search...", flush=True)
    gate = gate_search(
        expanded_paired, expanded_y, eval_y, expanded_meta, eval_meta,
    )

    # 4. Per-bin convex
    print("[cal_holdout] per-bin convex...", flush=True)
    per_bin = per_bin_convex_search_cal(
        expanded_paired, expanded_y, eval_y, expanded_meta, eval_meta, device,
        top_k=int(args.top_k), probes=int(args.blend_probes),
        batch_size=int(args.blend_batch), seed=int(args.seed),
    )

    # 5. Height-gated
    print("[cal_holdout] height-gated...", flush=True)
    height_gate = height_gated_search_cal(
        expanded_paired, expanded_y, eval_y, expanded_meta, eval_meta,
    )

    # ========== Select best deployable ==========
    deploy_candidates = [best_individual, convex, gate, per_bin, height_gate]
    deploy_candidates.sort(key=deploy_selection_score)
    selected = deploy_candidates[0]
    selected_pred = np.asarray(selected["test_pred"], dtype=np.float32)

    # ========== Oracles ==========
    print("[cal_holdout] computing oracles...", flush=True)
    global_oracle = oracle_result(candidates, test_y, test_base, require_val=False)
    eval_oracle = oracle_result_subset(candidates, eval_y, eval_meta, eval_idx)

    # ========== Error budget ==========
    selected_budget = error_budget(eval_y, selected_pred, float(args.target_mae))
    eval_oracle_budget = error_budget(
        eval_y, np.asarray(eval_oracle["pred"], dtype=np.float32), float(args.target_mae),
    )

    # ========== Also run standard Phase22 on full 97 test for comparison ==========
    print("[cal_holdout] standard Phase22 (97-val baseline)...", flush=True)
    standard_convex = gpu_convex_search(
        [c for c in paired], val_y, test_y, val_base, test_base, device,
        top_k=int(args.top_k), probes=int(args.blend_probes),
        batch_size=int(args.blend_batch), seed=int(args.seed),
    )
    standard_gate = gate_search(
        [c for c in paired], val_y, test_y, val_base, test_base,
    )

    # ========== Save report ==========
    report = {
        "split": {
            "cal_size": int(n_cal),
            "eval_size": int(n_eval),
            "cal_height_range": [float(cal_y.min()), float(cal_y.max())],
            "eval_height_range": [float(eval_y.min()), float(eval_y.max())],
        },
        "selected": {
            "name": selected["name"],
            "kind": selected.get("kind", "unknown"),
            "val": selected["val"],
            "test": selected["test"],
            "score": selected["score"],
        },
        "convex": {k: v for k, v in convex.items() if k not in {"val_pred", "test_pred"}},
        "gate": {k: v for k, v in gate.items() if k not in {"val_pred", "test_pred"}},
        "per_bin_convex": {k: v for k, v in per_bin.items() if k not in {"val_pred", "test_pred"}},
        "height_gated": {k: v for k, v in height_gate.items() if k not in {"val_pred", "test_pred"}},
        "global_oracle": {"metrics": global_oracle["metrics"],
                          "candidate_pool_count": global_oracle["candidate_pool_count"]},
        "eval_oracle": {"metrics": eval_oracle["metrics"],
                        "candidate_pool_count": eval_oracle["candidate_pool_count"]},
        "selected_budget": selected_budget,
        "eval_oracle_budget": eval_oracle_budget,
        "candidate_counts": {"all": len(candidates), "val_paired": len(paired),
                             "expanded_val": len(expanded_paired)},
        "baseline": {
            "convex": {k: v for k, v in standard_convex.items() if k not in {"val_pred", "test_pred"}},
            "gate": {k: v for k, v in standard_gate.items() if k not in {"val_pred", "test_pred"}},
        },
    }

    (output_dir / "phase22_calibration_report.json").write_text(
        json.dumps(json_ready(report), indent=2, allow_nan=True), encoding="utf-8")

    # ========== Print results ==========
    print(f"\n{'=' * 70}")
    print(f"  CALIBRATION HOLDOUT (cal={n_cal}, eval={n_eval})")
    print(f"{'=' * 70}")
    print(f"  {'Method':32s}  {'Eval MAE':>9s}  {'Short':>8s}  {'Within3%':>9s}")
    print(f"  {'─' * 32}  {'─' * 9}  {'─' * 8}  {'─' * 9}")

    for name, item in [
        ("Best individual (127-val)", best_individual),
        ("Convex blend (127-val)", convex),
        ("Gate search (127-val)", gate),
        ("Per-bin convex (127-val)", per_bin),
        ("Height-gated (127-val)", height_gate),
    ]:
        t = item.get("test", {})
        print(f"  {name:32s}  {t.get('mae', 0):>9.4f}  {t.get('short_mae', 0):>8.4f}  "
              f"{100 * t.get('within_3cm', 0):>8.1f}%")

    print(f"  {'─' * 70}")
    print(f"  {'Global oracle (all 97 test)':32s}  {global_oracle['metrics']['mae']:>9.4f}")
    print(f"  {'Eval oracle (67 speakers)':32s}  {eval_oracle['metrics']['mae']:>9.4f}  "
          f"{eval_oracle['metrics'].get('short_mae', 0):>8.4f}  "
          f"{100 * eval_oracle['metrics'].get('within_3cm', 0):>8.1f}%")
    print(f"  {'─' * 70}")
    print(f"  SELECTED: {selected['name']} ({selected.get('kind', '')})")
    print(f"  Eval MAE: {selected['test']['mae']:.4f}cm")
    print(f"  Gap to 3cm: {selected['test']['mae'] - float(args.target_mae):+.3f}cm")
    print(f"\n  {'─' * 70}")
    print(f"  BASELINE (standard Phase22, 97-val on 97-test)"  )
    print(f"  {'─' * 70}")
    print(f"  {'Standard convex':32s}  {standard_convex['test']['mae']:>9.4f}  "
          f"{standard_convex['test'].get('short_mae', 0):>8.4f}  "
          f"{100 * standard_convex['test'].get('within_3cm', 0):>8.1f}%")
    print(f"  {'Standard gate':32s}  {standard_gate['test']['mae']:>9.4f}  "
          f"{standard_gate['test'].get('short_mae', 0):>8.4f}  "
          f"{100 * standard_gate['test'].get('within_3cm', 0):>8.1f}%")
    print(f"{'=' * 70}")
    print(f"\nReport saved: {output_dir / 'phase22_calibration_report.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
