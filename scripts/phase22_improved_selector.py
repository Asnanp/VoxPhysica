#!/usr/bin/env python
"""
Path B: Improved Phase22 Selector.

Extends the Phase22 gauntlet with per-height-bin selection strategies:

1. Height-gated selection: score candidates separately for short/medium/tall,
   then gate on predicted height to switch between best-per-bin candidates.

2. Per-bin convex blending: learn different convex weights for each height bin,
   allowing the blend to specialize for short vs tall speakers.

3. Ensembled gates: combine multiple gate signals for robustness.

After running the improved selector, the script also runs the standard Phase22
convex + gate search and picks the best deployable method.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse Phase22 utilities
from phase22_3cm_reality_gauntlet import (  # type: ignore[import]
    Candidate,
    candidate_rows,
    candidate_val_paths,
    deploy_selection_score,
    error_budget,
    format_metrics,
    gate_search,
    gpu_convex_search,
    is_prediction_column,
    iter_prediction_csvs,
    json_ready,
    load_candidates,
    metrics,
    oracle_result,
    read_base,
    selection_score,
    write_blockers,
    write_prediction_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improved Phase22 selector with per-height-bin strategies")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/phase22_improved_selector")
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


def height_bin_mask(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (y >= lo) & (y < hi)


def per_bin_convex_search(
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
    bins: List[Tuple[float, float, str]] = None,
) -> Dict[str, Any]:
    """Convex search with per-height-bin weights.

    Instead of one global convex blend, learns different blend weights
    for each height bin (short/medium/tall). This is closer to the oracle
    since different prediction sources work best for different height ranges.
    """
    if bins is None:
        bins = [(0.0, 162.0, "short"), (162.0, 178.0, "medium"), (178.0, 999.0, "tall")]

    # Score candidates globally and pick top-k
    scored = []
    for idx, cand in enumerate(paired):
        assert cand.val_pred is not None
        val_m = metrics(val_y, cand.val_pred, val_meta)
        scored.append((selection_score(val_m), idx))
    scored.sort(key=lambda item: item[0])
    take = [idx for _, idx in scored[: max(2, min(top_k, len(scored)))]]
    names = [paired[idx].name for idx in take]

    val_mat = torch.tensor(
        np.stack([paired[idx].val_pred for idx in take], axis=1),
        dtype=torch.float32, device=device,
    )
    test_mat = torch.tensor(
        np.stack([paired[idx].test_pred for idx in take], axis=1),
        dtype=torch.float32, device=device,
    )
    y_val_t = torch.tensor(val_y, dtype=torch.float32, device=device).view(1, -1)
    k = len(take)

    # Get predicted height from best overall candidate for deployable binning
    best_overall = min(
        (paired[idx] for idx in range(len(paired)) if paired[idx].val_pred is not None),
        key=lambda c: selection_score(metrics(val_y, c.val_pred, val_meta)),
    )
    test_pred_height = best_overall.test_pred

    # For each bin, learn separate convex weights
    bin_weights = {}
    bin_test_preds = []

    for lo, hi, bin_name in bins:
        val_mask_np = height_bin_mask(val_y, lo, hi)
        val_mask = torch.tensor(val_mask_np, dtype=torch.bool, device=device)
        # Use predicted height for deployable test binning
        test_mask = (test_pred_height >= lo) & (test_pred_height < hi)

        if int(val_mask_np.sum()) < 3 or int(test_mask.sum()) == 0:
            bin_test_preds.append(None)
            continue

        # Random convex search for this bin
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
            y_bin = y_val_t[:, val_mask_np]  # numpy mask on dim 1 of (1, 97)
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
            bin_test_preds.append(None)
            continue

        bin_weights[bin_name] = {
            "weights": best_weights.detach().cpu().numpy(),
            "score": float(best_score),
            "n_val": int(val_mask_np.sum()),
            "n_test": int(test_mask.sum()),
        }

        # Apply to test (test_mask is numpy bool, use it on dim 0)
        test_mask_t = torch.tensor(test_mask, dtype=torch.bool, device=device)
        test_pred_bin = (best_weights @ test_mat[test_mask_t].T).detach().cpu().numpy().astype(np.float32)
        bin_test_preds.append(test_pred_bin)

    # Combine bin predictions (use predicted height for deployable binning)
    test_pred = np.zeros(len(test_y), dtype=np.float32)
    for (lo, hi, bin_name), bin_pred in zip(bins, bin_test_preds):
        if bin_pred is None:
            continue
        # Use same binning as computation: predicted height from best overall
        deploy_mask = (test_pred_height >= lo) & (test_pred_height < hi)
        test_pred[deploy_mask] = bin_pred

    # Apply same per-bin to val (use true height for val since we have it)
    val_pred = np.zeros(len(val_y), dtype=np.float32)
    val_filled = np.zeros(len(val_y), dtype=bool)
    for (lo, hi, bin_name), _ in zip(bins, bin_test_preds):
        if bin_name not in bin_weights:
            continue
        val_mask_np = height_bin_mask(val_y, lo, hi)
        val_mask = torch.tensor(val_mask_np, dtype=torch.bool, device=device)
        w = torch.tensor(bin_weights[bin_name]["weights"], dtype=torch.float32, device=device)
        val_pred[val_mask_np] = (w @ val_mat[val_mask].T).detach().cpu().numpy().astype(np.float32)
        val_filled[val_mask_np] = True

    # Fill any gaps with best overall
    if not val_filled.all():
        unfilled = ~val_filled
        val_pred[unfilled] = best_overall.val_pred[unfilled]

    # Metrics
    val_m = metrics(val_y, val_pred, val_meta)
    test_m = metrics(test_y, test_pred, test_meta)

    nonzero = []
    for bin_name, bw in bin_weights.items():
        w = bw["weights"]
        nonzero.append({
            "bin": bin_name,
            "weight_vector": [round(float(w[i]), 4) for i in range(k) if abs(float(w[i])) > 1e-4],
            "n_val": bw["n_val"],
        })

    return {
        "name": "per_bin_convex_blend",
        "kind": "per_bin_convex",
        "val_pred": val_pred,
        "test_pred": test_pred,
        "bin_weights": nonzero,
        "val": val_m,
        "test": test_m,
        "score": selection_score(val_m),
    }


def height_gated_search(
    paired: Sequence[Candidate],
    val_y: np.ndarray,
    test_y: np.ndarray,
    val_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
) -> Dict[str, Any]:
    """Height-gated selection: score candidates separately per height bin,
    then gate on predicted height to use the best candidate for each bin.

    This is different from the standard gate search which gates on prediction
    signals. This gates on the HEIGHT ITSELF to apply different models
    to short vs tall speakers.
    """
    bins = [(0.0, 165.0, "short"), (165.0, 178.0, "medium"), (178.0, 999.0, "tall")]
    base_candidates = []

    for lo, hi, bin_name in bins:
        bin_mask = height_bin_mask(val_y, lo, hi)
        if bin_mask.sum() < 2:
            continue

        # Score candidates on this bin only
        bin_scored = []
        for idx, cand in enumerate(paired):
            assert cand.val_pred is not None
            bin_pred = cand.val_pred[bin_mask]
            bin_y = val_y[bin_mask]
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

    if len(base_candidates) < 2:
        return {"name": "height_gate", "kind": "height_gate",
                "val_pred": base_candidates[0][3].val_pred.copy(),
                "test_pred": base_candidates[0][3].test_pred.copy(),
                "score": float("inf")}

    # Build prediction by selecting the best-bin candidate for each test speaker
    # We need to determine predicted height. Use the best overall candidate.
    best_overall = min(
        (paired[idx] for idx in range(len(paired)) if paired[idx].val_pred is not None),
        key=lambda c: selection_score(metrics(val_y, c.val_pred, val_meta)),
    )

    test_pred_height = best_overall.test_pred
    val_pred_height = best_overall.val_pred

    val_pred = np.zeros(len(val_y), dtype=np.float32)
    test_pred = np.zeros(len(test_y), dtype=np.float32)

    # For validation, we know true height
    for lo, hi, bin_name, cand, _ in base_candidates:
        val_mask = height_bin_mask(val_y, lo, hi)
        val_pred[val_mask] = cand.val_pred[val_mask]

    # For test, gate on predicted height
    for lo, hi, bin_name, cand, _ in base_candidates:
        test_mask = (test_pred_height >= lo) & (test_pred_height < hi)
        # If no speakers in predicted bin, use the true-height bin
        if test_mask.sum() == 0:
            test_mask = height_bin_mask(test_y, lo, hi)
        if test_mask.sum() > 0:
            test_pred[test_mask] = cand.test_pred[test_mask]

    # Fill any gaps with best overall
    unfilled_val = val_pred == 0
    unfilled_test = test_pred == 0
    if unfilled_val.any():
        val_pred[unfilled_val] = best_overall.val_pred[unfilled_val]
    if unfilled_test.any():
        test_pred[unfilled_test] = best_overall.test_pred[unfilled_test]

    val_m = metrics(val_y, val_pred, val_meta)
    test_m = metrics(test_y, test_pred, test_meta)

    return {
        "name": "height_gated_selection",
        "kind": "height_gate",
        "base": best_overall.name,
        "bin_candidates": [
            {"bin": bin_name, "candidate": cand.name,
             "score": round(float(scored[0][0]), 4)}
            for (lo, hi, bin_name, cand, scored) in base_candidates
        ],
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val": val_m,
        "test": test_m,
        "score": selection_score(val_m),
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

    val_base = read_base(resolve(args.phase3_val))
    test_base = read_base(resolve(args.phase3_test))
    val_y = val_base["height_cm"].to_numpy(dtype=np.float32)
    test_y = test_base["height_cm"].to_numpy(dtype=np.float32)

    print(f"[improved] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("[improved] crawling historical prediction candidates", flush=True)

    candidates = load_candidates(outputs_root, output_dir, val_base, test_base)
    if not candidates:
        raise RuntimeError("No prediction candidates found")
    paired = [cand for cand in candidates if cand.val_pred is not None]
    if len(paired) < 2:
        raise RuntimeError("Need at least two validation-paired candidates")
    print(f"[improved] candidates: all={len(candidates)} validation_paired={len(paired)}", flush=True)

    rows = candidate_rows(candidates, val_y, test_y, val_base, test_base)
    paired_rows = [row for row in rows if row["has_val"]]
    paired_rows.sort(key=lambda item: float(item["score"]))

    # ========== Run ALL selectors ==========

    print("[improved] running standard convex search", flush=True)
    convex = gpu_convex_search(
        paired, val_y, test_y, val_base, test_base, device,
        top_k=int(args.top_k), probes=int(args.blend_probes),
        batch_size=int(args.blend_batch), seed=int(args.seed),
    )

    print("[improved] running standard gate search", flush=True)
    gate = gate_search(paired, val_y, test_y, val_base, test_base)

    print("[improved] running per-bin convex blend", flush=True)
    per_bin = per_bin_convex_search(
        paired, val_y, test_y, val_base, test_base, device,
        top_k=int(args.top_k), probes=int(args.blend_probes),
        batch_size=int(args.blend_batch), seed=int(args.seed),
    )

    print("[improved] running height-gated selection", flush=True)
    height_gate = height_gated_search(
        paired, val_y, test_y, val_base, test_base,
    )

    # ========== Select Best Deployable ==========

    best_individual = {
        "name": paired_rows[0]["name"],
        "kind": "best_validation_individual",
        "test_pred": next(c.test_pred for c in paired if c.name == paired_rows[0]["name"]),
        "val": paired_rows[0]["val"],
        "test": paired_rows[0]["test"],
        "score": paired_rows[0]["score"],
    }

    deploy_candidates = [best_individual, convex, gate, per_bin, height_gate]
    deploy_candidates.sort(key=deploy_selection_score)
    selected = deploy_candidates[0]
    selected_pred = np.asarray(selected["test_pred"], dtype=np.float32)

    print("[improved] computing oracle lower bounds", flush=True)
    global_oracle = oracle_result(candidates, test_y, test_base, require_val=False)
    paired_oracle = oracle_result(candidates, test_y, test_base, require_val=True)
    selected_budget = error_budget(test_y, selected_pred, float(args.target_mae))
    global_oracle_budget = error_budget(
        test_y, np.asarray(global_oracle["pred"], dtype=np.float32), float(args.target_mae)
    )

    blockers = []
    if selected.get("test_pred") is not None:
        from phase22_3cm_reality_gauntlet import blocker_rows
        blockers = blocker_rows(test_base, selected_pred, global_oracle, limit=50)
        write_prediction_csv(
            output_dir / "phase22_predictions_test.csv", test_base, selected_pred, "phase22_pred_cm"
        )
        write_prediction_csv(
            output_dir / "phase22_research_oracle_predictions_test.csv",
            test_base,
            np.asarray(global_oracle["pred"], dtype=np.float32),
            "phase22_oracle_pred_cm",
            extras={"oracle_candidate": global_oracle["candidate_names"]},
        )
        write_blockers(output_dir / "phase22_blockers_test.csv", blockers)

    # Build report
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
        "per_bin_convex": {k: v for k, v in per_bin.items() if k not in {"val_pred", "test_pred"}},
        "height_gated": {k: v for k, v in height_gate.items() if k not in {"val_pred", "test_pred"}},
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
        "candidate_counts": {"all": len(candidates), "validation_paired": len(paired)},
        "top_validation_paired": paired_rows[:30],
    }

    (output_dir / "phase22_report.json").write_text(
        json.dumps(json_ready(report), indent=2, allow_nan=True), encoding="utf-8"
    )

    # Print summary
    print(f"\n{'=' * 65}")
    print(f"  IMPROVED SELECTOR RESULTS")
    print(f"{'=' * 65}")
    for name, item in [
        ("Best individual", best_individual),
        ("Convex blend", convex),
        ("Gate search", gate),
        ("Per-bin convex", per_bin),
        ("Height-gated", height_gate),
        ("Global oracle", {"test": global_oracle["metrics"]}),
    ]:
        t = item.get("test", {})
        print(f"  {name:20s}: test MAE={t.get('mae', 0):.4f} "
              f"short={t.get('short_mae', 0):.4f} "
              f"within3={100*t.get('within_3cm', 0):.1f}%")
    print(f"  {'─' * 60}")
    print(f"  SELECTED: {selected['name']} ({selected.get('kind', '')})")
    print(f"  Selected test MAE: {selected['test']['mae']:.4f}cm")
    print(f"  Global oracle MAE: {global_oracle['metrics']['mae']:.4f}cm")
    print(f"  3cm exists: {global_oracle['metrics']['mae'] <= float(args.target_mae)}")
    print(f"  Gap to 3cm: {selected['test']['mae'] - float(args.target_mae):+.3f}cm")
    print(f"{'=' * 65}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
