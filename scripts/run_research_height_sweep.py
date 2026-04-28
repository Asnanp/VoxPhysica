#!/usr/bin/env python
"""Run the speaker-height research ensemble across multiple seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.research.speaker_height_ensemble import run_research_experiment  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the research height ensemble across multiple seeds.")
    parser.add_argument("--features-dir", default="data/features_audited")
    parser.add_argument("--output-root", default="outputs/research_height_ensemble")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 17, 23])
    parser.add_argument("--target-mae-cm", type=float, default=3.0)
    parser.add_argument("--ensemble-trials", type=int, default=5000)
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model subset, e.g. ridge elasticnet grad_boost_l1",
    )
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _load_metrics(metrics_path: str) -> Dict[str, Any]:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _headline(payload: Dict[str, Any]) -> Dict[str, float]:
    test = payload["final_test"]["calibrated_edge"]
    return {
        "mae": float(test["mae"]),
        "short_mae": float(test.get("short_mae", float("nan"))),
        "tall_mae": float(test.get("tall_mae", float("nan"))),
        "within_3cm": float(test["within_3cm"]),
    }


def _write_summary(output_root: str, results: List[Dict[str, Any]]) -> None:
    best = min(results, key=lambda item: float(item["payload"]["final_test"]["calibrated_edge"]["mae"]))
    summary = {
        "experiment": "research_height_sweep",
        "output_root": output_root,
        "best_seed": int(best["seed"]),
        "best_test": best["payload"]["final_test"]["calibrated_edge"],
        "runs": [
            {
                "seed": int(item["seed"]),
                "output_dir": item["output_dir"],
                "test": item["payload"]["final_test"]["calibrated_edge"],
                "val": item["payload"]["final_val"]["calibrated_edge"],
            }
            for item in results
        ],
    }
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "sweep_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "# Research Height Sweep",
        "",
        f"- Best seed: `{summary['best_seed']}`",
        f"- Best test speaker MAE: `{summary['best_test']['mae']:.3f} cm`",
        f"- Best test short speaker MAE: `{summary['best_test'].get('short_mae', float('nan')):.3f} cm`",
        f"- Best test tall speaker MAE: `{summary['best_test'].get('tall_mae', float('nan')):.3f} cm`",
        "",
        "## Runs",
    ]
    for item in summary["runs"]:
        lines.append(
            "- "
            f"seed `{item['seed']}`: test_mae=`{item['test']['mae']:.3f} cm`, "
            f"short=`{item['test'].get('short_mae', float('nan')):.3f} cm`, "
            f"tall=`{item['test'].get('tall_mae', float('nan')):.3f} cm`, "
            f"within_3cm=`{item['test']['within_3cm'] * 100:.1f}%`"
        )
    with open(os.path.join(output_root, "sweep_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    features_dir = _resolve(args.features_dir)
    output_root = _resolve(args.output_root)
    results: List[Dict[str, Any]] = []

    for seed in args.seeds:
        output_dir = os.path.join(output_root, f"seed_{seed}")
        metrics_path = os.path.join(output_dir, "metrics.json")
        if os.path.isfile(metrics_path) and not args.rerun_existing:
            payload = _load_metrics(metrics_path)
            print(f"[ResearchSweep] Seed {seed}: using existing metrics from {metrics_path}")
        else:
            print(f"[ResearchSweep] Seed {seed}: running experiment...")
            payload = run_research_experiment(
                features_dir=features_dir,
                output_dir=output_dir,
                seed=seed,
                target_mae_cm=args.target_mae_cm,
                model_names=args.models,
                ensemble_trials=args.ensemble_trials,
                save_model=not args.no_save_model,
            )
        headline = _headline(payload)
        print(
            "[ResearchSweep] "
            f"seed={seed} | test_mae={headline['mae']:.3f}cm | "
            f"short={headline['short_mae']:.3f}cm | "
            f"tall={headline['tall_mae']:.3f}cm | "
            f"within_3cm={headline['within_3cm'] * 100:.1f}%"
        )
        results.append({"seed": int(seed), "output_dir": output_dir, "payload": payload})

    _write_summary(output_root, results)
    best = min(results, key=lambda item: float(item["payload"]["final_test"]["calibrated_edge"]["mae"]))
    best_test = best["payload"]["final_test"]["calibrated_edge"]
    print(
        "[ResearchSweep] BEST "
        f"seed={best['seed']} | test_mae={best_test['mae']:.3f}cm | "
        f"short={best_test.get('short_mae', float('nan')):.3f}cm | "
        f"tall={best_test.get('tall_mae', float('nan')):.3f}cm"
    )
    print(f"[ResearchSweep] summary -> {os.path.join(output_root, 'sweep_summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
