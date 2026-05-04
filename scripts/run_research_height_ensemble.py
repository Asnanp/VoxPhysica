#!/usr/bin/env python
"""Run the direct speaker-level 3 cm research challenger."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.research.speaker_height_ensemble import run_research_experiment  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the speaker-level height ensemble.")
    parser.add_argument("--features-dir", default="data/features_audited")
    parser.add_argument("--output-dir", default="outputs/research_height_ensemble/seed_11")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--target-mae-cm", type=float, default=3.0)
    parser.add_argument("--ensemble-trials", type=int, default=5000)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model subset, e.g. ridge huber extra_trees hist_gbr_l1",
    )
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    features_dir = _resolve(args.features_dir)
    output_dir = _resolve(args.output_dir)
    try:
        payload = run_research_experiment(
            features_dir=features_dir,
            output_dir=output_dir,
            seed=args.seed,
            target_mae_cm=args.target_mae_cm,
            model_names=args.models,
            ensemble_trials=args.ensemble_trials,
            save_model=not args.no_save_model,
        )
    except FileNotFoundError as exc:
        print(f"[ResearchHeight] {exc}", file=sys.stderr)
        print(
            "[ResearchHeight] Expected audited splits at data/features_audited/{train,val,test}.",
            file=sys.stderr,
        )
        return 2
    val = payload["final_val"]["calibrated_edge"]
    test = payload["final_test"]["calibrated_edge"]
    print("[ResearchHeight] Finished speaker-level ensemble")
    print(f"[ResearchHeight] val_mae={val['mae']:.3f}cm | test_mae={test['mae']:.3f}cm")
    print(
        "[ResearchHeight] "
        f"test_short={test.get('short_mae', float('nan')):.3f}cm | "
        f"test_tall={test.get('tall_mae', float('nan')):.3f}cm | "
        f"within_3cm={test['within_3cm'] * 100:.1f}%"
    )
    print(f"[ResearchHeight] target_met={payload['target_met']}")
    print(f"[ResearchHeight] artifacts -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
