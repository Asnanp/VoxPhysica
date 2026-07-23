#!/usr/bin/env python
"""Run the direct speaker-level height research ensemble."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.speaker_height_ensemble import run_research_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run speaker-level height ensemble research.")
    parser.add_argument("--features-dir", default="data/features_v4_target_ssl")
    parser.add_argument("--output-dir", default="outputs/diagnostics/speaker_height_research_target_ssl")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--target-mae-cm", type=float, default=3.0)
    parser.add_argument("--ensemble-trials", type=int, default=5000)
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def _resolve(path: str | os.PathLike[str]) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def main() -> int:
    args = parse_args()
    payload = run_research_experiment(
        features_dir=_resolve(args.features_dir),
        output_dir=_resolve(args.output_dir),
        seed=int(args.seed),
        target_mae_cm=float(args.target_mae_cm),
        ensemble_trials=int(args.ensemble_trials),
        save_model=not bool(args.no_save_model),
    )
    final = payload["final_test"]["calibrated_edge"]
    print(
        "[speaker-research] final calibrated_edge test "
        f"mae={final['mae']:.3f} rmse={final['rmse']:.3f} "
        f"short={final.get('short_mae', float('nan')):.3f} "
        f"medium={final.get('medium_mae', float('nan')):.3f} "
        f"tall={final.get('tall_mae', float('nan')):.3f} "
        f"within3={final['within_3cm'] * 100:.1f}%"
    )
    print(json.dumps({"target_met": payload["target_met"], "output_dir": _resolve(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
