"""Run the leakage-resistant VoxPhysica 3 cm research experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.strict_height_pipeline import run_strict_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train on speaker-disjoint development data and perform one "
            "frozen test evaluation."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/strict_3cm_research"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use a reduced model search. This still evaluates test once, "
            "so use it only for an explicitly non-sealed smoke benchmark."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_strict_experiment(
        root=ROOT,
        output_dir=args.output_dir,
        seed=args.seed,
        folds=args.folds,
        quick=args.quick,
    )
    test = result["test"]
    metrics = test["metrics"]
    confidence = test["bootstrap_mae_95_ci"]
    print()
    print("VoxPhysica strict result")
    print(f"  test MAE: {metrics['mae_cm']:.3f} cm")
    print(
        "  95% bootstrap CI: "
        f"[{confidence['lower_95_cm']:.3f}, "
        f"{confidence['upper_95_cm']:.3f}] cm"
    )
    print(f"  3 cm target met: {test['three_cm_point_target_met']}")
    print(f"  4 cm target met: {test['four_cm_point_target_met']}")
    print(
        "  frozen recipe: "
        + json.dumps(result["selection"]["frozen_recipe"], sort_keys=True)
    )


if __name__ == "__main__":
    main()
