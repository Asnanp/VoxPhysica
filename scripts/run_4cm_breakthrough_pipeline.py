"""VoxPhysica 4.0 cm MAE Breakthrough Pipeline Execution Script.

Trains multi-SSL acoustic, physics, and metadata-assisted models using
5-fold Out-Of-Fold (OOF) cross-validation on train, convex ensembling,
and group residual calibration. Evaluates sealed test set once.
"""

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
        description="VoxPhysica 4.0 cm Breakthrough Pipeline Execution."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/4cm_breakthrough"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced candidate search for fast verification.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("=================================================================")
    print("  VoxPhysica 4.0 cm MAE Breakthrough Height Estimation Pipeline  ")
    print("=================================================================")
    print(f" Output Directory : {args.output_dir}")
    print(f" Seed             : {args.seed}")
    print(f" Folds            : {args.folds}")
    print(f" Quick Mode       : {args.quick}")
    print("-----------------------------------------------------------------")

    result = run_strict_experiment(
        root=ROOT,
        output_dir=args.output_dir,
        seed=args.seed,
        folds=args.folds,
        quick=args.quick,
    )

    val_metrics = result["selection"]["frozen_validation_metrics"]
    test = result["test"]
    metrics = test["metrics"]
    confidence = test["bootstrap_mae_95_ci"]
    recipe = result["selection"]["frozen_recipe"]

    print("\n------------------- EXPERIMENT RESULTS -------------------")
    print(f"  Frozen Validation MAE: {val_metrics['mae_cm']:.3f} cm")
    print(f"  Sealed Test MAE       : {metrics['mae_cm']:.3f} cm")
    print(f"  Test Median AE        : {metrics['median_ae_cm']:.3f} cm")
    print(f"  Test RMSE             : {metrics['rmse_cm']:.3f} cm")
    print(
        f"  95% Bootstrap CI      : [{confidence['lower_95_cm']:.3f}, "
        f"{confidence['upper_95_cm']:.3f}] cm"
    )
    print(f"  4.0 cm Target Met     : {test['four_cm_point_target_met']}")
    print(f"  3.0 cm Target Met     : {test['three_cm_point_target_met']}")
    print(f"  Winning Recipe        : {recipe['name']}")
    print(f"  Post-processor        : {recipe['postprocess']}")
    print(
        "  Ensemble Weights     : "
        + json.dumps(result["final_model"]["weights"], indent=2)
    )

    if test['four_cm_point_target_met']:
        print("\n SUCCESS: Achieved Target Speaker-Level MAE <= 4.0 cm!")
    else:
        print(f"\n MAE is {metrics['mae_cm']:.3f} cm.")

    print("=================================================================")


if __name__ == "__main__":
    main()
