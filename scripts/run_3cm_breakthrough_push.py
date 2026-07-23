"""VoxPhysica 3.0 cm MAE Breakthrough Research Pipeline Execution Script.

Runs physics-informed VTL feature augmentation, multi-view SSL representation fusion,
5-fold OOF convex meta-ensembling, and range-aware piecewise calibration.
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
        description="VoxPhysica 3.0 cm MAE Breakthrough Pipeline Execution."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/3cm_breakthrough_vtl"),
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
    print("  VoxPhysica 3.0 cm MAE Breakthrough Physics & MLOps Pipeline   ")
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
    slices = test.get("slices", {})

    print("\n------------------- EXPERIMENT RESULTS -------------------")
    print(f"  Frozen Validation MAE: {val_metrics['mae_cm']:.3f} cm")
    print(f"  Sealed Test MAE       : {metrics['mae_cm']:.3f} cm")
    print(f"  Test Median AE        : {metrics['median_ae_cm']:.3f} cm")
    print(f"  Test RMSE             : {metrics['rmse_cm']:.3f} cm")
    print(
        f"  95% Bootstrap CI      : [{confidence['lower_95_cm']:.3f}, "
        f"{confidence['upper_95_cm']:.3f}] cm"
    )
    print(f"  3.0 cm Target Met     : {test['three_cm_point_target_met']}")
    print(f"  Winning Recipe        : {recipe['name']}")
    print(f"  Post-processor        : {recipe['postprocess']}")

    if "short_lt160" in slices:
        short = slices["short_lt160"]
        print(f"  Short (<160cm) MAE   : {short['mae_cm']:.3f} cm (N={short['n']})")
    if "tall_ge175" in slices:
        tall = slices["tall_ge175"]
        print(f"  Tall (>=175cm) MAE   : {tall['mae_cm']:.3f} cm (N={tall['n']})")

    print("=================================================================")


if __name__ == "__main__":
    main()
