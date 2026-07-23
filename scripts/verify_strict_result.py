"""Independently verify a saved strict VoxPhysica result artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/strict_3cm_research"),
    )
    args = parser.parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )

    payload = json.loads(
        (output_dir / "strict_results.json").read_text(encoding="utf-8")
    )
    with (output_dir / "predictions_test_once.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    target = np.asarray([float(row["height_cm"]) for row in rows])
    prediction = np.asarray([float(row["pred_height_cm"]) for row in rows])
    error = np.abs(target - prediction)
    recomputed = {
        "mae_cm": float(error.mean()),
        "median_ae_cm": float(np.median(error)),
        "p90_ae_cm": float(np.quantile(error, 0.90)),
        "rmse_cm": float(np.sqrt(np.mean(np.square(error)))),
        "within_3cm": float(np.mean(error <= 3.0)),
        "within_4cm": float(np.mean(error <= 4.0)),
    }
    recorded = payload["test"]["metrics"]
    for name, value in recomputed.items():
        if not np.isclose(value, recorded[name], atol=2e-6):
            raise AssertionError(
                f"{name} mismatch: CSV={value}, JSON={recorded[name]}"
            )

    for relative, expected in payload["integrity_manifest"].items():
        path = ROOT / relative
        if path.stat().st_size != expected["bytes"]:
            raise AssertionError(f"Size changed: {relative}")
        if sha256(path) != expected["sha256"]:
            raise AssertionError(f"Hash changed: {relative}")

    if len(rows) != payload["data"]["test_speakers"]:
        raise AssertionError("Prediction row count does not match test speaker count")

    print(
        f"verified {len(rows)} speakers; "
        f"MAE={recomputed['mae_cm']:.6f} cm; "
        "all input hashes match"
    )


if __name__ == "__main__":
    main()
