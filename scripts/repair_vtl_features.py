#!/usr/bin/env python
"""Repair stored VTL features without rebuilding audio features from WAV files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessing.feature_extractor import (  # noqa: E402
    robust_positive_median,
    robust_vtl_from_formant_spacing,
    vtl_spacing_bounds,
)

SPLITS = ("train", "val", "test")
FORMANT_SPACING_COL = 134
VTL_COL = 135


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/features_audited")
    parser.add_argument("--output-dir", default="data/features_vtl_fixed")
    parser.add_argument("--speed-of-sound", type=float, default=34000.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def iter_npz_files(split_dir: str, limit: int | None = None) -> Iterable[str]:
    count = 0
    for name in sorted(os.listdir(split_dir)):
        if not name.endswith(".npz"):
            continue
        yield os.path.join(split_dir, name)
        count += 1
        if limit is not None and count >= limit:
            break


def repair_payload(payload: Dict[str, Any], *, speed_of_sound: float) -> Tuple[Dict[str, Any], Dict[str, float]]:
    sequence = np.asarray(payload["sequence"], dtype=np.float32).copy()
    if sequence.ndim != 2 or sequence.shape[1] <= VTL_COL:
        raise ValueError(f"Expected sequence with at least {VTL_COL + 1} columns, got {sequence.shape}")

    old_vtl = sequence[:, VTL_COL].astype(np.float32, copy=True)
    spacing = sequence[:, FORMANT_SPACING_COL].astype(np.float32, copy=False)
    new_vtl = robust_vtl_from_formant_spacing(spacing, speed_of_sound=speed_of_sound)
    sequence[:, VTL_COL] = new_vtl

    min_spacing, max_spacing = vtl_spacing_bounds(speed_of_sound=speed_of_sound)
    invalid_spacing = (
        (~np.isfinite(spacing))
        | (spacing <= 0.0)
        | (spacing < min_spacing)
        | (spacing > max_spacing)
    )
    invalid_vtl = (~np.isfinite(new_vtl)) | (new_vtl <= 0.0)

    payload = dict(payload)
    payload["sequence"] = sequence.astype(np.float32, copy=False)
    payload["vtl_mean"] = np.asarray(robust_positive_median(new_vtl), dtype=np.float32)
    payload["invalid_spacing_rate"] = np.asarray(float(np.mean(invalid_spacing)), dtype=np.float32)
    payload["invalid_vtl_rate"] = np.asarray(float(np.mean(invalid_vtl)), dtype=np.float32)
    payload["vtl_repair_tag"] = np.asarray("robust_vtl_v1")

    old_valid = old_vtl[np.isfinite(old_vtl) & (old_vtl > 0.0)]
    new_valid = new_vtl[np.isfinite(new_vtl) & (new_vtl > 0.0)]
    stats = {
        "old_vtl_mean": float(old_valid.mean()) if old_valid.size else 0.0,
        "old_vtl_max": float(old_valid.max()) if old_valid.size else 0.0,
        "new_vtl_mean": float(new_valid.mean()) if new_valid.size else 0.0,
        "new_vtl_max": float(new_valid.max()) if new_valid.size else 0.0,
    }
    return payload, stats


def load_payload(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def write_payload(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.npz"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, path)


def copy_metadata(input_dir: str, output_dir: str) -> None:
    for name in (
        "target_stats.json",
        "feature_contract.json",
        "feature_diagnostics.json",
        "build_manifest.json",
    ):
        src = os.path.join(input_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, name))


def main() -> int:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    copy_metadata(input_dir, output_dir)

    totals = {
        "files": 0,
        "old_vtl_mean_sum": 0.0,
        "old_vtl_max": 0.0,
        "new_vtl_mean_sum": 0.0,
        "new_vtl_max": 0.0,
    }
    for split in SPLITS:
        src_split = os.path.join(input_dir, split)
        dst_split = os.path.join(output_dir, split)
        if not os.path.isdir(src_split):
            continue
        os.makedirs(dst_split, exist_ok=True)
        split_count = 0
        for src_path in iter_npz_files(src_split, limit=args.limit):
            dst_path = os.path.join(dst_split, os.path.basename(src_path))
            if os.path.exists(dst_path) and not args.overwrite:
                continue
            payload, stats = repair_payload(load_payload(src_path), speed_of_sound=args.speed_of_sound)
            write_payload(dst_path, payload)
            split_count += 1
            totals["files"] += 1
            totals["old_vtl_mean_sum"] += stats["old_vtl_mean"]
            totals["old_vtl_max"] = max(totals["old_vtl_max"], stats["old_vtl_max"])
            totals["new_vtl_mean_sum"] += stats["new_vtl_mean"]
            totals["new_vtl_max"] = max(totals["new_vtl_max"], stats["new_vtl_max"])
        print(f"{split}: repaired {split_count} files")

    files = max(1, int(totals["files"]))
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "files_repaired": int(totals["files"]),
        "old_vtl_mean_avg": float(totals["old_vtl_mean_sum"] / files),
        "old_vtl_max": float(totals["old_vtl_max"]),
        "new_vtl_mean_avg": float(totals["new_vtl_mean_sum"] / files),
        "new_vtl_max": float(totals["new_vtl_max"]),
        "repair": "sequence column 135 recomputed from formant spacing column 134 with robust VTL clamp",
    }
    with open(os.path.join(output_dir, "vtl_repair_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
