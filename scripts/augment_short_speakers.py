#!/usr/bin/env python
"""
Create duplicated feature files for sampling experiments.

This script does not collect new people and must never increase the reported
unique-speaker count. For real short-speaker collection, consent, and audio QC,
use ``scripts/collect_short_speaker_data.py``.

Creates a new training directory with all original files PLUS 2 additional
augmented copies of each short-speaker (<165cm) .npz file. The copies are
identical to the originals — the on-the-fly augmentation pipeline in the
data loader applies different random transforms each epoch, so the model
sees effectively diverse versions of the short-speaker features.

Usage:
    python scripts/augment_short_speakers.py \\
        --source-dir data/features_vtl_ssl \\
        --output-dir data/features_vtl_ssl_augmented \\
        --short-threshold 165.0 \\
        --n-copies 2
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment short speakers by duplicating feature .npz files"
    )
    parser.add_argument(
        "--source-dir",
        default="data/features_vtl_ssl",
        help="Source feature directory with train/val/test splits",
    )
    parser.add_argument(
        "--output-dir",
        default="data/features_vtl_ssl_augmented",
        help="Output feature directory with augmented train split",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=165.0,
        help="Height threshold (cm) below which speakers are considered 'short'",
    )
    parser.add_argument(
        "--n-copies",
        type=int,
        default=2,
        help="Number of additional augmented copies to create per short speaker",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_dir)
    output_root = Path(args.output_dir)
    threshold = float(args.short_threshold)
    n_copies = int(args.n_copies)

    print("=" * 60)
    print("  SHORT FEATURE OVERSAMPLING (NOT NEW PEOPLE)")
    print("=" * 60)
    print(f"  Source:      {source_root}")
    print(f"  Output:      {output_root}")
    print(f"  Threshold:   <{threshold}cm")
    print(f"  Copies/short: {n_copies}")
    print()

    # Process each split
    for split in ("train", "val", "test"):
        src_dir = source_root / split
        dst_dir = output_root / split
        if not src_dir.exists():
            print(f"  [SKIP] {split}: source not found")
            continue

        os.makedirs(dst_dir, exist_ok=True)
        files = sorted(glob.glob(str(src_dir / "*.npz")))
        total = len(files)
        short_count = 0
        copied = 0

        for fpath in files:
            data = np.load(fpath, allow_pickle=True)
            height_cm = float(data["height_cm"])
            data.close()

            filename = os.path.basename(fpath)

            # Copy original
            shutil.copy2(fpath, str(dst_dir / filename))
            copied += 1

            # For short speakers in TRAIN only, create augmented copies
            if height_cm < threshold and split == "train":
                short_count += 1
                for i in range(1, n_copies + 1):
                    aug_name = filename.replace(".npz", f"_aug{i}.npz")
                    shutil.copy2(fpath, str(dst_dir / aug_name))
                    copied += 1

        print(f"  [{split:5s}] {total:5d} originals -> {short_count:4d} short x{n_copies} -> {copied:5d} total files")

    # Copy target_stats.json and other metadata
    for extra in ("target_stats.json", "ssl_info.json"):
        src = source_root / extra
        if src.exists():
            shutil.copy2(src, str(output_root / extra))
            print(f"  Copied {extra}")

    print()
    print("  Done! Oversampled features ready; unique people did not increase.")
    print(f"  New feature directory: {output_root}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
