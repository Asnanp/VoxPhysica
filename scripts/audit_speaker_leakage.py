#!/usr/bin/env python
"""Audit train/val/test feature splits for speaker leakage."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, Iterable, Set

import numpy as np


def _decode_np_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.ndim == 0:
            return str(value.item())
        return str(value.flat[0])
    return str(value)


def load_split_speaker_ids(split_dir: str) -> Set[str]:
    speaker_ids: Set[str] = set()
    for path in sorted(glob.glob(os.path.join(split_dir, "*.npz"))):
        with np.load(path, allow_pickle=True) as data:
            if "speaker_id" in data:
                speaker_id = _decode_np_value(data["speaker_id"]).strip()
            else:
                speaker_id = os.path.splitext(os.path.basename(path))[0].split("_aug", 1)[0]
        if speaker_id:
            speaker_ids.add(speaker_id)
    return speaker_ids


def speaker_leakage_report(feature_root: str, splits: Iterable[str] = ("train", "val", "test")) -> Dict[str, object]:
    split_sets: Dict[str, Set[str]] = {}
    for split in splits:
        split_dir = os.path.join(feature_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        split_sets[split] = load_split_speaker_ids(split_dir)

    overlaps: Dict[str, int] = {}
    examples: Dict[str, list[str]] = {}
    split_names = list(split_sets.keys())
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            key = f"{left}_{right}"
            shared = sorted(split_sets[left] & split_sets[right])
            overlaps[key] = len(shared)
            examples[key] = shared[:10]

    return {
        "feature_root": feature_root,
        "split_counts": {split: len(ids) for split, ids in split_sets.items()},
        "overlap_counts": overlaps,
        "overlap_examples": examples,
        "has_leakage": any(count > 0 for count in overlaps.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit train/val/test splits for speaker leakage.")
    parser.add_argument("--features-dir", required=True, help="Directory containing train/ val/ test split folders")
    args = parser.parse_args()

    feature_root = args.features_dir if os.path.isabs(args.features_dir) else os.path.abspath(args.features_dir)
    report = speaker_leakage_report(feature_root)

    print(f"[Leakage Audit] Feature root: {report['feature_root']}")
    for split, count in report["split_counts"].items():
        print(f"  {split}: {count} unique speakers")
    for pair, count in report["overlap_counts"].items():
        print(f"  overlap {pair}: {count}")
        examples = report["overlap_examples"].get(pair, [])
        if examples:
            print(f"    examples: {', '.join(examples)}")

    if report["has_leakage"]:
        print("[Leakage Audit] Speaker leakage detected.")
        return 1

    print("[Leakage Audit] No speaker leakage detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
