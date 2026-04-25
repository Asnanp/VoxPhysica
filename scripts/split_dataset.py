#!/usr/bin/env python
"""Create canonical train/val/test speaker splits with provenance metadata."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.audit_utils import (
    file_sha256,
    height_bin,
    normalize_path,
    safe_float,
    split_manifest_fingerprint,
    summarize_split_rows,
)

INPUT_CSV = "data/cleaned_dataset.csv"
OUTPUT_DIR = "data/splits"
SEED = 95

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

HEIGHT_BINS = [
    (0.0, 160.0, "short"),
    (160.0, 175.0, "medium"),
    (175.0, 1000.0, "tall"),
]

AGE_BIN_COUNT = 8
WEIGHT_BIN_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create canonical VocalMorph data splits.")
    parser.add_argument("--input_csv", default=INPUT_CSV)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--age_bins", type=int, default=AGE_BIN_COUNT)
    parser.add_argument("--weight_bins", type=int, default=WEIGHT_BIN_COUNT)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test_ratio", type=float, default=TEST_RATIO)
    return parser.parse_args()


def bin_value(value: float, bins: Sequence[Tuple[float, float, str]]) -> str:
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return "unknown"


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = index - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def build_quantile_bins(values: Sequence[float], n_bins: int, prefix: str) -> List[Tuple[float, float, str]]:
    finite_values = [value for value in values if value == value]
    if len(finite_values) < max(2, n_bins):
        return [(float("-inf"), float("inf"), f"{prefix}_all")]
    edges = [quantile(finite_values, idx / n_bins) for idx in range(n_bins + 1)]
    deduped = [edges[0]]
    for edge in edges[1:]:
        if edge > deduped[-1]:
            deduped.append(edge)
    if len(deduped) < 2:
        return [(float("-inf"), float("inf"), f"{prefix}_all")]
    bins = []
    for idx in range(len(deduped) - 1):
        lo = deduped[idx]
        hi = deduped[idx + 1]
        bins.append((lo, hi, f"{prefix}_q{idx + 1}"))
    lo, _, label = bins[-1]
    bins[-1] = (lo, float("inf"), label)
    return bins


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    a_sorted = sorted([value for value in a if value == value])
    b_sorted = sorted([value for value in b if value == value])
    if not a_sorted or not b_sorted:
        return 0.0
    na, nb = len(a_sorted), len(b_sorted)
    ia = ib = 0
    ca = cb = 0
    max_diff = 0.0
    while ia < na and ib < nb:
        if a_sorted[ia] <= b_sorted[ib]:
            value = a_sorted[ia]
            while ia < na and a_sorted[ia] == value:
                ia += 1
                ca += 1
        else:
            value = b_sorted[ib]
            while ib < nb and b_sorted[ib] == value:
                ib += 1
                cb += 1
        max_diff = max(max_diff, abs(ca / na - cb / nb))
    while ia < na:
        ia += 1
        ca += 1
        max_diff = max(max_diff, abs(ca / na - cb / nb))
    while ib < nb:
        ib += 1
        cb += 1
        max_diff = max(max_diff, abs(ca / na - cb / nb))
    return max_diff


def get_values(rows: Sequence[dict], key: str, *, nisp_only_for_weight: bool = False) -> List[float]:
    values: List[float] = []
    for row in rows:
        if nisp_only_for_weight and key == "weight_kg" and str(row.get("source", "")).upper() != "NISP":
            continue
        value = safe_float(row.get(key, ""))
        if value == value:
            values.append(value)
    return values


def source_counts(rows: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("source", "Unknown") or "Unknown")] += 1
    return dict(counts)


def evaluate_split(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    test_rows: Sequence[dict],
    all_rows: Sequence[dict],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, float]:
    metrics = {
        "ks_height_tv": ks_statistic(get_values(train_rows, "height_cm"), get_values(val_rows, "height_cm")),
        "ks_height_tt": ks_statistic(get_values(train_rows, "height_cm"), get_values(test_rows, "height_cm")),
        "ks_age_tv": ks_statistic(get_values(train_rows, "age"), get_values(val_rows, "age")),
        "ks_age_tt": ks_statistic(get_values(train_rows, "age"), get_values(test_rows, "age")),
        "ks_weight_tv": ks_statistic(
            get_values(train_rows, "weight_kg", nisp_only_for_weight=True),
            get_values(val_rows, "weight_kg", nisp_only_for_weight=True),
        ),
        "ks_weight_tt": ks_statistic(
            get_values(train_rows, "weight_kg", nisp_only_for_weight=True),
            get_values(test_rows, "weight_kg", nisp_only_for_weight=True),
        ),
    }
    metrics["avg_ks"] = sum(metrics.values()) / float(len(metrics))

    total_counts = source_counts(all_rows)
    split_counts = {
        "train": source_counts(train_rows),
        "val": source_counts(val_rows),
        "test": source_counts(test_rows),
    }
    penalty = 0.0
    for source, total in total_counts.items():
        penalty += abs(split_counts["train"].get(source, 0) - train_ratio * total) / max(1.0, total)
        penalty += abs(split_counts["val"].get(source, 0) - val_ratio * total) / max(1.0, total)
        penalty += abs(split_counts["test"].get(source, 0) - test_ratio * total) / max(1.0, total)
    metrics["src_penalty"] = penalty
    metrics["objective"] = metrics["avg_ks"] * 100.0 + penalty * 2.0
    metrics["nisp_test_count"] = float(split_counts["test"].get("NISP", 0))
    return metrics


def build_source_strata(rows: Sequence[dict], age_bins, weight_bins) -> Dict[str, List[dict]]:
    strata: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        gender = str(row.get("gender", "Unknown") or "Unknown")
        height = safe_float(row.get("height_cm", ""))
        age = safe_float(row.get("age", ""))
        weight = safe_float(row.get("weight_kg", ""))
        key = "|".join(
            [
                gender,
                bin_value(height if height == height else -1.0, HEIGHT_BINS),
                bin_value(age if age == age else -1.0, age_bins),
                bin_value(weight, weight_bins) if weight == weight else "weight_na",
            ]
        )
        strata[key].append(dict(row))
    return strata


def split_within_source(
    rows: Sequence[dict],
    age_bins,
    weight_bins,
    rng: random.Random,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[dict], List[dict], List[dict]]:
    strata = build_source_strata(rows, age_bins, weight_bins)
    target_train = int(round(train_ratio * len(rows)))
    target_val = int(round(val_ratio * len(rows)))
    target_test = len(rows) - target_train - target_val

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    test_rows: List[dict] = []
    leftovers: List[dict] = []

    for members in strata.values():
        items = list(members)
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        used = n_train + n_val + n_test
        train_rows.extend(items[:n_train])
        val_rows.extend(items[n_train : n_train + n_val])
        test_rows.extend(items[n_train + n_val : used])
        leftovers.extend(items[used:])

    deficits = {
        "train": target_train - len(train_rows),
        "val": target_val - len(val_rows),
        "test": target_test - len(test_rows),
    }
    for item in leftovers:
        choice = max(deficits, key=lambda key: deficits[key])
        if deficits[choice] <= 0:
            choice = "train"
        if choice == "train":
            train_rows.append(item)
        elif choice == "val":
            val_rows.append(item)
        else:
            test_rows.append(item)
        deficits[choice] -= 1

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def generate_split(
    rows: Sequence[dict],
    age_bins,
    weight_bins,
    seed: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[dict], List[dict], List[dict]]:
    rng = random.Random(seed)
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source", "Unknown") or "Unknown")].append(dict(row))

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    test_rows: List[dict] = []
    for source_rows in by_source.values():
        tr, va, te = split_within_source(
            source_rows,
            age_bins,
            weight_bins,
            rng,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_rows.extend(tr)
        val_rows.extend(va)
        test_rows.extend(te)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty split CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_split_bundle(rows_by_split: Dict[str, List[dict]], output_dir: str) -> Dict[str, Dict[str, str]]:
    artifacts: Dict[str, Dict[str, str]] = {}
    for split_name, rows in rows_by_split.items():
        canonical_path = os.path.join(output_dir, f"{split_name}_clean.csv")
        legacy_path = os.path.join(output_dir, f"{split_name}.csv")
        write_csv(rows, canonical_path)
        write_csv(rows, legacy_path)
        artifacts[split_name] = {
            "canonical_path": normalize_path(canonical_path),
            "legacy_path": normalize_path(legacy_path),
            "sha256": file_sha256(canonical_path),
            "fingerprint": split_manifest_fingerprint(rows),
        }
    return artifacts


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv if os.path.isabs(args.input_csv) else os.path.join(os.getcwd(), args.input_csv)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(os.getcwd(), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_csv, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    age_values = [safe_float(row.get("age", "")) for row in rows]
    age_values = [value for value in age_values if value == value]
    weight_values = [
        safe_float(row.get("weight_kg", ""))
        for row in rows
        if str(row.get("source", "")).upper() == "NISP"
    ]
    weight_values = [value for value in weight_values if value == value]
    age_bins = build_quantile_bins(age_values, args.age_bins, "age")
    weight_bins = build_quantile_bins(weight_values, args.weight_bins, "weight")

    results = []
    for offset in range(max(1, int(args.trials))):
        seed = int(args.seed) + offset
        train_rows, val_rows, test_rows = generate_split(
            rows,
            age_bins,
            weight_bins,
            seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        metrics = evaluate_split(
            train_rows,
            val_rows,
            test_rows,
            rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        metrics["seed"] = seed
        results.append({"train": train_rows, "val": val_rows, "test": test_rows, "metrics": metrics})

    results.sort(key=lambda item: item["metrics"]["objective"])
    best = results[0]
    rows_by_split = {
        "train": best["train"],
        "val": best["val"],
        "test": best["test"],
    }
    artifacts = write_split_bundle(rows_by_split, output_dir)

    leak_counts = {
        "train_val": len({row["speaker_id"] for row in rows_by_split["train"]} & {row["speaker_id"] for row in rows_by_split["val"]}),
        "train_test": len({row["speaker_id"] for row in rows_by_split["train"]} & {row["speaker_id"] for row in rows_by_split["test"]}),
        "val_test": len({row["speaker_id"] for row in rows_by_split["val"]} & {row["speaker_id"] for row in rows_by_split["test"]}),
    }

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": normalize_path(input_csv),
        "output_dir": normalize_path(output_dir),
        "seed_sweep": {
            "base_seed": int(args.seed),
            "trials": int(args.trials),
            "winning_seed": int(best["metrics"]["seed"]),
        },
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "metrics": best["metrics"],
        "leak_check": leak_counts,
        "split_artifacts": artifacts,
        "split_summaries": {
            split_name: summarize_split_rows(split_rows)
            for split_name, split_rows in rows_by_split.items()
        },
        "speaker_lists": {
            split_name: sorted(row["speaker_id"] for row in split_rows)
            for split_name, split_rows in rows_by_split.items()
        },
        "height_bins": [label for _, _, label in HEIGHT_BINS],
        "age_bins": age_bins,
        "weight_bins": weight_bins,
    }

    metadata_path = os.path.join(output_dir, "split_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    stats_lines = [
        "=== VocalMorph Canonical Split Statistics ===",
        "",
        f"Winning seed: {best['metrics']['seed']}",
        f"Average KS: {best['metrics']['avg_ks']:.4f}",
        f"Objective: {best['metrics']['objective']:.4f}",
        f"Source penalty: {best['metrics']['src_penalty']:.4f}",
        f"Leak check: {leak_counts}",
        "",
    ]
    for split_name in ("train", "val", "test"):
        summary = metadata["split_summaries"][split_name]
        stats_lines.append(f"{split_name.upper()}: {summary['rows']} speakers")
        stats_lines.append(f"  Sources: {summary['source_counts']}")
        stats_lines.append(f"  Genders: {summary['gender_counts']}")
        stats_lines.append(f"  Height bins: {summary['height_bin_counts']}")
        stats_lines.append(f"  SHA256: {artifacts[split_name]['sha256']}")
        stats_lines.append("")
    with open(os.path.join(output_dir, "split_stats.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(stats_lines))

    print("\n".join(stats_lines))
    print(f"[OK] Canonical split metadata saved to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
