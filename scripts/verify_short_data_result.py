#!/usr/bin/env python
"""Independently verify VoxPhysica short-speaker collection artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-dir",
        default="outputs/short_speaker_collection_v1",
    )
    parser.add_argument(
        "--train-manifest",
        default="data/splits/train_plus_short_support.csv",
    )
    parser.add_argument("--base-train", default="data/splits/train_clean.csv")
    parser.add_argument("--validation", default="data/splits/val_clean.csv")
    parser.add_argument("--historical-test", default="data/splits/test_clean.csv")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def ids(data: list[dict[str, str]]) -> set[str]:
    return {row["speaker_id"].strip() for row in data}


def main() -> int:
    args = parse_args()
    collection_dir = resolve(args.collection_dir)
    report = json.loads(
        (collection_dir / "short_data_audit.json").read_text(encoding="utf-8")
    )
    speakers = rows(collection_dir / "short_speakers_accepted.csv")
    clips = rows(collection_dir / "short_clips_accepted.csv")
    combined = rows(resolve(args.train_manifest))
    base = rows(resolve(args.base_train))
    validation = rows(resolve(args.validation))
    historical_test = rows(resolve(args.historical_test))

    speaker_ids = [row["speaker_id"].strip() for row in speakers]
    duplicate_speakers = [
        speaker_id
        for speaker_id, count in Counter(speaker_ids).items()
        if count != 1
    ]
    if duplicate_speakers:
        raise RuntimeError(f"Duplicate accepted speakers: {duplicate_speakers[:10]}")

    clip_speakers = Counter(row["speaker_id"].strip() for row in clips)
    unknown_clip_speakers = sorted(set(clip_speakers) - set(speaker_ids))
    if unknown_clip_speakers:
        raise RuntimeError(f"Clip rows without speaker row: {unknown_clip_speakers[:10]}")
    below_minimum = {
        speaker_id: count for speaker_id, count in clip_speakers.items() if count < 5
    }
    if below_minimum:
        raise RuntimeError(f"Speakers below five clips: {below_minimum}")

    if any(row.get("qc_pass") != "yes" for row in clips):
        raise RuntimeError("Accepted clip manifest contains a failed-QC row")
    if any(not Path(row["audio_path"]).is_file() for row in clips):
        raise RuntimeError("Accepted clip manifest contains a missing audio file")

    hash_owners: defaultdict[str, set[str]] = defaultdict(set)
    for row in clips:
        digest = row["sha256"].strip()
        if len(digest) != 64:
            raise RuntimeError("Accepted clip has invalid SHA-256")
        hash_owners[digest].add(row["speaker_id"].strip())
    cross_speaker_hashes = {
        digest: owners for digest, owners in hash_owners.items() if len(owners) > 1
    }
    if cross_speaker_hashes:
        raise RuntimeError("One audio hash is assigned to multiple speakers")

    sealed_ids = ids(validation) | ids(historical_test)
    overlap = sorted(set(speaker_ids) & sealed_ids)
    if overlap:
        raise RuntimeError(f"Accepted speaker overlaps validation/test: {overlap[:10]}")
    if set(speaker_ids) & ids(base):
        raise RuntimeError("Accepted support overlaps base train")

    expected_combined = ids(base) | set(speaker_ids)
    if ids(combined) != expected_combined:
        raise RuntimeError("Combined train IDs do not equal base plus support")

    accepted_report = report["accepted"]
    checks = {
        "speaker_count": (len(speakers), int(accepted_report["speakers"])),
        "clip_count": (len(clips), int(accepted_report["clips"])),
        "combined_train_count": (
            len(ids(combined)),
            int(report["combined_train_speakers"]),
        ),
    }
    mismatches = {
        key: values for key, values in checks.items() if values[0] != values[1]
    }
    if mismatches:
        raise RuntimeError(f"Report mismatch: {mismatches}")

    duration_hours = sum(float(row["duration_s"]) for row in clips) / 3600.0
    if abs(duration_hours - float(accepted_report["duration_hours"])) > 1e-5:
        raise RuntimeError("Duration does not reproduce report")

    print(
        f"[verify-short-data] speakers={len(speakers)} clips={len(clips)} "
        f"hours={duration_hours:.4f} combined_train={len(ids(combined))}"
    )
    print("[verify-short-data] speaker and hash integrity passed")
    print("[verify-short-data] validation/test overlap=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
