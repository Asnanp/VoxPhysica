#!/usr/bin/env python
"""Extract HeightCeleb-labeled VoxCeleb1 clips from WebDataset tar shards.

The Zenodo WebDataset release stores samples as `key.json` + `key.wav`, where
the JSON contains `speakerid`. This script keeps only speakers present in the
HeightCeleb CSV and writes a normal `wav/idxxxxx/*.wav` tree that the existing
VocalMorph external-speaker preparation script can consume.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HeightCeleb speakers from VoxCeleb WebDataset tars.")
    parser.add_argument("--heightceleb-csv", required=True)
    parser.add_argument("--tar", action="append", required=True, help="WebDataset tar shard. Can be repeated.")
    parser.add_argument("--output-root", required=True, help="Destination wav root, e.g. .../VoxCeleb1WebDataset/wav")
    parser.add_argument("--split", default="TRAIN", help="HeightCeleb split to keep, or ALL.")
    parser.add_argument("--max-clips-per-speaker", type=int, default=40)
    parser.add_argument("--report", default="data/external/heightceleb_webdataset_extract_report.json")
    return parser.parse_args()


def _target_speakers(heightceleb_csv: Path, split: str) -> Set[str]:
    with heightceleb_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"VoxCeleb1 ID", "split"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise RuntimeError(f"HeightCeleb CSV is missing columns: {sorted(missing)}")
    split_upper = str(split).upper()
    speakers = set()
    for row in rows:
        if split_upper != "ALL" and str(row.get("split", "")).upper() != split_upper:
            continue
        speaker_id = str(row.get("VoxCeleb1 ID", "")).strip()
        if speaker_id:
            speakers.add(speaker_id)
    return speakers


def _existing_counts(output_root: Path, target_speakers: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for speaker_id in target_speakers:
        speaker_dir = output_root / speaker_id
        if speaker_dir.is_dir():
            counts[speaker_id] = sum(1 for path in speaker_dir.rglob("*.wav") if path.is_file())
    return counts


def _sample_key(name: str) -> str:
    return PurePosixPath(name).stem


def _read_labels(tar: tarfile.TarFile, members: list[tarfile.TarInfo]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for member in members:
        if not member.isfile() or not member.name.lower().endswith(".json"):
            continue
        extracted = tar.extractfile(member)
        if extracted is None:
            continue
        try:
            payload = json.loads(extracted.read().decode("utf-8", errors="replace"))
        except Exception:
            continue
        speaker_id = str(payload.get("speakerid", "") or payload.get("speaker_id", "")).strip()
        if speaker_id:
            labels[_sample_key(member.name)] = speaker_id
    return labels


def _extract_tar(
    tar_path: Path,
    *,
    output_root: Path,
    target_speakers: Set[str],
    counts: Counter[str],
    max_clips_per_speaker: int,
) -> Dict[str, int]:
    stats = Counter()
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        labels = _read_labels(tar, members)
        stats["json_labels"] = len(labels)
        for member in members:
            if not member.isfile() or not member.name.lower().endswith(".wav"):
                continue
            stats["wav_seen"] += 1
            speaker_id = labels.get(_sample_key(member.name), "")
            if speaker_id not in target_speakers:
                continue
            if counts[speaker_id] >= max_clips_per_speaker:
                stats["speaker_full"] += 1
                continue
            out_dir = output_root / speaker_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / PurePosixPath(member.name).name
            if out_path.exists():
                counts[speaker_id] += 1
                stats["already_exists"] += 1
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            with out_path.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            counts[speaker_id] += 1
            stats["wav_extracted"] += 1
    return dict(stats)


def main() -> int:
    args = parse_args()
    heightceleb_csv = Path(args.heightceleb_csv).resolve()
    output_root = Path(args.output_root).resolve()
    report_path = Path(args.report)
    target_speakers = _target_speakers(heightceleb_csv, args.split)
    output_root.mkdir(parents=True, exist_ok=True)

    counts = _existing_counts(output_root, target_speakers)
    tar_reports = []
    for raw_tar in args.tar:
        tar_path = Path(raw_tar).resolve()
        if not tar_path.is_file():
            raise FileNotFoundError(f"Missing tar shard: {tar_path}")
        before = sum(counts.values())
        stats = _extract_tar(
            tar_path,
            output_root=output_root,
            target_speakers=target_speakers,
            counts=counts,
            max_clips_per_speaker=int(args.max_clips_per_speaker),
        )
        after = sum(counts.values())
        tar_reports.append(
            {
                "tar": str(tar_path),
                "added_clips": after - before,
                "stats": stats,
            }
        )
        complete = sum(1 for speaker_id in target_speakers if counts[speaker_id] >= int(args.max_clips_per_speaker))
        print(
            f"[webdataset-extract] {tar_path.name}: added={after - before} "
            f"clips_total={after} complete_speakers={complete}/{len(target_speakers)}"
        )

    complete_speakers = sum(1 for speaker_id in target_speakers if counts[speaker_id] >= int(args.max_clips_per_speaker))
    speakers_with_clips = sum(1 for speaker_id in target_speakers if counts[speaker_id] > 0)
    report = {
        "heightceleb_csv": str(heightceleb_csv),
        "output_root": str(output_root),
        "target_speakers": len(target_speakers),
        "speakers_with_clips": speakers_with_clips,
        "complete_speakers": complete_speakers,
        "clips_total": int(sum(counts.values())),
        "max_clips_per_speaker": int(args.max_clips_per_speaker),
        "remaining_without_clips": sorted(speaker_id for speaker_id in target_speakers if counts[speaker_id] == 0)[:200],
        "tars": tar_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[webdataset-extract] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
