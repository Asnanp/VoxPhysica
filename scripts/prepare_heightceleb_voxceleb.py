#!/usr/bin/env python
"""Create a VocalMorph external-speaker CSV from HeightCeleb + local VoxCeleb audio.

HeightCeleb provides VoxCeleb1 speaker IDs and heights. It does not contain the
VoxCeleb audio files themselves, so this script expects a local VoxCeleb audio
root such as a directory containing `wav/id10001/.../*.wav`.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HeightCeleb/VoxCeleb rows for train-only ingestion.")
    parser.add_argument("--voxceleb-root", required=True)
    parser.add_argument("--heightceleb-csv", default=None)
    parser.add_argument("--output-csv", default="data/external/celebrity_speakers.csv")
    parser.add_argument("--split", default="TRAIN", help="HeightCeleb split to use, or ALL.")
    parser.add_argument("--min-clips", type=int, default=2)
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--max-speakers", type=int, default=0)
    parser.add_argument("--report", default="data/external/heightceleb_prepare_report.json")
    return parser.parse_args()


def _download_heightceleb() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="stachu86/HeightCeleb",
        filename="height_celeb.csv",
        repo_type="dataset",
    )


def _speaker_dir_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    common_roots = [
        root,
        root / "wav",
        root / "vox1_dev_wav" / "wav",
        root / "vox1_test_wav" / "wav",
        root / "vox1" / "wav",
        root / "vox1" / "vox1_dev_wav" / "wav",
        root / "vox1" / "vox1_test_wav" / "wav",
        root / "voxceleb1" / "wav",
        root / "VoxCeleb1" / "wav",
        root / "VoxCeleb1" / "vox1_dev_wav" / "wav",
        root / "VoxCeleb1" / "vox1_test_wav" / "wav",
    ]
    for base in common_roots:
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and child.name.startswith("id"):
                index.setdefault(child.name, child)

    # Some mirrors/extractions add an extra wrapper folder. Keep scanning after
    # common roots so test-only and dev-only folders can be indexed together.
    for child in root.rglob("id*"):
        if child.is_dir() and child.name.startswith("id"):
            index.setdefault(child.name, child)
    return index


def _audio_files(speaker_dir: Path, max_clips: int) -> List[str]:
    files = [
        path
        for path in sorted(speaker_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if max_clips > 0:
        files = files[:max_clips]
    return [str(path.resolve()) for path in files]


def _gender(sex: str) -> str:
    return "Male" if str(sex).strip().upper().startswith("M") else "Female"


def main() -> int:
    args = parse_args()
    voxceleb_root = Path(args.voxceleb_root).resolve()
    if not voxceleb_root.is_dir():
        raise FileNotFoundError(f"VoxCeleb root not found: {voxceleb_root}")

    heightceleb_csv = args.heightceleb_csv or _download_heightceleb()
    df = pd.read_csv(heightceleb_csv)
    required = {"VoxCeleb1 ID", "height", "sex", "split"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"HeightCeleb CSV is missing columns: {sorted(missing)}")
    if str(args.split).upper() != "ALL":
        df = df[df["split"].astype(str).str.upper() == str(args.split).upper()].copy()

    speaker_dirs = _speaker_dir_index(voxceleb_root)
    rows = []
    skipped = Counter()
    for _, row in df.iterrows():
        speaker_id = str(row["VoxCeleb1 ID"]).strip()
        speaker_dir = speaker_dirs.get(speaker_id)
        if speaker_dir is None:
            skipped["missing_speaker_dir"] += 1
            continue
        clips = _audio_files(speaker_dir, int(args.max_clips))
        if len(clips) < int(args.min_clips):
            skipped["too_few_clips"] += 1
            continue
        rows.append(
            {
                "speaker_id": f"CELEB_{speaker_id}",
                "height_cm": f"{float(row['height']):.4f}",
                "gender": _gender(str(row["sex"])),
                "age": "",
                "weight_kg": "",
                "audio_paths": "|".join(clips),
                "audio_dir": "",
                "audio_glob": "",
                "source": "CELEB",
            }
        )
        if int(args.max_speakers) > 0 and len(rows) >= int(args.max_speakers):
            break

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "speaker_id",
                "height_cm",
                "gender",
                "age",
                "weight_kg",
                "audio_paths",
                "audio_dir",
                "audio_glob",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "heightceleb_csv": str(heightceleb_csv),
        "voxceleb_root": str(voxceleb_root),
        "output_csv": str(output_csv.resolve()),
        "rows": len(rows),
        "skipped": dict(sorted(skipped.items())),
        "gender_counts": dict(sorted(Counter(row["gender"] for row in rows).items())),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[heightceleb] wrote {output_csv} speakers={len(rows)} skipped={dict(skipped)}")
    print(f"[heightceleb] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
