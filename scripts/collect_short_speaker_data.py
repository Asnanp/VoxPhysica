#!/usr/bin/env python
"""Build VoxPhysica's quality-controlled short-speaker support manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.short_data_collection import (  # noqa: E402
    AudioQCPolicy,
    CollectionResult,
    SPEAKER_OUTPUT_FIELDS,
    CLIP_OUTPUT_FIELDS,
    assert_unique_and_disjoint,
    assert_unique_audio_hashes,
    build_train_support_manifest,
    collect_consented_measured,
    collect_public_heightceleb,
    read_csv_rows,
    sha256_file,
    summarize,
    write_csv_rows,
    write_markdown_report,
    CANONICAL_FIELDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect real short-speaker support from licensed public metadata "
            "and optional consented measured participants."
        )
    )
    parser.add_argument(
        "--public-csv",
        default="data/external/celebrity_speakers.csv",
        help="HeightCeleb/VoxCeleb speaker manifest already available locally.",
    )
    parser.add_argument(
        "--consented-csv",
        default=None,
        help="Optional pseudonymous participant intake CSV.",
    )
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Optional root for relative audio paths.",
    )
    parser.add_argument(
        "--short-threshold-cm",
        type=float,
        default=160.0,
        help="Exclusive height threshold for the strict short group.",
    )
    parser.add_argument("--min-clips", type=int, default=5)
    parser.add_argument("--max-clips", type=int, default=40)
    parser.add_argument(
        "--base-train-csv",
        default="data/splits/train_clean.csv",
    )
    parser.add_argument("--val-csv", default="data/splits/val_clean.csv")
    parser.add_argument("--test-csv", default="data/splits/test_clean.csv")
    parser.add_argument(
        "--output-dir",
        default="outputs/short_speaker_collection_v1",
    )
    parser.add_argument(
        "--train-output",
        default="data/splits/train_plus_short_support.csv",
        help="Canonical train manifest with accepted development/train-only support.",
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=30.0,
    )
    return parser.parse_args()


def empty_result() -> CollectionResult:
    return CollectionResult([], [], [], [])


def resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> int:
    args = parse_args()
    if not (130.0 <= float(args.short_threshold_cm) <= 175.0):
        raise ValueError("--short-threshold-cm must be between 130 and 175")
    if int(args.min_clips) < 1:
        raise ValueError("--min-clips must be positive")
    if int(args.max_clips) and int(args.max_clips) < int(args.min_clips):
        raise ValueError("--max-clips must be zero or at least --min-clips")

    public_csv = resolve(args.public_csv)
    consented_csv = resolve(args.consented_csv)
    audio_root = resolve(args.audio_root)
    base_train_path = resolve(args.base_train_csv)
    val_path = resolve(args.val_csv)
    test_path = resolve(args.test_csv)
    output_dir = resolve(args.output_dir)
    train_output = resolve(args.train_output)
    assert public_csv is not None
    assert base_train_path is not None
    assert val_path is not None
    assert test_path is not None
    assert output_dir is not None
    assert train_output is not None

    for required in (public_csv, base_train_path, val_path, test_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    policy = AudioQCPolicy(
        min_duration_s=float(args.min_duration_s),
        max_duration_s=float(args.max_duration_s),
    )
    print(
        f"[short-data] auditing public support below "
        f"{float(args.short_threshold_cm):.1f} cm",
        flush=True,
    )
    public = collect_public_heightceleb(
        public_csv,
        repo_root=ROOT,
        audio_root=audio_root,
        short_threshold_cm=float(args.short_threshold_cm),
        min_clips=int(args.min_clips),
        max_clips=int(args.max_clips),
        policy=policy,
    )

    consented = empty_result()
    if consented_csv is not None:
        if not consented_csv.is_file():
            raise FileNotFoundError(consented_csv)
        print(f"[short-data] auditing consented intake {consented_csv}", flush=True)
        consented = collect_consented_measured(
            consented_csv,
            repo_root=ROOT,
            audio_root=audio_root,
            short_threshold_cm=float(args.short_threshold_cm),
            min_clips=int(args.min_clips),
            max_clips=int(args.max_clips),
            policy=policy,
        )

    train_rows = read_csv_rows(base_train_path)
    val_rows = read_csv_rows(val_path)
    test_rows = read_csv_rows(test_path)
    speakers = public.speakers + consented.speakers
    clips = public.clips + consented.clips

    assert_unique_and_disjoint(
        speakers,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
    )
    assert_unique_audio_hashes(clips)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "short_speakers_accepted.csv",
        speakers,
        SPEAKER_OUTPUT_FIELDS,
    )
    write_csv_rows(
        output_dir / "short_clips_accepted.csv",
        clips,
        CLIP_OUTPUT_FIELDS,
    )
    rejected_speakers = public.rejected_speakers + consented.rejected_speakers
    rejected_clips = public.rejected_clips + consented.rejected_clips
    rejection_speaker_fields = sorted(
        {key for row in rejected_speakers for key in row} or {"speaker_id", "reason"}
    )
    write_csv_rows(
        output_dir / "short_speakers_rejected.csv",
        rejected_speakers,
        rejection_speaker_fields,
    )
    write_csv_rows(
        output_dir / "short_clips_rejected.csv",
        rejected_clips,
        CLIP_OUTPUT_FIELDS,
    )

    combined_train = build_train_support_manifest(train_rows, speakers)
    write_csv_rows(train_output, combined_train, CANONICAL_FIELDS)

    hashes = {
        "public_csv": sha256_file(public_csv),
        "base_train_csv": sha256_file(base_train_path),
        "validation_csv": sha256_file(val_path),
        "historical_test_csv": sha256_file(test_path),
    }
    if consented_csv is not None:
        hashes["consented_csv"] = sha256_file(consented_csv)

    report = summarize(
        train_rows=train_rows,
        public=public,
        consented=consented,
        short_threshold_cm=float(args.short_threshold_cm),
        policy=policy,
        input_hashes=hashes,
    )
    report["outputs"] = {
        "speaker_manifest": str(output_dir / "short_speakers_accepted.csv"),
        "clip_manifest": str(output_dir / "short_clips_accepted.csv"),
        "train_manifest": str(train_output),
    }
    report["combined_train_speakers"] = len(combined_train)

    report_path = output_dir / "short_data_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(output_dir / "SHORT_DATA_REPORT.md", report)

    print(
        f"[short-data] accepted speakers={len(speakers)} clips={len(clips)} "
        f"combined_train={len(combined_train)}",
        flush=True,
    )
    print(f"[short-data] report={output_dir / 'SHORT_DATA_REPORT.md'}", flush=True)
    print(f"[short-data] train manifest={train_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
