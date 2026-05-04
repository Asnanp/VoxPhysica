#!/usr/bin/env python
"""Prepare train-only external speakers for VocalMorph feature building.

Expected CSV columns:
  speaker_id,height_cm,gender,audio_paths

Optional columns:
  source,weight_kg,age,audio_dir,audio_glob

`audio_paths` is a pipe-separated list. Paths may be absolute, relative to
--audio-root, or relative to the repository root. External speakers are appended
only to the train manifest; validation/test remain sealed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.audit_utils import height_bin, read_csv_rows  # noqa: E402


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}
CANONICAL_FIELDS = ["speaker_id", "source", "gender", "height_cm", "weight_kg", "age", "audio_paths"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append vetted external speakers to train manifest.")
    parser.add_argument("--external-csv", default="data/external/celebrity_speakers.csv")
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--base-train-csv", default="data/splits/train_clean.csv")
    parser.add_argument("--val-csv", default="data/splits/val_clean.csv")
    parser.add_argument("--test-csv", default="data/splits/test_clean.csv")
    parser.add_argument("--output-train-csv", default="data/splits/train_plus_external.csv")
    parser.add_argument("--report", default="data/splits/train_plus_external_report.json")
    parser.add_argument("--source-name", default="CELEB")
    parser.add_argument("--min-clips", type=int, default=2)
    parser.add_argument("--max-clips", type=int, default=40)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--allow-train-overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: str | os.PathLike[str], *, audio_root: Optional[Path] = None) -> Path:
    raw = Path(str(path).strip().strip('"'))
    if raw.is_absolute():
        return raw
    if audio_root is not None:
        candidate = audio_root / raw
        if candidate.exists():
            return candidate
    return ROOT / raw


def _template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
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
        writer.writerow(
            {
                "speaker_id": "CELEB_example_001",
                "height_cm": "170.0",
                "gender": "Female",
                "age": "",
                "weight_kg": "",
                "audio_paths": "C:/path/to/clip1.wav|C:/path/to/clip2.wav",
                "audio_dir": "",
                "audio_glob": "",
                "source": "CELEB",
            }
        )


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(str(value).strip())
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clean_gender(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"m", "male", "1"}:
        return "Male"
    if text in {"f", "female", "0"}:
        return "Female"
    raise ValueError(f"gender must be Male/Female, got {value!r}")


def _repo_or_abs(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
        return str(rel).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def _paths_from_row(row: Mapping[str, str], *, audio_root: Optional[Path], max_clips: int) -> List[str]:
    raw_paths: List[str] = []
    audio_paths = str(row.get("audio_paths", "") or "").strip()
    if audio_paths:
        raw_paths.extend(part.strip() for part in audio_paths.split("|") if part.strip())

    audio_dir = str(row.get("audio_dir", "") or "").strip()
    if audio_dir:
        base = _resolve(audio_dir, audio_root=audio_root)
        if base.is_dir():
            for child in sorted(base.rglob("*")):
                if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS:
                    raw_paths.append(str(child))

    audio_glob = str(row.get("audio_glob", "") or "").strip()
    if audio_glob:
        pattern = str(_resolve(audio_glob, audio_root=audio_root))
        raw_paths.extend(sorted(glob.glob(pattern, recursive=True)))

    seen: set[str] = set()
    out: List[str] = []
    for raw in raw_paths:
        path = _resolve(raw, audio_root=audio_root)
        key = str(path.resolve()).lower() if path.exists() else str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            out.append(_repo_or_abs(path))
        if max_clips > 0 and len(out) >= max_clips:
            break
    return out


def _speaker_ids(rows: Sequence[Mapping[str, str]]) -> set[str]:
    return {str(row.get("speaker_id", "") or "").strip() for row in rows if str(row.get("speaker_id", "") or "").strip()}


def _load_external_rows(
    path: Path,
    *,
    audio_root: Optional[Path],
    source_name: str,
    min_clips: int,
    max_clips: int,
) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    rows = read_csv_rows(str(path))
    prepared: List[Dict[str, str]] = []
    skipped: Counter[str] = Counter()
    warnings: List[str] = []

    for row_idx, row in enumerate(rows, start=2):
        raw_sid = str(row.get("speaker_id", "") or "").strip()
        if not raw_sid:
            skipped["missing_speaker_id"] += 1
            continue
        speaker_id = raw_sid if raw_sid.upper().startswith("CELEB_") else f"CELEB_{raw_sid}"
        height_cm = _safe_float(row.get("height_cm"))
        if not (120.0 <= height_cm <= 230.0):
            skipped["bad_height"] += 1
            warnings.append(f"row {row_idx}: skipped {speaker_id}, height_cm={row.get('height_cm')!r}")
            continue
        try:
            gender = _clean_gender(row.get("gender"))
        except ValueError as exc:
            skipped["bad_gender"] += 1
            warnings.append(f"row {row_idx}: skipped {speaker_id}, {exc}")
            continue
        paths = _paths_from_row(row, audio_root=audio_root, max_clips=max_clips)
        if len(paths) < int(min_clips):
            skipped["too_few_audio_clips"] += 1
            warnings.append(f"row {row_idx}: skipped {speaker_id}, clips={len(paths)}")
            continue
        prepared.append(
            {
                "speaker_id": speaker_id,
                "source": str(row.get("source", "") or source_name).strip().upper() or source_name.upper(),
                "gender": gender,
                "height_cm": f"{height_cm:.4f}",
                "weight_kg": str(row.get("weight_kg", "") or "").strip(),
                "age": str(row.get("age", "") or "").strip(),
                "audio_paths": "|".join(paths),
            }
        )

    duplicate_ids = [sid for sid, count in Counter(row["speaker_id"] for row in prepared).items() if count > 1]
    if duplicate_ids:
        raise RuntimeError(f"Duplicate external speaker_id values: {duplicate_ids[:10]}")

    report = {
        "input_rows": len(rows),
        "prepared_rows": len(prepared),
        "skipped": dict(sorted(skipped.items())),
        "warnings": warnings[:200],
        "height_bins": dict(sorted(Counter(height_bin(float(row["height_cm"])) for row in prepared).items())),
        "gender_counts": dict(sorted(Counter(row["gender"] for row in prepared).items())),
    }
    return prepared, report


def main() -> int:
    args = parse_args()
    external_csv = _resolve(args.external_csv)
    if not external_csv.exists():
        _template(external_csv)
        print(f"[external-speakers] Created template: {external_csv}")
        print("[external-speakers] Fill it with real train-only speakers, then rerun this command.")
        return 0

    audio_root = _resolve(args.audio_root) if args.audio_root else None
    base_train_path = _resolve(args.base_train_csv)
    val_path = _resolve(args.val_csv)
    test_path = _resolve(args.test_csv)
    output_train_path = _resolve(args.output_train_csv)
    report_path = _resolve(args.report)

    base_train = read_csv_rows(str(base_train_path))
    val_rows = read_csv_rows(str(val_path))
    test_rows = read_csv_rows(str(test_path))
    external_rows, external_report = _load_external_rows(
        external_csv,
        audio_root=audio_root,
        source_name=str(args.source_name),
        min_clips=int(args.min_clips),
        max_clips=int(args.max_clips),
    )
    if not external_rows and not args.allow_empty:
        raise RuntimeError(
            "No valid external speakers were prepared. Fill the external CSV with "
            "real audio paths, or pass --allow-empty only for a dry plumbing check."
        )

    train_ids = _speaker_ids(base_train)
    sealed_ids = _speaker_ids(val_rows) | _speaker_ids(test_rows)
    external_ids = _speaker_ids(external_rows)
    sealed_overlap = sorted(external_ids & sealed_ids)
    if sealed_overlap:
        raise RuntimeError(
            "External speaker overlap with sealed validation/test split: "
            + ", ".join(sealed_overlap[:20])
        )
    train_overlap = sorted(external_ids & train_ids)
    if train_overlap and not args.allow_train_overwrite:
        raise RuntimeError(
            "External speaker overlap with existing train split. "
            "Rename/remove duplicates or pass --allow-train-overwrite: "
            + ", ".join(train_overlap[:20])
        )

    if train_overlap and args.allow_train_overwrite:
        base_train = [row for row in base_train if str(row.get("speaker_id", "") or "").strip() not in set(train_overlap)]

    output_train_path.parent.mkdir(parents=True, exist_ok=True)
    combined = list(base_train) + external_rows
    with output_train_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        for row in combined:
            writer.writerow({field: row.get(field, "") for field in CANONICAL_FIELDS})

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_csv": str(external_csv),
        "base_train_csv": str(base_train_path),
        "output_train_csv": str(output_train_path),
        "val_csv": str(val_path),
        "test_csv": str(test_path),
        "base_train_speakers": len(train_ids),
        "external_speakers": len(external_rows),
        "combined_train_speakers": len(_speaker_ids(combined)),
        "external": external_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[external-speakers] wrote {output_train_path}")
    print(f"[external-speakers] external speakers={len(external_rows)} combined train={len(_speaker_ids(combined))}")
    print(f"[external-speakers] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
