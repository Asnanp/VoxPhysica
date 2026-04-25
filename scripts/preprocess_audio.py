#!/usr/bin/env python
"""
VocalMorph audio preprocessing pipeline.

Reads:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test.csv

Processes every file in `audio_paths` with this exact order:
  1) Resample to 16kHz
  2) RMS normalize to -20 dBFS (target RMS=0.1)
  3) VAD trim (webrtcvad, 30ms frames, aggressiveness=2)
  4) Duration filtering after VAD:
       <1.5s -> drop
       >10s  -> truncate to first 10s
       else  -> keep

Writes:
  data/audio_clean/{train,val,test}/... (mirrors original folder structure)
  data/splits/audio_clean_manifest.csv
  data/splits/dropped_audio.csv
  data/splits/train_clean.csv
  data/splits/val_clean.csv
  data/splits/test_clean.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import webrtcvad
except ImportError as exc:
    raise ImportError(
        "webrtcvad is required. Install with: pip install webrtcvad>=2.0.10"
    ) from exc


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGET_SR = 16000
TARGET_RMS = 10 ** (-20.0 / 20.0)  # 0.1
VAD_FRAME_MS = 30
VAD_AGGRESSIVENESS = 2
FLAG_SHORT_VOICED_SEC = 1.0
MIN_KEEP_SEC = 1.5
MAX_KEEP_SEC = 10.0

DEFAULT_SPLITS_DIR = "data/splits"
DEFAULT_CLEAN_ROOT = "data/audio_clean"


@dataclass
class FileResult:
    speaker_id: str
    source: str
    original_path: str
    clean_path: str
    original_duration: float
    clean_duration: float
    vad_duration: float
    dropped: bool
    truncated: bool
    flagged_short_voiced: bool
    reason: str = ""


@dataclass
class SourceAgg:
    processed: int = 0
    dropped: int = 0
    truncated: int = 0
    flagged_short_voiced: int = 0
    before_durations: List[float] = field(default_factory=list)
    after_vad_durations: List[float] = field(default_factory=list)
    after_final_durations: List[float] = field(default_factory=list)


@dataclass
class RunStats:
    processed: int = 0
    dropped: int = 0
    truncated: int = 0
    flagged_short_voiced: int = 0
    before_durations: List[float] = field(default_factory=list)
    after_vad_durations: List[float] = field(default_factory=list)
    after_final_durations: List[float] = field(default_factory=list)
    per_source: Dict[str, SourceAgg] = field(default_factory=lambda: defaultdict(SourceAgg))

    def update(self, r: FileResult):
        src = r.source or "Unknown"
        src_agg = self.per_source[src]

        self.processed += 1
        src_agg.processed += 1

        self.before_durations.append(r.original_duration)
        self.after_vad_durations.append(r.vad_duration)
        src_agg.before_durations.append(r.original_duration)
        src_agg.after_vad_durations.append(r.vad_duration)

        if r.flagged_short_voiced:
            self.flagged_short_voiced += 1
            src_agg.flagged_short_voiced += 1

        if r.truncated:
            self.truncated += 1
            src_agg.truncated += 1

        if r.dropped:
            self.dropped += 1
            src_agg.dropped += 1
        else:
            self.after_final_durations.append(r.clean_duration)
            src_agg.after_final_durations.append(r.clean_duration)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess VocalMorph audio from split CSVs")
    p.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR)
    p.add_argument("--clean_root", default=DEFAULT_CLEAN_ROOT)
    p.add_argument("--max_files", type=int, default=None, help="Optional debug limit")
    return p.parse_args()


def resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(ROOT, path_str)


def to_rel_from_root(path_str: str) -> str:
    if not os.path.isabs(path_str):
        return path_str.replace("\\", "/")

    try:
        rel = os.path.relpath(path_str, ROOT)
        if not rel.startswith(".."):
            return rel.replace("\\", "/")
    except Exception:
        pass

    drive, tail = os.path.splitdrive(path_str)
    drive_token = drive.replace(":", "") if drive else "abs"
    tail = tail.lstrip("\\/")
    return os.path.join("_abs", drive_token, tail).replace("\\", "/")


def parse_audio_paths(audio_paths: str) -> List[str]:
    if not audio_paths:
        return []
    return [p.strip() for p in audio_paths.split("|") if p.strip()]


def read_split_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_split_csv(path: str, rows: List[dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return audio.astype(np.float32, copy=False)
    return librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)


def rms_normalize(audio: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    scale = TARGET_RMS / (rms + 1e-9)
    out = audio * scale
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def vad_trim(audio: np.ndarray, sr: int, vad: webrtcvad.Vad) -> np.ndarray:
    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError(f"Unsupported sample rate for WebRTC VAD: {sr}")

    frame_len = int(sr * (VAD_FRAME_MS / 1000.0))
    if frame_len <= 0 or len(audio) < frame_len:
        return np.array([], dtype=np.float32)

    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)

    voiced_chunks: List[np.ndarray] = []
    for start in range(0, len(pcm_i16) - frame_len + 1, frame_len):
        frame = pcm_i16[start : start + frame_len]
        if vad.is_speech(frame.tobytes(), sr):
            voiced_chunks.append(frame)

    if not voiced_chunks:
        return np.array([], dtype=np.float32)

    trimmed_i16 = np.concatenate(voiced_chunks)
    return (trimmed_i16.astype(np.float32) / 32768.0).astype(np.float32)


def seconds(num_samples: int, sr: int = TARGET_SR) -> float:
    if sr <= 0:
        return 0.0
    return float(num_samples) / float(sr)


def mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def ascii_hist(values: List[float], bins: int = 12, width: int = 48) -> str:
    if not values:
        return "  (no kept files)"

    vmin = min(values)
    vmax = max(values)
    if np.isclose(vmin, vmax):
        return f"  [{vmin:.2f}-{vmax:.2f}] {'#' * width} {len(values)}"

    edges = np.linspace(vmin, vmax, bins + 1)
    counts = np.zeros(bins, dtype=int)
    for v in values:
        idx = np.searchsorted(edges, v, side="right") - 1
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1

    max_count = int(np.max(counts)) if len(counts) else 1
    lines = []
    for i, c in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        bar_len = int((c / max_count) * width) if max_count > 0 else 0
        lines.append(f"  [{lo:5.2f}-{hi:5.2f}] {'#' * bar_len:<{width}} {int(c)}")
    return "\n".join(lines)


def process_file(
    speaker_id: str,
    source: str,
    split_name: str,
    original_path: str,
    vad: webrtcvad.Vad,
) -> FileResult:
    abs_in = resolve_path(original_path)
    rel_original = to_rel_from_root(original_path)
    rel_clean = os.path.join("data", "audio_clean", split_name, rel_original).replace("\\", "/")
    abs_out = os.path.join(ROOT, rel_clean)

    if not os.path.exists(abs_in):
        return FileResult(
            speaker_id=speaker_id,
            source=source,
            original_path=original_path,
            clean_path="",
            original_duration=0.0,
            clean_duration=0.0,
            vad_duration=0.0,
            dropped=True,
            truncated=False,
            flagged_short_voiced=True,
            reason="file_not_found",
        )

    try:
        audio, sr = load_audio(abs_in)
    except Exception as e:
        return FileResult(
            speaker_id=speaker_id,
            source=source,
            original_path=original_path,
            clean_path="",
            original_duration=0.0,
            clean_duration=0.0,
            vad_duration=0.0,
            dropped=True,
            truncated=False,
            flagged_short_voiced=True,
            reason=f"read_error:{e.__class__.__name__}",
        )

    original_duration = seconds(len(audio), sr)

    # STEP 1 - resample to 16k
    audio = resample_to_16k(audio, sr=sr)

    # STEP 2 - RMS normalize to -20 dBFS (target RMS=0.1)
    audio = rms_normalize(audio)

    # STEP 3 - VAD silence trimming (voiced-only)
    voiced = vad_trim(audio, sr=TARGET_SR, vad=vad)
    vad_duration = seconds(len(voiced), TARGET_SR)
    flagged_short_voiced = vad_duration < FLAG_SHORT_VOICED_SEC

    # STEP 4 - duration filter after VAD
    if vad_duration < MIN_KEEP_SEC:
        reason = "<1.5s_after_vad"
        if flagged_short_voiced:
            reason = "<1.0s_voiced_after_vad"
        return FileResult(
            speaker_id=speaker_id,
            source=source,
            original_path=original_path,
            clean_path="",
            original_duration=original_duration,
            clean_duration=vad_duration,
            vad_duration=vad_duration,
            dropped=True,
            truncated=False,
            flagged_short_voiced=flagged_short_voiced,
            reason=reason,
        )

    truncated = False
    if vad_duration > MAX_KEEP_SEC:
        truncated = True
        max_samples = int(MAX_KEEP_SEC * TARGET_SR)
        voiced = voiced[:max_samples]

    clean_duration = seconds(len(voiced), TARGET_SR)

    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    sf.write(abs_out, voiced, TARGET_SR, subtype="PCM_16")

    return FileResult(
        speaker_id=speaker_id,
        source=source,
        original_path=original_path,
        clean_path=rel_clean,
        original_duration=original_duration,
        clean_duration=clean_duration,
        vad_duration=vad_duration,
        dropped=False,
        truncated=truncated,
        flagged_short_voiced=flagged_short_voiced,
        reason="",
    )


def process_split(
    split_name: str,
    split_csv_abs: str,
    vad: webrtcvad.Vad,
    stats: RunStats,
    manifest_rows: List[dict],
    dropped_rows: List[dict],
    max_files: Optional[int] = None,
) -> List[dict]:
    rows = read_split_csv(split_csv_abs)
    if not rows:
        return []

    out_rows: List[dict] = []
    files_seen = 0

    for row in tqdm(rows, desc=f"{split_name} rows"):
        speaker_id = row.get("speaker_id", "")
        source = row.get("source", "Unknown")
        audio_paths = parse_audio_paths(row.get("audio_paths", ""))

        clean_paths: List[str] = []
        for p in audio_paths:
            if max_files is not None and files_seen >= max_files:
                break
            files_seen += 1

            r = process_file(
                speaker_id=speaker_id,
                source=source,
                split_name=split_name,
                original_path=p,
                vad=vad,
            )
            stats.update(r)

            manifest_rows.append(
                {
                    "speaker_id": r.speaker_id,
                    "source": r.source,
                    "original_path": r.original_path,
                    "clean_path": r.clean_path,
                    "original_duration": f"{r.original_duration:.6f}",
                    "clean_duration": f"{r.clean_duration:.6f}",
                    "dropped": str(bool(r.dropped)),
                }
            )

            if r.dropped:
                dropped_rows.append(
                    {
                        "speaker_id": r.speaker_id,
                        "path": r.original_path,
                        "reason": r.reason,
                        "duration": f"{r.clean_duration:.6f}",
                    }
                )
            else:
                clean_paths.append(r.clean_path)

        if max_files is not None and files_seen >= max_files:
            # Stop collecting more files, but still emit row with processed subset.
            pass

        new_row = dict(row)
        new_row["audio_paths"] = "|".join(clean_paths)
        out_rows.append(new_row)

        if max_files is not None and files_seen >= max_files:
            break

    return out_rows


def print_stats(stats: RunStats):
    print("\n=== VocalMorph Audio Preprocess Stats ===")
    print(f"Total files processed : {stats.processed}")
    print(f"Total dropped (<1.5s): {stats.dropped}")
    print(f"Total truncated (>10s): {stats.truncated}")
    print(f"Flagged voiced <1.0s : {stats.flagged_short_voiced}")
    print(f"Mean duration before  : {mean_or_zero(stats.before_durations):.3f}s")
    print(f"Mean duration after VAD: {mean_or_zero(stats.after_vad_durations):.3f}s")
    print(f"Mean final kept duration: {mean_or_zero(stats.after_final_durations):.3f}s")

    print("\nPer-source stats:")
    for src in sorted(stats.per_source.keys()):
        agg = stats.per_source[src]
        print(
            f"  {src:<6} | files={agg.processed:6d} "
            f"dropped={agg.dropped:6d} truncated={agg.truncated:6d} "
            f"mean_before={mean_or_zero(agg.before_durations):.3f}s "
            f"mean_after_vad={mean_or_zero(agg.after_vad_durations):.3f}s "
            f"mean_final={mean_or_zero(agg.after_final_durations):.3f}s"
        )

    print("\nDuration distribution histogram (kept files):")
    print(ascii_hist(stats.after_final_durations, bins=12, width=52))


def main():
    args = parse_args()

    splits_dir_abs = resolve_path(args.splits_dir)
    clean_root_abs = resolve_path(args.clean_root)
    os.makedirs(clean_root_abs, exist_ok=True)

    split_to_csv = {
        "train": os.path.join(splits_dir_abs, "train.csv"),
        "val": os.path.join(splits_dir_abs, "val.csv"),
        "test": os.path.join(splits_dir_abs, "test.csv"),
    }

    for split_name, p in split_to_csv.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split CSV: {p}")

    vad = webrtcvad.Vad()
    vad.set_mode(VAD_AGGRESSIVENESS)

    stats = RunStats()
    manifest_rows: List[dict] = []
    dropped_rows: List[dict] = []

    split_clean_rows: Dict[str, List[dict]] = {}
    split_fieldnames: Dict[str, List[str]] = {}

    for split_name in ["train", "val", "test"]:
        csv_path = split_to_csv[split_name]
        rows = read_split_csv(csv_path)
        split_fieldnames[split_name] = list(rows[0].keys()) if rows else []
        split_clean_rows[split_name] = process_split(
            split_name=split_name,
            split_csv_abs=csv_path,
            vad=vad,
            stats=stats,
            manifest_rows=manifest_rows,
            dropped_rows=dropped_rows,
            max_files=args.max_files,
        )

    manifest_path = os.path.join(splits_dir_abs, "audio_clean_manifest.csv")
    dropped_path = os.path.join(splits_dir_abs, "dropped_audio.csv")
    train_clean_path = os.path.join(splits_dir_abs, "train_clean.csv")
    val_clean_path = os.path.join(splits_dir_abs, "val_clean.csv")
    test_clean_path = os.path.join(splits_dir_abs, "test_clean.csv")

    write_split_csv(
        manifest_path,
        manifest_rows,
        fieldnames=[
            "speaker_id",
            "source",
            "original_path",
            "clean_path",
            "original_duration",
            "clean_duration",
            "dropped",
        ],
    )

    write_split_csv(
        dropped_path,
        dropped_rows,
        fieldnames=["speaker_id", "path", "reason", "duration"],
    )

    write_split_csv(train_clean_path, split_clean_rows["train"], split_fieldnames["train"])
    write_split_csv(val_clean_path, split_clean_rows["val"], split_fieldnames["val"])
    write_split_csv(test_clean_path, split_clean_rows["test"], split_fieldnames["test"])

    print_stats(stats)
    print("\nWrote:")
    print(f"  {manifest_path}")
    print(f"  {dropped_path}")
    print(f"  {train_clean_path}")
    print(f"  {val_clean_path}")
    print(f"  {test_clean_path}")


if __name__ == "__main__":
    main()
