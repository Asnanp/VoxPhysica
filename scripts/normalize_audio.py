"""
VocalMorph - Audio Normalization Script

Goals for "10/10" audio quality:
  - All WAVs at a single sample rate (default: 16 kHz)
  - Mono channel
  - Consistent RMS loudness (no extremely quiet or clipping files)
  - Enforce a minimum duration per clip

This script:
  * Reads an input speaker CSV (default: data/cleaned_dataset.csv)
  * Loads each WAV in audio_paths
  * Resamples to target_sr and converts to mono
  * RMS-normalizes to a target RMS
  * Saves to a mirrored directory tree under output_root
  * Writes a new CSV with updated audio_paths pointing to normalized files
  * Drops speakers that end up with too few usable utterances
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
try:
    import librosa
except ImportError:
    librosa = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and resample VocalMorph audio")
    parser.add_argument("--input_csv", default="data/cleaned_dataset.csv",
                        help="Input cleaned CSV with audio_paths")
    parser.add_argument("--output_csv", default="data/cleaned_dataset_normalized.csv",
                        help="Output CSV with normalized audio_paths")
    parser.add_argument("--output_root", default="data/normalized",
                        help="Root directory where normalized WAVs will be written")
    parser.add_argument("--target_sr", type=int, default=16000,
                        help="Target sample rate for all audio")
    parser.add_argument("--target_rms", type=float, default=0.03,
                        help="Target RMS amplitude after normalization (0-1)")
    parser.add_argument("--min_duration", type=float, default=1.0,
                        help="Minimum duration (seconds) for a clip to be kept")
    parser.add_argument("--min_utterances", type=int, default=5,
                        help="Minimum number of clips per speaker to keep the speaker")
    return parser.parse_args()


def ensure_librosa_available():
    if librosa is None:
        raise RuntimeError(
            "librosa is required for resampling but is not installed.\n"
            "Please run: pip install librosa"
        )


def compute_rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float32) ** 2)))


def normalize_rms(y: np.ndarray, target_rms: float) -> np.ndarray:
    rms = compute_rms(y)
    if rms <= 0.0 or target_rms <= 0.0:
        return y
    scale = target_rms / rms
    y_norm = y.astype(np.float32) * scale
    # safety clip
    y_norm = np.clip(y_norm, -1.0, 1.0)
    return y_norm


def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    # average channels
    return np.mean(y, axis=1, dtype=np.float32)


def resample_audio(y: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if sr == target_sr:
        return y, sr
    ensure_librosa_available()
    y_resampled = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return y_resampled.astype(np.float32), target_sr


def make_normalized_path(output_root: str, original_path: str) -> str:
    # Preserve relative layout under output_root; strip leading "data/" if present.
    norm = original_path.replace("\\", "/")
    if norm.startswith("data/"):
        norm = norm[len("data/") :]
    out_path = os.path.join(output_root, norm)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def process_clip(
    in_path: str,
    out_path: str,
    target_sr: int,
    target_rms: float,
    min_duration: float,
) -> bool:
    if not os.path.exists(in_path):
        print(f"[MISS] {in_path} (file not found)")
        return False
    try:
        y, sr = sf.read(in_path, always_2d=False)
    except Exception as e:
        print(f"[ERR] Could not read {in_path}: {e}")
        return False

    y = to_mono(np.asarray(y, dtype=np.float32))
    y, sr_new = resample_audio(y, sr, target_sr)

    duration = len(y) / float(sr_new)
    if duration < min_duration:
        print(f"[SKIP] {in_path} duration {duration:.2f}s < {min_duration}s")
        return False

    y = normalize_rms(y, target_rms)

    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, y, sr_new, subtype="PCM_16")
    except Exception as e:
        print(f"[ERR] Failed to write {out_path}: {e}")
        return False

    return True


def normalize_audio_for_csv(args: argparse.Namespace):
    import csv

    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("=== VocalMorph Audio Normalizer ===")
    print(f"Input CSV   : {args.input_csv}")
    print(f"Output CSV  : {args.output_csv}")
    print(f"Output root : {args.output_root}")
    print(f"Target SR   : {args.target_sr} Hz")
    print(f"Target RMS  : {args.target_rms}")
    print(f"Min duration: {args.min_duration}s\n")

    os.makedirs(args.output_root, exist_ok=True)

    kept_rows: List[Dict] = []
    removed: List[Tuple[str, str]] = []

    for idx, row in enumerate(rows, 1):
        speaker_id = row.get("speaker_id", "")
        audio_str = row.get("audio_paths", "") or ""
        raw_paths = [p for p in audio_str.split("|") if p.strip()]

        if not raw_paths:
            removed.append((speaker_id, "no audio_paths"))
            print(f"[DROP] {speaker_id} — no audio_paths")
            continue

        new_paths: List[str] = []
        for rel in raw_paths:
            in_path = rel
            if not os.path.isabs(in_path):
                in_path = os.path.join(os.getcwd(), rel)

            out_path = make_normalized_path(args.output_root, rel)
            ok = process_clip(
                in_path=in_path,
                out_path=out_path,
                target_sr=args.target_sr,
                target_rms=args.target_rms,
                min_duration=args.min_duration,
            )
            if ok:
                # store path relative to project root
                rel_out = os.path.relpath(out_path, os.getcwd()).replace("\\", "/")
                new_paths.append(rel_out)

        if len(new_paths) < args.min_utterances:
            removed.append(
                (speaker_id, f"only {len(new_paths)} normalized clips (min {args.min_utterances})")
            )
            print(
                f"[DROP] {speaker_id} — only {len(new_paths)} normalized clips "
                f"(min {args.min_utterances})"
            )
            continue

        row["audio_paths"] = "|".join(new_paths)
        kept_rows.append(row)

    fieldnames = rows[0].keys() if rows else []
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print("\n=== Summary ===")
    print(f"Speakers in input : {len(rows)}")
    print(f"Speakers kept     : {len(kept_rows)}")
    print(f"Speakers dropped  : {len(removed)}")
    if removed:
        print("Dropped (sample up to 20):")
        for sid, reason in removed[:20]:
            print(f"  {sid}: {reason}")


def main():
    args = parse_args()
    normalize_audio_for_csv(args)


if __name__ == "__main__":
    main()

