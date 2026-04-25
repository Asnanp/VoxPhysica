#!/usr/bin/env python
"""
Prepare NISP dataset into train/val/test feature .npz files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.audio_enhancement import MicrophoneEnhancementConfig
from src.preprocessing.feature_extractor import FeatureConfig, extract_all_features, load_audio

warnings.filterwarnings("ignore")


LANGUAGE_DIRS = {
    "Hindi": ("Hindi_master", ["Hindi", "English_Hindi"]),
    "Kannada": ("Kannada_master", ["Kannada", "English_Kannada"]),
    "Malayalam": ("Malayalam_master", ["Malayalam", "English_Malayalam"]),
    "Tamil": ("Tamil_master", ["Tamil", "English_Tamil"]),
    "Telugu": ("Telugu_master", ["Telugu", "English_Telugu"]),
}


def parse_spkrinfo(spkrinfo_path: str) -> pd.DataFrame:
    rows = []
    with open(spkrinfo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                rows.append(
                    {
                        "speaker_id": parts[0],
                        "gender": 1 if parts[1].lower() == "male" else 0,
                        "mother_tongue": parts[2],
                        "height_cm": float(parts[3]),
                        "weight_kg": float(parts[6]),
                        "age": float(parts[7]),
                    }
                )
            except (ValueError, IndexError) as e:
                warnings.warn(f"Skipping malformed row: {line} | {e}")

    df = pd.DataFrame(rows)
    print(f"[NISP] Parsed {len(df)} speakers")
    return df


def parse_split_file(split_path: str) -> set:
    ids = set()
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def find_audio_files(
    nisp_dir: str,
    speaker_id: str,
    mother_tongue: str,
    use_english: bool = True,
    use_native: bool = True,
    max_files: int = 5,
) -> List[Tuple[str, str]]:
    found = []
    nisp_path = Path(nisp_dir)

    lang_info = LANGUAGE_DIRS.get(mother_tongue)
    if lang_info is None:
        warnings.warn(f"Unknown language for speaker {speaker_id}")
        return []

    master_dir, subdirs = lang_info
    for subdir in subdirs:
        is_english = "English" in subdir
        if is_english and not use_english:
            continue
        if (not is_english) and not use_native:
            continue

        recs_path = nisp_path / master_dir / subdir / "RECS"
        if not recs_path.exists():
            recs_path = nisp_path / master_dir / subdir
            if not recs_path.exists():
                continue

        spk_files = list(recs_path.glob(f"{speaker_id}*.wav"))
        spk_files += list(recs_path.glob(f"{speaker_id.lower()}*.wav"))

        spk_subdir = recs_path / speaker_id
        if spk_subdir.exists():
            spk_files += list(spk_subdir.glob("*.wav"))

        rec_type = "english" if is_english else "native"
        found.extend((str(p), rec_type) for p in spk_files)

    found_english = [(p, t) for p, t in found if t == "english"]
    found_native = [(p, t) for p, t in found if t == "native"]

    result = found_english[:max_files]
    remaining = max_files - len(result)
    if remaining > 0:
        result += found_native[:remaining]

    return result


def extract_speaker_features(
    speaker_row: dict,
    nisp_dir: str,
    config: FeatureConfig,
    use_english: bool = True,
    use_native: bool = True,
    max_files_per_speaker: int = 5,
    enhance_audio: bool = True,
) -> list:
    audio_files = find_audio_files(
        nisp_dir=nisp_dir,
        speaker_id=speaker_row["speaker_id"],
        mother_tongue=speaker_row.get("mother_tongue", "Hindi"),
        use_english=use_english,
        use_native=use_native,
        max_files=max_files_per_speaker,
    )

    if not audio_files:
        return []

    results = []
    enhancement_cfg = MicrophoneEnhancementConfig(enabled=enhance_audio)
    for audio_path, rec_type in audio_files:
        audio = load_audio(
            audio_path,
            config.sample_rate,
            max_duration=10.0,
            min_duration=1.5,
            enhance=enhance_audio,
            enhancement_config=enhancement_cfg,
        )
        if audio is None:
            continue

        feats = extract_all_features(audio, config)
        feats["speaker_id"] = speaker_row["speaker_id"]
        feats["rec_type"] = rec_type
        feats["source"] = "NISP"
        feats["height_cm"] = speaker_row["height_cm"]
        feats["weight_kg"] = speaker_row["weight_kg"]
        feats["age"] = speaker_row["age"]
        feats["gender"] = speaker_row["gender"]
        results.append(feats)

    return results


def prepare_nisp(
    nisp_dir: str,
    output_dir: str,
    use_english: bool = True,
    use_native: bool = False,
    max_per_speaker: int = 5,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    enhance_audio: bool = True,
):
    nisp_dir = os.path.expandvars(nisp_dir)
    print(f"NISP dir: {nisp_dir}")
    print(f"Output dir: {output_dir}")
    print(f"English={use_english} Native={use_native} max_per_speaker={max_per_speaker}")

    spkrinfo_path = os.path.join(nisp_dir, "total_spkrinfo.list")
    if not os.path.exists(spkrinfo_path):
        raise FileNotFoundError(f"total_spkrinfo.list not found: {spkrinfo_path}")

    df = parse_spkrinfo(spkrinfo_path)

    train_ids, test_ids = set(), set()
    train_path = os.path.join(nisp_dir, "train_spkrID")
    test_path = os.path.join(nisp_dir, "test_spkrID")
    if os.path.exists(train_path):
        train_ids = parse_split_file(train_path)
    if os.path.exists(test_path):
        test_ids = parse_split_file(test_path)

    train_out = os.path.join(output_dir, "train")
    val_out = os.path.join(output_dir, "val")
    test_out = os.path.join(output_dir, "test")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    config = FeatureConfig(sample_rate=sample_rate, n_mfcc=n_mfcc)

    processed_train = processed_val = processed_test = 0
    skipped = 0
    stats_vals: Dict[str, List[float]] = {"height": [], "weight": [], "age": []}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing speakers"):
        speaker_id = row["speaker_id"]
        if speaker_id in test_ids:
            split, out_dir = "test", test_out
        elif speaker_id in train_ids:
            split, out_dir = "train", train_out
        else:
            split, out_dir = "val", val_out

        feats_list = extract_speaker_features(
            speaker_row=row.to_dict(),
            nisp_dir=nisp_dir,
            config=config,
            use_english=use_english,
            use_native=use_native,
            max_files_per_speaker=max_per_speaker,
            enhance_audio=enhance_audio,
        )

        if not feats_list:
            skipped += 1
            continue

        for i, feats in enumerate(feats_list):
            file_id = f"{speaker_id}_{i:02d}"
            out_path = os.path.join(out_dir, f"{file_id}.npz")
            np.savez(
                out_path,
                sequence=feats["sequence"],
                f0_mean=np.float32(feats["f0_mean"]),
                formant_spacing_mean=np.float32(feats["formant_spacing_mean"]),
                vtl_mean=np.float32(feats["vtl_mean"]),
                jitter=np.float32(feats["jitter"]),
                shimmer=np.float32(feats["shimmer"]),
                hnr=np.float32(feats["hnr"]),
                height_cm=np.float32(feats["height_cm"]),
                weight_kg=np.float32(feats["weight_kg"]),
                age=np.float32(feats["age"]),
                gender=np.int64(feats["gender"]),
                speaker_id=feats["speaker_id"],
                rec_type=feats["rec_type"],
                source="NISP",
            )

        if split == "train":
            processed_train += 1
            stats_vals["height"].append(float(row["height_cm"]))
            stats_vals["weight"].append(float(row["weight_kg"]))
            stats_vals["age"].append(float(row["age"]))
        elif split == "val":
            processed_val += 1
        else:
            processed_test += 1

    stats = {}
    for key, vals in stats_vals.items():
        arr = np.array(vals, dtype=np.float32)
        stats[key] = {"mean": float(arr.mean()), "std": float(arr.std() + 1e-9)}

    stats_path = os.path.join(output_dir, "target_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nNISP preparation complete")
    print(f"Train speakers processed: {processed_train}")
    print(f"Val speakers processed:   {processed_val}")
    print(f"Test speakers processed:  {processed_test}")
    print(f"Skipped (no audio):       {skipped}")
    print(f"Stats saved: {stats_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare NISP dataset for VocalMorph")
    parser.add_argument("--nisp_dir", type=str, default="data/NISP-Dataset")
    parser.add_argument("--output_dir", type=str, default="data/features")
    parser.add_argument("--use_english", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_native", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_per_speaker", type=int, default=5)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mfcc", type=int, default=40)
    parser.add_argument("--disable_enhancement", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_nisp(
        nisp_dir=args.nisp_dir,
        output_dir=args.output_dir,
        use_english=args.use_english,
        use_native=args.use_native,
        max_per_speaker=args.max_per_speaker,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        enhance_audio=not args.disable_enhancement,
    )
