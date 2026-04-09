"""
VocalMorph - Dataset Cleaning Script (enhanced)

Goal: produce a "10/10" quality speaker table from unified_dataset.csv.

Cleaning steps (speaker-level):
    1. Drop speakers with missing or non-numeric age
    2. Enforce numeric ranges for height, weight, age
    3. Flag and remove weight outliers (< WEIGHT_MIN or > WEIGHT_MAX)
    4. Remove TIMIT SA1/SA2 sentences from audio paths (same script for all speakers)
    5. Verify gender vs filename consistency (NISP only, warn only)
    6. Drop speakers with too few usable utterances (MIN_UTTERANCES)
    7. Optional: z-score based outlier removal for height/weight/age
"""

import argparse
import csv
import math
import os
from typing import Dict, List, Tuple

INPUT_CSV_DEFAULT = "data/unified_dataset.csv"
OUTPUT_CSV_DEFAULT = "data/cleaned_dataset.csv"

# ─────────────────────────────────────────────
# Thresholds (can be overridden via CLI)
# ─────────────────────────────────────────────
WEIGHT_MIN = 40.0   # kg — below this is likely an error
WEIGHT_MAX = 115.0  # kg — above this is extreme but implausible in this corpus
HEIGHT_MIN = 140.0  # cm
HEIGHT_MAX = 210.0  # cm
AGE_MIN = 18.0      # yrs
AGE_MAX = 75.0      # yrs

MIN_UTTERANCES = 5  # drop speakers with fewer usable WAVs

# Z-score outlier removal (speaker-level)
ENABLE_Z_OUTLIERS = True
Z_THRESHOLD = 3.0


def is_sa_sentence(path: str) -> bool:
    """Returns True if the WAV file is a TIMIT SA sentence (SA1 or SA2)."""
    filename = os.path.basename(path).upper()
    return filename in ("SA1.WAV", "SA2.WAV")


def infer_gender_from_path_nisp(path: str) -> str:
    """
    Extract gender from NISP filename.
    e.g. Hin_0069_Eng_m_0000.wav → Male
         Mal_0028_Mal_f_0001.wav → Female
    """
    fname = os.path.basename(path)
    parts = fname.replace(".wav", "").split("_")
    # Format: {Lang}_{ID}_{ContentLang}_{gender}_{utter}
    if len(parts) >= 4:
        g = parts[3].lower()
        if g == "m":
            return "Male"
        if g == "f":
            return "Female"
    return "Unknown"


def safe_float(value: str):
    try:
        return float(value)
    except Exception:
        return math.nan


def z_stats(values: List[float]) -> Tuple[float, float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return mean, math.sqrt(var)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VocalMorph dataset cleaner (enhanced)")
    parser.add_argument("--input_csv", default=INPUT_CSV_DEFAULT, help="Input unified CSV")
    parser.add_argument("--output_csv", default=OUTPUT_CSV_DEFAULT, help="Output cleaned CSV")
    parser.add_argument("--min_utterances", type=int, default=MIN_UTTERANCES)
    parser.add_argument("--height_min", type=float, default=HEIGHT_MIN)
    parser.add_argument("--height_max", type=float, default=HEIGHT_MAX)
    parser.add_argument("--weight_min", type=float, default=WEIGHT_MIN)
    parser.add_argument("--weight_max", type=float, default=WEIGHT_MAX)
    parser.add_argument("--age_min", type=float, default=AGE_MIN)
    parser.add_argument("--age_max", type=float, default=AGE_MAX)
    parser.add_argument("--disable_z_outliers", action="store_true",
                        help="Disable z-score based outlier removal")
    parser.add_argument("--z_threshold", type=float, default=Z_THRESHOLD,
                        help="Z-score threshold for outlier removal")
    return parser.parse_args()


def initial_clean(rows: List[Dict], args: argparse.Namespace):
    """First-pass cleaning: structural filters + range checks."""
    cleaned: List[Dict] = []
    removed: List[Tuple[str, str]] = []

    print("=== VocalMorph Dataset Cleaner (pass 1: structural + ranges) ===\n")
    print(f"Input speakers: {len(rows)}\n")

    for row in rows:
        speaker_id = row.get("speaker_id", "")
        source = row.get("source", "")
        gender = row.get("gender", "")
        age_str = str(row.get("age", "")).strip()
        weight_str = str(row.get("weight_kg", "")).strip()
        height_str = str(row.get("height_cm", "")).strip()
        audio_str = row.get("audio_paths", "") or ""

        # 1) Age: must exist and be numeric + in range
        if not age_str:
            removed.append((speaker_id, "missing age"))
            print(f"[DROP] {speaker_id} — missing age")
            continue
        age_val = safe_float(age_str)
        if math.isnan(age_val):
            removed.append((speaker_id, f"non-numeric age: {age_str!r}"))
            print(f"[DROP] {speaker_id} — non-numeric age {age_str!r}")
            continue
        if not (args.age_min <= age_val <= args.age_max):
            removed.append((speaker_id, f"age {age_val} outside [{args.age_min}, {args.age_max}]"))
            print(f"[DROP] {speaker_id} — age {age_val} outside safe range")
            continue

        # 2) Height: must exist and be numeric + in range
        if not height_str:
            removed.append((speaker_id, "missing height"))
            print(f"[DROP] {speaker_id} — missing height")
            continue
        height_val = safe_float(height_str)
        if math.isnan(height_val):
            removed.append((speaker_id, f"non-numeric height: {height_str!r}"))
            print(f"[DROP] {speaker_id} — non-numeric height {height_str!r}")
            continue
        if not (args.height_min <= height_val <= args.height_max):
            removed.append((speaker_id, f"height {height_val} outside [{args.height_min}, {args.height_max}]"))
            print(f"[DROP] {speaker_id} — height {height_val} outside safe range")
            continue

        # 3) Weight: only enforced for rows that have it (NISP)
        weight_val = math.nan
        if weight_str:
            weight_val = safe_float(weight_str)
            if math.isnan(weight_val):
                removed.append((speaker_id, f"non-numeric weight: {weight_str!r}"))
                print(f"[DROP] {speaker_id} — non-numeric weight {weight_str!r}")
                continue
            if weight_val < args.weight_min:
                removed.append((speaker_id, f"weight {weight_val}kg < {args.weight_min}kg"))
                print(f"[DROP] {speaker_id} — weight {weight_val}kg is below threshold")
                continue
            if weight_val > args.weight_max:
                removed.append((speaker_id, f"weight {weight_val}kg > {args.weight_max}kg"))
                print(f"[DROP] {speaker_id} — weight {weight_val}kg is above threshold")
                continue

        # 4) Remove TIMIT SA1/SA2 sentences
        paths = [p for p in audio_str.split("|") if p.strip()]
        if source == "TIMIT":
            before = len(paths)
            paths = [p for p in paths if not is_sa_sentence(p)]
            after = len(paths)
            if before != after:
                print(f"[INFO] {speaker_id} — removed {before - after} SA1/SA2 sentences")

        # 5) Gender vs filename consistency check (NISP only, warn only)
        if source == "NISP" and paths:
            file_gender = infer_gender_from_path_nisp(paths[0])
            if file_gender != "Unknown" and file_gender != gender:
                print(
                    f"[WARN] {speaker_id} — metadata gender '{gender}' "
                    f"vs filename gender '{file_gender}' — keeping metadata value"
                )

        # 6) Minimum utterances check
        if len(paths) < args.min_utterances:
            removed.append((speaker_id, f"only {len(paths)} utterances after cleaning"))
            print(f"[DROP] {speaker_id} — only {len(paths)} usable utterances")
            continue

        row["age"] = str(int(round(age_val)))
        row["height_cm"] = f"{height_val:.2f}"
        if not math.isnan(weight_val):
            row["weight_kg"] = f"{weight_val:.2f}"
        row["audio_paths"] = "|".join(paths)
        cleaned.append(row)

    return cleaned, removed


def z_score_filter(rows: List[Dict], args: argparse.Namespace):
    """Optional second-pass: drop extreme z-score outliers for height/weight/age."""
    if args.disable_z_outliers or not rows:
        return rows, []

    print("\n=== VocalMorph Dataset Cleaner (pass 2: z-score outliers) ===")

    heights = [safe_float(r.get("height_cm", "")) for r in rows]
    ages = [safe_float(r.get("age", "")) for r in rows]
    weights = [safe_float(r.get("weight_kg", "")) for r in rows]

    h_mean, h_std = z_stats(heights)
    a_mean, a_std = z_stats(ages)
    w_mean, w_std = z_stats(weights)

    print(
        f"Height mean={h_mean:.2f} std={h_std:.2f} | "
        f"Age mean={a_mean:.2f} std={a_std:.2f} | "
        f"Weight mean={w_mean:.2f} std={w_std:.2f}"
    )

    kept: List[Dict] = []
    removed: List[Tuple[str, str]] = []

    for r in rows:
        sid = r.get("speaker_id", "")
        h = safe_float(r.get("height_cm", ""))
        a = safe_float(r.get("age", ""))
        w = safe_float(r.get("weight_kg", ""))

        drop_reason = None

        if h_std > 0 and not math.isnan(h):
            z_h = abs((h - h_mean) / h_std)
            if z_h > args.z_threshold:
                drop_reason = f"height z={z_h:.2f} > {args.z_threshold}"

        if drop_reason is None and a_std > 0 and not math.isnan(a):
            z_a = abs((a - a_mean) / a_std)
            if z_a > args.z_threshold:
                drop_reason = f"age z={z_a:.2f} > {args.z_threshold}"

        if drop_reason is None and w_std > 0 and not math.isnan(w):
            z_w = abs((w - w_mean) / w_std)
            if z_w > args.z_threshold:
                drop_reason = f"weight z={z_w:.2f} > {args.z_threshold}"

        if drop_reason:
            removed.append((sid, drop_reason))
            print(f"[DROP] {sid} — {drop_reason}")
        else:
            kept.append(r)

    return kept, removed


def summarize(rows: List[Dict], removed: List[Tuple[str, str]], input_count: int, output_csv: str):
    fieldnames = ["speaker_id", "source", "gender",
                  "height_cm", "weight_kg", "age", "audio_paths"]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    nisp_count = sum(1 for r in rows if r.get("source") == "NISP")
    timit_count = sum(1 for r in rows if r.get("source") == "TIMIT")

    heights = [safe_float(r.get("height_cm", "")) for r in rows]
    heights = [h for h in heights if not math.isnan(h)]
    weights = [safe_float(r.get("weight_kg", "")) for r in rows if r.get("weight_kg")]
    weights = [w for w in weights if not math.isnan(w)]
    ages = [safe_float(r.get("age", "")) for r in rows]
    ages = [a for a in ages if not math.isnan(a)]

    print(f"\n{'=' * 55}")
    print(f"✅ Cleaned dataset saved → {output_csv}")
    print(f"   Input speakers  : {input_count}")
    print(f"   Removed         : {len(removed)}")
    print(f"   Final speakers  : {len(rows)}")
    print(f"   NISP            : {nisp_count}")
    print(f"   TIMIT           : {timit_count}")

    if heights:
        print(
            f"\n📊 Final Stats (speaker-level):\n"
            f"   Height → min: {min(heights):.1f} | max: {max(heights):.1f} | "
            f"mean: {sum(heights) / len(heights):.1f} cm"
        )
    if weights:
        print(
            f"   Weight → min: {min(weights):.1f} | max: {max(weights):.1f} | "
            f"mean: {sum(weights) / len(weights):.1f} kg"
        )
    if ages:
        print(
            f"   Age    → min: {min(ages):.0f} | max: {max(ages):.0f} | "
            f"mean: {sum(ages) / len(ages):.1f}"
        )

    if removed:
        print(f"\n🗑️  Removed speakers (sample up to 30):")
        for sid, reason in removed[:30]:
            print(f"   {sid}: {reason}")


def main():
    args = parse_args()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cleaned_1, removed_1 = initial_clean(rows, args)
    cleaned_2, removed_2 = z_score_filter(cleaned_1, args)

    all_removed = removed_1 + removed_2
    summarize(cleaned_2, all_removed, input_count=len(rows), output_csv=args.output_csv)


if __name__ == "__main__":
    main()