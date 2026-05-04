"""
VocalMorph - Dataset Merge Script (v2 - fixed language master scoping)
Merges NISP and TIMIT into a unified CSV for the dataloader.

Output schema:
    speaker_id | source | gender | height_cm | weight_kg | age | audio_paths
"""

import os
import glob
import csv
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG — update these paths to your local setup
# ─────────────────────────────────────────────
NISP_ROOT       = "data/NISP-Dataset"
NISP_SPKRINFO   = os.path.join(NISP_ROOT, "total_spkrinfo.list")

TIMIT_ROOT      = "data/TIMIT"
TIMIT_SPKRINFO  = os.path.join(TIMIT_ROOT, "DOC", "SPKRINFO.TXT")

OUTPUT_CSV      = "data/unified_dataset.csv"

# Map NISP language prefix → master folder name
LANG_MASTER_MAP = {
    "Hin": "Hindi_master",
    "Kan": "Kannada_master",
    "Mal": "Malayalam_master",
    "Tam": "Tamil_master",
    "Tel": "Telugu_master",
}


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def imperial_to_cm(height_str: str) -> float:
    """Convert '5'11"' → 180.34 cm"""
    height_str = height_str.strip().replace('"', '')
    if "'" in height_str:
        parts = height_str.split("'")
        feet = int(parts[0])
        inches = int(parts[1]) if parts[1] else 0
    else:
        feet, inches = int(height_str), 0
    return round((feet * 12 + inches) * 2.54, 2)


def derive_age(rec_date: str, birth_date: str) -> int:
    """Derive age from RecDate and BirthDate (MM/DD/YY format)"""
    try:
        rec  = datetime.strptime(rec_date.strip(),   "%m/%d/%y")
        born = datetime.strptime(birth_date.strip(), "%m/%d/%y")
        # Fix 2-digit year ambiguity (TIMIT dates are 1900s)
        if rec.year  > 2000: rec  = rec.replace(year=rec.year  - 100)
        if born.year > 2000: born = born.replace(year=born.year - 100)
        return (rec - born).days // 365
    except Exception:
        return None


def normalize_gender_nisp(g: str) -> str:
    g = g.strip().lower()
    if g in ("male", "m"):   return "Male"
    if g in ("female", "f"): return "Female"
    return "Unknown"


def normalize_gender_timit(g: str) -> str:
    g = g.strip().upper()
    if g == "M": return "Male"
    if g == "F": return "Female"
    return "Unknown"


# ─────────────────────────────────────────────
# NISP PARSER
# ─────────────────────────────────────────────
def parse_nisp(nisp_root: str, spkrinfo_path: str) -> list[dict]:
    """
    Parse total_spkrinfo.list and glob WAV paths ONLY from the correct
    language master folder for each speaker.

    Expected column order (adjust indices if your file differs):
        Speaker_ID  Gender  Mother_Tongue  Height  Shoulder  Waist  Weight  Age  ...
        e.g.: Hin_0001  Male  Hindi  172.0  38.0  80.0  68.5  24  ...
    """
    records = []

    with open(spkrinfo_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("#") or line.startswith("Speaker_ID"):
                continue

            cols = line.split()
            if len(cols) < 8:
                continue

            try:
                speaker_id = cols[0]           # e.g. Hin_0001
                gender     = normalize_gender_nisp(cols[1])
                # cols[2]  = Mother_Tongue     (skip)
                height_cm  = float(cols[3])
                # cols[4]  = Shoulder          (skip)
                # cols[5]  = Waist             (skip)
                weight_kg  = float(cols[6])
                age        = round(float(cols[7]))  # handles 24.24 → 24
            except (ValueError, IndexError):
                print(f"[NISP] Skipping malformed line: {line}")
                continue

            # ── KEY FIX: scope glob to the speaker's own language master ──
            lang_prefix   = speaker_id.split("_")[0]    # "Hin", "Mal", etc.
            spk_num       = speaker_id.split("_")[1]    # "0001", "0028", etc.
            master_folder = LANG_MASTER_MAP.get(lang_prefix)

            if not master_folder:
                print(f"[NISP] Unknown language prefix for {speaker_id}, skipping.")
                continue

            # Glob both native language and English sub-folders within this master only
            wav_pattern = os.path.join(
                nisp_root, master_folder, "*", "RECS", spk_num, "*.wav"
            )
            audio_paths = glob.glob(wav_pattern, recursive=False)

            if not audio_paths:
                print(f"[NISP] No WAVs found for {speaker_id} in {master_folder}, skipping.")
                continue

            # Normalise to forward slashes for cross-platform consistency
            audio_paths = [p.replace("\\", "/") for p in sorted(audio_paths)]

            records.append({
                "speaker_id":  f"NISP_{speaker_id}",
                "source":      "NISP",
                "gender":      gender,
                "height_cm":   height_cm,
                "weight_kg":   weight_kg,
                "age":         age,
                "audio_paths": "|".join(audio_paths),
            })

    print(f"[NISP] Parsed {len(records)} speakers.")
    return records


# ─────────────────────────────────────────────
# TIMIT PARSER
# ─────────────────────────────────────────────
def parse_timit(timit_root: str, spkrinfo_path: str) -> list[dict]:
    """
    Parse SPKRINFO.TXT and glob WAV paths per speaker.
    Columns: ID  Sex  DR  Use  RecDate  BirthDate  Ht  Race  Edu  [Comments]
    """
    records = []

    with open(spkrinfo_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            cols = line.split()
            if len(cols) < 7:
                continue

            try:
                speaker_id = cols[0]    # e.g. ABC0
                sex        = cols[1]    # M or F
                rec_date   = cols[4]    # MM/DD/YY
                birth_date = cols[5]    # MM/DD/YY
                ht_str     = cols[6]    # e.g. 5'11"
            except IndexError:
                print(f"[TIMIT] Skipping malformed line: {line}")
                continue

            height_cm = imperial_to_cm(ht_str)
            age       = derive_age(rec_date, birth_date)
            gender    = normalize_gender_timit(sex)

            # Speaker folders are named [M/F][INITIALS+DIGIT], e.g. MABC0 or FABC0
            folder_name = f"{sex.upper()}{speaker_id.upper()}"
            wav_pattern = os.path.join(
                timit_root, "*", "*", folder_name, "*.WAV"
            )
            audio_paths = glob.glob(wav_pattern, recursive=False)

            if not audio_paths:
                print(f"[TIMIT] No WAVs found for {speaker_id} ({folder_name}), skipping.")
                continue

            audio_paths = [p.replace("\\", "/") for p in sorted(audio_paths)]

            records.append({
                "speaker_id":  f"TIMIT_{speaker_id}",
                "source":      "TIMIT",
                "gender":      gender,
                "height_cm":   height_cm,
                "weight_kg":   None,        # Not available in TIMIT
                "age":         age,
                "audio_paths": "|".join(audio_paths),
            })

    print(f"[TIMIT] Parsed {len(records)} speakers.")
    return records


# ─────────────────────────────────────────────
# MERGE + SAVE
# ─────────────────────────────────────────────
def merge_and_save(nisp_records, timit_records, output_path: str):
    all_records = nisp_records + timit_records

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["speaker_id", "source", "gender",
                  "height_cm", "weight_kg", "age", "audio_paths"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n✅ Unified dataset saved → {output_path}")
    print(f"   Total speakers : {len(all_records)}")
    print(f"   NISP           : {len(nisp_records)}")
    print(f"   TIMIT          : {len(timit_records)}")

    # Quick stats
    heights = [r["height_cm"] for r in all_records if r["height_cm"]]
    weights = [r["weight_kg"] for r in all_records if r["weight_kg"]]
    ages    = [r["age"]       for r in all_records if r["age"]]

    if heights:
        print(f"\n📊 Stats:")
        print(f"   Height → min: {min(heights):.1f} cm | max: {max(heights):.1f} cm | mean: {sum(heights)/len(heights):.1f} cm")
    if weights:
        print(f"   Weight → min: {min(weights):.1f} kg | max: {max(weights):.1f} kg | mean: {sum(weights)/len(weights):.1f} kg  [NISP only]")
    if ages:
        print(f"   Age    → min: {min(ages)} | max: {max(ages)} | mean: {sum(ages)/len(ages):.1f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== VocalMorph Dataset Merger v2 ===\n")

    nisp_records  = parse_nisp(NISP_ROOT, NISP_SPKRINFO)
    timit_records = parse_timit(TIMIT_ROOT, TIMIT_SPKRINFO)

    merge_and_save(nisp_records, timit_records, OUTPUT_CSV)