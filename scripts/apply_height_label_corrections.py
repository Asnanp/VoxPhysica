#!/usr/bin/env python
"""Apply verified height-label corrections to split CSVs.

Fill `corrected_height_cm` and set `decision` to `correct` in the template
created by Phase 5. This script writes new split CSVs and never overwrites the
canonical originals.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply verified height corrections to split CSVs.")
    parser.add_argument("--corrections", default="outputs/phase5_label_domain_audit/height_label_corrections_template.csv")
    parser.add_argument("--train-csv", default="data/splits/train_clean.csv")
    parser.add_argument("--val-csv", default="data/splits/val_clean.csv")
    parser.add_argument("--test-csv", default="data/splits/test_clean.csv")
    parser.add_argument("--output-dir", default="data/splits_phase5_corrected")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corrections: Dict[Tuple[str, str], str] = {}
    for row in read_rows(resolve(args.corrections)):
        decision = str(row.get("decision", "")).strip().lower()
        corrected = str(row.get("corrected_height_cm", "")).strip()
        if decision not in {"correct", "use", "apply"} or not corrected:
            continue
        float(corrected)
        corrections[(str(row.get("split", "")).strip(), str(row.get("speaker_id", "")).strip())] = corrected

    out_dir = resolve(args.output_dir)
    total = 0
    for split, csv_arg in (("train", args.train_csv), ("val", args.val_csv), ("test", args.test_csv)):
        rows = read_rows(resolve(csv_arg))
        fieldnames = list(rows[0].keys()) if rows else ["speaker_id", "source", "gender", "height_cm", "weight_kg", "age", "audio_paths"]
        for row in rows:
            key = (split, str(row.get("speaker_id", "")).strip())
            if key in corrections:
                row["height_cm"] = corrections[key]
                total += 1
        write_rows(out_dir / f"{split}_clean.csv", rows, fieldnames)
    print(f"[phase5] applied {total} verified height corrections")
    print(f"[phase5] wrote corrected splits to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
