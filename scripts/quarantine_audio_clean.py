#!/usr/bin/env python
"""
Safely clean VocalMorph cleaned-audio assets without deleting files.

What it does:
1) Normalizes and de-duplicates `audio_paths` in:
     data/splits/train_clean.csv
     data/splits/val_clean.csv
     data/splits/test_clean.csv
2) Removes missing paths from CSV rows.
3) Moves unreferenced/orphan WAV files from `data/audio_clean` to:
     data/audio_quarantine/orphan/...
4) Optionally detects duplicate-content WAV files (hash-based) and moves extras to:
     data/audio_quarantine/duplicate/...
5) Writes audit CSV:
     data/splits/audio_quarantine_audit.csv

Important:
- Files are MOVED, not deleted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SPLITS_DIR = "data/splits"
DEFAULT_AUDIO_ROOT = "data/audio_clean"
DEFAULT_QUARANTINE_ROOT = "data/audio_quarantine"


@dataclass
class AuditRow:
    reason: str
    original_path: str
    moved_to: str
    details: str = ""


def parse_args():
    p = argparse.ArgumentParser(description="Move unused/duplicate cleaned audio files to quarantine")
    p.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR)
    p.add_argument("--audio_root", default=DEFAULT_AUDIO_ROOT)
    p.add_argument("--quarantine_root", default=DEFAULT_QUARANTINE_ROOT)
    p.add_argument("--check_duplicates", action="store_true", help="Enable content-hash duplicate detection")
    return p.parse_args()


def resolve(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(ROOT, path_str)


def to_rel_root(path_str: str) -> str:
    if not os.path.isabs(path_str):
        return path_str.replace("\\", "/")
    rel = os.path.relpath(path_str, ROOT)
    if rel.startswith(".."):
        drive, tail = os.path.splitdrive(path_str)
        drive = drive.replace(":", "") if drive else "abs"
        tail = tail.lstrip("\\/")
        rel = os.path.join("_abs", drive, tail)
    return rel.replace("\\", "/")


def parse_paths(audio_paths: str) -> List[str]:
    if not audio_paths:
        return []
    return [x.strip().replace("\\", "/") for x in audio_paths.split("|") if x.strip()]


def write_csv(path: str, rows: List[dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def move_to_quarantine(abs_src: str, quarantine_root_abs: str, reason: str) -> str:
    rel = to_rel_root(abs_src)
    abs_dst = os.path.join(quarantine_root_abs, reason, rel)
    os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
    shutil.move(abs_src, abs_dst)
    return abs_dst


def collect_actual_wavs(audio_root_abs: str) -> Set[str]:
    out: Set[str] = set()
    for dp, _, fns in os.walk(audio_root_abs):
        for n in fns:
            if n.lower().endswith(".wav"):
                out.add(os.path.normpath(os.path.join(dp, n)))
    return out


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_split_csvs(
    splits_dir_abs: str,
    split_files: List[str],
) -> Tuple[Set[str], List[AuditRow], Dict[str, Dict[str, int]]]:
    referenced: Set[str] = set()
    audit: List[AuditRow] = []
    stats: Dict[str, Dict[str, int]] = {}

    for fn in split_files:
        p = os.path.join(splits_dir_abs, fn)
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        rows_out: List[dict] = []
        removed_paths = 0
        deduped_paths = 0
        removed_rows = 0

        for r in rows:
            raw = parse_paths(r.get("audio_paths", ""))
            uniq = list(OrderedDict((x, None) for x in raw).keys())
            deduped_paths += max(0, len(raw) - len(uniq))

            valid: List[str] = []
            for x in uniq:
                ax = x if os.path.isabs(x) else os.path.join(ROOT, x)
                ax = os.path.normpath(ax)
                if os.path.exists(ax):
                    valid.append(to_rel_root(x))
                    referenced.add(ax)
                else:
                    removed_paths += 1
                    audit.append(
                        AuditRow(
                            reason="missing_in_csv",
                            original_path=x,
                            moved_to="",
                            details=f"removed_from_{fn}",
                        )
                    )

            if not valid:
                removed_rows += 1
                continue

            r2 = dict(r)
            r2["audio_paths"] = "|".join(valid)
            rows_out.append(r2)

        write_csv(p, rows_out, fieldnames)
        stats[fn] = {
            "rows_before": len(rows),
            "rows_after": len(rows_out),
            "rows_removed": removed_rows,
            "paths_removed": removed_paths,
            "paths_deduped": deduped_paths,
        }

    return referenced, audit, stats


def quarantine_orphans(
    actual_wavs: Set[str],
    referenced_wavs: Set[str],
    quarantine_root_abs: str,
) -> List[AuditRow]:
    audit: List[AuditRow] = []
    orphans = sorted(actual_wavs - referenced_wavs)
    for p in orphans:
        dst = move_to_quarantine(p, quarantine_root_abs, reason="orphan")
        audit.append(
            AuditRow(
                reason="orphan",
                original_path=to_rel_root(p),
                moved_to=to_rel_root(dst),
                details="not_referenced_in_clean_csvs",
            )
        )
    return audit


def quarantine_duplicate_content(
    wavs_to_check: Set[str],
    quarantine_root_abs: str,
) -> List[AuditRow]:
    by_size: Dict[int, List[str]] = defaultdict(list)
    for p in wavs_to_check:
        by_size[os.path.getsize(p)].append(p)

    audit: List[AuditRow] = []
    for group in by_size.values():
        if len(group) < 2:
            continue

        by_hash: Dict[str, List[str]] = defaultdict(list)
        for p in group:
            by_hash[file_md5(p)].append(p)

        for same_hash in by_hash.values():
            if len(same_hash) < 2:
                continue

            same_hash_sorted = sorted(same_hash)
            keep = same_hash_sorted[0]
            for dup in same_hash_sorted[1:]:
                dst = move_to_quarantine(dup, quarantine_root_abs, reason="duplicate")
                audit.append(
                    AuditRow(
                        reason="duplicate",
                        original_path=to_rel_root(dup),
                        moved_to=to_rel_root(dst),
                        details=f"same_content_as:{to_rel_root(keep)}",
                    )
                )
    return audit


def main():
    args = parse_args()
    splits_dir_abs = resolve(args.splits_dir)
    audio_root_abs = resolve(args.audio_root)
    quarantine_root_abs = resolve(args.quarantine_root)
    os.makedirs(quarantine_root_abs, exist_ok=True)

    split_files = ["train_clean.csv", "val_clean.csv", "test_clean.csv"]
    for fn in split_files:
        p = os.path.join(splits_dir_abs, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split CSV: {p}")

    referenced, csv_audit, csv_stats = normalize_split_csvs(splits_dir_abs, split_files)
    actual_before = collect_actual_wavs(audio_root_abs)

    orphan_audit = quarantine_orphans(actual_before, referenced, quarantine_root_abs)

    # Re-scan after orphan moves.
    actual_after_orphans = collect_actual_wavs(audio_root_abs)
    dup_audit: List[AuditRow] = []
    if args.check_duplicates:
        dup_audit = quarantine_duplicate_content(actual_after_orphans, quarantine_root_abs)

    audit_rows = csv_audit + orphan_audit + dup_audit
    audit_csv = os.path.join(splits_dir_abs, "audio_quarantine_audit.csv")
    write_csv(
        audit_csv,
        [
            {
                "reason": a.reason,
                "original_path": a.original_path,
                "moved_to": a.moved_to,
                "details": a.details,
            }
            for a in audit_rows
        ],
        fieldnames=["reason", "original_path", "moved_to", "details"],
    )

    print("=== Audio Quarantine Cleanup ===")
    for fn in split_files:
        st = csv_stats[fn]
        print(
            f"{fn}: rows {st['rows_before']} -> {st['rows_after']} | "
            f"rows_removed={st['rows_removed']} paths_removed={st['paths_removed']} "
            f"paths_deduped={st['paths_deduped']}"
        )
    print(f"Orphan files moved    : {len(orphan_audit)}")
    print(f"Duplicate files moved : {len(dup_audit)}")
    print(f"Audit CSV             : {audit_csv}")
    print(f"Quarantine root       : {quarantine_root_abs}")


if __name__ == "__main__":
    main()

