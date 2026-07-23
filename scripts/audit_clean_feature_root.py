#!/usr/bin/env python
"""Audit and create a cleaned feature root without touching originals."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and hardlink a cleaned feature root.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--original-train-only",
        action="store_true",
        help="Drop static augmented train copies. Val/test are never augmented by this script.",
    )
    parser.add_argument("--expected-dim", type=int, default=None)
    parser.add_argument("--min-height-cm", type=float, default=135.0)
    parser.add_argument("--max-height-cm", type=float, default=205.0)
    parser.add_argument("--min-duration-s", type=float, default=1.45)
    parser.add_argument("--min-capture-quality", type=float, default=0.50)
    parser.add_argument("--min-speech-ratio", type=float, default=0.20)
    parser.add_argument("--max-clipped-ratio", type=float, default=0.20)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.shape == ():
            return str(value.item())
        return str(value.reshape(-1)[0])
    return str(value)


def safe_float(data: Mapping[str, Any], key: str) -> float:
    if key not in data:
        return float("nan")
    try:
        value = data[key]
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return float("nan")
            return float(value.item() if value.shape == () else value.reshape(-1)[0])
        return float(value)
    except Exception:
        return float("nan")


def height_bin(height: float) -> str:
    if not math.isfinite(height):
        return "unknown"
    if height < 160.0:
        return "short"
    if height < 175.0:
        return "medium"
    return "tall"


def is_augmented(data: Mapping[str, Any], path: Path) -> bool:
    if "is_augmented" in data:
        value = safe_float(data, "is_augmented")
        if math.isfinite(value):
            return bool(int(value))
    if "augmentation_tag" in data:
        tag = decode(data["augmentation_tag"]).strip().lower()
        if tag and tag != "original":
            return True
    return "_aug" in path.stem.lower()


def feature_files(split_dir: Path) -> List[Path]:
    if not split_dir.is_dir():
        return []
    return sorted(path for path in split_dir.iterdir() if path.suffix.lower() == ".npz")


def load_record(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    reasons: List[str] = []
    try:
        with np.load(path, allow_pickle=True) as data:
            sequence = np.asarray(data["sequence"], dtype=np.float32) if "sequence" in data else np.asarray([])
            height = safe_float(data, "height_cm")
            duration = safe_float(data, "duration_s")
            quality = safe_float(data, "capture_quality_score")
            speech_ratio = safe_float(data, "speech_ratio")
            clipped = safe_float(data, "clipped_ratio")
            f0 = safe_float(data, "f0_mean")
            spacing = safe_float(data, "formant_spacing_mean")
            vtl = safe_float(data, "vtl_mean")
            source = decode(data["source"]).upper() if "source" in data else path.name.split("_", 1)[0].upper()
            speaker_id = decode(data["speaker_id"]).strip() if "speaker_id" in data else path.stem.rsplit("_", 1)[0]
            gender = safe_float(data, "gender")
            aug = is_augmented(data, path)
    except Exception as exc:
        return {}, [f"load_error:{type(exc).__name__}"]

    record = {
        "path": str(path),
        "name": path.name,
        "speaker_id": speaker_id,
        "source": source,
        "height_cm": float(height),
        "height_bin": height_bin(height),
        "gender": int(gender) if math.isfinite(gender) else -1,
        "duration_s": float(duration) if math.isfinite(duration) else float("nan"),
        "capture_quality_score": float(quality) if math.isfinite(quality) else float("nan"),
        "speech_ratio": float(speech_ratio) if math.isfinite(speech_ratio) else float("nan"),
        "clipped_ratio": float(clipped) if math.isfinite(clipped) else float("nan"),
        "f0_mean": float(f0) if math.isfinite(f0) else float("nan"),
        "formant_spacing_mean": float(spacing) if math.isfinite(spacing) else float("nan"),
        "vtl_mean": float(vtl) if math.isfinite(vtl) else float("nan"),
        "is_augmented": bool(aug),
        "sequence_shape": list(sequence.shape),
        "sequence_dim": int(sequence.shape[1]) if sequence.ndim == 2 else 0,
        "sequence_frames": int(sequence.shape[0]) if sequence.ndim == 2 else 0,
        "sequence_nonfinite": int(np.size(sequence) - np.isfinite(sequence).sum()) if sequence.size else 0,
    }
    return record, reasons


def reject_reasons(record: Mapping[str, Any], split: str, args: argparse.Namespace) -> List[str]:
    reasons: List[str] = []
    if split == "train" and args.original_train_only and bool(record.get("is_augmented")):
        reasons.append("static_augmented_train_copy")

    shape = record.get("sequence_shape") or []
    if len(shape) != 2 or int(record.get("sequence_frames", 0)) <= 0:
        reasons.append("bad_sequence_shape")
    if args.expected_dim is not None and int(record.get("sequence_dim", 0)) != int(args.expected_dim):
        reasons.append("wrong_sequence_dim")
    if int(record.get("sequence_nonfinite", 0)) > 0:
        reasons.append("nonfinite_sequence")

    height = float(record.get("height_cm", float("nan")))
    if not math.isfinite(height) or height < args.min_height_cm or height > args.max_height_cm:
        reasons.append("bad_height")

    duration = float(record.get("duration_s", float("nan")))
    if math.isfinite(duration) and duration < args.min_duration_s:
        reasons.append("too_short_duration")

    quality = float(record.get("capture_quality_score", float("nan")))
    if math.isfinite(quality) and quality < args.min_capture_quality:
        reasons.append("low_capture_quality")

    speech_ratio = float(record.get("speech_ratio", float("nan")))
    if math.isfinite(speech_ratio) and speech_ratio < args.min_speech_ratio:
        reasons.append("low_speech_ratio")

    clipped = float(record.get("clipped_ratio", float("nan")))
    if math.isfinite(clipped) and clipped > args.max_clipped_ratio:
        reasons.append("too_clipped")
    return reasons


def summarize_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    source_counts = Counter(str(r.get("source", "UNKNOWN")) for r in records)
    bin_counts = Counter(str(r.get("height_bin", "unknown")) for r in records)
    gender_counts = Counter(str(r.get("gender", -1)) for r in records)
    speaker_counts = Counter(str(r.get("speaker_id", "")) for r in records)
    by_source_bin_gender = Counter(
        (str(r.get("source", "UNKNOWN")), str(r.get("height_bin", "unknown")), str(r.get("gender", -1)))
        for r in records
    )
    return {
        "files": len(records),
        "speakers": len([speaker for speaker in speaker_counts if speaker]),
        "source_counts": dict(sorted(source_counts.items())),
        "height_bin_counts": dict(sorted(bin_counts.items())),
        "gender_counts": dict(sorted(gender_counts.items())),
        "source_bin_gender_counts": {
            "|".join(key): value for key, value in sorted(by_source_bin_gender.items())
        },
    }


def target_stats(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    tracks: Dict[str, List[float]] = {"height": [], "age": [], "weight": []}
    for record in records:
        path = Path(str(record["path"]))
        with np.load(path, allow_pickle=True) as data:
            height = safe_float(data, "height_cm")
            age = safe_float(data, "age")
            weight = safe_float(data, "weight_kg")
            source = decode(data["source"]).upper() if "source" in data else ""
        if math.isfinite(height):
            tracks["height"].append(float(height))
        if math.isfinite(age):
            tracks["age"].append(float(age))
        if math.isfinite(weight) and source == "NISP":
            tracks["weight"].append(float(weight))

    def summary(values: List[float], default_mean: float = 0.0, default_std: float = 1.0):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return {"mean": default_mean, "std": default_std}
        return {"mean": float(arr.mean()), "std": float(arr.std() + 1e-9)}

    return {
        "height": summary(tracks["height"]),
        "age": summary(tracks["age"]),
        "weight": summary(tracks["weight"]),
    }


def prepare_output(output_root: Path, allow_overwrite: bool) -> None:
    resolved = output_root.resolve()
    expected_data = (ROOT / "data").resolve()
    if resolved.parent != expected_data:
        raise SystemExit(f"Refusing output outside data/: {resolved}")
    if output_root.exists():
        if not allow_overwrite:
            raise SystemExit(f"Output exists; pass --allow-overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        (output_root / split).mkdir(parents=True, exist_ok=False)


def link_file(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Feature Cleaning Report",
        "",
        f"- Input root: `{report['input_root']}`",
        f"- Output root: `{report['output_root']}`",
        f"- Original train only: `{report['policy']['original_train_only']}`",
        "",
        "## Counts",
    ]
    for split in ("train", "val", "test"):
        split_report = report["splits"][split]
        lines.append(
            f"- `{split}`: kept={split_report['kept']['files']} files, "
            f"rejected={split_report['rejected_files']} files, "
            f"speakers={split_report['kept']['speakers']}"
        )
        lines.append(f"  - Kept bins: `{json.dumps(split_report['kept']['height_bin_counts'], sort_keys=True)}`")
        if split_report["reject_reasons"]:
            lines.append(f"  - Reject reasons: `{json.dumps(split_report['reject_reasons'], sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- This script does not touch original feature files.",
            "- Val/test are kept honest; cleaning is intended for the training root.",
            "- The main removed population is static augmented train copies, which were overweighting repeated speakers and height/source/gender priors.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = resolve(args.input_root)
    output_root = resolve(args.output_root)
    if not input_root.is_dir():
        raise SystemExit(f"Missing input root: {input_root}")

    prepare_output(output_root, args.allow_overwrite)
    report: Dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "policy": {
            "original_train_only": bool(args.original_train_only),
            "expected_dim": args.expected_dim,
            "min_height_cm": args.min_height_cm,
            "max_height_cm": args.max_height_cm,
            "min_duration_s": args.min_duration_s,
            "min_capture_quality": args.min_capture_quality,
            "min_speech_ratio": args.min_speech_ratio,
            "max_clipped_ratio": args.max_clipped_ratio,
        },
        "splits": {},
    }

    kept_train_records: List[Mapping[str, Any]] = []
    for split in ("train", "val", "test"):
        kept: List[Dict[str, Any]] = []
        rejected_files = 0
        reject_counter: Counter[str] = Counter()
        rejected_examples: Dict[str, List[str]] = defaultdict(list)
        for path in feature_files(input_root / split):
            record, load_reasons = load_record(path)
            reasons = list(load_reasons)
            if record:
                reasons.extend(reject_reasons(record, split, args))
            if reasons:
                rejected_files += 1
                for reason in reasons:
                    reject_counter[reason] += 1
                    if len(rejected_examples[reason]) < 10:
                        rejected_examples[reason].append(path.name)
                continue
            kept.append(record)
            link_file(path, output_root / split / path.name)

        if split == "train":
            kept_train_records = kept
        report["splits"][split] = {
            "input_files": len(feature_files(input_root / split)),
            "rejected_files": rejected_files,
            "reject_reasons": dict(sorted(reject_counter.items())),
            "rejected_examples": {k: v for k, v in sorted(rejected_examples.items())},
            "kept": summarize_records(kept),
        }

    stats = target_stats(kept_train_records)
    (output_root / "target_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    report["target_stats"] = stats

    for name in (
        "feature_contract.json",
        "feature_diagnostics.json",
        "build_manifest.json",
        "vtl_repair_summary.json",
        "ssl_fusion_manifest.json",
    ):
        src = input_root / name
        if src.exists():
            shutil.copy2(src, output_root / name)

    report_path = output_root / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(output_root / "cleaning_report.md", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
