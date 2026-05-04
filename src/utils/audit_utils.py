"""Shared utilities for the strict audited VocalMorph pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

CANONICAL_SPLITS: Tuple[str, ...] = ("train", "val", "test")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def decode_np_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.ndim == 0:
            return str(value.item())
        return str(value.flat[0])
    return str(value)


def json_fingerprint(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _path_suffix(path: str, depth: int = 3) -> Tuple[str, ...]:
    normalized = normalize_path(path)
    parts: List[str] = []
    while True:
        head, tail = os.path.split(normalized)
        if tail:
            parts.append(tail)
        if not head or head == normalized:
            break
        normalized = head
    return tuple(reversed(parts[:depth]))


def _same_relocatable_split_path(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if normalize_path(left) == normalize_path(right):
        return True
    return _path_suffix(left, depth=3) == _path_suffix(right, depth=3)


def iter_row_audio_paths(row: Mapping[str, Any]) -> List[str]:
    field = str(row.get("audio_paths", "") or "")
    return [part.strip() for part in field.split("|") if part.strip()]


def height_bin(height_cm: float) -> str:
    if not np.isfinite(height_cm):
        return "unknown"
    if height_cm < 160.0:
        return "short"
    if height_cm < 175.0:
        return "medium"
    return "tall"


def duration_bin(duration_s: float) -> str:
    if not np.isfinite(duration_s):
        return "unknown"
    if duration_s < 2.5:
        return "short"
    if duration_s < 5.0:
        return "medium"
    return "long"


def quality_bucket(score: float) -> str:
    if not np.isfinite(score):
        return "unknown"
    if score < 0.50:
        return "low"
    if score < 0.75:
        return "medium"
    return "high"


def summarize_split_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    speakers = [str(row.get("speaker_id", "") or "") for row in rows]
    source_counts = Counter(str(row.get("source", "UNKNOWN") or "UNKNOWN").upper() for row in rows)
    gender_counts = Counter(str(row.get("gender", "UNKNOWN") or "UNKNOWN").title() for row in rows)
    height_counts = Counter(height_bin(safe_float(row.get("height_cm"))) for row in rows)
    return {
        "rows": int(len(rows)),
        "unique_speakers": int(len({speaker for speaker in speakers if speaker})),
        "source_counts": dict(sorted(source_counts.items())),
        "gender_counts": dict(sorted(gender_counts.items())),
        "height_bin_counts": dict(sorted(height_counts.items())),
    }


def split_manifest_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    materialized: List[Dict[str, Any]] = []
    for row in rows:
        materialized.append({str(key): str(value) for key, value in dict(row).items()})
    return json_fingerprint({"rows": materialized})


def leakage_report_for_rows(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    verify_audio_hashes: bool = False,
    root_dir: Optional[str] = None,
) -> Dict[str, Any]:
    speaker_sets: Dict[str, set[str]] = {}
    audio_sets: Dict[str, set[str]] = {}
    hash_sets: Dict[str, set[str]] = {}
    hash_examples: Dict[str, Dict[str, str]] = {}

    for split_name, rows in split_rows.items():
        speaker_set: set[str] = set()
        audio_set: set[str] = set()
        hash_set: set[str] = set()
        split_hash_examples: Dict[str, str] = {}
        for row in rows:
            speaker_id = str(row.get("speaker_id", "") or "").strip()
            if speaker_id:
                speaker_set.add(speaker_id)
            for raw_path in iter_row_audio_paths(row):
                full_path = raw_path
                if root_dir and not os.path.isabs(full_path):
                    full_path = os.path.join(root_dir, full_path)
                normalized = normalize_path(full_path)
                audio_set.add(normalized)
                if verify_audio_hashes and os.path.isfile(normalized):
                    digest = file_sha256(normalized)
                    hash_set.add(digest)
                    split_hash_examples.setdefault(digest, normalized)
        speaker_sets[split_name] = speaker_set
        audio_sets[split_name] = audio_set
        hash_sets[split_name] = hash_set
        hash_examples[split_name] = split_hash_examples

    report: Dict[str, Any] = {
        "split_counts": {
            split: {
                "speakers": len(speaker_sets.get(split, set())),
                "audio_paths": len(audio_sets.get(split, set())),
                "audio_hashes": len(hash_sets.get(split, set())) if verify_audio_hashes else 0,
            }
            for split in split_rows.keys()
        },
        "speaker_overlap_counts": {},
        "audio_path_overlap_counts": {},
        "audio_hash_overlap_counts": {},
        "speaker_overlap_examples": {},
        "audio_path_overlap_examples": {},
        "audio_hash_overlap_examples": {},
        "has_leakage": False,
    }

    split_names = list(split_rows.keys())
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            key = f"{left}_{right}"
            shared_speakers = sorted(speaker_sets[left] & speaker_sets[right])
            shared_audio = sorted(audio_sets[left] & audio_sets[right])
            report["speaker_overlap_counts"][key] = len(shared_speakers)
            report["audio_path_overlap_counts"][key] = len(shared_audio)
            report["speaker_overlap_examples"][key] = shared_speakers[:10]
            report["audio_path_overlap_examples"][key] = shared_audio[:10]

            shared_hashes = sorted(hash_sets[left] & hash_sets[right]) if verify_audio_hashes else []
            report["audio_hash_overlap_counts"][key] = len(shared_hashes)
            report["audio_hash_overlap_examples"][key] = [
                {
                    "sha256": digest,
                    left: hash_examples[left].get(digest, ""),
                    right: hash_examples[right].get(digest, ""),
                }
                for digest in shared_hashes[:10]
            ]
            if shared_speakers or shared_audio or shared_hashes:
                report["has_leakage"] = True

    return report


def assert_no_split_leakage(report: Mapping[str, Any]) -> None:
    for key, count in dict(report.get("speaker_overlap_counts", {})).items():
        if int(count) > 0:
            raise AssertionError(f"speaker leakage detected for split pair {key}: {count}")
    for key, count in dict(report.get("audio_path_overlap_counts", {})).items():
        if int(count) > 0:
            raise AssertionError(f"audio-path leakage detected for split pair {key}: {count}")
    for key, count in dict(report.get("audio_hash_overlap_counts", {})).items():
        if int(count) > 0:
            raise AssertionError(f"audio-content leakage detected for split pair {key}: {count}")


def feature_files(split_dir: str) -> List[str]:
    return sorted(
        os.path.join(split_dir, name)
        for name in os.listdir(split_dir)
        if name.lower().endswith(".npz")
    )


def load_feature_speaker_ids(split_dir: str) -> set[str]:
    speaker_ids: set[str] = set()
    if not os.path.isdir(split_dir):
        return speaker_ids
    for path in feature_files(split_dir):
        with np.load(path, allow_pickle=True) as data:
            if "speaker_id" in data:
                speaker_id = decode_np_value(data["speaker_id"]).strip()
            else:
                speaker_id = os.path.splitext(os.path.basename(path))[0].split("_aug", 1)[0]
        if speaker_id:
            speaker_ids.add(speaker_id)
    return speaker_ids


def compute_target_stats_from_feature_dir(train_dir: str) -> Dict[str, Dict[str, float]]:
    height_values: List[float] = []
    age_values: List[float] = []
    weight_values: List[float] = []

    for path in feature_files(train_dir):
        with np.load(path, allow_pickle=True) as data:
            height = safe_float(data["height_cm"]) if "height_cm" in data else float("nan")
            age = safe_float(data["age"]) if "age" in data else float("nan")
            weight = safe_float(data["weight_kg"]) if "weight_kg" in data else float("nan")
            source = decode_np_value(data["source"]) if "source" in data else ""

        if np.isfinite(height):
            height_values.append(height)
        if np.isfinite(age):
            age_values.append(age)
        if np.isfinite(weight) and source.upper() == "NISP":
            weight_values.append(weight)

    def _summary(values: Sequence[float], default_mean: float = 0.0, default_std: float = 1.0) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return {"mean": float(default_mean), "std": float(default_std)}
        return {"mean": float(arr.mean()), "std": float(arr.std() + 1e-9)}

    return {
        "height": _summary(height_values),
        "age": _summary(age_values),
        "weight": _summary(weight_values),
    }


def compute_feature_diagnostics(split_dir: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "file_count": 0,
        "unique_speakers": 0,
        "sequence_dim": 0,
        "nonfinite_sequence_files": 0,
        "nonfinite_scalar_files": 0,
        "zero_voiced_f0_files": 0,
        "invalid_spacing_files": 0,
        "invalid_vtl_files": 0,
        "zero_praat_tail_files": 0,
        "augmented_files": 0,
        "quality_rejected_files": 0,
        "per_speaker_clip_count": {},
        "scalar_summary": {},
        "zero_variance_dims_count": 0,
        "zero_variance_dims": [],
    }
    if not os.path.isdir(split_dir):
        return report

    speaker_counter: Counter[str] = Counter()
    pooled_means: List[np.ndarray] = []
    scalar_tracks: MutableMapping[str, List[float]] = defaultdict(list)

    for path in feature_files(split_dir):
        with np.load(path, allow_pickle=True) as data:
            sequence = np.asarray(data["sequence"], dtype=np.float32)
            speaker_id = decode_np_value(data["speaker_id"]).strip() if "speaker_id" in data else ""
            f0_mean = safe_float(data["f0_mean"]) if "f0_mean" in data else float("nan")
            spacing = safe_float(data["formant_spacing_mean"]) if "formant_spacing_mean" in data else float("nan")
            vtl = safe_float(data["vtl_mean"]) if "vtl_mean" in data else float("nan")
            duration = safe_float(data["duration_s"]) if "duration_s" in data else float("nan")
            speech_ratio = safe_float(data["speech_ratio"]) if "speech_ratio" in data else float("nan")
            snr_db = safe_float(data["snr_db_estimate"]) if "snr_db_estimate" in data else float("nan")
            capture_quality = safe_float(data["capture_quality_score"]) if "capture_quality_score" in data else float("nan")
            is_augmented = bool(int(safe_float(data["is_augmented"]))) if "is_augmented" in data else ("_aug" in os.path.basename(path))
            quality_ok = data["quality_ok"] if "quality_ok" in data else None
            if isinstance(quality_ok, np.ndarray) and quality_ok.shape == ():
                quality_ok = bool(quality_ok.item())
            elif quality_ok is not None:
                quality_ok = bool(quality_ok)

        report["file_count"] += 1
        report["sequence_dim"] = max(report["sequence_dim"], int(sequence.shape[1] if sequence.ndim == 2 else 0))
        if speaker_id:
            speaker_counter[speaker_id] += 1
        if is_augmented:
            report["augmented_files"] += 1
        if quality_ok is False:
            report["quality_rejected_files"] += 1
        if not np.isfinite(sequence).all():
            report["nonfinite_sequence_files"] += 1
        if any(
            not np.isfinite(value)
            for value in (f0_mean, spacing, vtl)
            if value == value
        ):
            report["nonfinite_scalar_files"] += 1
        if not np.isfinite(f0_mean) or f0_mean <= 0.0:
            report["zero_voiced_f0_files"] += 1
        if not np.isfinite(spacing) or spacing <= 0.0:
            report["invalid_spacing_files"] += 1
        if not np.isfinite(vtl) or vtl <= 0.0:
            report["invalid_vtl_files"] += 1
        if (not np.isfinite(f0_mean) or f0_mean <= 0.0) and (not np.isfinite(spacing) or spacing <= 0.0) and (not np.isfinite(vtl) or vtl <= 0.0):
            report["zero_praat_tail_files"] += 1
        if sequence.ndim == 2 and sequence.size > 0:
            pooled_means.append(sequence.mean(axis=0))
        for key, value in (
            ("duration_s", duration),
            ("speech_ratio", speech_ratio),
            ("snr_db_estimate", snr_db),
            ("capture_quality_score", capture_quality),
            ("f0_mean", f0_mean),
            ("formant_spacing_mean", spacing),
            ("vtl_mean", vtl),
        ):
            if np.isfinite(value):
                scalar_tracks[key].append(float(value))

    report["unique_speakers"] = int(len(speaker_counter))
    report["per_speaker_clip_count"] = dict(sorted(speaker_counter.items()))
    for key, values in scalar_tracks.items():
        arr = np.asarray(values, dtype=np.float32)
        report["scalar_summary"][key] = {
            "count": int(arr.size),
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size else float("nan"),
            "min": float(arr.min()) if arr.size else float("nan"),
            "max": float(arr.max()) if arr.size else float("nan"),
        }

    if pooled_means:
        pooled = np.stack(pooled_means, axis=0).astype(np.float32)
        variances = pooled.var(axis=0)
        zero_dims = np.flatnonzero(variances <= 1e-12)
        report["zero_variance_dims_count"] = int(zero_dims.size)
        report["zero_variance_dims"] = zero_dims.astype(int).tolist()

    return report


def compare_scalar_drift(
    reference: Mapping[str, Any],
    other: Mapping[str, Any],
    keys: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    drift: Dict[str, Dict[str, float]] = {}
    ref_summary = dict(reference.get("scalar_summary", {}))
    other_summary = dict(other.get("scalar_summary", {}))
    for key in keys:
        ref = ref_summary.get(key, {})
        current = other_summary.get(key, {})
        ref_mean = safe_float(ref.get("mean"))
        ref_std = safe_float(ref.get("std"))
        other_mean = safe_float(current.get("mean"))
        if not np.isfinite(ref_mean) or not np.isfinite(other_mean):
            continue
        denom = abs(ref_std) if np.isfinite(ref_std) and abs(ref_std) > 1e-9 else 1.0
        drift[key] = {
            "reference_mean": float(ref_mean),
            "other_mean": float(other_mean),
            "abs_mean_shift": float(abs(other_mean - ref_mean)),
            "std_units": float(abs(other_mean - ref_mean) / denom),
        }
    return drift


def feature_contract_payload(
    *,
    feature_config: Mapping[str, Any],
    split_files: Mapping[str, str],
    target_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "feature_config": dict(feature_config),
        "feature_config_fingerprint": json_fingerprint(dict(feature_config)),
        "split_files": {key: normalize_path(value) for key, value in split_files.items()},
        "split_hashes": {
            key: file_sha256(normalize_path(value)) for key, value in split_files.items()
        },
    }
    if target_stats is not None:
        payload["target_stats_fingerprint"] = json_fingerprint(dict(target_stats))
    return payload


def validate_feature_contract(
    *,
    feature_root: str,
    expected_feature_config: Mapping[str, Any],
    expected_split_files: Mapping[str, str],
    require_target_stats: bool = True,
) -> Dict[str, Any]:
    contract_path = os.path.join(feature_root, "feature_contract.json")
    if not os.path.exists(contract_path):
        raise FileNotFoundError(f"Missing feature contract: {contract_path}")
    with open(contract_path, "r", encoding="utf-8") as handle:
        contract = json.load(handle)

    expected_payload = feature_contract_payload(
        feature_config=expected_feature_config,
        split_files=expected_split_files,
    )
    if contract.get("feature_config_fingerprint") != expected_payload["feature_config_fingerprint"]:
        raise RuntimeError("Feature contract mismatch: current config fingerprint differs from audited build.")

    for split_name, split_path in expected_payload["split_files"].items():
        actual_path = normalize_path(contract.get("split_files", {}).get(split_name, ""))
        if not _same_relocatable_split_path(actual_path, split_path):
            raise RuntimeError(
                f"Feature contract mismatch for {split_name}: expected split file {split_path}, got {actual_path or 'missing'}"
            )
        actual_hash = contract.get("split_hashes", {}).get(split_name)
        expected_hash = expected_payload["split_hashes"][split_name]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Feature contract mismatch for {split_name}: split manifest hash changed since build."
            )

    if require_target_stats and not os.path.exists(os.path.join(feature_root, "target_stats.json")):
        raise FileNotFoundError(f"Missing target_stats.json in {feature_root}")
    return contract


__all__ = [
    "CANONICAL_SPLITS",
    "assert_no_split_leakage",
    "compare_scalar_drift",
    "compute_feature_diagnostics",
    "compute_target_stats_from_feature_dir",
    "decode_np_value",
    "duration_bin",
    "feature_contract_payload",
    "feature_files",
    "file_sha256",
    "height_bin",
    "json_fingerprint",
    "leakage_report_for_rows",
    "load_feature_speaker_ids",
    "normalize_path",
    "quality_bucket",
    "read_csv_rows",
    "safe_float",
    "split_manifest_fingerprint",
    "summarize_split_rows",
    "validate_feature_contract",
]
