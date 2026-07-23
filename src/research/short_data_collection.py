"""Consent- and license-aware short-speaker data collection for VoxPhysica.

The module keeps three concepts separate:

* a person is a unique speaker, never an augmented clip;
* public HeightCeleb labels are noisy train-only estimates;
* consented measured participants may be assigned to development or a sealed test.

No audio is copied. Output manifests reference source files and contain only
pseudonymous speaker identifiers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CANONICAL_FIELDS = [
    "speaker_id",
    "source",
    "gender",
    "height_cm",
    "weight_kg",
    "age",
    "audio_paths",
]

SPEAKER_OUTPUT_FIELDS = CANONICAL_FIELDS + [
    "collection_role",
    "collection_source",
    "height_measurement_type",
    "label_quality",
    "metadata_license",
    "audio_license",
    "consent_basis",
    "public_release_allowed",
    "valid_clip_count",
    "total_duration_s",
]

CLIP_OUTPUT_FIELDS = [
    "speaker_id",
    "audio_path",
    "sha256",
    "duration_s",
    "sample_rate_hz",
    "channels",
    "sample_width_bytes",
    "rms_dbfs",
    "peak_fraction",
    "clipping_fraction",
    "qc_pass",
    "qc_reasons",
]

FORBIDDEN_PII_COLUMNS = {
    "name",
    "full_name",
    "email",
    "email_address",
    "phone",
    "phone_number",
    "address",
    "government_id",
    "aadhaar",
    "passport",
}

YES_VALUES = {"1", "true", "yes", "y", "consented"}
SUPPORTED_MODEL_GENDERS = {"Male", "Female"}


@dataclass(frozen=True)
class AudioQCPolicy:
    min_duration_s: float = 2.0
    max_duration_s: float = 30.0
    min_sample_rate_hz: int = 8_000
    max_rms_dbfs: float = -3.0
    min_rms_dbfs: float = -50.0
    min_peak_fraction: float = 0.01
    max_clipping_fraction: float = 0.01


@dataclass
class AudioQCResult:
    audio_path: str
    sha256: str = ""
    duration_s: float = math.nan
    sample_rate_hz: int = 0
    channels: int = 0
    sample_width_bytes: int = 0
    rms_dbfs: float = math.nan
    peak_fraction: float = math.nan
    clipping_fraction: float = math.nan
    qc_pass: bool = False
    qc_reasons: str = ""


@dataclass
class CollectionResult:
    speakers: list[dict[str, str]]
    clips: list[dict[str, str]]
    rejected_speakers: list[dict[str, str]]
    rejected_clips: list[dict[str, str]]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def clean_gender(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"m", "male", "man"}:
        return "Male"
    if text in {"f", "female", "woman"}:
        return "Female"
    if text in {"other", "nonbinary", "non-binary"}:
        return "Other"
    if text in {"prefer not to say", "prefer_not_to_say", "unknown", ""}:
        return "Prefer not to say"
    return str(value).strip()


def is_yes(value: Any) -> bool:
    return str(value or "").strip().lower() in YES_VALUES


def resolve_audio_path(raw: str, *, repo_root: Path, audio_root: Path | None) -> Path:
    path = Path(str(raw).strip().strip('"'))
    if path.is_absolute():
        return path
    if audio_root is not None:
        candidate = audio_root / path
        if candidate.exists():
            return candidate
    return repo_root / path


def _pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if sample_width == 3:
        byte_data = np.frombuffer(raw, dtype=np.uint8)
        usable = (len(byte_data) // 3) * 3
        triplets = byte_data[:usable].reshape(-1, 3).astype(np.int32)
        values = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8_388_608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2_147_483_648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width}")


def audit_wav(path: Path, policy: AudioQCPolicy) -> AudioQCResult:
    result = AudioQCResult(audio_path=str(path.resolve()))
    reasons: list[str] = []
    if not path.is_file():
        result.qc_reasons = "missing_file"
        return result

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result.sha256 = digest.hexdigest()

    try:
        with wave.open(str(path), "rb") as handle:
            channels = int(handle.getnchannels())
            sample_rate = int(handle.getframerate())
            sample_width = int(handle.getsampwidth())
            n_frames = int(handle.getnframes())
            compression = str(handle.getcomptype())
            raw = handle.readframes(n_frames)
    except (wave.Error, OSError, EOFError) as exc:
        result.qc_reasons = f"unreadable_wav:{type(exc).__name__}"
        return result

    result.channels = channels
    result.sample_rate_hz = sample_rate
    result.sample_width_bytes = sample_width
    result.duration_s = n_frames / sample_rate if sample_rate > 0 else math.nan

    if compression != "NONE":
        reasons.append("compressed_wav")
    if channels not in {1, 2}:
        reasons.append("unsupported_channels")
    if sample_rate < policy.min_sample_rate_hz:
        reasons.append("sample_rate_too_low")
    if not math.isfinite(result.duration_s) or result.duration_s < policy.min_duration_s:
        reasons.append("too_short")
    if math.isfinite(result.duration_s) and result.duration_s > policy.max_duration_s:
        reasons.append("too_long")

    try:
        samples = _pcm_to_float(raw, sample_width)
    except ValueError:
        reasons.append("unsupported_sample_width")
        samples = np.asarray([], dtype=np.float64)

    if samples.size:
        absolute = np.abs(samples)
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        result.rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
        result.peak_fraction = float(np.max(absolute))
        result.clipping_fraction = float(np.mean(absolute >= 0.999))

        if result.rms_dbfs < policy.min_rms_dbfs:
            reasons.append("near_silent")
        if result.rms_dbfs > policy.max_rms_dbfs:
            reasons.append("excessive_level")
        if result.peak_fraction < policy.min_peak_fraction:
            reasons.append("low_peak")
        if result.clipping_fraction > policy.max_clipping_fraction:
            reasons.append("clipped")
    else:
        reasons.append("no_pcm_samples")

    result.qc_pass = not reasons
    result.qc_reasons = "|".join(reasons)
    return result


def clip_row(speaker_id: str, result: AudioQCResult) -> dict[str, str]:
    return {
        "speaker_id": speaker_id,
        "audio_path": result.audio_path,
        "sha256": result.sha256,
        "duration_s": _fmt_float(result.duration_s, 6),
        "sample_rate_hz": str(result.sample_rate_hz),
        "channels": str(result.channels),
        "sample_width_bytes": str(result.sample_width_bytes),
        "rms_dbfs": _fmt_float(result.rms_dbfs, 4),
        "peak_fraction": _fmt_float(result.peak_fraction, 8),
        "clipping_fraction": _fmt_float(result.clipping_fraction, 8),
        "qc_pass": "yes" if result.qc_pass else "no",
        "qc_reasons": result.qc_reasons,
    }


def _fmt_float(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _audio_paths(
    row: Mapping[str, str],
    *,
    repo_root: Path,
    audio_root: Path | None,
    max_clips: int,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for raw in str(row.get("audio_paths", "") or "").split("|"):
        raw = raw.strip()
        if not raw:
            continue
        path = resolve_audio_path(raw, repo_root=repo_root, audio_root=audio_root)
        key = str(path.resolve()).lower() if path.exists() else str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
        if max_clips > 0 and len(paths) >= max_clips:
            break
    return paths


def _audit_speaker_audio(
    speaker_id: str,
    paths: Sequence[Path],
    *,
    policy: AudioQCPolicy,
    min_clips: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], float]:
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    duration = 0.0
    seen_hashes: set[str] = set()

    for path in paths:
        result = audit_wav(path, policy)
        row = clip_row(speaker_id, result)
        if result.qc_pass and result.sha256 in seen_hashes:
            row["qc_pass"] = "no"
            row["qc_reasons"] = "duplicate_audio_hash_within_speaker"
        if row["qc_pass"] == "yes":
            seen_hashes.add(row["sha256"])
            accepted.append(row)
            duration += safe_float(row["duration_s"], 0.0)
        else:
            rejected.append(row)

    if len(accepted) < min_clips:
        for row in accepted:
            row["qc_pass"] = "no"
            row["qc_reasons"] = "speaker_below_min_valid_clips"
            rejected.append(row)
        accepted = []
        duration = 0.0
    return accepted, rejected, duration


def collect_public_heightceleb(
    csv_path: Path,
    *,
    repo_root: Path,
    audio_root: Path | None,
    short_threshold_cm: float,
    min_clips: int,
    max_clips: int,
    policy: AudioQCPolicy,
) -> CollectionResult:
    rows = read_csv_rows(csv_path)
    speakers: list[dict[str, str]] = []
    clips: list[dict[str, str]] = []
    rejected_speakers: list[dict[str, str]] = []
    rejected_clips: list[dict[str, str]] = []

    for row_number, row in enumerate(rows, start=2):
        speaker_id = str(row.get("speaker_id", "") or "").strip()
        height_cm = safe_float(row.get("height_cm"))
        gender = clean_gender(row.get("gender"))
        if not speaker_id:
            rejected_speakers.append({"row": str(row_number), "reason": "missing_speaker_id"})
            continue
        if not (120.0 <= height_cm < short_threshold_cm):
            continue
        if gender not in SUPPORTED_MODEL_GENDERS:
            rejected_speakers.append(
                {"speaker_id": speaker_id, "reason": "unsupported_gender_for_current_model"}
            )
            continue

        paths = _audio_paths(
            row,
            repo_root=repo_root,
            audio_root=audio_root,
            max_clips=max_clips,
        )
        accepted, rejected, duration = _audit_speaker_audio(
            speaker_id,
            paths,
            policy=policy,
            min_clips=min_clips,
        )
        rejected_clips.extend(rejected)
        if len(accepted) < min_clips:
            rejected_speakers.append(
                {
                    "speaker_id": speaker_id,
                    "reason": "too_few_valid_clips",
                    "valid_clips": str(len(accepted)),
                }
            )
            continue

        valid_paths = [clip["audio_path"] for clip in accepted]
        speakers.append(
            {
                "speaker_id": speaker_id,
                "source": "HEIGHTCELEB",
                "gender": gender,
                "height_cm": f"{height_cm:.4f}",
                "weight_kg": str(row.get("weight_kg", "") or "").strip(),
                "age": str(row.get("age", "") or "").strip(),
                "audio_paths": "|".join(valid_paths),
                "collection_role": "train_support_only",
                "collection_source": "HeightCeleb/VoxCeleb1",
                "height_measurement_type": "internet_estimate",
                "label_quality": "noisy_public_estimate",
                "metadata_license": "CC-BY-4.0",
                "audio_license": "VoxCeleb terms; audio not redistributed",
                "consent_basis": "existing_public_research_dataset",
                "public_release_allowed": "manifest_only",
                "valid_clip_count": str(len(accepted)),
                "total_duration_s": f"{duration:.3f}",
            }
        )
        clips.extend(accepted)

    return CollectionResult(speakers, clips, rejected_speakers, rejected_clips)


def validate_consent_schema(fieldnames: Iterable[str]) -> None:
    normalized = {str(name).strip().lower() for name in fieldnames}
    found = sorted(normalized & FORBIDDEN_PII_COLUMNS)
    if found:
        raise ValueError(
            "Consent intake must be pseudonymous. Remove direct PII columns: "
            + ", ".join(found)
        )


def collect_consented_measured(
    csv_path: Path,
    *,
    repo_root: Path,
    audio_root: Path | None,
    short_threshold_cm: float,
    min_clips: int,
    max_clips: int,
    policy: AudioQCPolicy,
) -> CollectionResult:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_consent_schema(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    speakers: list[dict[str, str]] = []
    clips: list[dict[str, str]] = []
    rejected_speakers: list[dict[str, str]] = []
    rejected_clips: list[dict[str, str]] = []

    for row_number, row in enumerate(rows, start=2):
        speaker_id = str(row.get("participant_code", "") or "").strip()
        age = safe_float(row.get("age_years"))
        h1 = safe_float(row.get("height_measurement_1_cm"))
        h2 = safe_float(row.get("height_measurement_2_cm"))
        gender = clean_gender(row.get("gender"))
        role = str(row.get("collection_role", "development") or "development").strip().lower()

        reason = ""
        if not speaker_id:
            reason = "missing_participant_code"
        elif age < 18.0:
            reason = "adult_participants_only"
        elif not is_yes(row.get("consent_audio_research")):
            reason = "audio_research_consent_missing"
        elif not is_yes(row.get("consent_model_training")):
            reason = "model_training_consent_missing"
        elif not (120.0 <= h1 <= 210.0 and 120.0 <= h2 <= 210.0):
            reason = "invalid_repeated_height"
        elif abs(h1 - h2) > 1.0:
            reason = "height_measurements_disagree_gt_1cm"
        elif role not in {"development", "sealed_test"}:
            reason = "collection_role_must_be_development_or_sealed_test"

        height_cm = (h1 + h2) / 2.0 if math.isfinite(h1) and math.isfinite(h2) else math.nan
        if not reason and height_cm >= short_threshold_cm:
            continue
        if reason:
            rejected_speakers.append(
                {"row": str(row_number), "speaker_id": speaker_id, "reason": reason}
            )
            continue

        paths = _audio_paths(
            row,
            repo_root=repo_root,
            audio_root=audio_root,
            max_clips=max_clips,
        )
        accepted, rejected, duration = _audit_speaker_audio(
            speaker_id,
            paths,
            policy=policy,
            min_clips=min_clips,
        )
        rejected_clips.extend(rejected)
        if len(accepted) < min_clips:
            rejected_speakers.append(
                {
                    "speaker_id": speaker_id,
                    "reason": "too_few_valid_clips",
                    "valid_clips": str(len(accepted)),
                }
            )
            continue

        valid_paths = [clip["audio_path"] for clip in accepted]
        speakers.append(
            {
                "speaker_id": speaker_id,
                "source": "VOXPHYSICA_CONSENTED",
                "gender": gender,
                "height_cm": f"{height_cm:.4f}",
                "weight_kg": "",
                "age": f"{age:.1f}",
                "audio_paths": "|".join(valid_paths),
                "collection_role": role,
                "collection_source": "VoxPhysica prospective collection",
                "height_measurement_type": "two_measurement_average",
                "label_quality": "measured",
                "metadata_license": "participant consent",
                "audio_license": "participant consent",
                "consent_basis": "written_informed_consent",
                "public_release_allowed": (
                    "yes" if is_yes(row.get("consent_public_release")) else "no"
                ),
                "valid_clip_count": str(len(accepted)),
                "total_duration_s": f"{duration:.3f}",
            }
        )
        clips.extend(accepted)

    return CollectionResult(speakers, clips, rejected_speakers, rejected_clips)


def assert_unique_and_disjoint(
    support_speakers: Sequence[Mapping[str, str]],
    *,
    train_rows: Sequence[Mapping[str, str]],
    val_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
) -> None:
    support_ids = [str(row.get("speaker_id", "")).strip() for row in support_speakers]
    duplicates = sorted(sid for sid, count in Counter(support_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate collected speaker IDs: {duplicates[:20]}")

    sealed_ids = {
        str(row.get("speaker_id", "")).strip()
        for row in list(val_rows) + list(test_rows)
    }
    overlap = sorted(set(support_ids) & sealed_ids)
    if overlap:
        raise ValueError(
            "Collected speaker overlap with validation/test: " + ", ".join(overlap[:20])
        )

    train_ids = {str(row.get("speaker_id", "")).strip() for row in train_rows}
    train_overlap = sorted(set(support_ids) & train_ids)
    if train_overlap:
        raise ValueError(
            "Collected speaker overlap with existing train: "
            + ", ".join(train_overlap[:20])
        )


def assert_unique_audio_hashes(clips: Sequence[Mapping[str, str]]) -> None:
    owners: defaultdict[str, set[str]] = defaultdict(set)
    for row in clips:
        digest = str(row.get("sha256", "") or "").strip()
        if digest:
            owners[digest].add(str(row.get("speaker_id", "")).strip())
    cross_speaker = {
        digest: sorted(speakers)
        for digest, speakers in owners.items()
        if len(speakers) > 1
    }
    if cross_speaker:
        examples = list(cross_speaker.items())[:5]
        raise ValueError(f"Audio hash assigned to multiple speakers: {examples}")


def build_train_support_manifest(
    base_train: Sequence[Mapping[str, str]],
    collected: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    support = [
        row
        for row in collected
        if str(row.get("collection_role", "")).strip().lower()
        in {"development", "train_support_only"}
        and str(row.get("gender", "")).strip() in SUPPORTED_MODEL_GENDERS
    ]
    combined = [
        {field: str(row.get(field, "") or "") for field in CANONICAL_FIELDS}
        for row in list(base_train) + support
    ]
    return combined


def summarize(
    *,
    train_rows: Sequence[Mapping[str, str]],
    public: CollectionResult,
    consented: CollectionResult,
    short_threshold_cm: float,
    policy: AudioQCPolicy,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    speakers = public.speakers + consented.speakers
    clips = public.clips + consented.clips
    measured = [
        row for row in consented.speakers
        if row.get("height_measurement_type") == "two_measurement_average"
    ]
    roles = Counter(row.get("collection_role", "unknown") for row in measured)
    genders = Counter(row.get("gender", "unknown") for row in speakers)
    sources = Counter(row.get("collection_source", "unknown") for row in speakers)

    base_short = [
        row for row in train_rows
        if safe_float(row.get("height_cm")) < short_threshold_cm
    ]
    target = {
        "development_measured_short": 120,
        "sealed_test_measured_short": 80,
        "development_measured_short_male": 30,
        "sealed_test_measured_short_male": 15,
    }
    measured_dev = [
        row for row in measured if row.get("collection_role") == "development"
    ]
    measured_test = [
        row for row in measured if row.get("collection_role") == "sealed_test"
    ]
    measured_dev_male = [row for row in measured_dev if row.get("gender") == "Male"]
    measured_test_male = [row for row in measured_test if row.get("gender") == "Male"]
    achieved = {
        "development_measured_short": len(measured_dev),
        "sealed_test_measured_short": len(measured_test),
        "development_measured_short_male": len(measured_dev_male),
        "sealed_test_measured_short_male": len(measured_test_male),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "pilot_public_support_ready_measured_collection_pending"
            if public.speakers and not measured
            else "collection_in_progress"
        ),
        "short_threshold_cm_exclusive": short_threshold_cm,
        "person_count_definition": "unique speaker_id; augmented clips never count as people",
        "base_train": {
            "speakers": len(train_rows),
            "short_speakers": len(base_short),
            "short_gender_counts": dict(
                Counter(row.get("gender", "unknown") for row in base_short)
            ),
        },
        "accepted": {
            "speakers": len(speakers),
            "clips": len(clips),
            "duration_hours": sum(
                safe_float(row.get("duration_s"), 0.0) for row in clips
            ) / 3600.0,
            "gender_counts": dict(genders),
            "source_counts": dict(sources),
            "measured_role_counts": dict(roles),
        },
        "public_heightceleb": {
            "accepted_speakers": len(public.speakers),
            "accepted_clips": len(public.clips),
            "rejected_speakers": len(public.rejected_speakers),
            "rejected_clips": len(public.rejected_clips),
            "train_only": True,
            "label_warning": (
                "Heights are internet-derived estimates. The HeightCeleb authors "
                "recommend precise measured data for testing."
            ),
        },
        "consented_measured": {
            "accepted_speakers": len(consented.speakers),
            "accepted_clips": len(consented.clips),
            "rejected_speakers": len(consented.rejected_speakers),
            "rejected_clips": len(consented.rejected_clips),
        },
        "pilot_quota": {
            "target": target,
            "achieved": achieved,
            "remaining": {
                key: max(0, target[key] - achieved[key]) for key in target
            },
        },
        "audio_qc_policy": asdict(policy),
        "input_sha256": dict(input_hashes),
        "ethics": {
            "adult_only": True,
            "direct_pii_prohibited": True,
            "public_audio_copied": False,
            "height_is_sensitive_soft_biometric": True,
            "institutional_ethics_review_required_before_recruitment": True,
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_markdown_report(path: Path, report: Mapping[str, Any]) -> None:
    accepted = report["accepted"]
    base = report["base_train"]
    public = report["public_heightceleb"]
    consented = report["consented_measured"]
    remaining = report["pilot_quota"]["remaining"]
    lines = [
        "# VoxPhysica Short-Speaker Data Audit",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Outcome",
        "",
        (
            f"The pipeline accepted **{accepted['speakers']} unique short speakers** "
            f"and **{accepted['clips']} quality-controlled clips** "
            f"({accepted['duration_hours']:.2f} hours)."
        ),
        (
            f"The original training split has {base['short_speakers']} speakers below "
            f"{report['short_threshold_cm_exclusive']:.1f} cm."
        ),
        "",
        "## Public train-only support",
        "",
        (
            f"HeightCeleb contributed {public['accepted_speakers']} speakers and "
            f"{public['accepted_clips']} clips after audio QC."
        ),
        (
            "These heights are internet-derived estimates and are restricted to "
            "train support. They are not a measured external test set."
        ),
        "",
        "## Prospective measured collection",
        "",
        (
            f"Accepted measured participants: {consented['accepted_speakers']}. "
            "Before recruitment, obtain institutional ethics approval and written "
            "informed consent."
        ),
        (
            "Remaining pilot quota: "
            + ", ".join(f"{key}={value}" for key, value in remaining.items())
            + "."
        ),
        "",
        "## Integrity controls",
        "",
        "- Every person is counted once by pseudonymous speaker ID.",
        "- Validation and historical-test speaker IDs are rejected.",
        "- Duplicate audio hashes across speakers are rejected.",
        "- Direct identifiers are prohibited in the intake CSV.",
        "- Audio is referenced in place and is not copied or redistributed.",
        "- Synthetic augmentation is not counted as new people.",
        "",
        "## Interpretation",
        "",
        (
            "This collection closes part of the female short-speaker training gap, "
            "but it does not close the short-male or measured-label gap. A new "
            "sealed test must contain measured heights and must not be used during "
            "model selection."
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
