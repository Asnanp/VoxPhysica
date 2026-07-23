import csv
import wave
from pathlib import Path

import numpy as np
import pytest

from src.research.short_data_collection import (
    AudioQCPolicy,
    assert_unique_and_disjoint,
    audit_wav,
    build_train_support_manifest,
    collect_consented_measured,
    collect_public_heightceleb,
    validate_consent_schema,
)


def _write_wav(
    path: Path,
    *,
    seconds: float = 2.5,
    silent: bool = False,
    frequency_hz: float = 220.0,
) -> None:
    sample_rate = 16_000
    count = int(sample_rate * seconds)
    if silent:
        samples = np.zeros(count, dtype=np.int16)
    else:
        time = np.arange(count, dtype=np.float64) / sample_rate
        samples = (
            0.2 * np.sin(2.0 * np.pi * frequency_hz * time) * 32767
        ).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype("<i2").tobytes())


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_audio_qc_accepts_clean_pcm_and_rejects_silence(tmp_path: Path):
    clean = tmp_path / "clean.wav"
    silent = tmp_path / "silent.wav"
    _write_wav(clean)
    _write_wav(silent, silent=True)

    accepted = audit_wav(clean, AudioQCPolicy())
    rejected = audit_wav(silent, AudioQCPolicy())

    assert accepted.qc_pass
    assert accepted.duration_s == pytest.approx(2.5)
    assert len(accepted.sha256) == 64
    assert not rejected.qc_pass
    assert "near_silent" in rejected.qc_reasons


def test_public_collection_filters_height_and_counts_people_once(tmp_path: Path):
    clips = []
    for index in range(3):
        clip = tmp_path / f"short_{index}.wav"
        _write_wav(clip, frequency_hz=220.0 + 10.0 * index)
        clips.append(str(clip))
    tall_clip = tmp_path / "tall.wav"
    _write_wav(tall_clip)

    manifest = tmp_path / "public.csv"
    fields = ["speaker_id", "height_cm", "gender", "audio_paths"]
    _write_csv(
        manifest,
        fields,
        [
            {
                "speaker_id": "CELEB_short",
                "height_cm": "155",
                "gender": "Female",
                "audio_paths": "|".join(clips),
            },
            {
                "speaker_id": "CELEB_tall",
                "height_cm": "180",
                "gender": "Male",
                "audio_paths": str(tall_clip),
            },
        ],
    )

    result = collect_public_heightceleb(
        manifest,
        repo_root=tmp_path,
        audio_root=None,
        short_threshold_cm=160.0,
        min_clips=2,
        max_clips=10,
        policy=AudioQCPolicy(),
    )

    assert [row["speaker_id"] for row in result.speakers] == ["CELEB_short"]
    assert len(result.clips) == 3
    assert result.speakers[0]["valid_clip_count"] == "3"
    assert result.speakers[0]["collection_role"] == "train_support_only"


def test_consent_intake_requires_adult_consent_and_repeated_measurement(tmp_path: Path):
    clips = []
    for index in range(2):
        clip = tmp_path / f"participant_{index}.wav"
        _write_wav(clip, frequency_hz=260.0 + 10.0 * index)
        clips.append(str(clip))

    intake = tmp_path / "intake.csv"
    fields = [
        "participant_code",
        "age_years",
        "gender",
        "height_measurement_1_cm",
        "height_measurement_2_cm",
        "collection_role",
        "consent_audio_research",
        "consent_model_training",
        "consent_public_release",
        "audio_paths",
    ]
    _write_csv(
        intake,
        fields,
        [
            {
                "participant_code": "VSP-001",
                "age_years": "21",
                "gender": "Male",
                "height_measurement_1_cm": "157.2",
                "height_measurement_2_cm": "157.6",
                "collection_role": "development",
                "consent_audio_research": "yes",
                "consent_model_training": "yes",
                "consent_public_release": "no",
                "audio_paths": "|".join(clips),
            },
            {
                "participant_code": "VSP-002",
                "age_years": "17",
                "gender": "Female",
                "height_measurement_1_cm": "151.0",
                "height_measurement_2_cm": "151.2",
                "collection_role": "development",
                "consent_audio_research": "yes",
                "consent_model_training": "yes",
                "consent_public_release": "no",
                "audio_paths": "|".join(clips),
            },
        ],
    )

    result = collect_consented_measured(
        intake,
        repo_root=tmp_path,
        audio_root=None,
        short_threshold_cm=160.0,
        min_clips=2,
        max_clips=10,
        policy=AudioQCPolicy(),
    )

    assert len(result.speakers) == 1
    assert result.speakers[0]["speaker_id"] == "VSP-001"
    assert float(result.speakers[0]["height_cm"]) == pytest.approx(157.4)
    assert result.speakers[0]["height_measurement_type"] == "two_measurement_average"
    assert any(row["reason"] == "adult_participants_only" for row in result.rejected_speakers)


def test_direct_identifier_columns_are_prohibited():
    with pytest.raises(ValueError, match="pseudonymous"):
        validate_consent_schema(["participant_code", "height_measurement_1_cm", "email"])


def test_collected_ids_cannot_overlap_validation_or_test():
    support = [{"speaker_id": "sealed-person"}]
    with pytest.raises(ValueError, match="validation/test"):
        assert_unique_and_disjoint(
            support,
            train_rows=[{"speaker_id": "train-person"}],
            val_rows=[{"speaker_id": "sealed-person"}],
            test_rows=[],
        )


def test_train_manifest_excludes_sealed_and_unsupported_gender():
    base = [
        {
            "speaker_id": "base",
            "source": "NISP",
            "gender": "Female",
            "height_cm": "155",
            "weight_kg": "",
            "age": "23",
            "audio_paths": "base.wav",
        }
    ]
    collected = [
        {
            "speaker_id": "dev",
            "source": "VOXPHYSICA_CONSENTED",
            "gender": "Male",
            "height_cm": "158",
            "weight_kg": "",
            "age": "25",
            "audio_paths": "dev.wav",
            "collection_role": "development",
        },
        {
            "speaker_id": "sealed",
            "source": "VOXPHYSICA_CONSENTED",
            "gender": "Female",
            "height_cm": "154",
            "weight_kg": "",
            "age": "26",
            "audio_paths": "sealed.wav",
            "collection_role": "sealed_test",
        },
        {
            "speaker_id": "other",
            "source": "VOXPHYSICA_CONSENTED",
            "gender": "Other",
            "height_cm": "156",
            "weight_kg": "",
            "age": "24",
            "audio_paths": "other.wav",
            "collection_role": "development",
        },
    ]

    combined = build_train_support_manifest(base, collected)

    assert [row["speaker_id"] for row in combined] == ["base", "dev"]
