from pathlib import Path

import numpy as np

from src.research.speaker_height_ensemble import (
    load_research_tables,
    run_research_experiment,
)
import scripts.build_feature_splits as build_feature_splits


def _write_feature(path: Path, speaker_id: str, height_cm: float, clip_idx: int) -> None:
    base = height_cm / 200.0 + clip_idx * 0.01
    sequence = np.stack(
        [
            np.linspace(base, base + 0.2, 8, dtype=np.float32),
            np.linspace(base * 0.5, base * 0.5 + 0.1, 8, dtype=np.float32),
            np.full(8, base * 1.5, dtype=np.float32),
            np.sin(np.linspace(0, 1, 8, dtype=np.float32) + base),
        ],
        axis=1,
    )
    np.savez(
        path,
        sequence=sequence.astype(np.float32),
        speaker_id=np.array(speaker_id, dtype=object),
        height_cm=np.float32(height_cm),
        gender=np.int64(1 if height_cm >= 170 else 0),
        source=np.array("NISP", dtype=object),
        f0_mean=np.float32(220.0 - height_cm * 0.4),
        formant_spacing_mean=np.float32(1300.0 - height_cm),
        vtl_mean=np.float32(height_cm / 6.7),
        jitter=np.float32(0.01),
        shimmer=np.float32(0.02),
        hnr=np.float32(20.0),
        duration_s=np.float32(3.0 + clip_idx),
        voiced_ratio=np.float32(0.9),
        invalid_spacing_rate=np.float32(0.0),
        invalid_vtl_rate=np.float32(0.0),
        speech_ratio=np.float32(0.95),
        snr_db_estimate=np.float32(28.0),
        capture_quality_score=np.float32(0.9),
        distance_cm_estimate=np.float32(18.0),
        distance_confidence=np.float32(0.8),
        clipped_ratio=np.float32(0.0),
        quality_ok=np.bool_(True),
    )


def _make_feature_store(root: Path) -> None:
    splits = {
        "train": [150, 156, 162, 168, 174, 180, 186, 192],
        "val": [152, 166, 178, 190],
        "test": [154, 170, 182, 194],
    }
    for split, heights in splits.items():
        split_dir = root / split
        split_dir.mkdir(parents=True)
        for idx, height in enumerate(heights):
            speaker_id = f"{split}_spk_{idx:02d}"
            for clip_idx in range(2):
                _write_feature(split_dir / f"{speaker_id}_{clip_idx:03d}.npz", speaker_id, float(height), clip_idx)


def test_load_research_tables_builds_one_row_per_speaker(tmp_path):
    feature_root = tmp_path / "features"
    _make_feature_store(feature_root)

    tables = load_research_tables(str(feature_root))

    assert tables["train"].x.shape[0] == 8
    assert tables["val"].x.shape[0] == 4
    assert tables["test"].x.shape[0] == 4
    assert tables["train"].x.shape[1] == tables["test"].x.shape[1]
    assert "speaker_n_clips" in tables["train"].feature_names


def test_run_research_experiment_writes_metrics_and_predictions(tmp_path):
    feature_root = tmp_path / "features"
    output_dir = tmp_path / "out"
    _make_feature_store(feature_root)

    payload = run_research_experiment(
        features_dir=str(feature_root),
        output_dir=str(output_dir),
        seed=11,
        model_names=("ridge", "huber"),
        ensemble_trials=20,
        save_model=False,
    )

    assert payload["speaker_counts"]["train"] == 8
    assert "calibrated_edge" in payload["final_test"]
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "predictions_test.csv").exists()
    assert (output_dir / "summary.md").exists()


def test_feature_builder_falls_back_from_cleaned_nisp_to_raw_tree(tmp_path, monkeypatch):
    raw_path = tmp_path / "data" / "NISP-Dataset" / "Tamil_master" / "sample.wav"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"RIFF")
    cleaned_path = "data/audio_clean/train/data/NISP-Dataset/Tamil_master/sample.wav"
    monkeypatch.setattr(build_feature_splits, "ROOT", str(tmp_path))

    resolved = build_feature_splits._resolve_audio_path(cleaned_path)

    assert Path(resolved) == raw_path
