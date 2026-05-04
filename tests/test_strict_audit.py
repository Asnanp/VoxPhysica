import json

import numpy as np
import pytest
import torch

from src.preprocessing.feature_extractor import build_feature_config
from src.utils.audit_utils import feature_contract_payload, validate_feature_contract
from src.utils.metrics import compute_metrics


def test_build_feature_config_reads_strict_audit_flags():
    config = {
        "data": {"sample_rate": 22050},
        "features": {
            "strict": True,
            "require_parselmouth": True,
            "mfcc": {"n_mfcc": 24, "n_fft": 1024, "hop_length": 256, "win_length": 512},
            "formants": {"n_formants": 5, "max_formant": 6000, "window_length": 0.03},
            "pitch": {"min_pitch": 70, "max_pitch": 400},
            "vtl": {"speed_of_sound": 33000},
        },
    }
    feature_config = build_feature_config(config)
    assert feature_config.sample_rate == 22050
    assert feature_config.n_mfcc == 24
    assert feature_config.strict is True
    assert feature_config.require_parselmouth is True
    assert feature_config.n_formants == 5


def test_validate_feature_contract_detects_split_hash_drift(tmp_path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    train_csv = split_dir / "train_clean.csv"
    val_csv = split_dir / "val_clean.csv"
    test_csv = split_dir / "test_clean.csv"
    for path in (train_csv, val_csv, test_csv):
        path.write_text("speaker_id\nspk_a\n", encoding="utf-8")
    (feature_root / "target_stats.json").write_text(json.dumps({"height": {"mean": 170.0, "std": 7.0}}), encoding="utf-8")

    feature_config = {"sample_rate": 16000, "n_mfcc": 40, "strict": True}
    contract = feature_contract_payload(
        feature_config=feature_config,
        split_files={
            "train": str(train_csv),
            "val": str(val_csv),
            "test": str(test_csv),
        },
    )
    (feature_root / "feature_contract.json").write_text(json.dumps(contract), encoding="utf-8")

    train_csv.write_text("speaker_id\nspk_b\n", encoding="utf-8")
    try:
        validate_feature_contract(
            feature_root=str(feature_root),
            expected_feature_config=feature_config,
            expected_split_files={
                "train": str(train_csv),
                "val": str(val_csv),
                "test": str(test_csv),
            },
        )
    except RuntimeError as exc:
        assert "split manifest hash changed" in str(exc)
    else:
        raise AssertionError("Expected validate_feature_contract to reject split hash drift.")


def test_validate_feature_contract_accepts_relocated_project_root(tmp_path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    split_dir = tmp_path / "current_project" / "data" / "splits"
    split_dir.mkdir(parents=True)
    train_csv = split_dir / "train_clean.csv"
    val_csv = split_dir / "val_clean.csv"
    test_csv = split_dir / "test_clean.csv"
    for path in (train_csv, val_csv, test_csv):
        path.write_text("speaker_id\nspk_a\n", encoding="utf-8")
    (feature_root / "target_stats.json").write_text(
        json.dumps({"height": {"mean": 170.0, "std": 7.0}}), encoding="utf-8"
    )

    feature_config = {"sample_rate": 16000, "n_mfcc": 40, "strict": True}
    contract = feature_contract_payload(
        feature_config=feature_config,
        split_files={
            "train": str(train_csv),
            "val": str(val_csv),
            "test": str(test_csv),
        },
    )
    old_root = tmp_path / "old_project"
    contract["split_files"] = {
        name: str(old_root / "data" / "splits" / path.name)
        for name, path in {
            "train": train_csv,
            "val": val_csv,
            "test": test_csv,
        }.items()
    }
    (feature_root / "feature_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )

    validate_feature_contract(
        feature_root=str(feature_root),
        expected_feature_config=feature_config,
        expected_split_files={
            "train": str(train_csv),
            "val": str(val_csv),
            "test": str(test_csv),
        },
    )


def test_compute_metrics_reports_rmse_median_and_subgroups():
    all_preds = {
        "height": [torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)],
        "weight": [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)],
        "age": [torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)],
        "gender_pred": [torch.tensor([0, 1, 1], dtype=torch.long)],
    }
    all_targets = {
        "height_raw": [torch.tensor([0.0, 2.0, 1.0], dtype=torch.float32)],
        "weight_raw": [torch.tensor([1.0, 1.0, 4.0], dtype=torch.float32)],
        "age_raw": [torch.tensor([2.0, 5.0, 5.0], dtype=torch.float32)],
        "gender": [torch.tensor([0, 1, 0], dtype=torch.long)],
        "source_id": [torch.tensor([1, 0, 1], dtype=torch.long)],
        "duration_s": [torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32)],
        "capture_quality_score": [torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32)],
    }

    metrics = compute_metrics(all_preds, all_targets, target_stats=None)
    assert metrics["height_mae"] == pytest.approx(2.0 / 3.0)
    assert "height_rmse" in metrics
    assert "height_median_ae" in metrics
    assert "height_source_nisp_mae" in metrics
    assert "height_duration_short_mae" in metrics
    assert "height_quality_low_mae" in metrics
