from pathlib import Path

import numpy as np

from scripts.audit_speaker_leakage import speaker_leakage_report
from scripts.run_omega_ladder import OMEGA_STAGES
from scripts.run_v2_ablations import build_ablation_config, summarize_metrics


def _write_feature(path: Path, speaker_id: str):
    np.savez(path, sequence=np.zeros((4, 8), dtype=np.float32), speaker_id=np.array(speaker_id, dtype=object))


def test_speaker_leakage_report_detects_overlap(tmp_path):
    feature_root = tmp_path / "features"
    for split in ("train", "val", "test"):
        (feature_root / split).mkdir(parents=True)

    _write_feature(feature_root / "train" / "a_001.npz", "spk_a")
    _write_feature(feature_root / "val" / "b_001.npz", "spk_b")
    _write_feature(feature_root / "test" / "a_999.npz", "spk_a")

    report = speaker_leakage_report(str(feature_root))
    assert report["has_leakage"] is True
    assert report["overlap_counts"]["train_test"] == 1


def test_build_ablation_config_targets_speaker_metric(tmp_path):
    base_config = {
        "training": {"epochs": 5},
        "logging": {"tensorboard": {}, "checkpoint": {}},
        "model": {"v2": {"toggles": {"use_cross_attention": True}}},
    }
    run_dir = tmp_path / "baseline" / "seed_11"
    config = build_ablation_config(
        base_config,
        {"model.v2.toggles.use_cross_attention": False},
        run_dir=str(run_dir),
        seed=11,
    )
    assert config["training"]["seed"] == 11
    assert config["training"]["early_stopping"]["monitor"] == "height_mae_speaker"
    assert config["logging"]["checkpoint"]["monitor"] == "height_mae_speaker"
    assert config["model"]["v2"]["toggles"]["use_cross_attention"] is False


def test_summarize_metrics_reports_mean_and_std():
    records = [
        {
            "ablation": "baseline",
            "final_val": {"height_mae_speaker": 2.4, "height_calibration_mae": 0.8, "height_uncertainty_error_corr": 0.5},
            "final_test": {"height_mae_speaker": 2.6},
        },
        {
            "ablation": "baseline",
            "final_val": {"height_mae_speaker": 2.8, "height_calibration_mae": 0.9, "height_uncertainty_error_corr": 0.4},
            "final_test": {"height_mae_speaker": 2.7},
        },
    ]
    summary = summarize_metrics(records)
    baseline = summary["baseline"]
    assert baseline["n_runs"] == 2.0
    assert baseline["final_val_height_mae_speaker_mean"] == 2.6
    assert baseline["final_test_height_mae_speaker_mean"] == 2.65


def test_omega_stage3e_is_registered_with_stable_scheduler():
    stage = OMEGA_STAGES["stage3e_height_only_stable_bin_weighted"]
    overrides = stage["overrides"]
    assert stage["monitor"] == "height_mae_speaker"
    assert overrides["training.speaker_batching.enabled"] is False
    assert overrides["training.scheduler.type"] == "cosine_annealing"
    assert overrides["training.scheduler.T_max"] == 10
    assert overrides["training.speaker_alignment.enable_pooled_height"] is False
    assert overrides["training.speaker_alignment.height_bin_loss_weight_short"] == 1.25


def test_omega_stage3f_is_registered_as_stage3c_style_long_run():
    stage = OMEGA_STAGES["stage3f_height_only_long_stable"]
    overrides = stage["overrides"]
    assert stage["monitor"] == "height_mae_speaker"
    assert overrides["training.speaker_batching.enabled"] is False
    assert overrides["training.scheduler.type"] == "cosine_annealing"
    assert overrides["training.scheduler.T_max"] == 50
    assert overrides["training.loss.task_weights.height"] == 4.0
    assert overrides["training.speaker_alignment.enable_pooled_height"] is False


def test_omega_stage4_proper_v2_height_first_is_registered():
    stage = OMEGA_STAGES["stage4_proper_v2_height_first"]
    overrides = stage["overrides"]
    assert stage["monitor"] == "height_mae_speaker"
    assert overrides["model.v2.toggles.use_physics_branch"] is False
    assert overrides["model.v2.toggles.use_height_context_refiner"] is True
    assert overrides["model.v2.toggles.use_height_bin_aux"] is True
    assert overrides["model.v2.loss_weights.height_bin_aux"] == 0.25
    assert overrides["training.speaker_batching.enabled"] is False
    assert overrides["training.scheduler.type"] == "cosine_annealing"


def test_omega_stage3g_is_registered_with_stronger_regularization():
    stage = OMEGA_STAGES["stage3g_height_only_strong_reg"]
    overrides = stage["overrides"]
    assert stage["monitor"] == "height_mae_speaker"
    assert overrides["model.v2.dropout"] == 0.35
    assert overrides["model.v2.branch_dropout"] == 0.20
    assert overrides["model.v2.drop_path_rate"] == 0.10
    assert overrides["training.scheduler.type"] == "cosine_annealing"
    assert overrides["training.scheduler.T_max"] == 50
    assert overrides["training.loss.focal_after_epoch"] == 3
    assert overrides["training.loss.focal_ema_decay"] == 0.90
    assert overrides["training.ema.decay"] == 0.999
    assert overrides["training.speaker_alignment.height_bin_loss_weight_short"] == 3.5
