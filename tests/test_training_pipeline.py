import os
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.train import _coerce_int, _resolve_resume_checkpoint
from src.models.vocalmorphv2 import (
    AblationToggles,
    LossWeights,
    ModelHyperparameters,
    VocalMorphV2,
)
from src.preprocessing.dataset import collate_fn
from src.training.trainer import VocalMorphTrainer


def _make_sequence(time_steps: int = 12, input_dim: int = 136) -> torch.Tensor:
    seq = torch.randn(time_steps, input_dim, dtype=torch.float32)
    formant_idxs = [125, 127, 129, 131]
    for idx, value in zip(formant_idxs, [500.0, 1500.0, 2500.0, 3500.0]):
        seq[:, idx] = value
    seq[:, 133] = 140.0
    seq[:, 134] = 700.0
    seq[:, 135] = 16.5
    return seq


class _TinyDataset(Dataset):
    def __init__(self, target_stats):
        self.target_stats = target_stats
        self.samples = []
        raw_rows = [
            ("spk_a", 158.0, 62.0, 24.0, 0),
            ("spk_a", 158.0, 62.0, 24.0, 0),
            ("spk_b", 176.0, 74.0, 39.0, 1),
            ("spk_b", 176.0, 74.0, 39.0, 1),
        ]
        for idx, (speaker_id, height_raw, weight_raw, age_raw, gender) in enumerate(raw_rows):
            self.samples.append(
                {
                    "sequence": _make_sequence(),
                    "height": torch.tensor(
                        (height_raw - target_stats["height"]["mean"]) / target_stats["height"]["std"],
                        dtype=torch.float32,
                    ),
                    "weight": torch.tensor(
                        (weight_raw - target_stats["weight"]["mean"]) / target_stats["weight"]["std"],
                        dtype=torch.float32,
                    ),
                    "age": torch.tensor(
                        (age_raw - target_stats["age"]["mean"]) / target_stats["age"]["std"],
                        dtype=torch.float32,
                    ),
                    "gender": torch.tensor(gender, dtype=torch.long),
                    "height_raw": torch.tensor(height_raw, dtype=torch.float32),
                    "weight_raw": torch.tensor(weight_raw, dtype=torch.float32),
                    "age_raw": torch.tensor(age_raw, dtype=torch.float32),
                    "f0_mean": torch.tensor(140.0, dtype=torch.float32),
                    "formant_spacing_mean": torch.tensor(700.0, dtype=torch.float32),
                    "vtl_mean": torch.tensor(16.5, dtype=torch.float32),
                    "weight_mask": torch.tensor(1.0, dtype=torch.float32),
                    "source_id": torch.tensor(1, dtype=torch.long),
                    "speaker_id": speaker_id,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def test_coerce_int_supports_simple_arithmetic():
    assert _coerce_int(4, "x") == 4
    assert _coerce_int("16*2", "x") == 32
    assert _coerce_int("(8 + 4) // 2", "x") == 6
    with pytest.raises(ValueError):
        _coerce_int("3/2", "x")


def test_v2_trainer_uses_native_loss_and_reports_speaker_metrics(tmp_path):
    torch.manual_seed(5)
    target_stats = {
        "height": {"mean": 172.0, "std": 8.0},
        "weight": {"mean": 68.0, "std": 10.0},
        "age": {"mean": 31.0, "std": 7.0},
    }
    dataset = _TinyDataset(target_stats)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = VocalMorphV2(
        input_dim=136,
        ecapa_channels=64,
        ecapa_scale=4,
        conformer_d_model=32,
        conformer_heads=4,
        conformer_blocks=1,
        dropout=0.1,
        target_stats=target_stats,
    )

    config = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "num_workers": 0,
            "device": "cpu",
            "mixed_precision": False,
            "ema": {"enabled": True, "decay": 0.99, "use_for_eval": True},
            "loss": {"type": "vtsl_v2", "use_gender_class_weights": False},
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]},
            "scheduler": {"T_0": 2, "T_mult": 1, "eta_min": 1e-5},
            "early_stopping": {"enabled": False, "monitor": "height_mae_speaker", "mode": "min"},
            "gradient_clipping": {"enabled": True, "max_norm": 1.0},
        },
        "evaluation": {"inference": {"use_ensemble": True, "deterministic": True, "n_crops": 2, "crop_size": 8}},
        "logging": {
            "tensorboard": {"log_dir": str(tmp_path / "logs")},
            "checkpoint": {"dir": str(tmp_path / "ckpts"), "save_top_k": 1, "monitor": "height_mae_speaker", "mode": "min"},
        },
        "physics": {"vtl_height_constraint": {"ratio": 6.7}},
    }

    trainer = VocalMorphTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        target_stats=target_stats,
    )

    assert trainer.use_native_v2_loss is True
    assert trainer.use_ema is True
    train_losses = trainer._train_epoch(epoch=1)
    assert torch.isfinite(torch.tensor(train_losses["total"]))
    assert trainer.ema_state

    val_metrics = trainer._val_epoch_on(loader)
    assert "height_mae" in val_metrics
    assert "height_mae_speaker" in val_metrics
    assert "height_mae_speaker_omega" in val_metrics
    assert "height_edge_balance_gap_speaker_mae" in val_metrics
    assert "height_balanced_frontier_speaker_mae" in val_metrics
    assert "height_edge_guarded_frontier_speaker_mae" in val_metrics
    assert "height_calibration_mae" in val_metrics
    assert "height_interval_68" in val_metrics
    assert torch.isfinite(torch.tensor(val_metrics["total"]))
    assert torch.isfinite(torch.tensor(val_metrics["height_mae_speaker"]))
    assert torch.isfinite(torch.tensor(val_metrics["height_mae_speaker_omega"]))
    assert torch.isfinite(torch.tensor(val_metrics["height_edge_balance_gap_speaker_mae"]))
    assert torch.isfinite(torch.tensor(val_metrics["height_balanced_frontier_speaker_mae"]))
    assert torch.isfinite(torch.tensor(val_metrics["height_edge_guarded_frontier_speaker_mae"]))
    assert torch.isfinite(torch.tensor(val_metrics["height_calibration_mae"]))
    trainer.close()


def test_trainer_applies_focal_overrides_to_native_v2_loss(tmp_path):
    torch.manual_seed(6)
    target_stats = {
        "height": {"mean": 172.0, "std": 8.0},
        "weight": {"mean": 68.0, "std": 10.0},
        "age": {"mean": 31.0, "std": 7.0},
    }
    dataset = _TinyDataset(target_stats)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = VocalMorphV2(
        input_dim=136,
        ecapa_channels=64,
        ecapa_scale=4,
        conformer_d_model=32,
        conformer_heads=4,
        conformer_blocks=1,
        dropout=0.1,
        target_stats=target_stats,
    )

    config = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "device": "cpu",
            "mixed_precision": False,
            "ema": {"enabled": False, "decay": 0.99, "use_for_eval": False},
            "loss": {
                "type": "vtsl_v2",
                "use_gender_class_weights": False,
                "focal_after_epoch": 3,
                "focal_ema_decay": 0.90,
            },
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]},
            "scheduler": {"type": "cosine_annealing", "T_max": 4, "eta_min": 1e-5},
            "early_stopping": {"enabled": False, "monitor": "height_mae_speaker", "mode": "min"},
            "gradient_clipping": {"enabled": True, "max_norm": 1.0},
        },
        "evaluation": {"inference": {"use_ensemble": False, "deterministic": True, "n_crops": 1}},
        "logging": {
            "tensorboard": {"log_dir": str(tmp_path / "logs")},
            "checkpoint": {"dir": str(tmp_path / "ckpts"), "save_top_k": 1, "monitor": "height_mae_speaker", "mode": "min"},
        },
        "physics": {"vtl_height_constraint": {"ratio": 6.7}},
    }

    trainer = VocalMorphTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        target_stats=target_stats,
    )
    focal_target = getattr(trainer.criterion, "base_loss", trainer.criterion)
    assert focal_target.focal_after_epoch == 3
    assert focal_target.focal_ema_decay == pytest.approx(0.90)
    trainer.close()


def test_trainer_writes_recovery_checkpoints_and_can_resume(tmp_path):
    torch.manual_seed(7)
    target_stats = {
        "height": {"mean": 172.0, "std": 8.0},
        "weight": {"mean": 68.0, "std": 10.0},
        "age": {"mean": 31.0, "std": 7.0},
    }
    dataset = _TinyDataset(target_stats)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    def _build_model():
        return VocalMorphV2(
            input_dim=136,
            ecapa_channels=64,
            ecapa_scale=4,
            conformer_d_model=32,
            conformer_heads=4,
            conformer_blocks=1,
            dropout=0.1,
            target_stats=target_stats,
        )

    config = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "device": "cpu",
            "mixed_precision": False,
            "allow_tf32": False,
            "ema": {"enabled": True, "decay": 0.99, "use_for_eval": True},
            "loss": {"type": "vtsl_v2", "use_gender_class_weights": False},
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]},
            "scheduler": {"T_0": 2, "T_mult": 1, "eta_min": 1e-5},
            "early_stopping": {"enabled": False, "monitor": "height_mae_speaker", "mode": "min"},
            "gradient_clipping": {"enabled": True, "max_norm": 1.0},
            "seed": 7,
        },
        "evaluation": {"inference": {"use_ensemble": True, "deterministic": True, "n_crops": 2, "crop_size": 8}},
        "logging": {
            "tensorboard": {"log_dir": str(tmp_path / "logs")},
            "checkpoint": {"dir": str(tmp_path / "ckpts"), "save_top_k": 2, "monitor": "height_mae_speaker", "mode": "min"},
        },
        "physics": {"vtl_height_constraint": {"ratio": 6.7}},
    }

    trainer = VocalMorphTrainer(
        model=_build_model(),
        train_loader=loader,
        val_loader=loader,
        config=config,
        target_stats=target_stats,
        train_eval_loader=loader,
    )
    trainer.train()
    trainer.close()

    ckpt_dir = tmp_path / "ckpts"
    assert (ckpt_dir / "last.ckpt").exists()
    assert (ckpt_dir / "last_good.ckpt").exists()
    assert (ckpt_dir / "best.ckpt").exists()
    assert (tmp_path / "metrics.jsonl").exists()

    payload = torch.load(ckpt_dir / "last.ckpt", map_location="cpu", weights_only=False)
    assert payload["epoch"] == 1
    assert "optimizer_state_dict" in payload
    assert "scheduler_state_dict" in payload
    assert "scaler_state_dict" in payload
    assert "rng_state" in payload
    assert "config" in payload
    assert payload["global_step"] >= 1

    resolved = _resolve_resume_checkpoint(config)
    assert resolved == str(ckpt_dir / "last.ckpt")

    resumed = VocalMorphTrainer(
        model=_build_model(),
        train_loader=loader,
        val_loader=loader,
        config=config,
        target_stats=target_stats,
        train_eval_loader=loader,
    )
    resumed.restore_from_checkpoint(payload)
    assert resumed.start_epoch == 2
    assert resumed.last_completed_epoch == 1
    assert resumed.optimizer_step_count == payload["optimizer_step_count"]
    resumed.close()


def test_trainer_accepts_old_checkpoint_without_reliability_tower(tmp_path):
    torch.manual_seed(9)
    target_stats = {
        "height": {"mean": 172.0, "std": 8.0},
        "weight": {"mean": 68.0, "std": 10.0},
        "age": {"mean": 31.0, "std": 7.0},
    }
    dataset = _TinyDataset(target_stats)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = VocalMorphV2(
        input_dim=136,
        ecapa_channels=64,
        ecapa_scale=4,
        conformer_d_model=32,
        conformer_heads=4,
        conformer_blocks=1,
        dropout=0.1,
        target_stats=target_stats,
    )

    config = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "device": "cpu",
            "mixed_precision": False,
            "allow_tf32": False,
            "ema": {"enabled": False, "decay": 0.99, "use_for_eval": False},
            "loss": {"type": "vtsl_v2", "use_gender_class_weights": False},
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]},
            "scheduler": {"T_0": 2, "T_mult": 1, "eta_min": 1e-5},
            "early_stopping": {"enabled": False, "monitor": "height_mae_speaker", "mode": "min"},
            "gradient_clipping": {"enabled": True, "max_norm": 1.0},
            "seed": 9,
        },
        "evaluation": {"inference": {"use_ensemble": False, "deterministic": True, "n_crops": 1}},
        "logging": {
            "tensorboard": {"log_dir": str(tmp_path / "logs")},
            "checkpoint": {"dir": str(tmp_path / "ckpts"), "save_top_k": 1, "monitor": "height_mae_speaker", "mode": "min"},
        },
        "physics": {"vtl_height_constraint": {"ratio": 6.7}},
    }

    trainer = VocalMorphTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        target_stats=target_stats,
    )
    stripped_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("reliability_tower.")
    }
    trainer._load_model_checkpoint_state(stripped_state)
    trainer.close()


def test_stage3d_alignment_losses_and_feature_smoothing_stay_finite(tmp_path):
    torch.manual_seed(11)
    target_stats = {
        "height": {"mean": 172.0, "std": 8.0},
        "weight": {"mean": 68.0, "std": 10.0},
        "age": {"mean": 31.0, "std": 7.0},
    }
    dataset = _TinyDataset(target_stats)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = VocalMorphV2(
        input_dim=136,
        ecapa_channels=64,
        ecapa_scale=4,
        conformer_d_model=32,
        conformer_heads=4,
        conformer_blocks=1,
        dropout=0.1,
        target_stats=target_stats,
    )

    config = {
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "device": "cpu",
            "mixed_precision": False,
            "feature_smoothing_std": 0.012,
            "lr_warmup_epochs": 1,
            "lr_warmup_start_factor": 0.25,
            "ema": {"enabled": True, "decay": 0.99, "use_for_eval": True},
            "loss": {"type": "vtsl_v2", "use_gender_class_weights": False},
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]},
            "scheduler": {"T_0": 2, "T_mult": 1, "eta_min": 1e-5},
            "speaker_alignment": {
                "enable_pooled_height": True,
                "enable_consistency": True,
                "enable_ranking": True,
                "pooling_method": "mean",
                "consistency_mode": "variance",
                "warmup_start_epoch": 2,
                "warmup_end_epoch": 2,
                "pooled_height_weight_max": 0.20,
                "consistency_weight_max": 0.08,
                "ranking_weight_max": 0.06,
                "ranking_min_height_delta_cm": 4.0,
                "ranking_margin_cm": 1.0,
                "height_bin_loss_start_epoch": 1,
                "height_bin_loss_weight_short": 1.35,
                "height_bin_loss_weight_medium": 1.0,
                "height_bin_loss_weight_tall": 1.10,
            },
            "early_stopping": {"enabled": False, "monitor": "height_mae_speaker", "mode": "min"},
            "gradient_clipping": {"enabled": True, "max_norm": 1.0},
        },
        "evaluation": {"inference": {"use_ensemble": False, "deterministic": True, "n_crops": 1}},
        "logging": {
            "tensorboard": {"log_dir": str(tmp_path / "logs")},
            "checkpoint": {"dir": str(tmp_path / "ckpts"), "save_top_k": 1, "monitor": "height_mae_speaker", "mode": "min"},
        },
        "physics": {"vtl_height_constraint": {"ratio": 6.7}},
    }

    trainer = VocalMorphTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        target_stats=target_stats,
    )

    train_losses = trainer._train_epoch(epoch=2)
    assert torch.isfinite(torch.tensor(train_losses["total"]))
    assert "speaker_pooled_height" in train_losses
    assert "speaker_clip_consistency" in train_losses
    assert "speaker_height_ranking" in train_losses
    assert torch.isfinite(torch.tensor(train_losses["speaker_pooled_height"]))
    assert torch.isfinite(torch.tensor(train_losses["speaker_clip_consistency"]))
    assert torch.isfinite(torch.tensor(train_losses["speaker_height_ranking"]))
    trainer.close()


def test_trainer_supports_monotonic_cosine_scheduler_without_restart(tmp_path):
    torch.manual_seed(13)
    target_stats = {
        "height": {"mean": 172.0, "std": 8.0},
        "weight": {"mean": 68.0, "std": 10.0},
        "age": {"mean": 31.0, "std": 7.0},
    }
    dataset = _TinyDataset(target_stats)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = VocalMorphV2(
        input_dim=136,
        ecapa_channels=64,
        ecapa_scale=4,
        conformer_d_model=32,
        conformer_heads=4,
        conformer_blocks=1,
        dropout=0.1,
        target_stats=target_stats,
    )

    config = {
        "training": {
            "epochs": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "device": "cpu",
            "mixed_precision": False,
            "ema": {"enabled": False, "decay": 0.99, "use_for_eval": False},
            "loss": {"type": "vtsl_v2", "use_gender_class_weights": False},
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]},
            "scheduler": {"type": "cosine_annealing", "T_max": 6, "eta_min": 1e-5},
            "early_stopping": {
                "enabled": False,
                "monitor": "height_mae_speaker",
                "mode": "min",
            },
            "gradient_clipping": {"enabled": True, "max_norm": 1.0},
        },
        "evaluation": {
            "inference": {
                "use_ensemble": False,
                "deterministic": True,
                "n_crops": 1,
            }
        },
        "logging": {
            "tensorboard": {"log_dir": str(tmp_path / "logs")},
            "checkpoint": {
                "dir": str(tmp_path / "ckpts"),
                "save_top_k": 1,
                "monitor": "height_mae_speaker",
                "mode": "min",
            },
        },
        "physics": {"vtl_height_constraint": {"ratio": 6.7}},
    }

    trainer = VocalMorphTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        target_stats=target_stats,
    )

    lrs = []
    for epoch in range(1, 4):
        losses = trainer._train_epoch(epoch=epoch)
        assert torch.isfinite(torch.tensor(losses["total"]))
        lrs.append(trainer.optimizer.param_groups[0]["lr"])

    assert trainer.scheduler.__class__.__name__ == "CosineAnnealingLR"
    assert lrs[1] <= lrs[0]
    assert lrs[2] <= lrs[1]
    trainer.close()


def test_height_first_proper_v2_path_emits_auxiliary_height_bin_loss(tmp_path):
    torch.manual_seed(17)
    target_stats = {
        "height": {"mean": 172.0, "std": 8.0},
        "weight": {"mean": 68.0, "std": 10.0},
        "age": {"mean": 31.0, "std": 7.0},
    }
    dataset = _TinyDataset(target_stats)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = VocalMorphV2(
        input_dim=136,
        target_stats=target_stats,
        toggles=AblationToggles(
            use_physics_branch=False,
            use_cross_attention=False,
            use_reliability_gate=False,
            use_height_prior=False,
            use_height_adapter=True,
            use_domain_adv=False,
            use_diversity_loss=False,
            use_feature_mixup=False,
            use_feature_normalization=True,
            use_acoustic_physics_consistency=False,
            use_ranking_loss=False,
            use_speaker_consistency=False,
            use_uncertainty_calibration=False,
            use_shoulder_head=False,
            use_waist_head=False,
            use_kendall_weights=False,
            use_height_context_refiner=True,
            use_height_bin_aux=True,
        ),
        loss_weights=LossWeights(
            height=4.0,
            weight=0.0,
            age=0.0,
            shoulder=0.0,
            waist=0.0,
            gender=0.0,
            vtsl=0.0,
            physics_penalty=0.0,
            domain_adv=0.0,
            ranking=0.0,
            diversity=0.0,
            speaker_consistency=0.0,
            uncertainty_calibration=0.0,
            height_bin_aux=0.25,
        ),
        hyperparameters=ModelHyperparameters(
            ecapa_channels=64,
            ecapa_scale=4,
            conformer_d_model=32,
            conformer_heads=4,
            conformer_blocks=1,
            dropout=0.1,
            branch_dropout=0.05,
            height_context_hidden_dim=64,
            height_context_blocks=2,
            height_context_scale=0.35,
            height_bin_hidden_dim=32,
            height_bin_classes=3,
            regression_hidden_dim=64,
            fused_dim=64,
            physics_embedding_dim=128,
            physics_fusion_dim=32,
            height_adapter_hidden_dim=32,
            physics_gate_hidden_dim=32,
        ),
    )

    config = {
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "device": "cpu",
            "mixed_precision": False,
            "feature_smoothing_std": 0.008,
            "lr_warmup_epochs": 1,
            "lr_warmup_start_factor": 0.25,
            "ema": {"enabled": False, "decay": 0.99, "use_for_eval": False},
            "loss": {"type": "vtsl_v2", "use_gender_class_weights": False},
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]},
            "scheduler": {"type": "cosine_annealing", "T_max": 4, "eta_min": 1e-5},
            "speaker_alignment": {
                "enable_pooled_height": False,
                "enable_consistency": False,
                "enable_ranking": False,
                "height_bin_loss_start_epoch": 1,
                "height_bin_loss_weight_short": 1.35,
                "height_bin_loss_weight_medium": 1.0,
                "height_bin_loss_weight_tall": 1.10,
            },
            "early_stopping": {"enabled": False, "monitor": "height_mae_speaker", "mode": "min"},
            "gradient_clipping": {"enabled": True, "max_norm": 1.0},
        },
        "evaluation": {"inference": {"use_ensemble": False, "deterministic": True, "n_crops": 1}},
        "logging": {
            "tensorboard": {"log_dir": str(tmp_path / "logs")},
            "checkpoint": {"dir": str(tmp_path / "ckpts"), "save_top_k": 1, "monitor": "height_mae_speaker", "mode": "min"},
        },
        "physics": {"vtl_height_constraint": {"ratio": 6.7}},
    }

    trainer = VocalMorphTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        target_stats=target_stats,
    )

    train_losses = trainer._train_epoch(epoch=1)
    assert "height_bin_aux" in train_losses
    assert torch.isfinite(torch.tensor(train_losses["total"]))
    assert torch.isfinite(torch.tensor(train_losses["height_bin_aux"]))
    trainer.close()
