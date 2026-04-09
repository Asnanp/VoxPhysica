import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.vocalmorphv2 import (
    AblationToggles,
    BayesianHeightHead,
    ConformerBlock,
    PhysicsFeatureSpec,
    PositiveLinear,
    ProbabilisticRegressionHead,
    VocalMorphV2,
    VocalTractSimulatorLossV2,
    _sanitize_padding_mask,
    _validate_class_labels,
    _validate_sequence_inputs,
    aggregate_by_speaker,
    build_vocalmorph_v2,
)


def _make_features(batch_size: int = 4, time_steps: int = 16, input_dim: int = 136) -> torch.Tensor:
    torch.manual_seed(7)
    features = torch.randn(batch_size, time_steps, input_dim, dtype=torch.float32)
    spec = PhysicsFeatureSpec()
    for idx, value in [
        (spec.formant_freq_idx(0), 500.0),
        (spec.formant_freq_idx(1), 1500.0),
        (spec.formant_freq_idx(2), 2500.0),
        (spec.formant_freq_idx(3), 3500.0),
        (spec.f0_idx, 140.0),
        (spec.spacing_idx, 700.0),
        (spec.vtl_idx, 16.5),
    ]:
        features[:, :, idx] = value + 0.01 * torch.randn(batch_size, time_steps)
    return features


def _make_batch(batch_size: int = 4, time_steps: int = 16, input_dim: int = 136):
    features = _make_features(batch_size=batch_size, time_steps=time_steps, input_dim=input_dim)
    padding_mask = torch.zeros(batch_size, time_steps, dtype=torch.bool)
    if time_steps > 3:
        padding_mask[0, -2:] = True
        padding_mask[-1, -1:] = True
    domain = torch.tensor([0, 0, 1, 1][:batch_size], dtype=torch.long)
    gender = torch.tensor([0, 0, 1, 1][:batch_size], dtype=torch.long)
    targets = {
        "height": torch.linspace(-0.5, 0.5, batch_size),
        "weight": torch.linspace(-0.25, 0.25, batch_size),
        "age": torch.linspace(-1.0, 1.0, batch_size),
        "shoulder": torch.linspace(-0.1, 0.3, batch_size),
        "waist": torch.linspace(-0.2, 0.2, batch_size),
        "height_raw": torch.linspace(160.0, 176.0, batch_size),
        "weight_raw": torch.linspace(58.0, 75.0, batch_size),
        "age_raw": torch.linspace(22.0, 46.0, batch_size),
        "shoulder_raw": torch.linspace(39.0, 46.0, batch_size),
        "waist_raw": torch.linspace(70.0, 92.0, batch_size),
        "gender": gender.clone(),
        "domain": domain.clone(),
        "weight_mask": torch.ones(batch_size),
        "shoulder_mask": torch.ones(batch_size),
        "waist_mask": torch.ones(batch_size),
        "f0_mean": torch.full((batch_size,), 140.0),
        "formant_spacing_mean": torch.full((batch_size,), 700.0),
        "vtl_mean": torch.full((batch_size,), 16.5),
    }
    return features, padding_mask, domain, targets


@pytest.fixture
def small_model():
    torch.manual_seed(11)
    model = VocalMorphV2(
        input_dim=136,
        ecapa_channels=128,
        ecapa_scale=4,
        conformer_d_model=64,
        conformer_heads=4,
        conformer_blocks=2,
        dropout=0.1,
        target_stats={
            "height": {"mean": 170.0, "std": 8.0},
            "weight": {"mean": 68.0, "std": 11.0},
            "age": {"mean": 33.0, "std": 8.0},
            "shoulder": {"mean": 43.0, "std": 4.0},
            "waist": {"mean": 81.0, "std": 9.0},
        },
    )
    return model


def test_sanitize_padding_mask_repairs_all_padded_rows():
    mask = torch.ones(2, 5, dtype=torch.bool)
    fixed = _sanitize_padding_mask(mask)
    assert fixed.dtype == torch.bool
    assert torch.equal(fixed[:, 0], torch.zeros(2, dtype=torch.bool))
    assert fixed[:, 1:].all()


def test_validate_sequence_inputs_and_class_labels():
    features = _make_features(batch_size=2, time_steps=4)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    validated_mask = _validate_sequence_inputs(features, mask, expected_feature_dim=136)
    assert torch.equal(validated_mask, mask)

    with pytest.raises(ValueError):
        _validate_sequence_inputs(features[:, :, :8], expected_feature_dim=136)

    with pytest.raises(ValueError):
        _validate_class_labels(torch.tensor([0.0, 2.5]), "gender", 2)


def test_positive_linear_and_physics_feature_spec():
    layer = PositiveLinear(5, 3)
    x = torch.randn(2, 5)
    y = layer(x)
    assert y.shape == (2, 3)
    assert torch.all(F.softplus(layer.weight_raw) > 0)

    spec = PhysicsFeatureSpec(n_mfcc=13, include_delta=True, include_delta_delta=False, n_formants=4)
    assert spec.formant_freq_idx(0) < spec.f0_idx < spec.spacing_idx < spec.minimum_input_dim


def test_conformer_block_and_probabilistic_heads_are_finite():
    block = ConformerBlock(d_model=32, n_heads=4, dropout=0.1)
    x = torch.randn(2, 7, 32)
    mask = torch.zeros(2, 7, dtype=torch.bool)
    y = block(x, padding_mask=mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    head = ProbabilisticRegressionHead(in_dim=32, hidden_dim=24, dropout=0.1)
    out = head(torch.randn(3, 32))
    assert out["mu"].shape == (3,)
    assert out["var"].shape == (3,)
    assert torch.all(out["var"] > 0)

    height_head = BayesianHeightHead(in_dim=32, hidden_dim=24, dropout=0.1)
    height_out = height_head(
        torch.randn(3, 32),
        physics_residual=torch.full((3,), 0.1),
        prior_residual=torch.full((3,), -0.1),
    )
    assert height_out["mu"].shape == (3,)
    assert height_out["mu_base"].shape == (3,)
    assert torch.isfinite(height_out["logvar"]).all()


@pytest.mark.parametrize("batch_size,time_steps", [(1, 1), (2, 5), (4, 17)])
def test_forward_shape_contracts_and_finite_outputs(small_model, batch_size, time_steps):
    features = _make_features(batch_size=batch_size, time_steps=time_steps)
    mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
    mask[:, 0] = False
    domain = torch.zeros(batch_size, dtype=torch.long)
    small_model.eval()

    out = small_model(
        features,
        padding_mask=mask,
        domain=domain,
        return_diagnostics=True,
        return_attention_maps=True,
    )
    for key in ("height", "weight", "age", "shoulder", "waist"):
        assert out[key].shape == (batch_size,)
        assert out[f"{key}_var"].shape == (batch_size,)
        assert torch.isfinite(out[key]).all()
        assert torch.isfinite(out[f"{key}_var"]).all()
    assert out["gender_logits"].shape == (batch_size, 2)
    assert out["quality_score"].shape == (batch_size,)
    assert out["valid_frames"].shape == (batch_size,)
    assert isinstance(out["cross_attention_maps"], list)
    assert "diagnostics" in out


def test_forward_with_partial_targets_and_loss_is_safe(small_model):
    features, mask, domain, targets = _make_batch(batch_size=4, time_steps=12)
    partial_targets = {
        "height": targets["height"],
        "gender": targets["gender"],
        "domain": targets["domain"],
    }
    out = small_model(features, padding_mask=mask, domain=domain, targets=partial_targets)
    assert "losses" in out
    for value in out["losses"].values():
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_backward_pass_has_finite_gradients(small_model):
    small_model.train()
    features, mask, domain, targets = _make_batch(batch_size=4, time_steps=12)
    out = small_model(features, padding_mask=mask, domain=domain, targets=targets)
    out["losses"]["total"].backward()
    grads = [param.grad for param in small_model.parameters() if param.requires_grad and param.grad is not None]
    assert grads, "Expected at least one gradient tensor"
    for grad in grads:
        assert torch.isfinite(grad).all()


def test_loss_module_handles_missing_targets():
    model = VocalMorphV2(
        input_dim=136,
        ecapa_channels=128,
        ecapa_scale=4,
        conformer_d_model=64,
        conformer_heads=4,
        conformer_blocks=1,
        dropout=0.1,
    )
    features, mask, domain, targets = _make_batch(batch_size=4, time_steps=8)
    preds = model(features, padding_mask=mask, domain=domain, return_aux=False)
    criterion = VocalTractSimulatorLossV2()
    losses = criterion(
        preds,
        {
            "height": targets["height"],
            "gender": targets["gender"],
            "domain": targets["domain"],
        },
    )
    for key in ("height_nll", "weight", "age", "shoulder", "waist", "gender", "domain_adv", "total"):
        assert key in losses
        assert torch.isfinite(losses[key])


def test_uncertainty_inference_returns_all_targets_and_variances(small_model):
    features, mask, domain, _ = _make_batch(batch_size=4, time_steps=12)
    deterministic = small_model.predict_with_uncertainty(
        features,
        padding_mask=mask,
        domain=domain,
        deterministic=True,
        n_crops=2,
        crop_size=8,
    )
    stochastic = small_model.predict_with_uncertainty(
        features,
        padding_mask=mask,
        domain=domain,
        deterministic=False,
        n_samples=3,
        n_crops=2,
        crop_size=8,
    )
    for result in (deterministic, stochastic):
        for key in ("height", "weight", "age", "shoulder", "waist"):
            assert result[key]["mean"].shape == (4,)
            assert result[key]["var"].shape == (4,)
            assert result[key]["std"].shape == (4,)
            assert torch.all(result[key]["var"] >= 0)
            assert "aleatoric_var" in result[key]
            assert "epistemic_var" in result[key]
        assert result["gender"]["probs"].shape == (4, 2)
        assert result["gender"]["pred"].shape == (4,)


def test_speaker_aggregation_mean_and_inverse_variance():
    speaker_ids = ["a", "a", "b", "b"]
    preds = {
        "height": torch.tensor([0.1, 0.2, -0.1, -0.2]),
        "weight": torch.tensor([0.0, 0.3, -0.2, -0.1]),
        "gender_probs": torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]]),
    }
    variances = {
        "height": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "weight": torch.tensor([0.2, 0.2, 0.1, 0.1]),
        "gender_probs": None,
    }
    mean_result = aggregate_by_speaker(speaker_ids, preds, variances=None, method="mean")
    iv_result = aggregate_by_speaker(speaker_ids, preds, variances=variances, method="inverse_variance")
    omega_result = aggregate_by_speaker(
        speaker_ids,
        preds,
        variances=variances,
        quality=torch.tensor([0.9, 0.7, 0.8, 0.6]),
        metadata={
            "capture_quality_score": torch.tensor([0.9, 0.8, 0.7, 0.6]),
            "speech_ratio": torch.tensor([0.8, 0.75, 0.7, 0.65]),
            "snr_db_estimate": torch.tensor([20.0, 18.0, 17.0, 16.0]),
            "duration_s": torch.tensor([3.0, 3.5, 2.8, 2.5]),
            "voiced_ratio": torch.tensor([0.8, 0.82, 0.7, 0.68]),
            "clipped_ratio": torch.tensor([0.0, 0.0, 0.01, 0.02]),
            "distance_cm_estimate": torch.tensor([18.0, 20.0, 17.0, 19.0]),
            "valid_frames": torch.tensor([80.0, 75.0, 82.0, 78.0]),
        },
        method="omega_robust_reliability_pool",
        target_stats={"height": {"mean": 170.0, "std": 8.0}},
    )
    assert set(mean_result["speaker"].keys()) == {"a", "b"}
    assert set(iv_result["speaker"].keys()) == {"a", "b"}
    assert set(omega_result["speaker"].keys()) == {"a", "b"}
    assert torch.isfinite(iv_result["speaker"]["a"]["height"])
    assert torch.isfinite(iv_result["speaker"]["b"]["weight"])
    assert torch.isfinite(omega_result["speaker"]["a"]["height"])
    assert "clip_reliability" in omega_result["utterance"]


def test_mixup_respects_gender_and_domain_compatibility(small_model):
    small_model.train()
    features, mask, domain, targets = _make_batch(batch_size=4, time_steps=10)
    mixed_features, mixed_mask, mixed_targets, mixup = small_model._apply_feature_mixup(
        features,
        mask,
        domain=domain,
        targets=targets,
        enabled=True,
    )
    assert mixed_features.shape == features.shape
    assert mixed_mask is not None and mixed_mask.shape == mask.shape
    assert mixed_targets is not None
    applied_indices = torch.where(mixup["applied"])[0]
    assert applied_indices.numel() > 0
    for idx in applied_indices.tolist():
        partner = int(mixup["pair_index"][idx].item())
        assert int(targets["gender"][idx].item()) == int(targets["gender"][partner].item())
        assert int(domain[idx].item()) == int(domain[partner].item())


@pytest.mark.parametrize(
    "toggle_overrides",
    [
        {},
        {"use_physics_branch": False, "use_cross_attention": False, "use_domain_adv": False},
        {"use_shoulder_head": False, "use_waist_head": False, "use_height_prior": False},
    ],
)
def test_ablation_toggle_matrix_runs(toggle_overrides):
    config = {
        "model": {
            "input_dim": 136,
            "v2": {
                "ecapa_channels": 128,
                "ecapa_scale": 4,
                "conformer_d_model": 64,
                "conformer_heads": 4,
                "conformer_blocks": 1,
                "dropout": 0.1,
                "toggles": toggle_overrides,
            }
        }
    }
    model = build_vocalmorph_v2(config)
    features, mask, domain, _ = _make_batch(batch_size=4, time_steps=8)
    out = model(features, padding_mask=mask, domain=domain, return_diagnostics=True)
    for key in ("height", "weight", "age", "shoulder", "waist"):
        assert out[key].shape == (4,)
        assert torch.isfinite(out[key]).all()
    assert torch.isfinite(out["quality_score"]).all()
