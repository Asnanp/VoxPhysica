import pytest
import torch

from src.models.vocalmorphv2 import VocalMorphV2 as ShimVocalMorphV2
from src.models.vocalmorph_v2 import AblationToggles, VocalMorphV2, build_vocalmorph_v2

from .conftest import make_batch, make_features


@pytest.mark.parametrize("batch_size,time_steps", [(1, 1), (2, 5), (4, 12)])
def test_forward_shapes_include_optional_targets(small_model, batch_size, time_steps):
    small_model.eval()
    features = make_features(batch_size=batch_size, time_steps=time_steps)
    mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
    mask[:, 0] = False
    domain = torch.zeros(batch_size, dtype=torch.long)
    with torch.no_grad():
        out = small_model(features, padding_mask=mask, domain=domain, return_diagnostics=True, return_attention_maps=True)
    for key in ("height", "weight", "age", "shoulder", "waist"):
        assert out[key].shape == (batch_size,)
        assert out[f"{key}_var"].shape == (batch_size,)
        assert torch.isfinite(out[key]).all()
    assert out["physics_gate"].shape == (batch_size,)
    assert out["physics_confidence"].shape == (batch_size,)
    assert out["height_features"].shape == (batch_size, small_model.hyperparameters.fused_dim)
    assert out["height_prior_summary"].shape == (batch_size, 3)
    assert torch.all(out["physics_gate"] >= small_model.hyperparameters.physics_gate_floor)
    assert torch.all(out["physics_gate"] <= 1.0)
    assert torch.all(out["physics_confidence"] >= 0.0)
    assert torch.all(out["physics_confidence"] <= 1.0)
    assert torch.all(out["spacing_confidence"] >= 0.0)
    assert torch.all(out["vtl_confidence"] >= 0.0)
    assert torch.all(out["formant_stability_score"] >= 0.0)
    assert out["gender_logits"].shape == (batch_size, 2)
    assert isinstance(out["cross_attention_maps"], list)
    assert "diagnostics" in out


def test_forward_with_partial_targets_and_backward(small_model):
    small_model.train()
    features, mask, domain, targets = make_batch(batch_size=4, time_steps=10)
    partial_targets = {
        "height": targets["height"],
        "gender": targets["gender"],
        "domain": targets["domain"],
    }
    out = small_model(features, padding_mask=mask, domain=domain, targets=partial_targets)
    assert "losses" in out
    out["losses"]["total"].backward()
    grads = [param.grad for param in small_model.parameters() if param.requires_grad and param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_forward_with_speaker_targets_exposes_stability_losses(small_model):
    small_model.train()
    features, mask, domain, targets = make_batch(batch_size=4, time_steps=10)
    out = small_model(features, padding_mask=mask, domain=domain, targets=targets, enable_mixup=False)
    assert "losses" in out
    assert "speaker_consistency" in out["losses"]
    assert "uncertainty_calibration" in out["losses"]
    assert torch.isfinite(out["losses"]["speaker_consistency"])
    assert torch.isfinite(out["losses"]["uncertainty_calibration"])
    assert torch.isfinite(out["losses"]["vtsl"])
    assert torch.allclose(out["speaker_mean_multi_formant_vtl"][0], out["speaker_mean_multi_formant_vtl"][1], atol=1e-5)
    assert torch.allclose(out["speaker_mean_formant_spacing"][0], out["speaker_mean_formant_spacing"][1], atol=1e-5)
    assert torch.allclose(out["speaker_mean_formant_stability"][0], out["speaker_mean_formant_stability"][1], atol=1e-5)
    assert torch.allclose(out["speaker_mean_multi_formant_vtl"][2], out["speaker_mean_multi_formant_vtl"][3], atol=1e-5)


def test_builder_and_backward_compatibility_imports():
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
                "toggles": {"use_physics_branch": False, "use_cross_attention": False},
            },
        }
    }
    model = build_vocalmorph_v2(config)
    assert isinstance(model, VocalMorphV2)
    assert issubclass(ShimVocalMorphV2, VocalMorphV2)


def test_ablation_toggle_matrix_runs():
    model = VocalMorphV2(
        input_dim=136,
        ecapa_channels=128,
        ecapa_scale=4,
        conformer_d_model=64,
        conformer_heads=4,
        conformer_blocks=1,
        dropout=0.1,
        toggles=AblationToggles(use_physics_branch=False, use_cross_attention=False, use_domain_adv=False),
    )
    features, mask, domain, _ = make_batch(batch_size=4, time_steps=8)
    out = model(features, padding_mask=mask, domain=domain)
    assert out["height"].shape == (4,)
    assert torch.isfinite(out["quality_score"]).all()
