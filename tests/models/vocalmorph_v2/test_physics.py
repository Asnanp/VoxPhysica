import torch

from src.models.vocalmorph_v2 import PhysicsPath, PhysicsFeatureSpec

from .conftest import make_features


def test_physics_path_returns_expected_keys_and_shapes():
    features = make_features(batch_size=3, time_steps=10)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    path = PhysicsPath(spec=PhysicsFeatureSpec())
    out = path(features, padding_mask=mask)
    for key in (
        "physics_input",
        "physics_embedding",
        "formant_spacing_pred",
        "observed_formant_spacing",
        "derived_formant_spacing",
        "imputed_formant_spacing",
        "physics_input_raw",
        "physics_input_normalized",
        "physics_confidence",
        "formant_spacing_consistency",
        "vtl_consistency",
        "observed_vtl",
        "derived_vtl",
        "imputed_vtl",
        "physics_reliability",
        "multi_formant_vtl",
        "vtl_from_f1_f4",
        "vtl_from_avg_spacing",
        "spacing_std",
        "formant_ratio_f2_f1",
        "formant_ratio_f3_f2",
        "formant_ratio_f4_f3",
        "spacing_confidence",
        "vtl_confidence",
        "formant_stability_score",
        "physics_residual_confidence",
    ):
        assert key in out
    assert out["physics_embedding"].shape == (3, 128)
    assert out["physics_input"].shape == (3, 19)
    assert out["physics_input_raw"].shape == (3, 19)
    assert out["physics_input_normalized"].shape == (3, 19)
    assert out["formant_spacing_pred"].shape == (3,)
    assert out["physics_confidence"].shape == (3,)
    assert out["physics_reliability"].shape == (3, 8)
    assert torch.isfinite(out["physics_embedding"]).all()
    assert torch.isfinite(out["physics_input_normalized"]).all()
    assert torch.all(
        (out["physics_confidence"] >= 0.0) & (out["physics_confidence"] <= 1.0)
    )
    assert torch.all(
        (out["formant_spacing_consistency"] >= 0.0)
        & (out["formant_spacing_consistency"] <= 1.0)
    )
    assert torch.all((out["vtl_consistency"] >= 0.0) & (out["vtl_consistency"] <= 1.0))
    assert torch.all(
        (out["spacing_confidence"] >= 0.0) & (out["spacing_confidence"] <= 1.0)
    )
    assert torch.all((out["vtl_confidence"] >= 0.0) & (out["vtl_confidence"] <= 1.0))
    assert torch.all(
        (out["formant_stability_score"] >= 0.0)
        & (out["formant_stability_score"] <= 1.0)
    )
    assert torch.all(
        (out["physics_residual_confidence"] >= 0.0)
        & (out["physics_residual_confidence"] <= 1.0)
    )
    assert torch.all(out["formant_ratio_f2_f1"] > 0.0)
    assert torch.all(out["formant_ratio_f3_f2"] > 0.0)
    assert torch.all(out["formant_ratio_f4_f3"] > 0.0)


def test_physics_path_remains_finite_when_signals_are_missing():
    features = torch.zeros(2, 8, 136, dtype=torch.float32)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    path = PhysicsPath(spec=PhysicsFeatureSpec())
    out = path(features, padding_mask=mask)
    assert torch.isfinite(out["physics_input"]).all()
    assert torch.isfinite(out["physics_embedding"]).all()
    assert torch.isfinite(out["formant_spacing_pred"]).all()
    assert torch.isfinite(out["physics_confidence"]).all()
