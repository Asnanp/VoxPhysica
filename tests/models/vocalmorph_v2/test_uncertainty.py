import torch

from src.models.vocalmorph_v2 import aggregate_by_speaker

from .conftest import make_batch


def test_uncertainty_returns_per_target_variance(small_model):
    features, mask, domain, _ = make_batch(batch_size=4, time_steps=12)
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
            assert "aleatoric_var" in result[key]
            assert "epistemic_var" in result[key]
            assert torch.all(result[key]["var"] >= 0)


def test_speaker_aggregation_inverse_variance_path():
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
    result = aggregate_by_speaker(speaker_ids, preds, variances=variances, method="inverse_variance")
    assert set(result["speaker"].keys()) == {"a", "b"}
    assert torch.isfinite(result["speaker"]["a"]["height"])
    assert torch.isfinite(result["speaker"]["b"]["weight"])
