import torch
import pytest

from src.models.vocalmorph_v2 import VocalMorphV2


def make_features(batch_size: int = 4, time_steps: int = 16, input_dim: int = 136) -> torch.Tensor:
    torch.manual_seed(7)
    features = torch.randn(batch_size, time_steps, input_dim, dtype=torch.float32)
    for idx, value in [(125, 500.0), (127, 1500.0), (129, 2500.0), (131, 3500.0), (133, 140.0), (134, 700.0), (135, 16.5)]:
        features[:, :, idx] = value + 0.01 * torch.randn(batch_size, time_steps)
    return features


def make_batch(batch_size: int = 4, time_steps: int = 16, input_dim: int = 136):
    features = make_features(batch_size=batch_size, time_steps=time_steps, input_dim=input_dim)
    padding_mask = torch.zeros(batch_size, time_steps, dtype=torch.bool)
    if time_steps > 3:
        padding_mask[0, -2:] = True
        padding_mask[-1, -1:] = True
    domain = torch.tensor([0, 0, 1, 1][:batch_size], dtype=torch.long)
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
        "gender": torch.tensor([0, 0, 1, 1][:batch_size], dtype=torch.long),
        "domain": domain.clone(),
        "speaker_idx": torch.tensor([0, 0, 1, 1][:batch_size], dtype=torch.long),
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
    return VocalMorphV2(
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
