import pytest
import torch

from src.models.vocalmorph_v2 import _sanitize_padding_mask, _validate_class_labels, _validate_sequence_inputs

from .conftest import make_features


def test_sanitize_padding_mask_repairs_all_padded_rows():
    mask = torch.ones(2, 5, dtype=torch.bool)
    fixed = _sanitize_padding_mask(mask)
    assert fixed.dtype == torch.bool
    assert torch.equal(fixed[:, 0], torch.zeros(2, dtype=torch.bool))
    assert fixed[:, 1:].all()


def test_validate_sequence_inputs_and_labels():
    features = make_features(batch_size=2, time_steps=4)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    validated_mask = _validate_sequence_inputs(features, mask, expected_feature_dim=136)
    assert torch.equal(validated_mask, mask)

    with pytest.raises(ValueError):
        _validate_sequence_inputs(features[:, :, :8], expected_feature_dim=136)

    with pytest.raises(ValueError):
        _validate_class_labels(torch.tensor([0.0, 2.5]), "gender", 2)
