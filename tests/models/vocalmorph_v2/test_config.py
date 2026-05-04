from src.models.vocalmorph_v2 import (
    LossWeights,
    ModelHyperparameters,
    PhysicsConstants,
    PhysicsFeatureSpec,
    SpeakerAlignmentConfig,
    dataclass_from_mapping,
)


def test_feature_spec_layout_indices_are_consistent():
    spec = PhysicsFeatureSpec(n_mfcc=13, include_delta=True, include_delta_delta=False, n_formants=4)
    assert spec.formant_freq_idx(0) < spec.f0_idx < spec.spacing_idx < spec.minimum_input_dim


def test_dataclass_from_mapping_ignores_unknown_keys():
    constants = dataclass_from_mapping(PhysicsConstants, {"default_spacing_hz": 710.0, "unknown": 1})
    assert constants.default_spacing_hz == 710.0

    hyper = dataclass_from_mapping(ModelHyperparameters, {"conformer_d_model": 64, "foo": "bar"})
    assert hyper.conformer_d_model == 64


def test_model_hyperparameters_accept_mixup_probability():
    hyper = dataclass_from_mapping(ModelHyperparameters, {"mixup_probability": 0.35})
    assert hyper.mixup_probability == 0.35


def test_bin_aware_height_config_fields_are_mapped():
    weights = dataclass_from_mapping(LossWeights, {"height_bin": 0.4})
    alignment = dataclass_from_mapping(
        SpeakerAlignmentConfig,
        {
            "height_bin_prior_blend": 0.35,
            "height_bin_prior_short_cm": 154.0,
        },
    )

    assert weights.height_bin == 0.4
    assert alignment.height_bin_prior_blend == 0.35
    assert alignment.height_bin_prior_short_cm == 154.0
