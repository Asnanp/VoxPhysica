"""Structured configuration objects for VocalMorph V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, TypeVar


@dataclass(frozen=True)
class PhysicsConstants:
    """Physical and numerical constants used by the physics-aware branches."""

    eps: float = 1e-6
    speed_of_sound_cm_per_s: float = 34000.0
    spacing_min_hz: float = 250.0
    spacing_max_hz: float = 1150.0
    vtl_min_cm: float = 10.0
    vtl_max_cm: float = 35.0
    signal_floor: float = 1e-4
    ratio_floor: float = 1e-3
    prior_tanh_scale: float = 0.35
    physics_residual_tanh_scale: float = 0.25
    domain_correction_base: float = 0.85
    domain_correction_range: float = 0.30
    default_spacing_hz: float = 700.0
    ranking_height_threshold_cm: float = 2.0

    @property
    def spacing_range_hz(self) -> float:
        return self.spacing_max_hz - self.spacing_min_hz

    @property
    def vtl_range_cm(self) -> float:
        return self.vtl_max_cm - self.vtl_min_cm

    @property
    def default_vtl_cm(self) -> float:
        return self.speed_of_sound_cm_per_s / (2.0 * self.default_spacing_hz)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LossWeights:
    """Loss weights for the multi-task objective."""

    height: float = 3.0
    weight: float = 1.0
    age: float = 1.0
    shoulder: float = 1.0
    waist: float = 1.0
    gender: float = 0.5
    vtsl: float = 0.2
    physics_penalty: float = 0.1
    domain_adv: float = 0.05
    ranking: float = 0.075
    diversity: float = 1.0
    speaker_consistency: float = 0.05
    uncertainty_calibration: float = 0.025

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregationConfig:
    """Speaker-level aggregation controls for strict and Omega evaluation."""

    method: str = "legacy_inverse_variance"
    omega_mad_z: float = 2.5
    omega_min_survivors: int = 3
    omega_huber_delta_scale: float = 1.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReliabilityConfig:
    """Controls handcrafted or learned clip-reliability estimation."""

    mode: str = "handcrafted"
    use_feature_drift: bool = True
    min_weight: float = 0.05
    hidden_dim: int = 32
    embedding_dim: int = 16

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpeakerAlignmentConfig:
    """Speaker-level objective alignment schedule."""

    enable_pooled_height: bool = False
    enable_consistency: bool = False
    enable_ranking: bool = False
    pooling_method: str = "omega"
    consistency_mode: str = "pairwise_weighted"
    warmup_start_epoch: int = 3
    warmup_end_epoch: int = 8
    pooled_height_weight_max: float = 0.0
    consistency_weight_max: float = 0.0
    ranking_weight_max: float = 0.0
    ranking_min_height_delta_cm: float = 2.0
    ranking_margin_cm: float = 0.5
    consistency_max_combined_std_cm: float = 5.0
    height_bin_loss_start_epoch: int = 1
    height_bin_loss_weight_short: float = 1.0
    height_bin_loss_weight_medium: float = 1.0
    height_bin_loss_weight_tall: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AblationToggles:
    """Runtime feature switches for ablations and controlled experiments."""

    use_physics_branch: bool = True
    use_cross_attention: bool = True
    use_reliability_gate: bool = True
    use_height_prior: bool = True
    use_height_adapter: bool = True
    use_domain_adv: bool = True
    use_diversity_loss: bool = True
    use_feature_mixup: bool = True
    use_feature_normalization: bool = True
    use_acoustic_physics_consistency: bool = True
    use_ranking_loss: bool = True
    use_speaker_consistency: bool = True
    use_uncertainty_calibration: bool = True
    use_shoulder_head: bool = True
    use_waist_head: bool = True
    use_kendall_weights: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelHyperparameters:
    """Grouped architectural hyperparameters for the full model."""

    ecapa_channels: int = 512
    ecapa_scale: int = 8
    conformer_d_model: int = 256
    conformer_heads: int = 8
    conformer_blocks: int = 4
    dropout: float = 0.30
    branch_dropout: float = 0.10
    drop_path_rate: float = 0.05
    layer_scale_init: float = 1e-4
    rel_pos_max_distance: int = 128
    cross_rel_pos_max_distance: int = 64
    pooling_hidden_dim: int = 128
    feature_norm_eps: float = 1e-5
    mixup_alpha: float = 0.2
    focal_after_epoch: int = 20
    ranking_margin: float = 0.10
    # Minimum clamp for exp(logvar) in the heteroscedastic NLL head. The
    # default preserves historical behavior; raising it prevents predicted
    # variance from collapsing toward zero, which otherwise lets the NLL
    # term `0.5 * logvar` drive total loss arbitrarily negative and makes
    # held-out NLL explode when the model is confidently wrong.
    nll_floor: float = 1e-6
    physics_embedding_dim: int = 128
    physics_fusion_dim: int = 128
    fused_dim: int = 256
    regression_hidden_dim: int = 192
    regression_var_scale: float = 0.5
    height_adapter_hidden_dim: int = 128
    height_adapter_scale: float = 0.25
    physics_gate_hidden_dim: int = 96
    physics_gate_floor: float = 0.10
    physics_gate_curriculum_epochs: int = 30
    uncertainty_samples: int = 10
    uncertainty_crops: int = 1
    domain_classes: int = 2
    gender_classes: int = 2

    @property
    def fusion_dim(self) -> int:
        return self.conformer_d_model + self.physics_fusion_dim

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicsFeatureSpec:
    """Feature layout assumption: [MFCCs | 5 spectral | formants | F0 | spacing | VTL]."""

    n_mfcc: int = 40
    include_delta: bool = True
    include_delta_delta: bool = True
    n_formants: int = 4

    @property
    def mfcc_dim(self) -> int:
        return self.n_mfcc * (
            1 + int(self.include_delta) + int(self.include_delta_delta)
        )

    @property
    def spectral_offset(self) -> int:
        return self.mfcc_dim

    @property
    def formant_offset(self) -> int:
        return self.spectral_offset + 5

    def formant_freq_idx(self, formant_number_zero_based: int) -> int:
        if not 0 <= formant_number_zero_based < self.n_formants:
            raise ValueError(
                f"formant_number_zero_based must be in [0, {self.n_formants - 1}], got {formant_number_zero_based}"
            )
        return self.formant_offset + (2 * formant_number_zero_based)

    def formant_bw_idx(self, formant_number_zero_based: int) -> int:
        return self.formant_freq_idx(formant_number_zero_based) + 1

    @property
    def f0_idx(self) -> int:
        return self.formant_offset + (self.n_formants * 2)

    @property
    def spacing_idx(self) -> int:
        return self.f0_idx + 1

    @property
    def vtl_idx(self) -> int:
        return self.spacing_idx + 1

    @property
    def minimum_input_dim(self) -> int:
        return self.vtl_idx + 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ConfigT = TypeVar("ConfigT")


def dataclass_from_mapping(
    dataclass_type: type[ConfigT], overrides: Optional[Mapping[str, Any]]
) -> ConfigT:
    """Instantiate a dataclass from a config mapping while ignoring unknown keys."""
    if overrides is None:
        return dataclass_type()
    field_names = set(dataclass_type.__dataclass_fields__.keys())
    filtered = {key: value for key, value in overrides.items() if key in field_names}
    return dataclass_type(**filtered)


_dataclass_from_mapping = dataclass_from_mapping


DEFAULT_PHYSICS_CONSTANTS = PhysicsConstants()
DEFAULT_LOSS_WEIGHTS = LossWeights()
EPS = DEFAULT_PHYSICS_CONSTANTS.eps


__all__ = [
    "AblationToggles",
    "AggregationConfig",
    "DEFAULT_LOSS_WEIGHTS",
    "DEFAULT_PHYSICS_CONSTANTS",
    "EPS",
    "LossWeights",
    "ModelHyperparameters",
    "PhysicsConstants",
    "PhysicsFeatureSpec",
    "ReliabilityConfig",
    "SpeakerAlignmentConfig",
    "_dataclass_from_mapping",
    "dataclass_from_mapping",
]
