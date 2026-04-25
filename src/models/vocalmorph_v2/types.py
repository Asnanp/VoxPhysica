"""Typed contracts for VocalMorph V2 inputs and outputs."""

from __future__ import annotations

from typing import Dict, List, TypedDict

import torch


class RegressionHeadOutput(TypedDict):
    mu: torch.Tensor
    var: torch.Tensor
    logvar: torch.Tensor


class RegressionUncertaintySummary(TypedDict):
    mean: torch.Tensor
    var: torch.Tensor
    std: torch.Tensor
    epistemic_var: torch.Tensor
    aleatoric_var: torch.Tensor


class VocalMorphTargets(TypedDict, total=False):
    height: torch.Tensor
    weight: torch.Tensor
    age: torch.Tensor
    shoulder: torch.Tensor
    waist: torch.Tensor
    height_raw: torch.Tensor
    weight_raw: torch.Tensor
    age_raw: torch.Tensor
    shoulder_raw: torch.Tensor
    waist_raw: torch.Tensor
    gender: torch.Tensor
    domain: torch.Tensor
    weight_mask: torch.Tensor
    shoulder_mask: torch.Tensor
    waist_mask: torch.Tensor
    f0_mean: torch.Tensor
    formant_spacing_mean: torch.Tensor
    vtl_mean: torch.Tensor
    speaker_idx: torch.Tensor
    epoch: int | torch.Tensor


class ForwardDiagnostics(TypedDict, total=False):
    quality_score: torch.Tensor
    valid_frames: torch.Tensor
    physics_confidence: torch.Tensor
    physics_gate: torch.Tensor
    physics_reliability: torch.Tensor
    formant_spacing_reliability: torch.Tensor
    vtl_reliability: torch.Tensor
    formant_spacing_source: torch.Tensor
    vtl_source: torch.Tensor
    formant_spacing_consistency: torch.Tensor
    vtl_consistency: torch.Tensor
    formant_spacing_consistency_error: torch.Tensor
    vtl_consistency_error: torch.Tensor
    cross_attention_maps: List[torch.Tensor]


MixupInfo = TypedDict(
    "MixupInfo",
    {
        "applied": torch.Tensor,
        "pair_index": torch.Tensor,
        "lambda": torch.Tensor,
    },
)


class SpeakerAggregationEntry(TypedDict, total=False):
    count: int
    quality: torch.Tensor
    height: torch.Tensor
    height_var: torch.Tensor
    height_std: torch.Tensor
    weight: torch.Tensor
    weight_var: torch.Tensor
    weight_std: torch.Tensor
    age: torch.Tensor
    age_var: torch.Tensor
    age_std: torch.Tensor
    shoulder: torch.Tensor
    shoulder_var: torch.Tensor
    shoulder_std: torch.Tensor
    waist: torch.Tensor
    waist_var: torch.Tensor
    waist_std: torch.Tensor
    gender_probs: torch.Tensor
    gender_pred: int


__all__ = [
    "ForwardDiagnostics",
    "MixupInfo",
    "RegressionHeadOutput",
    "RegressionUncertaintySummary",
    "SpeakerAggregationEntry",
    "VocalMorphTargets",
]
