import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.vocalmorph_v2.config import AggregationConfig, ReliabilityConfig
from src.models.vocalmorph_v2.reliability import (
    compose_handcrafted_clip_reliability,
    omega_reliability_pool,
)
from src.preprocessing.samplers import (
    BalancedHeightAwareSpeakerBatchSampler,
    GroupedSpeakerBatchSampler,
    HybridSpeakerBatchSampler,
)


def test_handcrafted_reliability_is_deterministic_and_bounded():
    metadata = {
        "capture_quality_score": torch.tensor([0.9, 0.4]),
        "speech_ratio": torch.tensor([0.8, 0.3]),
        "snr_db_estimate": torch.tensor([22.0, 4.0]),
        "clipped_ratio": torch.tensor([0.0, 0.10]),
        "distance_cm_estimate": torch.tensor([18.0, 50.0]),
        "voiced_ratio": torch.tensor([0.85, 0.1]),
        "duration_s": torch.tensor([4.0, 0.5]),
        "feature_drift_zscore": torch.tensor([0.2, 5.0]),
    }
    valid_frames = torch.tensor([96.0, 24.0])
    config = ReliabilityConfig(min_weight=0.05, use_feature_drift=True)

    first = compose_handcrafted_clip_reliability(
        metadata=metadata,
        valid_frames=valid_frames,
        crop_frames=96,
        pred_std_cm=torch.tensor([1.0, 6.0]),
        min_weight=config.min_weight,
        use_feature_drift=config.use_feature_drift,
    )
    second = compose_handcrafted_clip_reliability(
        metadata=metadata,
        valid_frames=valid_frames,
        crop_frames=96,
        pred_std_cm=torch.tensor([1.0, 6.0]),
        min_weight=config.min_weight,
        use_feature_drift=config.use_feature_drift,
    )

    assert torch.allclose(first["clip_reliability_prior"], second["clip_reliability_prior"])
    assert torch.all(first["clip_reliability_prior"] >= 0.05)
    assert torch.all(first["clip_reliability_prior"] <= 1.0)
    assert first["clip_reliability_prior"][0] > first["clip_reliability_prior"][1]


def test_omega_pool_rejects_large_outlier():
    pooled = omega_reliability_pool(
        torch.tensor([170.0, 171.0, 209.0]),
        clip_reliability=torch.tensor([0.9, 0.8, 0.9]),
        pred_var=torch.tensor([1.0, 1.0, 1.0]),
        config=AggregationConfig(omega_mad_z=2.5, omega_min_survivors=2, omega_huber_delta_scale=1.25),
    )

    assert abs(float(pooled["mean"].item()) - 170.5) < 1.5
    assert int(pooled["rejected_count"]) >= 1
    assert int(pooled["surviving_count"]) >= 2


def test_grouped_speaker_batch_sampler_is_deterministic():
    speaker_to_indices = {
        "spk_a": [0, 1, 2],
        "spk_b": [3, 4, 5],
        "spk_c": [6, 7, 8],
        "spk_d": [9, 10, 11],
    }
    sampler_a = GroupedSpeakerBatchSampler(
        speaker_to_indices,
        speakers_per_batch=2,
        clips_per_speaker=2,
        base_seed=17,
    )
    sampler_b = GroupedSpeakerBatchSampler(
        speaker_to_indices,
        speakers_per_batch=2,
        clips_per_speaker=2,
        base_seed=17,
    )
    sampler_a.set_epoch(3)
    sampler_b.set_epoch(3)

    batches_a = list(iter(sampler_a))
    batches_b = list(iter(sampler_b))
    assert batches_a == batches_b

    state = sampler_a.state_dict()
    sampler_b.load_state_dict(state)
    assert sampler_b.state_dict() == state


def test_height_balanced_sampler_supports_capped_gender_height_weights():
    speaker_to_indices = {
        "spk_short_f": [0, 1],
        "spk_short_m": [2, 3],
        "spk_medium_m": [4, 5],
    }
    speaker_metadata = {
        "spk_short_f": {"height_bin": "short", "gender": 0},
        "spk_short_m": {"height_bin": "short", "gender": 1},
        "spk_medium_m": {"height_bin": "medium", "gender": 1},
    }
    sampler = BalancedHeightAwareSpeakerBatchSampler(
        speaker_to_indices,
        speaker_metadata=speaker_metadata,
        speakers_per_batch=2,
        clips_per_speaker=1,
        height_bin_weights={"short": 2.0, "medium": 1.0, "tall": 1.0},
        gender_height_weights={"male_short": 3.0},
        max_speaker_weight=4.0,
        total_examples=6,
        base_seed=7,
    )

    weights = {
        speaker_id: sampler._speaker_weight(speaker_id, {0: 0, 1: 0})
        for speaker_id in speaker_to_indices
    }

    assert weights["spk_short_m"] == 4.0
    assert weights["spk_short_f"] > weights["spk_medium_m"]


def test_hybrid_speaker_batch_sampler_keeps_more_speaker_diversity():
    speaker_to_indices = {
        "spk_a": [0, 1, 2],
        "spk_b": [3, 4, 5],
        "spk_c": [6, 7, 8],
        "spk_d": [9, 10, 11],
        "spk_e": [12, 13, 14],
        "spk_f": [15, 16, 17],
    }
    sampler = HybridSpeakerBatchSampler(
        speaker_to_indices,
        paired_speakers_per_batch=2,
        singleton_speakers_per_batch=2,
        clips_per_speaker=2,
        base_seed=23,
    )
    sampler.set_epoch(1)

    batches = list(iter(sampler))
    assert batches

    first_batch = batches[0]
    assert len(first_batch) == 6

    index_to_speaker = {}
    for speaker_id, indices in speaker_to_indices.items():
        for idx in indices:
            index_to_speaker[idx] = speaker_id
    speaker_counts = {}
    for idx in first_batch:
        speaker_id = index_to_speaker[idx]
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

    count_histogram = sorted(speaker_counts.values())
    assert len(speaker_counts) == 4
    assert count_histogram == [1, 1, 2, 2]


def test_height_balanced_sampler_is_deterministic_and_preserves_budget():
    speaker_to_indices = {
        "spk_short_f": [0, 1, 2],
        "spk_short_m": [3, 4, 5],
        "spk_medium_f": [6, 7, 8],
        "spk_medium_m": [9, 10, 11],
        "spk_tall_m": [12, 13, 14],
    }
    speaker_metadata = {
        "spk_short_f": {"height_bin": "short", "gender": 0},
        "spk_short_m": {"height_bin": "short", "gender": 1},
        "spk_medium_f": {"height_bin": "medium", "gender": 0},
        "spk_medium_m": {"height_bin": "medium", "gender": 1},
        "spk_tall_m": {"height_bin": "tall", "gender": 1},
    }
    sampler_a = BalancedHeightAwareSpeakerBatchSampler(
        speaker_to_indices,
        speaker_metadata=speaker_metadata,
        speakers_per_batch=2,
        clips_per_speaker=2,
        total_examples=15,
        base_seed=29,
    )
    sampler_b = BalancedHeightAwareSpeakerBatchSampler(
        speaker_to_indices,
        speaker_metadata=speaker_metadata,
        speakers_per_batch=2,
        clips_per_speaker=2,
        total_examples=15,
        base_seed=29,
    )
    sampler_a.set_epoch(4)
    sampler_b.set_epoch(4)

    batches_a = list(iter(sampler_a))
    batches_b = list(iter(sampler_b))
    assert batches_a == batches_b
    assert len(sampler_a) == 4
    assert all(len(batch) == 4 for batch in batches_a)

    index_to_speaker = {}
    for speaker_id, indices in speaker_to_indices.items():
        for idx in indices:
            index_to_speaker[idx] = speaker_id
    for batch in batches_a:
        speakers = {index_to_speaker[idx] for idx in batch}
        assert len(speakers) == 2

    state = sampler_a.state_dict()
    sampler_b.load_state_dict(state)
    assert sampler_b.state_dict() == state
