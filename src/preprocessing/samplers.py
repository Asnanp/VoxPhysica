"""Deterministic speaker-aware sampling utilities for Omega experiments."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class WorkerSeedInitializer:
    """Pickle-safe callable for Windows DataLoader worker seeding."""

    def __init__(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)

    def __call__(self, worker_id: int) -> None:
        worker_seed = int(self.base_seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)


def build_worker_init_fn(base_seed: int):
    """Build a DataLoader worker init function with deterministic seeds."""
    return WorkerSeedInitializer(base_seed)


class GroupedSpeakerBatchSampler(Sampler[List[int]]):
    """Sample fixed speaker-count batches with multiple clips per speaker."""

    def __init__(
        self,
        speaker_to_indices: Mapping[str, Sequence[int]],
        *,
        speakers_per_batch: int,
        clips_per_speaker: int,
        base_seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if int(speakers_per_batch) < 1:
            raise ValueError("speakers_per_batch must be >= 1")
        if int(clips_per_speaker) < 1:
            raise ValueError("clips_per_speaker must be >= 1")
        normalized: Dict[str, List[int]] = {}
        for speaker_id, indices in speaker_to_indices.items():
            clean = [int(idx) for idx in indices]
            if clean:
                normalized[str(speaker_id)] = clean
        if not normalized:
            raise ValueError("speaker_to_indices must contain at least one speaker")

        self.speaker_to_indices = normalized
        self.speaker_ids = sorted(normalized.keys())
        self.speakers_per_batch = int(speakers_per_batch)
        self.clips_per_speaker = int(clips_per_speaker)
        self.base_seed = int(base_seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    @property
    def effective_batch_size(self) -> int:
        return self.speakers_per_batch * self.clips_per_speaker

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> Dict[str, int]:
        return {
            "epoch": int(self.epoch),
            "base_seed": int(self.base_seed),
            "speakers_per_batch": int(self.speakers_per_batch),
            "clips_per_speaker": int(self.clips_per_speaker),
        }

    def load_state_dict(self, state: Mapping[str, int] | None) -> None:
        if not state:
            return
        self.epoch = int(state.get("epoch", self.epoch))
        self.base_seed = int(state.get("base_seed", self.base_seed))

    def __len__(self) -> int:
        n_speakers = len(self.speaker_ids)
        if self.drop_last:
            return n_speakers // self.speakers_per_batch
        return int(math.ceil(n_speakers / self.speakers_per_batch))

    def _sample_indices_for_speaker(self, rng: random.Random, speaker_id: str) -> List[int]:
        candidates = list(self.speaker_to_indices[speaker_id])
        if len(candidates) >= self.clips_per_speaker:
            return rng.sample(candidates, self.clips_per_speaker)
        if not candidates:
            raise ValueError(f"Speaker '{speaker_id}' has no indices")
        picks = list(candidates)
        while len(picks) < self.clips_per_speaker:
            picks.append(rng.choice(candidates))
        rng.shuffle(picks)
        return picks

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.base_seed + self.epoch)
        speaker_order = list(self.speaker_ids)
        rng.shuffle(speaker_order)

        n_batches = len(self)
        for batch_idx in range(n_batches):
            start = batch_idx * self.speakers_per_batch
            end = start + self.speakers_per_batch
            speaker_chunk = speaker_order[start:end]
            if len(speaker_chunk) < self.speakers_per_batch and self.drop_last:
                continue
            batch_indices: List[int] = []
            for speaker_id in speaker_chunk:
                batch_indices.extend(self._sample_indices_for_speaker(rng, speaker_id))
            if batch_indices:
                yield batch_indices


def _weighted_choice_without_replacement(
    rng: random.Random,
    items: Sequence[str],
    weights: Sequence[float],
    k: int,
) -> List[str]:
    """Draw up to ``k`` distinct items using deterministic weighted sampling."""
    pool = list(items)
    pool_weights = [max(0.0, float(weight)) for weight in weights]
    chosen: List[str] = []
    k = max(0, min(int(k), len(pool)))
    while pool and len(chosen) < k:
        total = sum(pool_weights)
        if total <= 0.0:
            pick_idx = rng.randrange(len(pool))
        else:
            threshold = rng.random() * total
            pick_idx = 0
            cumulative = 0.0
            for idx, weight in enumerate(pool_weights):
                cumulative += weight
                if cumulative >= threshold:
                    pick_idx = idx
                    break
        chosen.append(pool.pop(pick_idx))
        pool_weights.pop(pick_idx)
    return chosen


class BalancedHeightAwareSpeakerBatchSampler(Sampler[List[int]]):
    """Speaker sampler for Stage 3d that preserves batch budget while upweighting hard slices.

    This sampler differs from the earlier grouped/hybrid variants in one critical way:
    the epoch length is tied to the dataset clip budget, not the number of unique
    speakers. That keeps the optimizer-step budget comparable to plain shuffled
    Stage 3c runs while still exposing same-speaker structure inside each batch.
    """

    def __init__(
        self,
        speaker_to_indices: Mapping[str, Sequence[int]],
        *,
        speaker_metadata: Mapping[str, Mapping[str, Any]],
        speakers_per_batch: int,
        clips_per_speaker: int,
        height_bin_weights: Optional[Mapping[str, float]] = None,
        total_examples: Optional[int] = None,
        balance_gender: bool = True,
        gender_balance_strength: float = 0.20,
        base_seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if int(speakers_per_batch) < 1:
            raise ValueError("speakers_per_batch must be >= 1")
        if int(clips_per_speaker) < 1:
            raise ValueError("clips_per_speaker must be >= 1")
        normalized: Dict[str, List[int]] = {}
        for speaker_id, indices in speaker_to_indices.items():
            clean = [int(idx) for idx in indices]
            if clean:
                normalized[str(speaker_id)] = clean
        if not normalized:
            raise ValueError("speaker_to_indices must contain at least one speaker")

        self.speaker_to_indices = normalized
        self.speaker_ids = sorted(normalized.keys())
        self.speakers_per_batch = int(speakers_per_batch)
        self.clips_per_speaker = int(clips_per_speaker)
        self.base_seed = int(base_seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.total_examples = int(
            total_examples
            if total_examples is not None
            else sum(len(indices) for indices in normalized.values())
        )
        if self.total_examples < 1:
            raise ValueError("total_examples must be >= 1")

        default_height_weights = {"short": 1.8, "medium": 1.0, "tall": 1.15}
        raw_height_weights = dict(height_bin_weights or {})
        self.height_bin_weights = {
            key: float(raw_height_weights.get(key, default_height_weights[key]))
            for key in default_height_weights
        }
        self.balance_gender = bool(balance_gender)
        self.gender_balance_strength = max(0.0, float(gender_balance_strength))

        self.speaker_height_bin: Dict[str, str] = {}
        self.speaker_gender: Dict[str, Optional[int]] = {}
        gender_counts = {0: 0, 1: 0}
        for speaker_id in self.speaker_ids:
            meta = dict(speaker_metadata.get(speaker_id, {}))
            height_bin = str(meta.get("height_bin", "medium")).strip().lower()
            if height_bin not in self.height_bin_weights:
                height_bin = "medium"
            self.speaker_height_bin[speaker_id] = height_bin

            gender_value = meta.get("gender")
            if gender_value in (0, 1):
                gender_int = int(gender_value)
            else:
                gender_int = None
            self.speaker_gender[speaker_id] = gender_int
            if gender_int in gender_counts:
                gender_counts[gender_int] += 1

        total_gender = float(sum(gender_counts.values()))
        self.gender_prior_weight: Dict[Optional[int], float] = {None: 1.0}
        if self.balance_gender and total_gender > 0 and min(gender_counts.values()) > 0:
            for gender, count in gender_counts.items():
                inverse_freq = total_gender / (2.0 * float(count))
                blended = 1.0 + self.gender_balance_strength * (inverse_freq - 1.0)
                self.gender_prior_weight[gender] = float(min(1.25, max(0.80, blended)))
        else:
            self.gender_prior_weight.update({0: 1.0, 1: 1.0})

    @property
    def effective_batch_size(self) -> int:
        return self.speakers_per_batch * self.clips_per_speaker

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "base_seed": int(self.base_seed),
            "speakers_per_batch": int(self.speakers_per_batch),
            "clips_per_speaker": int(self.clips_per_speaker),
            "total_examples": int(self.total_examples),
            "height_bin_weights": dict(self.height_bin_weights),
            "balance_gender": bool(self.balance_gender),
            "gender_balance_strength": float(self.gender_balance_strength),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        self.epoch = int(state.get("epoch", self.epoch))
        self.base_seed = int(state.get("base_seed", self.base_seed))

    def __len__(self) -> int:
        if self.drop_last:
            return max(1, self.total_examples // self.effective_batch_size)
        return max(1, int(math.ceil(self.total_examples / self.effective_batch_size)))

    def _sample_indices_for_speaker(
        self, rng: random.Random, speaker_id: str, n_clips: Optional[int] = None
    ) -> List[int]:
        n_clips = self.clips_per_speaker if n_clips is None else int(n_clips)
        candidates = list(self.speaker_to_indices[speaker_id])
        if len(candidates) >= n_clips:
            return rng.sample(candidates, n_clips)
        if not candidates:
            raise ValueError(f"Speaker '{speaker_id}' has no indices")
        picks = list(candidates)
        while len(picks) < n_clips:
            picks.append(rng.choice(candidates))
        rng.shuffle(picks)
        return picks

    def _speaker_weight(
        self,
        speaker_id: str,
        batch_gender_counts: Mapping[int, int],
    ) -> float:
        height_weight = self.height_bin_weights.get(
            self.speaker_height_bin.get(speaker_id, "medium"), 1.0
        )
        gender = self.speaker_gender.get(speaker_id)
        weight = height_weight * self.gender_prior_weight.get(gender, 1.0)
        if self.balance_gender and gender in (0, 1):
            current = int(batch_gender_counts.get(gender, 0))
            other = int(batch_gender_counts.get(1 - gender, 0))
            if current < other:
                weight *= 1.0 + self.gender_balance_strength
            elif current > other:
                weight *= max(0.50, 1.0 - self.gender_balance_strength)
        return float(max(weight, 1e-6))

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.base_seed + self.epoch)
        n_batches = len(self)
        for _ in range(n_batches):
            available = list(self.speaker_ids)
            batch_speakers: List[str] = []
            batch_gender_counts = {0: 0, 1: 0}
            while available and len(batch_speakers) < self.speakers_per_batch:
                candidate_weights = [
                    self._speaker_weight(speaker_id, batch_gender_counts)
                    for speaker_id in available
                ]
                picked = _weighted_choice_without_replacement(
                    rng, available, candidate_weights, 1
                )
                if not picked:
                    break
                speaker_id = picked[0]
                available.remove(speaker_id)
                batch_speakers.append(speaker_id)
                gender = self.speaker_gender.get(speaker_id)
                if gender in batch_gender_counts:
                    batch_gender_counts[gender] += 1
            if len(batch_speakers) < self.speakers_per_batch and self.drop_last:
                continue

            batch_indices: List[int] = []
            for speaker_id in batch_speakers:
                batch_indices.extend(self._sample_indices_for_speaker(rng, speaker_id))
            if batch_indices:
                rng.shuffle(batch_indices)
                yield batch_indices


class HybridSpeakerBatchSampler(Sampler[List[int]]):
    """Preserve some same-speaker pairing without collapsing speaker diversity.

    Each batch contains:
    - `paired_speakers_per_batch` speakers sampled with `clips_per_speaker` clips each
    - `singleton_speakers_per_batch` speakers sampled with a single clip each

    This keeps speaker-level structure available for future alignment losses while
    avoiding the aggressive identity collapse of the fully grouped sampler.
    """

    def __init__(
        self,
        speaker_to_indices: Mapping[str, Sequence[int]],
        *,
        paired_speakers_per_batch: int,
        singleton_speakers_per_batch: int,
        clips_per_speaker: int,
        base_seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if int(paired_speakers_per_batch) < 0:
            raise ValueError("paired_speakers_per_batch must be >= 0")
        if int(singleton_speakers_per_batch) < 0:
            raise ValueError("singleton_speakers_per_batch must be >= 0")
        if int(paired_speakers_per_batch) + int(singleton_speakers_per_batch) < 1:
            raise ValueError("hybrid sampler requires at least one speaker per batch")
        if int(clips_per_speaker) < 2:
            raise ValueError("clips_per_speaker must be >= 2 for hybrid speaker batches")

        normalized: Dict[str, List[int]] = {}
        for speaker_id, indices in speaker_to_indices.items():
            clean = [int(idx) for idx in indices]
            if clean:
                normalized[str(speaker_id)] = clean
        if not normalized:
            raise ValueError("speaker_to_indices must contain at least one speaker")

        self.speaker_to_indices = normalized
        self.speaker_ids = sorted(normalized.keys())
        self.paired_speakers_per_batch = int(paired_speakers_per_batch)
        self.singleton_speakers_per_batch = int(singleton_speakers_per_batch)
        self.clips_per_speaker = int(clips_per_speaker)
        self.base_seed = int(base_seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    @property
    def speakers_per_batch(self) -> int:
        return self.paired_speakers_per_batch + self.singleton_speakers_per_batch

    @property
    def effective_batch_size(self) -> int:
        return (
            self.paired_speakers_per_batch * self.clips_per_speaker
            + self.singleton_speakers_per_batch
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> Dict[str, int]:
        return {
            "epoch": int(self.epoch),
            "base_seed": int(self.base_seed),
            "paired_speakers_per_batch": int(self.paired_speakers_per_batch),
            "singleton_speakers_per_batch": int(self.singleton_speakers_per_batch),
            "clips_per_speaker": int(self.clips_per_speaker),
        }

    def load_state_dict(self, state: Mapping[str, int] | None) -> None:
        if not state:
            return
        self.epoch = int(state.get("epoch", self.epoch))
        self.base_seed = int(state.get("base_seed", self.base_seed))

    def __len__(self) -> int:
        n_speakers = len(self.speaker_ids)
        if self.drop_last:
            return n_speakers // self.speakers_per_batch
        return int(math.ceil(n_speakers / self.speakers_per_batch))

    def _sample_indices_for_speaker(
        self, rng: random.Random, speaker_id: str, n_clips: int
    ) -> List[int]:
        candidates = list(self.speaker_to_indices[speaker_id])
        if len(candidates) >= n_clips:
            return rng.sample(candidates, n_clips)
        if not candidates:
            raise ValueError(f"Speaker '{speaker_id}' has no indices")
        picks = list(candidates)
        while len(picks) < n_clips:
            picks.append(rng.choice(candidates))
        rng.shuffle(picks)
        return picks

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.base_seed + self.epoch)
        speaker_order = list(self.speaker_ids)
        rng.shuffle(speaker_order)

        n_batches = len(self)
        for batch_idx in range(n_batches):
            start = batch_idx * self.speakers_per_batch
            end = start + self.speakers_per_batch
            speaker_chunk = speaker_order[start:end]
            if len(speaker_chunk) < self.speakers_per_batch and self.drop_last:
                continue

            paired_count = min(len(speaker_chunk), self.paired_speakers_per_batch)
            paired_speakers = speaker_chunk[:paired_count]
            singleton_speakers = speaker_chunk[paired_count:]

            batch_indices: List[int] = []
            for speaker_id in paired_speakers:
                batch_indices.extend(
                    self._sample_indices_for_speaker(
                        rng, speaker_id, self.clips_per_speaker
                    )
                )
            for speaker_id in singleton_speakers:
                batch_indices.extend(self._sample_indices_for_speaker(rng, speaker_id, 1))

            if batch_indices:
                rng.shuffle(batch_indices)
                yield batch_indices


__all__ = [
    "BalancedHeightAwareSpeakerBatchSampler",
    "GroupedSpeakerBatchSampler",
    "HybridSpeakerBatchSampler",
    "WorkerSeedInitializer",
    "build_worker_init_fn",
]
