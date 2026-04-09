"""Deterministic speaker-aware sampling utilities for Omega experiments."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterator, List, Mapping, Sequence

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
    "GroupedSpeakerBatchSampler",
    "HybridSpeakerBatchSampler",
    "WorkerSeedInitializer",
    "build_worker_init_fn",
]
