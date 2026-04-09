"""
VocalMorph dataset utilities.

Loads pre-extracted .npz feature files from train/val/test split dirs.
Adds:
- source-aware metadata
- NISP-only weight masking for mixed NISP+TIMIT training
- on-the-fly feature-level augmentation for training
- helper for gender class weight computation
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .samplers import (
    GroupedSpeakerBatchSampler,
    HybridSpeakerBatchSampler,
    build_worker_init_fn,
)


@dataclass
class FeatureAugmentConfig:
    """On-the-fly feature-level augmentation parameters."""

    # Gaussian noise
    noise_p: float = 0.50
    noise_std: float = 0.02

    # Time masking (SpecAugment-style)
    time_mask_p: float = 0.40
    time_mask_max_frac: float = 0.10  # max 10% of sequence length

    # Feature dimension masking
    feat_mask_p: float = 0.30
    feat_mask_max_frac: float = 0.08  # max 8% of feature dims

    # Feature scaling (per-utterance gain)
    scale_p: float = 0.35
    scale_std: float = 0.08

    # Temporal jitter (random frame dropout)
    temporal_jitter_p: float = 0.25
    temporal_jitter_max_frac: float = 0.05  # max 5% of frames


class FeatureAugmenter:
    """Lightweight on-the-fly augmentation for pre-extracted feature sequences.

    Operates on numpy arrays of shape (T, D). Designed to be safe for
    anthropometric prediction — avoids destroying formant/f0 structure.
    """

    def __init__(self, config: Optional[FeatureAugmentConfig] = None):
        self.cfg = config or FeatureAugmentConfig()

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a feature sequence (T, D)."""
        if seq.ndim != 2 or seq.size == 0:
            return seq

        out = seq.copy()

        # Gaussian noise
        if np.random.random() < self.cfg.noise_p:
            noise = np.random.normal(0.0, self.cfg.noise_std, out.shape).astype(
                np.float32
            )
            out = out + noise

        # Time masking
        if np.random.random() < self.cfg.time_mask_p and out.shape[0] > 4:
            max_len = max(1, int(out.shape[0] * self.cfg.time_mask_max_frac))
            mask_len = np.random.randint(1, max_len + 1)
            start = np.random.randint(0, out.shape[0] - mask_len + 1)
            out[start : start + mask_len] = 0.0

        # Feature dimension masking
        if np.random.random() < self.cfg.feat_mask_p and out.shape[1] > 10:
            max_dims = max(1, int(out.shape[1] * self.cfg.feat_mask_max_frac))
            n_dims = np.random.randint(1, max_dims + 1)
            dims = np.random.choice(out.shape[1], n_dims, replace=False)
            out[:, dims] = 0.0

        # Feature scaling
        if np.random.random() < self.cfg.scale_p:
            scale = 1.0 + np.random.normal(0.0, self.cfg.scale_std)
            scale = np.clip(scale, 0.85, 1.15).astype(np.float32)
            out = out * scale

        # Temporal jitter (drop random frames)
        if np.random.random() < self.cfg.temporal_jitter_p and out.shape[0] > 8:
            max_drop = max(1, int(out.shape[0] * self.cfg.temporal_jitter_max_frac))
            n_drop = np.random.randint(1, max_drop + 1)
            drop_idx = np.random.choice(out.shape[0], n_drop, replace=False)
            keep_idx = np.setdiff1d(np.arange(out.shape[0]), drop_idx)
            out = out[keep_idx]

        return out.astype(np.float32, copy=False)


class VocalMorphDataset(Dataset):
    """
    Dataset of pre-extracted .npz feature files.
    Expected keys in each .npz:
      sequence, height_cm, weight_kg, age, gender
    Optional keys:
      source, speaker_id, f0_mean, formant_spacing_mean, vtl_mean
    """

    def __init__(
        self,
        features_dir: str,
        max_len: Optional[int] = None,
        target_stats: Optional[Dict] = None,
        crop_mode: str = "head",
        augment: bool = False,
        augment_config: Optional[FeatureAugmentConfig] = None,
    ):
        self.features_dir = features_dir
        self.max_len = max_len
        self.target_stats = target_stats
        self.crop_mode = str(crop_mode or "head").strip().lower()
        if self.crop_mode not in {"head", "center", "random"}:
            raise ValueError(
                f"crop_mode must be one of ['head', 'center', 'random'], got {crop_mode!r}"
            )
        self.augment = augment
        self.augmenter = FeatureAugmenter(augment_config) if augment else None

        self.file_paths = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
        if len(self.file_paths) == 0:
            raise ValueError(f"No .npz files found in: {features_dir}")
        self._speaker_to_indices: Optional[Dict[str, List[int]]] = None

        aug_label = " [augment=ON]" if augment else ""
        print(
            f"  [{os.path.basename(features_dir)}] {len(self.file_paths)} samples{aug_label}"
        )

    def _normalize(self, val: float, key: str) -> float:
        if self.target_stats is None:
            return float(val)
        s = self.target_stats.get(key, {})
        mean = float(s.get("mean", 0.0))
        std = float(s.get("std", 1.0))
        if std < 1e-12:
            return float(val - mean)
        return float((val - mean) / std)

    def denormalize(self, val: torch.Tensor, key: str) -> torch.Tensor:
        if self.target_stats is None:
            return val
        s = self.target_stats.get(key, {})
        return val * float(s.get("std", 1.0)) + float(s.get("mean", 0.0))

    @staticmethod
    def _decode_np_value(v) -> str:
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore")
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return ""
            if v.ndim == 0:
                return str(v.item())
            return str(v[0])
        return str(v)

    def _infer_source(self, data, speaker_id: str, file_path: str) -> str:
        if "source" in data:
            s = self._decode_np_value(data["source"]).strip().upper()
            if s in {"NISP", "TIMIT"}:
                return s
        sid = speaker_id.upper()
        if sid.startswith("TIMIT_"):
            return "TIMIT"
        if sid.startswith("NISP_"):
            return "NISP"
        name = os.path.basename(file_path).upper()
        if name.startswith("TIMIT_"):
            return "TIMIT"
        return "NISP"

    def __len__(self) -> int:
        return len(self.file_paths)

    @classmethod
    def _speaker_id_from_path(cls, path: str) -> str:
        base = os.path.splitext(os.path.basename(path))[0]
        if "_aug" in base:
            base = base.split("_aug", 1)[0]
        parts = base.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]

        data = np.load(path, allow_pickle=True)
        if "speaker_id" in data:
            return cls._decode_np_value(data["speaker_id"])
        return base

    def _crop_sequence(self, seq: np.ndarray) -> np.ndarray:
        if self.max_len is None or seq.shape[0] <= self.max_len:
            return seq

        overflow = int(seq.shape[0] - self.max_len)
        if self.crop_mode == "random":
            start = int(np.random.randint(0, overflow + 1))
        elif self.crop_mode == "center":
            start = overflow // 2
        else:
            start = 0
        return seq[start : start + self.max_len]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.file_paths[idx]
        data = np.load(path, allow_pickle=True)

        seq = data["sequence"].astype(np.float32)
        seq = self._crop_sequence(seq)

        # On-the-fly feature augmentation (training only)
        if self.augmenter is not None:
            seq = self.augmenter(seq)

        speaker_id = (
            self._decode_np_value(data["speaker_id"])
            if "speaker_id" in data
            else os.path.splitext(os.path.basename(path))[0]
        )
        source = self._infer_source(data, speaker_id=speaker_id, file_path=path)
        source_id = 1 if source == "NISP" else 0

        height_raw = float(data["height_cm"])
        age_raw = float(data["age"])

        weight_raw = np.nan
        if "weight_kg" in data:
            try:
                weight_raw = float(data["weight_kg"])
            except Exception:
                weight_raw = np.nan

        if "gender" in data:
            g = data["gender"]
            if isinstance(g, np.ndarray) and g.shape == ():
                g = g.item()
            if isinstance(g, str):
                gender = 1 if g.lower() == "male" else 0
            else:
                gender = int(g)
        else:
            gender = 0

        # NISP has reliable weight labels; TIMIT weight is missing.
        has_weight = bool(np.isfinite(weight_raw)) and source == "NISP"
        weight_mask = 1.0 if has_weight else 0.0
        norm_weight = self._normalize(weight_raw, "weight") if has_weight else 0.0

        f0_mean = float(data["f0_mean"]) if "f0_mean" in data else 0.0
        formant_spacing = (
            float(data["formant_spacing_mean"])
            if "formant_spacing_mean" in data
            else 0.0
        )
        vtl_mean = float(data["vtl_mean"]) if "vtl_mean" in data else 0.0
        duration_s = float(data["duration_s"]) if "duration_s" in data else float("nan")
        speech_ratio = float(data["speech_ratio"]) if "speech_ratio" in data else float("nan")
        snr_db_estimate = (
            float(data["snr_db_estimate"]) if "snr_db_estimate" in data else float("nan")
        )
        capture_quality_score = (
            float(data["capture_quality_score"])
            if "capture_quality_score" in data
            else float("nan")
        )
        voiced_ratio = float(data["voiced_ratio"]) if "voiced_ratio" in data else float("nan")
        clipped_ratio = float(data["clipped_ratio"]) if "clipped_ratio" in data else float("nan")
        distance_cm_estimate = (
            float(data["distance_cm_estimate"])
            if "distance_cm_estimate" in data
            else float("nan")
        )
        distance_confidence = (
            float(data["distance_confidence"])
            if "distance_confidence" in data
            else float("nan")
        )
        quality_ok = bool(data["quality_ok"]) if "quality_ok" in data else True

        return {
            "sequence": torch.from_numpy(seq),
            "height": torch.tensor(
                self._normalize(height_raw, "height"), dtype=torch.float32
            ),
            "weight": torch.tensor(norm_weight, dtype=torch.float32),
            "age": torch.tensor(self._normalize(age_raw, "age"), dtype=torch.float32),
            "gender": torch.tensor(gender, dtype=torch.long),
            "height_raw": torch.tensor(height_raw, dtype=torch.float32),
            "weight_raw": torch.tensor(
                weight_raw if np.isfinite(weight_raw) else float("nan"),
                dtype=torch.float32,
            ),
            "age_raw": torch.tensor(age_raw, dtype=torch.float32),
            "f0_mean": torch.tensor(f0_mean, dtype=torch.float32),
            "formant_spacing_mean": torch.tensor(formant_spacing, dtype=torch.float32),
            "vtl_mean": torch.tensor(vtl_mean, dtype=torch.float32),
            "duration_s": torch.tensor(duration_s, dtype=torch.float32),
            "speech_ratio": torch.tensor(speech_ratio, dtype=torch.float32),
            "snr_db_estimate": torch.tensor(snr_db_estimate, dtype=torch.float32),
            "capture_quality_score": torch.tensor(
                capture_quality_score, dtype=torch.float32
            ),
            "voiced_ratio": torch.tensor(voiced_ratio, dtype=torch.float32),
            "clipped_ratio": torch.tensor(clipped_ratio, dtype=torch.float32),
            "distance_cm_estimate": torch.tensor(
                distance_cm_estimate, dtype=torch.float32
            ),
            "distance_confidence": torch.tensor(
                distance_confidence, dtype=torch.float32
            ),
            "quality_ok": torch.tensor(1 if quality_ok else 0, dtype=torch.long),
            "weight_mask": torch.tensor(weight_mask, dtype=torch.float32),
            "source_id": torch.tensor(source_id, dtype=torch.long),
            "speaker_id": speaker_id,
        }

    def gender_counts(self) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for p in self.file_paths:
            data = np.load(p, allow_pickle=True)
            g = data["gender"] if "gender" in data else 0
            if isinstance(g, np.ndarray) and g.shape == ():
                g = g.item()
            if isinstance(g, str):
                g = 1 if g.lower() == "male" else 0
            counts[int(g)] = counts.get(int(g), 0) + 1
        return counts

    def infer_input_dim(self) -> int:
        sample = np.load(self.file_paths[0], allow_pickle=True)
        return int(sample["sequence"].shape[1])

    def speaker_to_indices(self) -> Dict[str, List[int]]:
        if self._speaker_to_indices is None:
            mapping: Dict[str, List[int]] = {}
            for idx, path in enumerate(self.file_paths):
                speaker_id = self._speaker_id_from_path(path)
                mapping.setdefault(speaker_id, []).append(idx)
            self._speaker_to_indices = mapping
        return self._speaker_to_indices


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences. Returns padding mask (True = padded)."""
    sequences = [item["sequence"] for item in batch]
    lengths = [s.shape[0] for s in sequences]
    max_len = max(lengths)

    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    padding_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < max_len:
            padding_mask[i, l:] = True

    scalar_keys = [
        "height",
        "weight",
        "age",
        "gender",
        "height_raw",
        "weight_raw",
        "age_raw",
        "f0_mean",
        "formant_spacing_mean",
        "vtl_mean",
        "duration_s",
        "speech_ratio",
        "snr_db_estimate",
        "capture_quality_score",
        "voiced_ratio",
        "clipped_ratio",
        "distance_cm_estimate",
        "distance_confidence",
        "quality_ok",
        "weight_mask",
        "source_id",
    ]
    default_scalars = {
        "duration_s": float("nan"),
        "speech_ratio": float("nan"),
        "snr_db_estimate": float("nan"),
        "capture_quality_score": float("nan"),
        "voiced_ratio": float("nan"),
        "clipped_ratio": float("nan"),
        "distance_cm_estimate": float("nan"),
        "distance_confidence": float("nan"),
        "quality_ok": 1,
        "weight_mask": 1.0,
        "source_id": 0,
    }

    result = {
        "sequence": padded,
        "padding_mask": padding_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "speaker_id": [item["speaker_id"] for item in batch],
    }
    for k in scalar_keys:
        values = []
        for item in batch:
            if k in item:
                values.append(item[k])
                continue
            sample_sequence = item["sequence"]
            default_value = default_scalars.get(k, 0.0)
            dtype = torch.long if k in {"gender", "source_id", "quality_ok"} else torch.float32
            values.append(torch.tensor(default_value, dtype=dtype, device=sample_sequence.device))
        result[k] = torch.stack(values)

    return result


def build_dataloaders_from_dirs(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    target_stats: Optional[Dict] = None,
    max_len: Optional[int] = None,
    train_crop_mode: str = "head",
    eval_crop_mode: str = "center",
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    train_augment: bool = False,
    augment_config: Optional[FeatureAugmentConfig] = None,
    speaker_batching: Optional[Dict[str, int]] = None,
    base_seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print("[VocalMorph Dataset] Loading splits:")
    train_ds = VocalMorphDataset(
        train_dir,
        max_len=max_len,
        target_stats=target_stats,
        crop_mode=train_crop_mode,
        augment=train_augment,
        augment_config=augment_config,
    )
    val_ds = VocalMorphDataset(
        val_dir, max_len=max_len, target_stats=target_stats, crop_mode=eval_crop_mode
    )
    test_ds = VocalMorphDataset(
        test_dir, max_len=max_len, target_stats=target_stats, crop_mode=eval_crop_mode
    )

    pin = torch.cuda.is_available()
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            num_workers > 0 if persistent_workers is None else persistent_workers
        )
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader_kwargs["worker_init_fn"] = build_worker_init_fn(int(base_seed))

    speaker_batching = dict(speaker_batching or {})
    speaker_batching_enabled = bool(speaker_batching.get("enabled", False))
    train_batch_sampler = None
    if speaker_batching_enabled:
        batching_mode = str(speaker_batching.get("mode", "grouped")).strip().lower()
        speakers_per_batch = int(speaker_batching.get("speakers_per_batch", 8))
        clips_per_speaker = int(speaker_batching.get("clips_per_speaker", 2))
        if batching_mode == "hybrid":
            train_batch_sampler = HybridSpeakerBatchSampler(
                train_ds.speaker_to_indices(),
                paired_speakers_per_batch=int(
                    speaker_batching.get("paired_speakers_per_batch", 4)
                ),
                singleton_speakers_per_batch=int(
                    speaker_batching.get("singleton_speakers_per_batch", 8)
                ),
                clips_per_speaker=clips_per_speaker,
                base_seed=int(base_seed),
                drop_last=False,
            )
        else:
            train_batch_sampler = GroupedSpeakerBatchSampler(
                train_ds.speaker_to_indices(),
                speakers_per_batch=speakers_per_batch,
                clips_per_speaker=clips_per_speaker,
                base_seed=int(base_seed),
                drop_last=False,
            )

    if train_batch_sampler is None:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


# Legacy helper kept for backward compatibility.
def build_dataloaders(
    features_dir: str,
    metadata_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    random_seed: int = 42,
    max_len: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[Dict]]:
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import shutil
    import tempfile

    df = pd.read_csv(metadata_path)
    all_ids = df["speaker_id"].tolist()
    genders = (
        df["gender"].apply(lambda x: 1 if str(x).lower() == "male" else 0).tolist()
    )

    train_ids, temp_ids, _, temp_genders = train_test_split(
        all_ids,
        genders,
        train_size=train_ratio,
        stratify=genders,
        random_state=random_seed,
    )
    val_size = val_ratio / (1 - train_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_size,
        stratify=temp_genders,
        random_state=random_seed,
    )

    tmpdir = tempfile.mkdtemp()
    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_dir = os.path.join(tmpdir, split)
        os.makedirs(split_dir)
        for sid in ids:
            src = os.path.join(features_dir, f"{sid}.npz")
            if os.path.exists(src):
                shutil.copy(src, split_dir)

    stats_path = os.path.join(features_dir, "target_stats.json")
    target_stats = None
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            target_stats = json.load(f)

    return build_dataloaders_from_dirs(
        os.path.join(tmpdir, "train"),
        os.path.join(tmpdir, "val"),
        os.path.join(tmpdir, "test"),
        batch_size=batch_size,
        num_workers=num_workers,
        target_stats=target_stats,
        max_len=max_len,
    ) + (target_stats,)
