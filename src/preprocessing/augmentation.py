"""Audio augmentation helpers for VocalMorph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import signal

try:
    from audiomentations import AddGaussianNoise, Compose, Gain, PitchShift, TimeStretch

    AUDIOMENTATIONS_AVAILABLE = True
except Exception:
    AUDIOMENTATIONS_AVAILABLE = False


@dataclass
class AugmentationConfig:
    noise_p: float = 0.35
    time_stretch_p: float = 0.35
    pitch_shift_p: float = 0.35
    gain_p: float = 0.25
    bandlimit_p: float = 0.25
    clip_p: float = 0.20
    dropout_p: float = 0.15
    colored_noise_p: float = 0.25
    distance_p: float = 0.20

    noise_min: float = 0.0005
    noise_max: float = 0.0100

    stretch_min: float = 0.90
    stretch_max: float = 1.10

    pitch_min: float = -2.0
    pitch_max: float = 2.0

    gain_min_db: float = -6.0
    gain_max_db: float = 6.0

    clip_min: float = 0.72
    clip_max: float = 0.96

    dropout_max_fraction: float = 0.08
    dropout_segments: int = 2

    colored_noise_min: float = 0.0003
    colored_noise_max: float = 0.0060

    lowpass_min_hz: float = 2600.0
    lowpass_max_hz: float = 6500.0
    highpass_min_hz: float = 40.0
    highpass_max_hz: float = 180.0
    filter_order: int = 4
    distance_lowpass_min_hz: float = 1800.0
    distance_lowpass_max_hz: float = 4200.0
    distance_min_gain: float = 0.45
    distance_max_gain: float = 0.85
    distance_reflections: int = 4
    distance_delay_ms_min: float = 12.0
    distance_delay_ms_max: float = 85.0
    distance_decay_min: float = 0.20
    distance_decay_max: float = 0.55


def build_augmenter(config: Optional[AugmentationConfig] = None):
    if not AUDIOMENTATIONS_AVAILABLE:
        return None
    cfg = config or AugmentationConfig()
    return Compose(
        [
            AddGaussianNoise(min_amplitude=cfg.noise_min, max_amplitude=cfg.noise_max, p=cfg.noise_p),
            TimeStretch(min_rate=cfg.stretch_min, max_rate=cfg.stretch_max, p=cfg.time_stretch_p),
            PitchShift(min_semitones=cfg.pitch_min, max_semitones=cfg.pitch_max, p=cfg.pitch_shift_p),
            Gain(min_gain_db=cfg.gain_min_db, max_gain_db=cfg.gain_max_db, p=cfg.gain_p),
        ]
    )


def _rng() -> np.random.Generator:
    return np.random.default_rng()


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    return np.clip(arr, -1.0, 1.0).astype(np.float32, copy=False)


def _colored_noise(length: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(length).astype(np.float32)
    cutoff = rng.uniform(600.0, min(2400.0, (sample_rate * 0.5) - 100.0))
    normalized = max(1e-4, cutoff / max((sample_rate * 0.5), 1.0))
    sos = signal.butter(2, normalized, btype="lowpass", output="sos")
    shaped = signal.sosfiltfilt(sos, noise) if length > 64 else signal.sosfilt(sos, noise)
    shaped = shaped.astype(np.float32, copy=False)
    shaped /= float(np.std(shaped) + 1e-6)
    return shaped


def _random_bandlimit(audio: np.ndarray, sample_rate: int, cfg: AugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    if audio.size == 0:
        return audio
    nyquist = 0.5 * float(sample_rate)
    mode = rng.choice(["lowpass", "highpass", "bandpass"])
    if mode == "lowpass":
        cutoff = min(rng.uniform(cfg.lowpass_min_hz, cfg.lowpass_max_hz), nyquist - 120.0)
        if cutoff <= 0.0:
            return audio
        sos = signal.butter(cfg.filter_order, cutoff / nyquist, btype="lowpass", output="sos")
    elif mode == "highpass":
        cutoff = min(rng.uniform(cfg.highpass_min_hz, cfg.highpass_max_hz), nyquist - 120.0)
        if cutoff <= 0.0:
            return audio
        sos = signal.butter(cfg.filter_order, cutoff / nyquist, btype="highpass", output="sos")
    else:
        low = rng.uniform(max(40.0, cfg.highpass_min_hz), min(220.0, nyquist * 0.15))
        high = rng.uniform(max(low + 800.0, cfg.lowpass_min_hz), min(cfg.lowpass_max_hz, nyquist - 120.0))
        if high <= low:
            return audio
        sos = signal.butter(cfg.filter_order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, audio) if audio.size > 64 else signal.sosfilt(sos, audio)
    return filtered.astype(np.float32, copy=False)


def _random_dropout(audio: np.ndarray, cfg: AugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    out = audio.copy()
    max_fraction = float(np.clip(cfg.dropout_max_fraction, 0.0, 0.40))
    max_seg_len = max(1, int(round(out.size * max_fraction)))
    for _ in range(max(1, int(cfg.dropout_segments))):
        seg_len = int(rng.integers(1, max_seg_len + 1))
        if seg_len >= out.size:
            break
        start = int(rng.integers(0, max(1, out.size - seg_len)))
        out[start : start + seg_len] *= float(rng.uniform(0.0, 0.35))
    return out.astype(np.float32, copy=False)


def _soft_clip(audio: np.ndarray, cfg: AugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    threshold = float(rng.uniform(cfg.clip_min, cfg.clip_max))
    if threshold <= 0.0:
        return audio
    out = np.clip(audio, -threshold, threshold) / threshold
    return out.astype(np.float32, copy=False)


def _simulate_far_microphone(audio: np.ndarray, sample_rate: int, cfg: AugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    if audio.size == 0:
        return audio

    out = np.asarray(audio, dtype=np.float32).copy()
    nyquist = 0.5 * float(sample_rate)
    cutoff = float(rng.uniform(cfg.distance_lowpass_min_hz, cfg.distance_lowpass_max_hz))
    cutoff = min(cutoff, nyquist - 120.0)
    if cutoff > 0.0:
        sos = signal.butter(cfg.filter_order, cutoff / nyquist, btype="lowpass", output="sos")
        out = signal.sosfiltfilt(sos, out) if out.size > 64 else signal.sosfilt(sos, out)
        out = out.astype(np.float32, copy=False)

    dry_gain = float(rng.uniform(cfg.distance_min_gain, cfg.distance_max_gain))
    mixed = out * dry_gain
    decay = float(rng.uniform(cfg.distance_decay_min, cfg.distance_decay_max))
    n_reflections = max(1, int(cfg.distance_reflections))
    for idx in range(n_reflections):
        delay_ms = float(rng.uniform(cfg.distance_delay_ms_min, cfg.distance_delay_ms_max))
        delay = max(1, int(round(sample_rate * delay_ms / 1000.0)))
        if delay >= mixed.size:
            break
        refl_gain = decay ** (idx + 1)
        mixed[delay:] += out[:-delay] * refl_gain

    return mixed.astype(np.float32, copy=False)


def apply_custom_augmentations(
    audio: np.ndarray,
    sample_rate: int,
    config: Optional[AugmentationConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    cfg = config or AugmentationConfig()
    local_rng = rng or _rng()
    out = np.asarray(audio, dtype=np.float32).copy()

    if out.size == 0:
        return out

    if local_rng.random() < cfg.colored_noise_p:
        amp = float(local_rng.uniform(cfg.colored_noise_min, cfg.colored_noise_max))
        out = out + amp * _colored_noise(out.size, sample_rate, local_rng)

    if local_rng.random() < cfg.distance_p:
        out = _simulate_far_microphone(out, sample_rate, cfg, local_rng)

    if local_rng.random() < cfg.bandlimit_p:
        out = _random_bandlimit(out, sample_rate, cfg, local_rng)

    if local_rng.random() < cfg.dropout_p:
        out = _random_dropout(out, cfg, local_rng)

    if local_rng.random() < cfg.clip_p:
        out = _soft_clip(out, cfg, local_rng)

    return _normalize_audio(out)


def apply_augmentations(
    audio: np.ndarray,
    sample_rate: int,
    augmenter,
    n_variants: int,
    config: Optional[AugmentationConfig] = None,
) -> List[np.ndarray]:
    """Generate `n_variants` augmented versions of one waveform."""
    if n_variants <= 0:
        return []

    cfg = config or AugmentationConfig()
    rng = _rng()
    variants: List[np.ndarray] = []
    for _ in range(n_variants):
        aug = np.asarray(audio, dtype=np.float32).copy()
        if augmenter is not None:
            aug = augmenter(samples=aug, sample_rate=sample_rate)
        aug = apply_custom_augmentations(aug, sample_rate, config=cfg, rng=rng)
        variants.append(aug.astype(np.float32, copy=False))
    return variants
