import os
import sys

import numpy as np
import soundfile as sf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.augmentation import AugmentationConfig, apply_custom_augmentations
from src.preprocessing.audio_enhancement import MicrophoneEnhancementConfig, enhance_microphone_audio
from src.preprocessing.feature_extractor import load_audio


def _make_noisy_speech(sample_rate: int = 16000, seconds: float = 3.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False, dtype=np.float32)
    speech = 0.20 * np.sin(2.0 * np.pi * 140.0 * t) + 0.08 * np.sin(2.0 * np.pi * 220.0 * t)
    envelope = np.zeros_like(t)
    envelope[int(0.5 * sample_rate) : int(2.5 * sample_rate)] = 1.0
    noise = 0.05 * np.random.default_rng(13).standard_normal(t.shape[0]).astype(np.float32)
    return (speech * envelope + noise).astype(np.float32)


def test_enhance_microphone_audio_returns_finite_waveform():
    audio = _make_noisy_speech()
    enhanced, report = enhance_microphone_audio(audio, 16000, MicrophoneEnhancementConfig())
    assert enhanced.ndim == 1
    assert enhanced.dtype == np.float32
    assert enhanced.size > 0
    assert np.isfinite(enhanced).all()
    assert np.max(np.abs(enhanced)) <= 1.0
    assert report.output_duration_s > 0.0
    assert report.snr_db_estimate == report.snr_db_estimate
    assert 0.0 <= report.capture_quality_score <= 1.0
    assert 10.0 <= report.distance_cm_estimate <= 80.0
    assert 0.0 <= report.distance_confidence <= 1.0
    assert report.distance_band in {"near", "mid", "far"}
    assert report.reverb_tail_ratio >= 0.0


def test_load_audio_returns_enhancement_metadata(tmp_path):
    wav_path = tmp_path / "sample.wav"
    sf.write(str(wav_path), _make_noisy_speech(seconds=3.5), 16000)
    loaded = load_audio(
        str(wav_path),
        target_sr=16000,
        min_duration=1.5,
        enhance=True,
        enhancement_config=MicrophoneEnhancementConfig(),
        return_metadata=True,
    )
    assert loaded is not None
    audio, metadata = loaded
    assert audio.ndim == 1
    assert np.isfinite(audio).all()
    assert metadata["enhancement"] is not None
    assert metadata["duration_s"] >= 1.5
    assert "distance_cm_estimate" in metadata["enhancement"]
    assert "quality_ok" in metadata["enhancement"]


def test_custom_augmentation_keeps_audio_finite():
    audio = _make_noisy_speech()
    augmented = apply_custom_augmentations(audio, 16000, AugmentationConfig())
    assert augmented.shape == audio.shape
    assert augmented.dtype == np.float32
    assert np.isfinite(augmented).all()
    assert np.max(np.abs(augmented)) <= 1.0


def test_distance_augmentation_keeps_audio_finite():
    audio = _make_noisy_speech()
    cfg = AugmentationConfig(
        colored_noise_p=0.0,
        bandlimit_p=0.0,
        dropout_p=0.0,
        clip_p=0.0,
        distance_p=1.0,
    )
    augmented = apply_custom_augmentations(audio, 16000, cfg)
    assert augmented.shape == audio.shape
    assert augmented.dtype == np.float32
    assert np.isfinite(augmented).all()
    assert np.max(np.abs(augmented)) <= 1.0
