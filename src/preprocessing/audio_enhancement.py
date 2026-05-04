"""Speech enhancement utilities for microphone and noisy speech recordings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy import signal

try:
    import webrtcvad

    WEBRTCVAD_AVAILABLE = True
except Exception:
    WEBRTCVAD_AVAILABLE = False


@dataclass
class MicrophoneEnhancementConfig:
    enabled: bool = True
    highpass_hz: float = 55.0
    lowpass_hz: float = 7600.0
    filter_order: int = 4
    vad_aggressiveness: int = 2
    vad_frame_ms: int = 30
    vad_keep_silence_ms: int = 120
    trim_db: float = 28.0
    target_rms: float = 0.10
    max_gain_db: float = 18.0
    peak_limit: float = 0.98
    preemphasis: float = 0.97
    spectral_gate_strength: float = 1.35
    spectral_gate_floor: float = 0.10
    noise_frame_percentile: float = 0.20
    smooth_freq_bins: int = 5
    smooth_time_frames: int = 7
    apply_vad_mask: bool = True
    min_speech_ratio: float = 0.45
    min_snr_db: float = 8.0
    max_clipped_ratio: float = 0.02
    max_distance_cm: float = 45.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpeechEnhancementReport:
    input_duration_s: float
    output_duration_s: float
    speech_ratio: float
    snr_db_estimate: float
    clipped_ratio: float
    used_vad: bool
    used_spectral_gate: bool
    capture_quality_score: float
    distance_cm_estimate: float
    distance_confidence: float
    distance_band: str
    hf_ratio_db: float
    reverb_tail_ratio: float
    quality_ok: bool
    quality_warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if arr.ndim != 1:
        raise ValueError(f"audio must be a 1-D waveform after mono conversion, got shape {tuple(arr.shape)}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _remove_dc(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    return (audio - float(audio.mean())).astype(np.float32, copy=False)


def _peak_normalize(audio: np.ndarray, peak_limit: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-8:
        return audio.astype(np.float32, copy=False)
    scale = min(float(peak_limit) / peak, 1.0)
    return (audio * scale).astype(np.float32, copy=False)


def _rms_normalize(audio: np.ndarray, target_rms: float, max_gain_db: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms < 1e-8:
        return audio.astype(np.float32, copy=False)
    max_scale = 10.0 ** (float(max_gain_db) / 20.0)
    scale = min(float(target_rms) / rms, max_scale)
    return (audio * scale).astype(np.float32, copy=False)


def _apply_sos_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int, mode: str) -> np.ndarray:
    nyquist = 0.5 * float(sample_rate)
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist:
        return audio.astype(np.float32, copy=False)
    normalized = float(cutoff_hz) / nyquist
    sos = signal.butter(int(order), normalized, btype=mode, output="sos")
    try:
        if audio.size > max(32, order * 8):
            filtered = signal.sosfiltfilt(sos, audio)
        else:
            filtered = signal.sosfilt(sos, audio)
    except ValueError:
        filtered = signal.sosfilt(sos, audio)
    return filtered.astype(np.float32, copy=False)


def _frame_rms(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float32)
    frame_length = max(128, int(sample_rate * 0.025))
    hop_length = max(64, int(sample_rate * 0.010))
    return librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0].astype(np.float32, copy=False)


def _rms_activity_envelope(audio: np.ndarray, sample_rate: int) -> Tuple[Optional[np.ndarray], float]:
    rms = _frame_rms(audio, sample_rate)
    if rms.size == 0:
        return None, 0.0

    noise_floor = float(np.quantile(rms, 0.20))
    speech_peak = float(np.quantile(rms, 0.90))
    threshold = max(noise_floor * 1.8, noise_floor + 0.15 * max(speech_peak - noise_floor, 1e-6))
    active = rms >= threshold
    if not bool(active.any()):
        active[int(np.argmax(rms))] = True

    hop_length = max(64, int(sample_rate * 0.010))
    sample_mask = np.repeat(active.astype(np.float32), hop_length)
    if sample_mask.size < audio.size:
        sample_mask = np.pad(sample_mask, (0, audio.size - sample_mask.size), mode="edge")
    sample_mask = sample_mask[: audio.size]
    return sample_mask.astype(np.float32, copy=False), float(sample_mask.mean()) if sample_mask.size else 0.0


def _estimate_noise_frames(magnitude: np.ndarray, percentile: float) -> np.ndarray:
    frame_energy = magnitude.mean(axis=0)
    if frame_energy.size == 0:
        return np.zeros((0,), dtype=bool)
    percentile = float(np.clip(percentile, 0.05, 0.80))
    threshold = np.quantile(frame_energy, percentile)
    mask = frame_energy <= threshold
    if not bool(mask.any()):
        mask[np.argmin(frame_energy)] = True
    return mask


def _smooth_mask(mask: np.ndarray, freq_bins: int, time_frames: int) -> np.ndarray:
    smoothed = mask
    if freq_bins > 1:
        kernel = np.ones((int(freq_bins), 1), dtype=np.float32) / float(freq_bins)
        smoothed = signal.convolve2d(smoothed, kernel, mode="same", boundary="symm")
    if time_frames > 1:
        kernel = np.ones((1, int(time_frames)), dtype=np.float32) / float(time_frames)
        smoothed = signal.convolve2d(smoothed, kernel, mode="same", boundary="symm")
    return smoothed.astype(np.float32, copy=False)


def spectral_gate(
    audio: np.ndarray,
    sample_rate: int,
    config: Optional[MicrophoneEnhancementConfig] = None,
) -> np.ndarray:
    cfg = config or MicrophoneEnhancementConfig()
    x = _to_mono_float32(audio)
    if x.size == 0:
        return x

    n_fft = 512
    hop_length = 160 if sample_rate >= 16000 else max(64, int(sample_rate * 0.01))
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    noise_frames = _estimate_noise_frames(magnitude, cfg.noise_frame_percentile)
    noise_profile = np.median(magnitude[:, noise_frames], axis=1, keepdims=True)

    denom = magnitude + 1e-6
    gain = 1.0 - float(cfg.spectral_gate_strength) * (noise_profile / denom)
    gain = np.clip(gain, float(cfg.spectral_gate_floor), 1.0).astype(np.float32, copy=False)
    gain = _smooth_mask(gain, cfg.smooth_freq_bins, cfg.smooth_time_frames)

    enhanced = librosa.istft(stft * gain, hop_length=hop_length, length=x.size)
    return enhanced.astype(np.float32, copy=False)


def _vad_envelope(
    audio: np.ndarray,
    sample_rate: int,
    config: MicrophoneEnhancementConfig,
) -> Tuple[Optional[np.ndarray], bool]:
    if not WEBRTCVAD_AVAILABLE or sample_rate not in {8000, 16000, 32000, 48000}:
        return None, False

    frame_len = int(sample_rate * (int(config.vad_frame_ms) / 1000.0))
    if frame_len <= 0 or audio.size < frame_len:
        return None, False

    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    vad = webrtcvad.Vad(int(np.clip(config.vad_aggressiveness, 0, 3)))

    voiced = []
    for start in range(0, pcm_i16.size - frame_len + 1, frame_len):
        frame = pcm_i16[start : start + frame_len]
        voiced.append(vad.is_speech(frame.tobytes(), sample_rate))

    if not voiced:
        return None, True

    voiced_mask = np.asarray(voiced, dtype=np.float32)
    keep_frames = max(0, int(round(config.vad_keep_silence_ms / max(config.vad_frame_ms, 1))))
    if keep_frames > 0:
        kernel = np.ones((keep_frames * 2) + 1, dtype=np.float32)
        voiced_mask = (np.convolve(voiced_mask, kernel, mode="same") > 0.0).astype(np.float32)

    sample_mask = np.repeat(voiced_mask, frame_len)
    if sample_mask.size < audio.size:
        sample_mask = np.pad(sample_mask, (0, audio.size - sample_mask.size), mode="edge")
    sample_mask = sample_mask[: audio.size]
    return sample_mask.astype(np.float32, copy=False), True


def _apply_activity_mask(
    audio: np.ndarray,
    envelope: Optional[np.ndarray],
    apply_mask: bool,
) -> Tuple[np.ndarray, float]:
    if envelope is None:
        return audio.astype(np.float32, copy=False), 0.0

    speech_ratio = float(envelope.mean()) if envelope.size else 0.0
    if not apply_mask or speech_ratio <= 0.0:
        return audio.astype(np.float32, copy=False), speech_ratio

    masked = (audio * envelope).astype(np.float32, copy=False)
    active = np.flatnonzero(envelope > 0.0)
    if active.size == 0:
        return masked, speech_ratio

    start = max(0, int(active[0]))
    end = min(masked.size, int(active[-1]) + 1)
    return masked[start:end].astype(np.float32, copy=False), speech_ratio


def _estimate_snr_db(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size == 0:
        return 0.0
    rms = _frame_rms(audio, sample_rate)
    if rms.size == 0:
        return 0.0
    low = np.quantile(rms, 0.15)
    high = np.quantile(rms, 0.85)
    noise_power = max(float(low) ** 2, 1e-8)
    signal_power = max(float(high) ** 2, noise_power + 1e-8)
    return float(10.0 * np.log10(signal_power / noise_power))


def _estimate_hf_ratio_db(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size < 128:
        return -30.0
    n_fft = 512 if audio.size >= 512 else 256
    hop_length = max(64, int(sample_rate * 0.010))
    power = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)) ** 2
    if power.size == 0:
        return -30.0
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    low_band = (freqs >= 250.0) & (freqs <= 1200.0)
    high_band = (freqs >= 2500.0) & (freqs <= min(6500.0, (sample_rate * 0.5) - 1.0))
    low_energy = float(power[low_band].mean()) if bool(low_band.any()) else 1e-8
    high_energy = float(power[high_band].mean()) if bool(high_band.any()) else 1e-8
    return float(10.0 * np.log10(max(high_energy, 1e-8) / max(low_energy, 1e-8)))


def _estimate_reverb_tail_ratio(audio: np.ndarray, sample_rate: int, envelope: Optional[np.ndarray]) -> float:
    if audio.size == 0 or envelope is None or envelope.size != audio.size:
        return 0.0

    active = envelope > 0.5
    if not bool(active.any()):
        return 0.0

    segment_ends = np.flatnonzero(active[:-1] & ~active[1:]) + 1
    if segment_ends.size == 0:
        return 0.0

    lookback = max(64, int(sample_rate * 0.040))
    tail_start = max(32, int(sample_rate * 0.030))
    tail_end = max(tail_start + 32, int(sample_rate * 0.180))
    ratios: List[float] = []
    for end_idx in segment_ends[:8]:
        speech_start = max(0, int(end_idx) - lookback)
        tail_a = min(audio.size, int(end_idx) + tail_start)
        tail_b = min(audio.size, int(end_idx) + tail_end)
        speech_chunk = audio[speech_start:end_idx]
        tail_chunk = audio[tail_a:tail_b]
        if speech_chunk.size < 32 or tail_chunk.size < 32:
            continue
        speech_energy = float(np.mean(np.square(speech_chunk, dtype=np.float64)))
        tail_energy = float(np.mean(np.square(tail_chunk, dtype=np.float64)))
        if speech_energy <= 1e-8:
            continue
        ratios.append(float(np.clip(tail_energy / speech_energy, 0.0, 1.5)))
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def _assess_capture_quality(
    audio: np.ndarray,
    sample_rate: int,
    speech_ratio: float,
    clipped_ratio: float,
    envelope: Optional[np.ndarray],
    config: MicrophoneEnhancementConfig,
) -> Dict[str, Any]:
    snr_db = _estimate_snr_db(audio, sample_rate)
    hf_ratio_db = _estimate_hf_ratio_db(audio, sample_rate)
    reverb_tail_ratio = _estimate_reverb_tail_ratio(audio, sample_rate, envelope)

    snr_score = float(np.clip((snr_db - 6.0) / 18.0, 0.0, 1.0))
    speech_score = float(np.clip((speech_ratio - 0.35) / 0.50, 0.0, 1.0))
    clip_score = 1.0 - float(np.clip(clipped_ratio / max(config.max_clipped_ratio, 1e-6), 0.0, 1.0))
    hf_score = float(np.clip((hf_ratio_db + 22.0) / 18.0, 0.0, 1.0))
    tail_score = 1.0 - float(np.clip(reverb_tail_ratio / 0.30, 0.0, 1.0))

    proximity_score = float(
        np.clip(
            0.35 * snr_score + 0.20 * speech_score + 0.20 * hf_score + 0.15 * tail_score + 0.10 * clip_score,
            0.0,
            1.0,
        )
    )
    distance_cm = float(np.clip(75.0 - (55.0 * proximity_score), 12.0, 80.0))
    if distance_cm <= 20.0:
        distance_band = "near"
    elif distance_cm <= 40.0:
        distance_band = "mid"
    else:
        distance_band = "far"

    confidence = float(
        np.clip(
            0.25 + 0.35 * speech_score + 0.20 * snr_score + 0.10 * tail_score,
            0.20,
            0.90,
        )
    )
    quality_score = float(
        np.clip(
            0.35 * snr_score + 0.20 * speech_score + 0.15 * clip_score + 0.15 * hf_score + 0.15 * tail_score,
            0.0,
            1.0,
        )
    )

    warnings: List[str] = []
    if speech_ratio < config.min_speech_ratio:
        warnings.append("low_speech_coverage")
    if snr_db < config.min_snr_db:
        warnings.append("low_snr")
    if clipped_ratio > config.max_clipped_ratio:
        warnings.append("input_clipping")
    if distance_cm > config.max_distance_cm:
        warnings.append("far_microphone")
    if reverb_tail_ratio > 0.30:
        warnings.append("reverb_heavy")

    return {
        "snr_db_estimate": snr_db,
        "hf_ratio_db": hf_ratio_db,
        "reverb_tail_ratio": reverb_tail_ratio,
        "capture_quality_score": quality_score,
        "distance_cm_estimate": distance_cm,
        "distance_confidence": confidence,
        "distance_band": distance_band,
        "quality_ok": len(warnings) == 0,
        "quality_warnings": warnings,
    }


def enhance_microphone_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: Optional[MicrophoneEnhancementConfig] = None,
) -> Tuple[np.ndarray, SpeechEnhancementReport]:
    cfg = config or MicrophoneEnhancementConfig()
    x = _to_mono_float32(audio)
    input_duration_s = float(x.size) / float(max(sample_rate, 1))
    input_clipped_ratio = float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0

    if not cfg.enabled or x.size == 0:
        diagnostics = _assess_capture_quality(
            x,
            sample_rate,
            speech_ratio=1.0 if x.size else 0.0,
            clipped_ratio=input_clipped_ratio,
            envelope=None,
            config=cfg,
        )
        report = SpeechEnhancementReport(
            input_duration_s=input_duration_s,
            output_duration_s=input_duration_s,
            speech_ratio=1.0 if x.size else 0.0,
            snr_db_estimate=diagnostics["snr_db_estimate"],
            clipped_ratio=input_clipped_ratio,
            used_vad=False,
            used_spectral_gate=False,
            capture_quality_score=diagnostics["capture_quality_score"],
            distance_cm_estimate=diagnostics["distance_cm_estimate"],
            distance_confidence=diagnostics["distance_confidence"],
            distance_band=diagnostics["distance_band"],
            hf_ratio_db=diagnostics["hf_ratio_db"],
            reverb_tail_ratio=diagnostics["reverb_tail_ratio"],
            quality_ok=diagnostics["quality_ok"],
            quality_warnings=diagnostics["quality_warnings"],
        )
        return x.astype(np.float32, copy=False), report

    x = _remove_dc(x)
    x = _peak_normalize(x, cfg.peak_limit)
    x = _apply_sos_filter(x, sample_rate, cfg.highpass_hz, cfg.filter_order, mode="highpass")
    x = _apply_sos_filter(x, sample_rate, cfg.lowpass_hz, cfg.filter_order, mode="lowpass")
    analysis_signal = x.copy()
    activity_envelope, used_vad = _vad_envelope(analysis_signal, sample_rate, cfg)
    speech_ratio = 0.0
    if activity_envelope is not None:
        speech_ratio = float(activity_envelope.mean()) if activity_envelope.size else 0.0
    else:
        activity_envelope, speech_ratio = _rms_activity_envelope(analysis_signal, sample_rate)
    x = signal.lfilter([1.0, -float(cfg.preemphasis)], [1.0], x).astype(np.float32, copy=False)
    x = spectral_gate(x, sample_rate, config=cfg)
    x, speech_ratio = _apply_activity_mask(x, activity_envelope, cfg.apply_vad_mask)
    if x.size:
        x, _ = librosa.effects.trim(x, top_db=float(cfg.trim_db))
    x = _rms_normalize(x, cfg.target_rms, cfg.max_gain_db)
    x = _peak_normalize(x, cfg.peak_limit)

    output_duration_s = float(x.size) / float(max(sample_rate, 1))
    clipped_ratio = float(np.mean(np.abs(x) >= cfg.peak_limit)) if x.size else 0.0
    diagnostics = _assess_capture_quality(
        analysis_signal,
        sample_rate,
        speech_ratio=speech_ratio,
        clipped_ratio=input_clipped_ratio,
        envelope=activity_envelope,
        config=cfg,
    )
    report = SpeechEnhancementReport(
        input_duration_s=input_duration_s,
        output_duration_s=output_duration_s,
        speech_ratio=speech_ratio,
        snr_db_estimate=diagnostics["snr_db_estimate"],
        clipped_ratio=clipped_ratio,
        used_vad=used_vad,
        used_spectral_gate=True,
        capture_quality_score=diagnostics["capture_quality_score"],
        distance_cm_estimate=diagnostics["distance_cm_estimate"],
        distance_confidence=diagnostics["distance_confidence"],
        distance_band=diagnostics["distance_band"],
        hf_ratio_db=diagnostics["hf_ratio_db"],
        reverb_tail_ratio=diagnostics["reverb_tail_ratio"],
        quality_ok=diagnostics["quality_ok"],
        quality_warnings=diagnostics["quality_warnings"],
    )
    return x.astype(np.float32, copy=False), report


__all__ = [
    "MicrophoneEnhancementConfig",
    "SpeechEnhancementReport",
    "WEBRTCVAD_AVAILABLE",
    "enhance_microphone_audio",
    "spectral_gate",
]
