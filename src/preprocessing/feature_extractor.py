"""
VocalMorph — Audio Feature Extraction Pipeline
===============================================
Extracts:
  - MFCCs (+ delta, delta-delta)
  - Formant frequencies (F1-F4) via Praat/parselmouth
  - Fundamental frequency (F0/pitch)
  - Spectral features (centroid, rolloff, flux, bandwidth, contrast)
  - Vocal Tract Length estimation
  - Voice quality (jitter, shimmer, HNR)

Author: Asnan P
"""

import os
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import librosa
import numpy as np

from .audio_enhancement import MicrophoneEnhancementConfig, enhance_microphone_audio

warnings.filterwarnings("ignore")

VTL_MIN_CM = 10.0
VTL_MAX_CM = 35.0

# Optional imports — degrade gracefully if not installed
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    warnings.warn("parselmouth not installed. Strict audited feature builds will fail closed.")


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    include_delta: bool = True
    include_delta_delta: bool = True
    n_formants: int = 4
    max_formant: float = 5500.0
    formant_window: float = 0.025
    min_pitch: float = 50.0
    max_pitch: float = 500.0
    speed_of_sound: float = 34000.0  # cm/s
    strict: bool = False
    require_parselmouth: bool = False
    normalize_spectral: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_feature_config(config: Mapping[str, Any]) -> FeatureConfig:
    """Construct the full feature contract from a repo config mapping."""
    data_cfg = dict(config.get("data", {}))
    feat_cfg = dict(config.get("features", {}))
    mfcc_cfg = dict(feat_cfg.get("mfcc", {}))
    formant_cfg = dict(feat_cfg.get("formants", {}))
    pitch_cfg = dict(feat_cfg.get("pitch", {}))
    vtl_cfg = dict(feat_cfg.get("vtl", {}))
    audit_cfg = dict(feat_cfg.get("audit", {}))
    return FeatureConfig(
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        n_mfcc=int(mfcc_cfg.get("n_mfcc", 40)),
        n_fft=int(mfcc_cfg.get("n_fft", 512)),
        hop_length=int(mfcc_cfg.get("hop_length", 160)),
        win_length=int(mfcc_cfg.get("win_length", 400)),
        include_delta=bool(mfcc_cfg.get("include_delta", True)),
        include_delta_delta=bool(mfcc_cfg.get("include_delta_delta", True)),
        n_formants=int(formant_cfg.get("n_formants", 4)),
        max_formant=float(formant_cfg.get("max_formant", 5500.0)),
        formant_window=float(
            formant_cfg.get("window_length", formant_cfg.get("formant_window", 0.025))
        ),
        min_pitch=float(pitch_cfg.get("min_pitch", 50.0)),
        max_pitch=float(pitch_cfg.get("max_pitch", 500.0)),
        speed_of_sound=float(vtl_cfg.get("speed_of_sound", 34000.0)),
        strict=bool(
            feat_cfg.get("strict", audit_cfg.get("strict_backend_check", False))
        ),
        require_parselmouth=bool(
            feat_cfg.get(
                "require_parselmouth",
                audit_cfg.get("require_parselmouth", False),
            )
        ),
        normalize_spectral=bool(feat_cfg.get("normalize_spectral", True)),
    )


def ensure_feature_backends(config: FeatureConfig) -> None:
    if (config.strict or config.require_parselmouth) and not PARSELMOUTH_AVAILABLE:
        raise RuntimeError(
            "parselmouth is required for the configured VocalMorph feature contract, "
            "but it is not installed in this environment."
        )


def vtl_spacing_bounds(
    speed_of_sound: float = 34000.0,
    min_vtl_cm: float = VTL_MIN_CM,
    max_vtl_cm: float = VTL_MAX_CM,
) -> Tuple[float, float]:
    min_spacing = float(speed_of_sound) / (2.0 * float(max_vtl_cm))
    max_spacing = float(speed_of_sound) / (2.0 * float(min_vtl_cm))
    return min_spacing, max_spacing


def robust_vtl_from_formant_spacing(
    formant_spacing: np.ndarray,
    *,
    speed_of_sound: float = 34000.0,
    min_vtl_cm: float = VTL_MIN_CM,
    max_vtl_cm: float = VTL_MAX_CM,
) -> np.ndarray:
    spacing = np.asarray(formant_spacing, dtype=np.float32)
    vtl = np.zeros_like(spacing, dtype=np.float32)
    finite_positive = np.isfinite(spacing) & (spacing > 0.0)
    if not np.any(finite_positive):
        return vtl

    min_spacing, max_spacing = vtl_spacing_bounds(
        speed_of_sound=speed_of_sound,
        min_vtl_cm=min_vtl_cm,
        max_vtl_cm=max_vtl_cm,
    )
    clipped_spacing = np.clip(spacing[finite_positive], min_spacing, max_spacing)
    vtl[finite_positive] = float(speed_of_sound) / (2.0 * clipped_spacing)
    return vtl.astype(np.float32, copy=False)


def robust_positive_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    valid = arr[np.isfinite(arr) & (arr > 0.0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


# ─────────────────────────────────────────────────────────────
# Audio Loading
# ─────────────────────────────────────────────────────────────

def load_audio(
    path: str,
    target_sr: int = 16000,
    max_duration: float = 10.0,
    min_duration: float = 2.0,
    enhance: bool = False,
    enhancement_config: Optional[MicrophoneEnhancementConfig] = None,
    return_metadata: bool = False,
) -> Optional[np.ndarray | Tuple[np.ndarray, Dict[str, object]]]:
    """
    Load audio file, resample to target_sr, validate duration.
    Returns mono float32 numpy array, or `(audio, metadata)` when `return_metadata=True`.
    """
    metadata: Dict[str, object] = {"enhancement": None}
    try:
        audio, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_duration)
        if enhance:
            audio, report = enhance_microphone_audio(audio, target_sr, config=enhancement_config)
            metadata["enhancement"] = report.to_dict()
        duration = len(audio) / target_sr
        if duration < min_duration:
            warnings.warn(f"Audio too short ({duration:.1f}s < {min_duration}s): {path}")
            return None
        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        audio = audio.astype(np.float32)
        if return_metadata:
            metadata["duration_s"] = float(duration)
            return audio, metadata
        return audio
    except Exception as e:
        warnings.warn(f"Failed to load {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# MFCC Features
# ─────────────────────────────────────────────────────────────

def extract_mfcc(
    audio: np.ndarray,
    config: FeatureConfig,
) -> np.ndarray:
    """
    Extract MFCCs + optional delta and delta-delta.
    Returns: (n_frames, n_mfcc * multiplier)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
    )  # (n_mfcc, T)

    features = [mfcc]

    if config.include_delta:
        delta = librosa.feature.delta(mfcc, order=1)
        features.append(delta)

    if config.include_delta_delta:
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.append(delta2)

    stacked = np.concatenate(features, axis=0)  # (n_mfcc * k, T)
    return stacked.T  # (T, n_mfcc * k)


# ─────────────────────────────────────────────────────────────
# Spectral Features
# ─────────────────────────────────────────────────────────────

def extract_spectral(audio: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """
    Extract frame-level spectral features.
    Returns: (n_frames, 5) — centroid, rolloff, flux, bandwidth, contrast_mean
    """
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length
    )  # (1, T)

    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length
    )  # (1, T)

    flux = librosa.onset.onset_strength(
        y=audio, sr=config.sample_rate, hop_length=config.hop_length
    )  # (T,)

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length
    )  # (1, T)

    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length
    ).mean(axis=0, keepdims=True)  # (1, T)

    features = np.concatenate([
        centroid, rolloff, flux[np.newaxis, :], bandwidth, contrast
    ], axis=0).T  # (T, 5)

    if config.normalize_spectral:
        features = (features - features.mean(0)) / (features.std(0) + 1e-9)
    return features


# ─────────────────────────────────────────────────────────────
# Praat-Based Features (Formants, F0, Voice Quality)
# ─────────────────────────────────────────────────────────────

def extract_praat_features(
    audio: np.ndarray,
    config: FeatureConfig,
) -> Dict[str, np.ndarray]:
    """
    Use parselmouth/Praat to extract:
      - Formants F1-F4 (Hz) per frame
      - F0 (pitch) per frame
      - Jitter (local), Shimmer (local), HNR
      - Formant spacing (Δf) per frame
      - VTL estimate per frame

    Returns dict of arrays, each shape (T, k).
    Strict audited runs fail closed when parselmouth is required but unavailable.
    """
    n_frames = int(np.ceil(len(audio) / config.hop_length))

    if not PARSELMOUTH_AVAILABLE:
        if config.strict or config.require_parselmouth:
            raise RuntimeError(
                "parselmouth is not available, so audited Praat/F0/VTL features cannot be extracted."
            )
        return {
            "formants": np.zeros((n_frames, config.n_formants * 2), dtype=np.float32),
            "f0": np.zeros((n_frames, 1), dtype=np.float32),
            "formant_spacing": np.zeros((n_frames, 1), dtype=np.float32),
            "vtl_estimate": np.zeros((n_frames, 1), dtype=np.float32),
            "jitter": np.zeros((1,), dtype=np.float32),
            "shimmer": np.zeros((1,), dtype=np.float32),
            "hnr": np.zeros((1,), dtype=np.float32),
        }

    snd = parselmouth.Sound(audio, sampling_frequency=config.sample_rate)

    # ── Formants ──
    formant_obj = call(snd, "To Formant (burg)", 0.0, config.n_formants, config.max_formant, config.formant_window, 50.0)
    times = np.arange(n_frames) * config.hop_length / config.sample_rate

    formant_freqs = np.zeros((n_frames, config.n_formants))
    formant_bw    = np.zeros((n_frames, config.n_formants))

    for i, t in enumerate(times):
        for j in range(1, config.n_formants + 1):
            try:
                freq = call(formant_obj, "Get value at time", j, t, "Hertz", "Linear")
                bw   = call(formant_obj, "Get bandwidth at time", j, t, "Hertz", "Linear")
                formant_freqs[i, j-1] = freq if not np.isnan(freq) else 0.0
                formant_bw[i, j-1]    = bw   if not np.isnan(bw) else 0.0
            except Exception:
                pass

    formants = np.concatenate([formant_freqs, formant_bw], axis=1)  # (T, n_formants*2)

    # ── Formant Spacing & VTL Estimate ──
    # Use mean spacing between consecutive formants
    # Δf = mean(F2-F1, F3-F2, ...) per frame
    diffs = np.diff(formant_freqs, axis=1)  # (T, n_formants-1)
    formant_spacing = diffs.mean(axis=1, keepdims=True)  # (T, 1) Hz

    # VTL = c / (2 * Δf)
    vtl_estimate = robust_vtl_from_formant_spacing(
        formant_spacing,
        speed_of_sound=config.speed_of_sound,
    )

    # ── F0/Pitch ──
    pitch_obj = call(snd, "To Pitch", 0.0, config.min_pitch, config.max_pitch)
    f0 = np.zeros((n_frames, 1))
    for i, t in enumerate(times):
        try:
            val = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
            f0[i, 0] = val if not np.isnan(val) else 0.0
        except Exception:
            pass

    # ── Voice Quality (global stats over whole utterance) ──
    try:
        point_process = call(snd, "To PointProcess (periodic, cc)", config.min_pitch, config.max_pitch)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, config.min_pitch, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        jitter = float(jitter) if not np.isnan(jitter) else 0.0
        shimmer = float(shimmer) if not np.isnan(shimmer) else 0.0
        hnr = float(hnr) if not np.isnan(hnr) else 0.0
    except Exception:
        jitter, shimmer, hnr = 0.0, 0.0, 0.0

    return {
        "formants": formants,                      # (T, n_formants*2)
        "f0": f0,                                  # (T, 1)
        "formant_spacing": formant_spacing,        # (T, 1)
        "vtl_estimate": vtl_estimate,              # (T, 1)
        "jitter": np.array([jitter]),              # (1,)
        "shimmer": np.array([shimmer]),            # (1,)
        "hnr": np.array([hnr]),                    # (1,)
    }


# ─────────────────────────────────────────────────────────────
# Full Feature Extraction
# ─────────────────────────────────────────────────────────────

def extract_all_features(
    audio: np.ndarray,
    config: FeatureConfig,
) -> Dict[str, np.ndarray]:
    """
    Full pipeline: extract all features for one audio clip.

    Returns:
        {
          "sequence":         (T, D)  — per-frame features for model input
          "f0_mean":          float   — mean F0 for physics constraint
          "formant_spacing":  float   — mean formant spacing for physics constraint
          "vtl_mean":         float   — mean VTL estimate
          "jitter":           float
          "shimmer":          float
          "hnr":              float
        }
    """
    ensure_feature_backends(config)
    mfcc = extract_mfcc(audio, config)               # (T, n_mfcc*k)
    spectral = extract_spectral(audio, config)        # (T, 5)
    praat = extract_praat_features(audio, config)

    # Align time axes (mfcc/spectral may differ from praat by 1-2 frames)
    T = min(mfcc.shape[0], spectral.shape[0], praat["formants"].shape[0], praat["f0"].shape[0])

    sequence = np.concatenate([
        mfcc[:T],                        # (T, n_mfcc * delta_mult)
        spectral[:T],                    # (T, 5)
        praat["formants"][:T],           # (T, n_formants*2)
        praat["f0"][:T],                 # (T, 1)
        praat["formant_spacing"][:T],    # (T, 1)
        praat["vtl_estimate"][:T],       # (T, 1)
    ], axis=1)  # (T, D_total)

    # Global physics stats
    f0_vals = praat["f0"][:T, 0]
    voiced = f0_vals[f0_vals > 0]
    f0_mean = float(voiced.mean()) if len(voiced) > 0 else 0.0

    fs_vals = praat["formant_spacing"][:T, 0]
    formant_spacing_mean = float(fs_vals[fs_vals > 0].mean()) if (fs_vals > 0).any() else 0.0

    vtl_vals = praat["vtl_estimate"][:T, 0]
    vtl_mean = robust_positive_median(vtl_vals)
    duration_s = float(len(audio) / max(config.sample_rate, 1))
    voiced_ratio = float(len(voiced) / max(T, 1))
    min_spacing, max_spacing = vtl_spacing_bounds(speed_of_sound=config.speed_of_sound)
    invalid_spacing_rate = (
        float(
            np.mean(
                (~np.isfinite(fs_vals))
                | (fs_vals <= 0.0)
                | (fs_vals < min_spacing)
                | (fs_vals > max_spacing)
            )
        )
        if T > 0
        else 1.0
    )
    invalid_vtl_rate = (
        float(np.mean((~np.isfinite(vtl_vals)) | (vtl_vals <= 0.0))) if T > 0 else 1.0
    )

    return {
        "sequence": sequence.astype(np.float32),
        "f0_mean": f0_mean,
        "formant_spacing_mean": formant_spacing_mean,
        "vtl_mean": vtl_mean,
        "jitter": float(praat["jitter"][0]),
        "shimmer": float(praat["shimmer"][0]),
        "hnr": float(praat["hnr"][0]),
        "duration_s": duration_s,
        "n_frames": int(T),
        "voiced_ratio": voiced_ratio,
        "invalid_spacing_rate": invalid_spacing_rate,
        "invalid_vtl_rate": invalid_vtl_rate,
        "praat_backend_available": bool(PARSELMOUTH_AVAILABLE),
    }


# ─────────────────────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────────────────────

def process_audio_file(
    audio_path: str,
    config: FeatureConfig,
    max_duration: float = 10.0,
    min_duration: float = 2.0,
    enhance: bool = False,
    enhancement_config: Optional[MicrophoneEnhancementConfig] = None,
) -> Optional[Dict]:
    """Load and extract features from a single audio file."""
    ensure_feature_backends(config)
    audio = load_audio(
        audio_path,
        config.sample_rate,
        max_duration,
        min_duration,
        enhance=enhance,
        enhancement_config=enhancement_config,
    )
    if audio is None:
        return None
    return extract_all_features(audio, config)


def process_dataset(
    audio_dir: str,
    metadata_path: str,
    output_path: str,
    config: Optional[FeatureConfig] = None,
) -> None:
    """
    Process all NISP audio files and save features to .npz files.

    Expected metadata.csv columns: speaker_id, height_cm, weight_kg, age, gender
    """
    import pandas as pd
    from tqdm import tqdm

    if config is None:
        config = FeatureConfig()

    df = pd.read_csv(metadata_path)
    os.makedirs(output_path, exist_ok=True)

    processed, skipped = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        speaker_id = row["speaker_id"]
        audio_path = os.path.join(audio_dir, f"{speaker_id}.wav")

        if not os.path.exists(audio_path):
            warnings.warn(f"Audio not found: {audio_path}")
            skipped += 1
            continue

        features = process_audio_file(audio_path, config)
        if features is None:
            skipped += 1
            continue

        out_file = os.path.join(output_path, f"{speaker_id}.npz")
        np.savez(
            out_file,
            sequence=features["sequence"],
            f0_mean=features["f0_mean"],
            formant_spacing_mean=features["formant_spacing_mean"],
            vtl_mean=features["vtl_mean"],
            jitter=features["jitter"],
            shimmer=features["shimmer"],
            hnr=features["hnr"],
            height_cm=row["height_cm"],
            weight_kg=row["weight_kg"],
            age=row["age"],
            gender=1 if str(row["gender"]).lower() == "male" else 0,
        )
        processed += 1

    print(f"\n✅ Feature extraction complete: {processed} processed, {skipped} skipped")
    print(f"📁 Saved to: {output_path}")


if __name__ == "__main__":
    # Quick test
    config = FeatureConfig()
    dummy_audio = np.random.randn(16000 * 3).astype(np.float32)
    feats = extract_all_features(dummy_audio, config)
    print("Feature extraction test:")
    print(f"  sequence shape: {feats['sequence'].shape}")
    print(f"  f0_mean: {feats['f0_mean']:.1f} Hz")
    print(f"  formant_spacing_mean: {feats['formant_spacing_mean']:.1f} Hz")
    print(f"  vtl_mean: {feats['vtl_mean']:.2f} cm")
