"""
VocalMorph-S: Speaker-Level Inference Module.

Converts clip-level predictions into robust speaker-level estimates using:
- Strict quality gating (SNR, speech ratio, physics reliability)
- MAD-based outlier rejection (Consistency Filter)
- Inverse-variance weighted aggregation
- Confidence-based abstention logic
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Sequence

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.predict import VocalMorphInference, VocalMorphPrediction, InferenceQualityGateConfig
from src.preprocessing.feature_extractor import extract_all_features, load_audio
from src.utils.audit_utils import decode_np_value

@dataclass
class SpeakerLevelPrediction:
    height_cm: float
    uncertainty_cm: float
    n_accepted_clips: int
    n_total_clips: int
    is_confident: bool
    status: str  # "SUCCESS", "ABSTAIN", "FAILED"
    message: str
    clip_details: List[Dict[str, Any]]

class SpeakerEvidenceAccumulator:
    def __init__(
        self,
        max_variance_threshold: float = 25.0,  # Max std of 5.0cm per clip
        max_outlier_mad_z: float = 1.5,        # Threshold for MAD outlier rejection
        target_uncertainty_cm: float = 2.5,    # Abstention threshold for high confidence
        min_required_clips: int = 2
    ):
        self.max_var = max_variance_threshold
        self.max_mad_z = max_outlier_mad_z
        self.target_sigma = target_uncertainty_cm
        self.min_clips = min_required_clips

    def aggregate(self, clip_preds: List[Dict[str, Any]]) -> SpeakerLevelPrediction:
        n_total = len(clip_preds)
        
        # 1. Reject clips with high internal variance
        valid_preds = [
            p for p in clip_preds 
            if (p['std']**2) <= self.max_var
        ]
        
        if len(valid_preds) < self.min_clips:
            return self._result(
                "ABSTAIN", f"Insufficient clips passed variance/quality check ({len(valid_preds)}/{n_total})",
                n_accepted=len(valid_preds), n_total=n_total
            )

        means = np.array([p['mean'] for p in valid_preds])
        stds = np.array([p['std'] for p in valid_preds])
        qualities = np.array([p.get('quality', 1.0) for p in valid_preds])

        # 2. Outlier Rejection using MAD (Median Absolute Deviation)
        median_val = np.median(means)
        mad = np.median(np.abs(means - median_val))
        mad = max(mad, 1e-4)
        
        # Modified Z-score: 0.6745 * (x - median) / MAD
        z_scores = 0.6745 * (np.abs(means - median_val) / mad)
        consistency_mask = z_scores <= self.max_mad_z
        
        if np.sum(consistency_mask) < self.min_clips:
            # Fallback to robust aggregation if MAD is too aggressive
            final_mean = median_val
            final_std = np.mean(stds) # Conservative fallback
            return self._result(
                "ABSTAIN", "Inconsistent evidence cluster. Median fallback used.",
                final_mean, final_std, int(np.sum(consistency_mask)), n_total
            )

        # 3. Inverse Variance Weighted Mean
        # Weight = Quality / Variance
        final_means = means[consistency_mask]
        final_stds = stds[consistency_mask]
        final_qualities = qualities[consistency_mask]
        
        variances = final_stds ** 2
        weights = final_qualities / (variances + 1e-6)
        weights /= np.sum(weights)

        speaker_mean = np.sum(final_means * weights)
        
        # Combined uncertainty: 1 / sqrt(sum(1/var))
        # We assume 50% correlation between clips from same speaker (conservative)
        theoretical_var = 1.0 / np.sum(1.0 / variances)
        correlation_factor = 1.5 # Penalty for intra-speaker correlation
        speaker_std = np.sqrt(theoretical_var * correlation_factor)

        # 4. Confidence Flag / Abstention
        is_confident = speaker_std <= self.target_sigma
        status = "SUCCESS" if is_confident else "ABSTAIN"
        msg = "High confidence prediction" if is_confident else f"Uncertainty (±{speaker_std:.2f}cm) above target"

        return self._result(
            status, msg, speaker_mean, speaker_std, 
            int(np.sum(consistency_mask)), n_total, is_confident,
            clip_details=valid_preds
        )

    def _result(self, status, msg, mean=0.0, std=0.0, n_accepted=0, n_total=0, is_confident=False, clip_details=None):
        return SpeakerLevelPrediction(
            height_cm=round(float(mean), 2),
            uncertainty_cm=round(float(std), 2),
            n_accepted_clips=n_accepted,
            n_total_clips=n_total,
            is_confident=is_confident,
            status=status,
            message=msg,
            clip_details=clip_details or []
        )

class VocalMorphSpeakerInference(VocalMorphInference):
    """Extends the base inference engine with speaker-level logic."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulator = SpeakerEvidenceAccumulator(
            target_uncertainty_cm=2.5,
            min_required_clips=2
        )

    def predict_speaker(self, audio_paths: List[str]) -> SpeakerLevelPrediction:
        prediction = self.predict_from_files(audio_paths, speaker_id="speaker_eval")
        if prediction is None:
            return self.accumulator._result(
                "FAILED",
                "Base strict inference rejected all clips.",
                n_accepted=0,
                n_total=len(audio_paths),
            )
        return SpeakerLevelPrediction(
            height_cm=float(prediction.height_cm),
            uncertainty_cm=float(prediction.height_std),
            n_accepted_clips=int(prediction.n_used_clips),
            n_total_clips=int(prediction.n_input_clips),
            is_confident=bool(prediction.accepted),
            status="SUCCESS" if prediction.accepted else "ABSTAIN",
            message="Shared Omega/legacy inference path",
            clip_details=[],
        )

def evaluate_speaker_level(
    engine: VocalMorphSpeakerInference, 
    test_dir: str, 
    clips_per_speaker: List[int] = [1, 3, 5, 10]
) -> Dict[int, float]:
    """
    Evaluates MAE at different levels of evidence accumulation.
    Expects test_dir to have speaker-prefixed npz or wav files.
    """
    import glob
    from collections import defaultdict

    def _speaker_id_from_npz(path: str) -> str:
        with np.load(path, allow_pickle=True) as data:
            if "speaker_id" in data:
                speaker_id = decode_np_value(data["speaker_id"]).strip()
                if speaker_id:
                    return speaker_id
        base = os.path.splitext(os.path.basename(path))[0]
        if "_aug" in base:
            base = base.split("_aug", 1)[0]
        parts = base.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else base
    
    # Group files by speaker
    files = glob.glob(os.path.join(test_dir, "*.npz")) # Using pre-extracted features for speed
    speaker_groups = defaultdict(list)
    for f in files:
        sid = _speaker_id_from_npz(f)
        speaker_groups[sid].append(f)
        
    results = {}
    
    for n in clips_per_speaker:
        errors = []
        print(f"Evaluating N={n} clips per speaker...")
        
        for sid, group in speaker_groups.items():
            if len(group) < n and n > 1: continue
            
            # Select first N clips
            selected = group[:n]
            
            # Mock the 'predict_speaker' but using pre-extracted features for speed in eval
            clip_preds = []
            true_height = None
            
            for f in selected:
                data = np.load(f, allow_pickle=True)
                true_height = float(data['height_cm'])
                
                # Run model on sequence
                seq = torch.from_numpy(data['sequence']).unsqueeze(0).to(engine.device)
                with torch.no_grad():
                    uc = engine.model.predict_with_uncertainty(seq, n_samples=engine.n_mc_samples)
                    h_mu = float(uc["height"]["mean"].item())
                    h_std = float(uc["height"]["std"].item())
                    h_std_scale = float(engine.target_stats.get("height", {}).get("std", 1.0)) if engine.target_stats else 1.0
                    
                    clip_preds.append({
                        "mean": engine._denorm(h_mu, "height"),
                        "std": h_std * h_std_scale,
                        "quality": 1.0 # Default for pre-extracted
                    })
            
            pred = engine.accumulator.aggregate(clip_preds)
            if pred.status == "SUCCESS" or (n == 1 and pred.height_cm > 0):
                errors.append(abs(pred.height_cm - true_height))
        
        results[n] = float(np.mean(errors)) if errors else 0.0
        print(f"  N={n} -> MAE: {results[n]:.2f} cm (samples: {len(errors)})")
        
    return results

if __name__ == "__main__":
    # Example usage / sanity check
    print("VocalMorph-S Module Initialized.")
