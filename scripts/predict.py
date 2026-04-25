#!/usr/bin/env python
"""VocalMorph inference pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.pibnn import build_model
from src.preprocessing.audio_enhancement import MicrophoneEnhancementConfig, enhance_microphone_audio
from src.preprocessing.feature_extractor import build_feature_config, extract_all_features, load_audio
from src.utils.audit_utils import safe_float, validate_feature_contract

GENDER_LABELS = {0: "Female", 1: "Male"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


@dataclass
class InferenceQualityGateConfig:
    enabled: bool = True
    strict: bool = True
    min_capture_quality_score: float = 0.55
    min_speech_ratio: float = 0.55
    min_snr_db: float = 10.0
    max_distance_cm: float = 30.0
    max_clipped_ratio: float = 0.03
    require_quality_ok: bool = False
    min_accepted_clips: int = 2
    max_height_std_cm: float = 8.0
    min_confidence_score: float = 0.35
    max_ood_zscore: float = 4.0


def quality_gate_decision(
    enhancement_meta: Optional[dict],
    gate_cfg: InferenceQualityGateConfig,
) -> Tuple[bool, List[str], float]:
    if not gate_cfg.enabled:
        return True, [], 1.0

    if not enhancement_meta:
        reasons = ["missing_quality_metadata"]
        accepted = not gate_cfg.strict
        return accepted, reasons, 0.20

    reasons: List[str] = []

    capture_score = enhancement_meta.get("capture_quality_score")
    speech_ratio = enhancement_meta.get("speech_ratio")
    snr_db = enhancement_meta.get("snr_db_estimate")
    distance_cm = enhancement_meta.get("distance_cm_estimate")
    clipped_ratio = enhancement_meta.get("clipped_ratio")
    quality_ok = enhancement_meta.get("quality_ok")

    if capture_score is not None and float(capture_score) < float(gate_cfg.min_capture_quality_score):
        reasons.append("low_capture_quality")
    if speech_ratio is not None and float(speech_ratio) < float(gate_cfg.min_speech_ratio):
        reasons.append("low_speech_ratio")
    if snr_db is not None and float(snr_db) < float(gate_cfg.min_snr_db):
        reasons.append("low_snr")
    if distance_cm is not None and float(distance_cm) > float(gate_cfg.max_distance_cm):
        reasons.append("far_microphone")
    if clipped_ratio is not None and float(clipped_ratio) > float(gate_cfg.max_clipped_ratio):
        reasons.append("clipped_input")
    if gate_cfg.require_quality_ok and quality_ok is not None and not bool(quality_ok):
        reasons.append("quality_check_failed")

    accepted = (len(reasons) == 0) or (not gate_cfg.strict)
    base_weight = float(np.clip(float(capture_score) if capture_score is not None else 0.50, 0.05, 1.0))
    if reasons and not gate_cfg.strict:
        base_weight *= 0.35
    return accepted, reasons, float(np.clip(base_weight, 0.05, 1.0))


@dataclass
class VocalMorphPrediction:
    height_cm: float
    height_std: float
    weight_kg: float
    weight_std: float
    age: float
    age_std: float
    gender: str
    gender_confidence: float
    speech_ratio: Optional[float] = None
    snr_db_estimate: Optional[float] = None
    capture_quality_score: Optional[float] = None
    distance_cm_estimate: Optional[float] = None
    distance_confidence: Optional[float] = None
    distance_band: Optional[str] = None
    quality_ok: Optional[bool] = None
    quality_warnings: Optional[List[str]] = None
    capture_advice: Optional[str] = None
    n_input_clips: int = 1
    n_used_clips: int = 1
    rejected_clips: int = 0
    rejected_reasons: Optional[List[str]] = None
    accepted: bool = True
    confidence_score: Optional[float] = None
    ood_score: Optional[float] = None
    canonical_feature_contract: Optional[str] = None

    def __str__(self):
        quality_line = ""
        clips_line = ""
        status_line = f"  Status : {'ACCEPTED' if self.accepted else 'REJECTED'}\n"
        if self.n_input_clips > 1:
            clips_line = (
                f"  Clips  : used={self.n_used_clips}/{self.n_input_clips}"
                f", rejected={self.rejected_clips}\n"
            )
            if self.rejected_reasons:
                clips_line += f"  Reject : {', '.join(self.rejected_reasons)}\n"
        if (
            self.speech_ratio is not None
            or self.snr_db_estimate is not None
            or self.capture_quality_score is not None
            or self.distance_cm_estimate is not None
        ):
            speech_ratio = f"{self.speech_ratio*100:.1f}%" if self.speech_ratio is not None else "n/a"
            snr = f"{self.snr_db_estimate:.1f} dB" if self.snr_db_estimate is not None else "n/a"
            capture = f"{self.capture_quality_score*100:.0f}%" if self.capture_quality_score is not None else "n/a"
            if self.distance_cm_estimate is not None:
                dist = f"~{self.distance_cm_estimate:.0f} cm"
                if self.distance_band:
                    dist = f"{dist} ({self.distance_band})"
            else:
                dist = "n/a"
            quality_line = (
                f"  Speech : ratio={speech_ratio}, est_snr={snr}\n"
                f"  Capture: score={capture}, mic={dist}\n"
            )
            if self.confidence_score is not None or self.ood_score is not None:
                conf = f"{self.confidence_score:.2f}" if self.confidence_score is not None else "n/a"
                ood = f"{self.ood_score:.2f}" if self.ood_score is not None else "n/a"
                quality_line += f"  Trust  : confidence={conf}, ood={ood}\n"
            if self.quality_ok is not None and not self.quality_ok:
                warnings = ", ".join(self.quality_warnings or []) if self.quality_warnings else "low capture quality"
                quality_line += f"  Warn   : {warnings}\n"
            if self.capture_advice:
                quality_line += f"  Advice : {self.capture_advice}\n"
        return (
            "\n" + "-" * 40 + "\n"
            + "  VocalMorph Prediction\n"
            + "-" * 40 + "\n"
            + status_line
            + f"  Height : {self.height_cm:.1f} +/- {self.height_std:.1f} cm\n"
            + f"  Weight : {self.weight_kg:.1f} +/- {self.weight_std:.1f} kg\n"
            + f"  Age    : {self.age:.1f} +/- {self.age_std:.1f} years\n"
            + f"  Gender : {self.gender} ({self.gender_confidence*100:.1f}% confidence)\n"
            + clips_line
            + quality_line
            + "-" * 40
        )


class VocalMorphInference:
    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        target_stats: Optional[dict] = None,
        device: str = "auto",
        n_mc_samples: int = 100,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.n_mc_samples = n_mc_samples
        self.reference_diagnostics = None
        self.feature_contract = None

        data_cfg = config.get("data", {})
        infer_cfg = config.get("inference", {})
        self.feature_root = (
            data_cfg.get("features_dir")
            if os.path.isabs(str(data_cfg.get("features_dir", "")))
            else os.path.join(ROOT, str(data_cfg.get("features_dir", "data/features_audited")))
        )
        split_manifest_cfg = data_cfg.get("split_manifests", {})
        self.split_manifests = {
            "train": (
                split_manifest_cfg.get("train")
                if os.path.isabs(str(split_manifest_cfg.get("train", "")))
                else os.path.join(ROOT, str(split_manifest_cfg.get("train", "data/splits/train_clean.csv")))
            ),
            "val": (
                split_manifest_cfg.get("val")
                if os.path.isabs(str(split_manifest_cfg.get("val", "")))
                else os.path.join(ROOT, str(split_manifest_cfg.get("val", "data/splits/val_clean.csv")))
            ),
            "test": (
                split_manifest_cfg.get("test")
                if os.path.isabs(str(split_manifest_cfg.get("test", "")))
                else os.path.join(ROOT, str(split_manifest_cfg.get("test", "data/splits/test_clean.csv")))
            ),
        }
        self.feature_config = build_feature_config(config)
        self.feature_contract = validate_feature_contract(
            feature_root=self.feature_root,
            expected_feature_config=self.feature_config.to_dict(),
            expected_split_files=self.split_manifests,
            require_target_stats=False,
        )
        if target_stats is None:
            stats_path = os.path.join(self.feature_root, "target_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r", encoding="utf-8") as handle:
                    target_stats = json.load(handle)
        self.target_stats = target_stats
        diagnostics_path = os.path.join(self.feature_root, "feature_diagnostics.json")
        if os.path.exists(diagnostics_path):
            with open(diagnostics_path, "r", encoding="utf-8") as handle:
                self.reference_diagnostics = json.load(handle)
        self.enhancement_config = MicrophoneEnhancementConfig(**infer_cfg.get("audio_enhancement", {}))
        self.deterministic_ensemble = bool(infer_cfg.get("deterministic_ensemble", True))
        self.crop_size = infer_cfg.get("crop_size")
        self.n_crops = max(1, int(infer_cfg.get("n_crops", 3)))
        self.quality_gate = InferenceQualityGateConfig(**infer_cfg.get("quality_gate", {}))

        self.model = build_model(config)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[VocalMorph] Model loaded from {checkpoint_path}")

    @staticmethod
    def _capture_advice(meta: Optional[dict]) -> Optional[str]:
        if not meta:
            return None
        advice: List[str] = []
        speech_ratio = meta.get("speech_ratio")
        snr_db = meta.get("snr_db_estimate")
        distance_cm = meta.get("distance_cm_estimate")
        quality_ok = meta.get("quality_ok")

        if speech_ratio is not None and float(speech_ratio) < 0.55:
            advice.append("speak continuously for a few seconds")
        if snr_db is not None and float(snr_db) < 10.0:
            advice.append("move to a quieter room")
        if distance_cm is not None and float(distance_cm) > 25.0:
            advice.append("hold the microphone within about 10-20 cm")
        if quality_ok is not None and not bool(quality_ok) and not advice:
            advice.append("re-record with a cleaner capture")
        if not advice:
            return None
        return "; ".join(advice)

    def _evaluate_clip_quality(self, enhancement_meta: Optional[dict]) -> Tuple[bool, List[str], float]:
        return quality_gate_decision(enhancement_meta, self.quality_gate)

    def _denorm(self, val: float, key: str) -> float:
        if self.target_stats is None:
            return val
        s = self.target_stats.get(key, {})
        return val * float(s.get("std", 1.0)) + float(s.get("mean", 0.0))

    def _compute_ood_score(
        self,
        feature_meta: Mapping[str, float],
        enhancement_meta: Optional[Mapping[str, float]],
    ) -> Optional[float]:
        if not self.reference_diagnostics:
            return None
        train_summary = (
            self.reference_diagnostics.get("splits", {})
            .get("train", {})
            .get("scalar_summary", {})
        )
        combined = {
            "duration_s": safe_float(feature_meta.get("duration_s")),
            "f0_mean": safe_float(feature_meta.get("f0_mean")),
            "formant_spacing_mean": safe_float(feature_meta.get("formant_spacing_mean")),
            "vtl_mean": safe_float(feature_meta.get("vtl_mean")),
            "speech_ratio": safe_float((enhancement_meta or {}).get("speech_ratio")),
            "snr_db_estimate": safe_float((enhancement_meta or {}).get("snr_db_estimate")),
            "capture_quality_score": safe_float((enhancement_meta or {}).get("capture_quality_score")),
        }
        z_scores: List[float] = []
        for key, value in combined.items():
            stats = train_summary.get(key, {})
            mean = safe_float(stats.get("mean"))
            std = safe_float(stats.get("std"))
            if not np.isfinite(value) or not np.isfinite(mean):
                continue
            scale = std if np.isfinite(std) and abs(std) > 1e-6 else 1.0
            z_scores.append(abs(value - mean) / scale)
        if not z_scores:
            return None
        return float(max(z_scores))

    @staticmethod
    def _confidence_score(
        *,
        height_std_cm: float,
        capture_quality_score: Optional[float],
        ood_score: Optional[float],
    ) -> float:
        uncertainty_term = 1.0 / (1.0 + max(0.0, float(height_std_cm)) / 8.0)
        quality_term = float(np.clip(capture_quality_score if capture_quality_score is not None else 0.5, 0.0, 1.0))
        ood_penalty = 1.0 / (1.0 + max(0.0, float(ood_score)) / 4.0) if ood_score is not None else 1.0
        return float(np.clip(0.5 * uncertainty_term + 0.5 * quality_term, 0.0, 1.0) * ood_penalty)

    def predict_from_file(self, audio_path: str) -> Optional[VocalMorphPrediction]:
        return self.predict_from_files([audio_path], speaker_id="user_single")

    def predict_from_files(self, audio_paths: Sequence[str], speaker_id: str = "user") -> Optional[VocalMorphPrediction]:
        if not audio_paths:
            print("[VocalMorph] No audio files provided.")
            return None

        accepted_audio: List[np.ndarray] = []
        accepted_meta: List[dict] = []
        clip_weights: List[float] = []
        rejected_reasons: List[str] = []

        for audio_path in audio_paths:
            loaded = load_audio(
                audio_path,
                self.feature_config.sample_rate,
                enhance=True,
                enhancement_config=self.enhancement_config,
                return_metadata=True,
            )
            if loaded is None:
                print(f"[VocalMorph] Failed to load: {audio_path}")
                rejected_reasons.append("load_failed")
                continue
            audio, metadata = loaded
            enhancement_meta = metadata.get("enhancement") if metadata else None
            accepted, reasons, weight = self._evaluate_clip_quality(enhancement_meta)
            if not accepted:
                tag = ",".join(reasons) if reasons else "quality_rejected"
                print(f"[VocalMorph] Rejected clip {audio_path}: {tag}")
                rejected_reasons.extend(reasons or ["quality_rejected"])
                continue
            accepted_audio.append(audio)
            accepted_meta.append(enhancement_meta or {})
            clip_weights.append(weight)

        required_clips = 1 if len(audio_paths) == 1 else max(1, int(self.quality_gate.min_accepted_clips))
        if len(accepted_audio) < required_clips:
            print(
                f"[VocalMorph] Not enough high-quality clips after gating: "
                f"{len(accepted_audio)}/{len(audio_paths)} accepted (need {required_clips})."
            )
            if self.quality_gate.strict:
                print("[VocalMorph] Re-record in a quiet room, closer to microphone, with continuous speech.")
                return None

        if not accepted_audio:
            print("[VocalMorph] No usable clips after quality gate.")
            return None

        return self._predict_batch(
            audios=accepted_audio,
            enhancement_meta_list=accepted_meta,
            quality_weights=clip_weights,
            speaker_id=speaker_id,
            n_input_clips=len(audio_paths),
            rejected_reasons=rejected_reasons,
        )

    def predict_from_mic(self, duration: float = 5.0) -> Optional[VocalMorphPrediction]:
        try:
            import sounddevice as sd
        except ImportError:
            print("[VocalMorph] sounddevice not installed. Run: pip install sounddevice")
            return None

        print(f"[VocalMorph] Recording {duration}s... Speak now")
        audio = sd.rec(
            int(duration * self.feature_config.sample_rate),
            samplerate=self.feature_config.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        enhanced, report = enhance_microphone_audio(audio.squeeze(), self.feature_config.sample_rate, self.enhancement_config)
        enhancement_meta = report.to_dict()
        accepted, reasons, _ = self._evaluate_clip_quality(enhancement_meta)
        if not accepted:
            print(f"[VocalMorph] Capture rejected: {', '.join(reasons) if reasons else 'quality gate'}")
            print("[VocalMorph] Try closer mic distance and a quieter room.")
            return None
        return self._predict_batch(
            audios=[enhanced],
            enhancement_meta_list=[enhancement_meta],
            quality_weights=[1.0],
            speaker_id="user_mic",
            n_input_clips=1,
            rejected_reasons=[],
        )

    @torch.no_grad()
    def _predict_batch(
        self,
        audios: Sequence[np.ndarray],
        enhancement_meta_list: Sequence[dict],
        quality_weights: Sequence[float],
        speaker_id: str,
        n_input_clips: int,
        rejected_reasons: Sequence[str],
    ) -> VocalMorphPrediction:
        feature_packets = [extract_all_features(audio, self.feature_config) for audio in audios]
        ood_scores = [
            self._compute_ood_score(packet, meta)
            for packet, meta in zip(feature_packets, enhancement_meta_list)
        ]
        sequences = [torch.from_numpy(packet["sequence"]) for packet in feature_packets]
        lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long, device=self.device)
        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0).to(self.device)
        max_len = int(padded.size(1))
        padding_mask = torch.arange(max_len, device=self.device).unsqueeze(0) >= lengths.unsqueeze(1)
        quality_tensor = torch.tensor(quality_weights, dtype=torch.float32, device=self.device)
        clip_metadata = {
            "duration_s": torch.tensor(
                [float(packet.get("duration_s", float(length) * self.feature_config.hop_length / self.feature_config.sample_rate)) for packet, length in zip(feature_packets, lengths.tolist())],
                dtype=torch.float32,
                device=self.device,
            ),
            "voiced_ratio": torch.tensor(
                [float(packet.get("voiced_ratio", 0.75)) for packet in feature_packets],
                dtype=torch.float32,
                device=self.device,
            ),
            "speech_ratio": torch.tensor(
                [float((meta or {}).get("speech_ratio", 0.70)) for meta in enhancement_meta_list],
                dtype=torch.float32,
                device=self.device,
            ),
            "snr_db_estimate": torch.tensor(
                [float((meta or {}).get("snr_db_estimate", 15.0)) for meta in enhancement_meta_list],
                dtype=torch.float32,
                device=self.device,
            ),
            "capture_quality_score": torch.tensor(
                [float((meta or {}).get("capture_quality_score", 0.50)) for meta in enhancement_meta_list],
                dtype=torch.float32,
                device=self.device,
            ),
            "clipped_ratio": torch.tensor(
                [float((meta or {}).get("clipped_ratio", 0.0)) for meta in enhancement_meta_list],
                dtype=torch.float32,
                device=self.device,
            ),
            "distance_cm_estimate": torch.tensor(
                [float((meta or {}).get("distance_cm_estimate", 18.0)) for meta in enhancement_meta_list],
                dtype=torch.float32,
                device=self.device,
            ),
            "distance_confidence": torch.tensor(
                [float((meta or {}).get("distance_confidence", 0.0)) for meta in enhancement_meta_list],
                dtype=torch.float32,
                device=self.device,
            ),
            "quality_ok": torch.tensor(
                [1.0 if (meta or {}).get("quality_ok", True) else 0.0 for meta in enhancement_meta_list],
                dtype=torch.float32,
                device=self.device,
            ),
            "valid_frames": lengths.to(dtype=torch.float32),
            "ood_zscore": torch.tensor(
                [float(score) if score is not None and np.isfinite(score) else float("nan") for score in ood_scores],
                dtype=torch.float32,
                device=self.device,
            ),
            "feature_drift_zscore": torch.tensor(
                [float(score) if score is not None and np.isfinite(score) else float("nan") for score in ood_scores],
                dtype=torch.float32,
                device=self.device,
            ),
        }

        if hasattr(self.model, "predict_with_uncertainty"):
            speaker_ids = [str(speaker_id)] * len(audios)
            uc = self.model.predict_with_uncertainty(
                padded,
                padding_mask=padding_mask,
                speaker_ids=speaker_ids,
                quality=quality_tensor,
                clip_metadata=clip_metadata,
                n_samples=self.n_mc_samples,
                deterministic=self.deterministic_ensemble,
                crop_size=self.crop_size,
                n_crops=self.n_crops,
                aggregation=getattr(getattr(self.model, "aggregation_config", None), "method", None),
            )
            speaker_result = uc.get("speaker", {}) or {}
            speaker_map = speaker_result.get("speaker", speaker_result) if isinstance(speaker_result, dict) else {}
            speaker_entry: Optional[Dict[str, torch.Tensor]] = speaker_map.get(str(speaker_id)) if isinstance(speaker_map, dict) else None
            if speaker_entry is not None:
                height_mean = float(speaker_entry["height"].detach().item())
                weight_mean = float(speaker_entry["weight"].detach().item())
                age_mean = float(speaker_entry["age"].detach().item())
                height_std = float(speaker_entry["height_std"].detach().item())
                weight_std = float(speaker_entry["weight_std"].detach().item())
                age_std = float(speaker_entry["age_std"].detach().item())
                gender_probs = speaker_entry["gender_probs"].detach()
                gender_idx = int(speaker_entry.get("gender_pred", int(torch.argmax(gender_probs).item())))
            else:
                height_mean = float(uc["height"]["mean"].mean().detach().item())
                weight_mean = float(uc["weight"]["mean"].mean().detach().item())
                age_mean = float(uc["age"]["mean"].mean().detach().item())
                height_std = float(uc["height"]["std"].mean().detach().item())
                weight_std = float(uc["weight"]["std"].mean().detach().item())
                age_std = float(uc["age"]["std"].mean().detach().item())
                gender_probs = uc["gender"]["probs"].mean(dim=0).detach()
                gender_idx = int(torch.argmax(gender_probs).item())
        else:
            try:
                out = self.model(padded, padding_mask=padding_mask)
            except TypeError:
                out = self.model(padded)
            norm_weights = quality_tensor / quality_tensor.sum().clamp(min=1e-6)
            height_mean = float((out["height"] * norm_weights).sum().detach().item())
            weight_mean = float((out["weight"] * norm_weights).sum().detach().item())
            age_mean = float((out["age"] * norm_weights).sum().detach().item())
            height_std = 0.0
            weight_std = 0.0
            age_std = 0.0
            gender_probs = (torch.softmax(out["gender_logits"], dim=-1) * norm_weights.unsqueeze(-1)).sum(dim=0).detach()
            gender_idx = int(torch.argmax(gender_probs).item())

        h_std_scale = float(self.target_stats.get("height", {}).get("std", 1.0)) if self.target_stats else 1.0
        w_std_scale = float(self.target_stats.get("weight", {}).get("std", 1.0)) if self.target_stats else 1.0
        a_std_scale = float(self.target_stats.get("age", {}).get("std", 1.0)) if self.target_stats else 1.0

        height = self._denorm(height_mean, "height")
        weight = self._denorm(weight_mean, "weight")
        age = self._denorm(age_mean, "age")

        weights = np.asarray(quality_weights, dtype=np.float64)
        weights = np.clip(weights, 1e-6, None)
        weights = weights / weights.sum()
        speech_ratio = None
        snr_db_estimate = None
        capture_quality_score = None
        distance_cm_estimate = None
        distance_confidence = None
        clipped_ratio = None
        quality_ok = True
        quality_warnings: List[str] = []

        if enhancement_meta_list:
            def _weighted_metric(key: str) -> Optional[float]:
                vals = [meta.get(key) for meta in enhancement_meta_list]
                valid = [(idx, float(v)) for idx, v in enumerate(vals) if v is not None]
                if not valid:
                    return None
                idxs = np.asarray([idx for idx, _ in valid], dtype=np.int64)
                vals_arr = np.asarray([v for _, v in valid], dtype=np.float64)
                w = weights[idxs]
                w = w / np.clip(w.sum(), 1e-12, None)
                return float((vals_arr * w).sum())

            speech_ratio = _weighted_metric("speech_ratio")
            snr_db_estimate = _weighted_metric("snr_db_estimate")
            capture_quality_score = _weighted_metric("capture_quality_score")
            distance_cm_estimate = _weighted_metric("distance_cm_estimate")
            distance_confidence = _weighted_metric("distance_confidence")
            clipped_ratio = _weighted_metric("clipped_ratio")

            for meta in enhancement_meta_list:
                if meta.get("quality_ok") is False:
                    quality_ok = False
                quality_warnings.extend([str(x) for x in meta.get("quality_warnings", [])])
        quality_warnings.extend([str(r) for r in rejected_reasons if r])
        quality_warnings = sorted(set(quality_warnings))
        distance_band = None
        if distance_cm_estimate is not None:
            if distance_cm_estimate <= 20.0:
                distance_band = "near"
            elif distance_cm_estimate <= 40.0:
                distance_band = "mid"
            else:
                distance_band = "far"

        aggregate_meta = {
            "speech_ratio": speech_ratio,
            "snr_db_estimate": snr_db_estimate,
            "capture_quality_score": capture_quality_score,
            "distance_cm_estimate": distance_cm_estimate,
            "distance_confidence": distance_confidence,
            "distance_band": distance_band,
            "quality_ok": quality_ok,
            "quality_warnings": quality_warnings,
            "clipped_ratio": clipped_ratio,
        }
        finite_ood_scores = [score for score in ood_scores if score is not None and np.isfinite(score)]
        ood_score = float(max(finite_ood_scores)) if finite_ood_scores else None
        height_std_cm = round(height_std * h_std_scale, 1)
        confidence_score = self._confidence_score(
            height_std_cm=height_std_cm,
            capture_quality_score=capture_quality_score,
            ood_score=ood_score,
        )
        decision_reasons = list(quality_warnings)
        if height_std_cm > float(self.quality_gate.max_height_std_cm):
            decision_reasons.append("high_predictive_uncertainty")
        if ood_score is not None and ood_score > float(self.quality_gate.max_ood_zscore):
            decision_reasons.append("feature_ood")
        if confidence_score < float(self.quality_gate.min_confidence_score):
            decision_reasons.append("low_confidence")
        decision_reasons = sorted(set(reason for reason in decision_reasons if reason))
        accepted = len(decision_reasons) == 0

        return VocalMorphPrediction(
            height_cm=round(height, 1),
            height_std=height_std_cm,
            weight_kg=round(weight, 1),
            weight_std=round(weight_std * w_std_scale, 1),
            age=round(age, 1),
            age_std=round(age_std * a_std_scale, 1),
            gender=GENDER_LABELS.get(gender_idx, str(gender_idx)),
            gender_confidence=round(float(gender_probs[gender_idx]), 4),
            speech_ratio=speech_ratio,
            snr_db_estimate=snr_db_estimate,
            capture_quality_score=capture_quality_score,
            distance_cm_estimate=distance_cm_estimate,
            distance_confidence=distance_confidence,
            distance_band=distance_band,
            quality_ok=quality_ok if enhancement_meta_list else None,
            quality_warnings=decision_reasons if decision_reasons else None,
            capture_advice=self._capture_advice(aggregate_meta),
            n_input_clips=int(n_input_clips),
            n_used_clips=int(len(audios)),
            rejected_clips=max(0, int(n_input_clips) - int(len(audios))),
            rejected_reasons=decision_reasons if decision_reasons else None,
            accepted=accepted,
            confidence_score=round(confidence_score, 4),
            ood_score=round(ood_score, 4) if ood_score is not None else None,
            canonical_feature_contract=self.feature_contract.get("feature_config_fingerprint")
            if self.feature_contract
            else None,
        )


def main():
    parser = argparse.ArgumentParser(description="VocalMorph Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/pibnn_base.yaml")
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--audio_list", type=str, default=None, help="Comma-separated list of audio paths for one speaker")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing multiple clips for one speaker")
    parser.add_argument("--speaker_id", type=str, default="user_001")
    parser.add_argument("--allow_low_quality", action="store_true", help="Disable strict quality rejection in inference")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--mc_samples", type=int, default=100)
    parser.add_argument("--target_stats", type=str, default=None)
    args = parser.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    target_stats = None
    if args.target_stats:
        import json

        with open(args.target_stats, "r") as f:
            target_stats = json.load(f)

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(ROOT, args.checkpoint)
    engine = VocalMorphInference(
        ckpt_path,
        config,
        target_stats=target_stats,
        n_mc_samples=args.mc_samples,
    )
    if args.allow_low_quality:
        engine.quality_gate.strict = False
        print("[VocalMorph] Quality gate strict mode disabled (--allow_low_quality).")

    audio_paths: List[str] = []
    if args.audio_list:
        audio_paths = [p.strip() for p in args.audio_list.split(",") if p.strip()]
    elif args.audio_dir:
        if not os.path.isdir(args.audio_dir):
            print(f"[VocalMorph] audio_dir does not exist: {args.audio_dir}")
            return
        audio_paths = [
            os.path.join(args.audio_dir, name)
            for name in sorted(os.listdir(args.audio_dir))
            if os.path.splitext(name.lower())[1] in SUPPORTED_AUDIO_EXTENSIONS
        ]
        if not audio_paths:
            print(f"[VocalMorph] No supported audio files found in directory: {args.audio_dir}")
            return
    elif args.audio:
        audio_paths = [args.audio]

    if audio_paths:
        result = engine.predict_from_files(audio_paths, speaker_id=args.speaker_id)
    else:
        result = engine.predict_from_mic(args.duration)
    if result:
        print(result)


if __name__ == "__main__":
    main()

