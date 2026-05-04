#!/usr/bin/env python
"""Run raw-audio perturbation stress tests for strict VocalMorph inference."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.predict import VocalMorphInference  # noqa: E402
from src.preprocessing.audio_enhancement import enhance_microphone_audio  # noqa: E402
from src.preprocessing.feature_extractor import load_audio  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VocalMorph robustness stress tests.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/pibnn_base.yaml")
    parser.add_argument("--test_csv", default="data/splits/test_clean.csv")
    parser.add_argument("--max_speakers", type=int, default=20)
    parser.add_argument("--report-out", default="audit/robustness_report.md")
    return parser.parse_args()


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def add_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-8)
    noise = np.random.normal(0.0, rms / (10 ** (10.0 / 20.0)), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def add_clipping(audio: np.ndarray, sr: int) -> np.ndarray:
    return np.clip(audio * 2.5, -0.35, 0.35).astype(np.float32)


def add_bandlimit(audio: np.ndarray, sr: int) -> np.ndarray:
    sos = signal.butter(6, [250.0, 2800.0], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, audio).astype(np.float32)


def add_short_crop(audio: np.ndarray, sr: int) -> np.ndarray:
    max_len = min(audio.shape[0], int(sr * 2.0))
    return audio[:max_len].astype(np.float32)


def add_silence_pad(audio: np.ndarray, sr: int) -> np.ndarray:
    pad = np.zeros(int(sr), dtype=np.float32)
    return np.concatenate([pad, audio, pad]).astype(np.float32)


def add_far_mic(audio: np.ndarray, sr: int) -> np.ndarray:
    sos = signal.butter(4, 2200.0, btype="lowpass", fs=sr, output="sos")
    filtered = signal.sosfilt(sos, audio).astype(np.float32)
    return np.clip(filtered * 0.45, -1.0, 1.0)


PERTURBATIONS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "clean": lambda audio, sr: audio.astype(np.float32, copy=False),
    "noise_10db": add_noise,
    "clipping": add_clipping,
    "bandlimit": add_bandlimit,
    "short_2s": add_short_crop,
    "silence_pad": add_silence_pad,
    "far_mic": add_far_mic,
}


def _load_test_rows(path: str, max_speakers: int) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[: max(1, int(max_speakers))]


def main() -> int:
    args = parse_args()
    ckpt_path = _resolve(args.checkpoint)
    config_path = _resolve(args.config)
    test_csv = _resolve(args.test_csv)
    report_out = _resolve(args.report_out)
    os.makedirs(os.path.dirname(report_out), exist_ok=True)

    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    engine = VocalMorphInference(ckpt_path, config)
    rows = _load_test_rows(test_csv, args.max_speakers)

    summary: Dict[str, Dict[str, float]] = {
        name: {"accepted": 0.0, "total": 0.0, "error_sum": 0.0}
        for name in PERTURBATIONS
    }

    for row in rows:
        audio_paths = [part.strip() for part in str(row.get("audio_paths", "")).split("|") if part.strip()]
        if not audio_paths:
            continue
        full_path = _resolve(audio_paths[0])
        loaded = load_audio(
            full_path,
            target_sr=int(config.get("data", {}).get("sample_rate", 16000)),
            enhance=False,
            return_metadata=False,
        )
        if loaded is None:
            continue
        audio = loaded
        target_height = float(row["height_cm"])
        speaker_id = str(row.get("speaker_id", "speaker"))
        for perturb_name, perturb_fn in PERTURBATIONS.items():
            perturbed = perturb_fn(audio.copy(), engine.feature_config.sample_rate)
            enhanced, report = enhance_microphone_audio(
                perturbed,
                engine.feature_config.sample_rate,
                engine.enhancement_config,
            )
            enhancement_meta = report.to_dict()
            accepted, reasons, weight = engine._evaluate_clip_quality(enhancement_meta)
            summary[perturb_name]["total"] += 1.0
            if not accepted and engine.quality_gate.strict:
                continue
            prediction = engine._predict_batch(
                audios=[enhanced],
                enhancement_meta_list=[enhancement_meta],
                quality_weights=[weight],
                speaker_id=speaker_id,
                n_input_clips=1,
                rejected_reasons=reasons,
            )
            if prediction.accepted:
                summary[perturb_name]["accepted"] += 1.0
                summary[perturb_name]["error_sum"] += abs(float(prediction.height_cm) - target_height)

    with open(report_out, "w", encoding="utf-8") as handle:
        handle.write("# Robustness Report\n\n")
        handle.write("## Findings\n")
        handle.write("- Stress tests use raw audio and the strict inference API, including quality gating, uncertainty, and OOD rejection.\n")
        handle.write("- Coverage below is the fraction of clips that remained accepted after the rejection logic.\n")
        handle.write("\n## Stress Test Table\n")
        handle.write("| perturbation | coverage | accepted MAE | notes |\n")
        handle.write("| --- | ---: | ---: | --- |\n")
        for name, row in summary.items():
            total = max(row["total"], 1.0)
            coverage = row["accepted"] / total
            mae = row["error_sum"] / row["accepted"] if row["accepted"] > 0 else float("nan")
            note = "strict rejection active"
            mae_text = f"{mae:.3f}" if np.isfinite(mae) else "nan"
            handle.write(f"| {name} | {coverage:.3f} | {mae_text} | {note} |\n")
        handle.write("\n## Fixes\n")
        handle.write("- Added raw-audio stress testing for noise, clipping, band-limit, short clips, silence padding, and far-microphone degradation.\n")
        handle.write("- Reused the inference rejection logic so the report measures coverage versus error instead of filtered MAE alone.\n")
        handle.write("\n## Remaining Risks\n")
        handle.write("- Final robustness claims still depend on a canonical audited checkpoint trained on rebuilt `data/features_audited` artifacts.\n")

    print(f"[Robustness] Wrote report to {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
