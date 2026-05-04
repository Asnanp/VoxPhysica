import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.predict import InferenceQualityGateConfig, quality_gate_decision


def test_quality_gate_accepts_good_capture():
    cfg = InferenceQualityGateConfig(strict=True)
    meta = {
        "capture_quality_score": 0.82,
        "speech_ratio": 0.88,
        "snr_db_estimate": 18.0,
        "distance_cm_estimate": 17.0,
        "clipped_ratio": 0.005,
        "quality_ok": True,
    }
    accepted, reasons, weight = quality_gate_decision(meta, cfg)
    assert accepted is True
    assert reasons == []
    assert 0.70 <= weight <= 1.0


def test_quality_gate_rejects_low_quality_in_strict_mode():
    cfg = InferenceQualityGateConfig(strict=True)
    meta = {
        "capture_quality_score": 0.30,
        "speech_ratio": 0.30,
        "snr_db_estimate": 4.0,
        "distance_cm_estimate": 55.0,
        "clipped_ratio": 0.09,
        "quality_ok": False,
    }
    accepted, reasons, weight = quality_gate_decision(meta, cfg)
    assert accepted is False
    assert "low_capture_quality" in reasons
    assert "low_snr" in reasons
    assert "far_microphone" in reasons
    assert weight >= 0.05


def test_quality_gate_soft_mode_downweights_bad_capture():
    cfg = InferenceQualityGateConfig(strict=False)
    meta = {
        "capture_quality_score": 0.35,
        "speech_ratio": 0.40,
        "snr_db_estimate": 7.0,
        "distance_cm_estimate": 42.0,
        "clipped_ratio": 0.02,
        "quality_ok": False,
    }
    accepted, reasons, weight = quality_gate_decision(meta, cfg)
    assert accepted is True
    assert reasons
    assert 0.05 <= weight < 0.35
