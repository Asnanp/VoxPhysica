import numpy as np

from scripts.repair_vtl_features import FORMANT_SPACING_COL, VTL_COL, repair_payload
from src.preprocessing.feature_extractor import robust_vtl_from_formant_spacing


def test_robust_vtl_clamps_bad_spacing_outliers():
    spacing = np.asarray([1.0, 1000.0, -5.0, np.nan], dtype=np.float32)
    vtl = robust_vtl_from_formant_spacing(spacing)
    assert np.isclose(vtl[0], 35.0)
    assert np.isclose(vtl[1], 17.0)
    assert vtl[2] == 0.0
    assert vtl[3] == 0.0


def test_repair_payload_updates_sequence_vtl_column_and_scalar():
    sequence = np.zeros((3, 136), dtype=np.float32)
    sequence[:, FORMANT_SPACING_COL] = np.asarray([1.0, 1000.0, 1200.0], dtype=np.float32)
    sequence[:, VTL_COL] = np.asarray([17000.0, 17.0, 14.1667], dtype=np.float32)
    repaired, stats = repair_payload({"sequence": sequence, "speaker_id": np.asarray("spk")}, speed_of_sound=34000.0)

    assert repaired["sequence"][:, VTL_COL].max() <= 35.0
    assert repaired["vtl_mean"] <= 35.0
    assert repaired["vtl_repair_tag"] == "robust_vtl_v1"
    assert stats["old_vtl_max"] > 1000.0
    assert stats["new_vtl_max"] <= 35.0
