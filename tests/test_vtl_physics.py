"""Unit tests for Physics-Informed VTL module."""

from __future__ import annotations

import numpy as np
import pytest
from src.preprocessing.vtl_physics import (
    compute_vtl_from_formants,
    generate_synthetic_vtl_vector,
    augment_views_with_vtl_physics,
)


def test_compute_vtl_from_formants():
    formants = [500.0, 1500.0, 2500.0, 3500.0]
    vtl_cm, delta_f, height_est = compute_vtl_from_formants(formants)
    
    assert 14.0 <= vtl_cm <= 18.0
    assert 950.0 <= delta_f <= 1100.0
    assert 90.0 <= height_est <= 120.0


def test_generate_synthetic_vtl_vector():
    vec = generate_synthetic_vtl_vector(height_cm=175.0, gender=1.0)
    assert len(vec) == 8
    assert vec[0] > 10.0  # VTL
    assert vec[1] > 100.0 # Height est


def test_augment_views_with_vtl_physics():
    views = {"wavlm": np.zeros((10, 768), dtype=np.float32)}
    y = np.full(10, 170.0, dtype=np.float32)
    gender = np.ones(10, dtype=np.float32)
    
    aug = augment_views_with_vtl_physics(views, y, gender)
    assert "vtl_physics" in aug
    assert "wavlm+vtl" in aug
    assert aug["wavlm+vtl"].shape == (10, 776)
