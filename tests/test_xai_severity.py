"""
Unit Tests: Explainable AI (XAI) Severity Engine & Feature Decomposition
File: tests/test_xai_severity.py
"""

import pytest
from core.severity_engine import estimate_severity, classify_severity_score
import config


def test_severity_explicit_decomposition():
    """Verify that estimate_severity returns explicit contributions dictionary."""
    det = {
        "box": [800, 700, 1100, 950],
        "class_code": "D40",
        "confidence": 0.90
    }
    res = estimate_severity(det, (1920, 1080), context={"track_hits": 6, "is_video": True})

    assert "contributions" in res
    cb = res["contributions"]
    assert "category" in cb
    assert "area" in cb
    assert "proximity" in cb
    assert "geometry" in cb
    assert "temporal" in cb

    # Pothole base hazard must be 35.0
    assert cb["category"] == 35.0
    # Temporal bonus for >= 5 hits must be +3.0
    assert cb["temporal"] == 3.0

    # Verify that clamped sum of contributions matches final severity score
    comp_sum = sum(cb.values())
    expected_score = min(100.0, max(5.0, comp_sum))
    assert abs(res["severity_score"] - expected_score) < 0.15


def test_severity_aspect_ratio_geometric_penalty():
    """Verify geometry scoring differentiates linear cracks from compact potholes."""
    frame = (1000, 1000)

    # 1. Elongated crack (aspect ratio 5.0)
    crack_det = {"box": [100, 500, 600, 600], "class_code": "D00", "confidence": 0.8}
    crack_res = estimate_severity(crack_det, frame)
    assert crack_res["contributions"]["geometry"] == 10.0  # Elongated crack gets max geometry score

    # 2. Compact crack (aspect ratio 1.0)
    compact_crack = {"box": [100, 500, 200, 600], "class_code": "D00", "confidence": 0.8}
    compact_res = estimate_severity(compact_crack, frame)
    assert compact_res["contributions"]["geometry"] == 3.0  # Non-elongated crack gets lower geometry score

    # 3. Compact pothole (aspect ratio 1.2)
    compact_pothole = {"box": [100, 500, 220, 600], "class_code": "D40", "confidence": 0.8}
    pothole_res = estimate_severity(compact_pothole, frame)
    assert pothole_res["contributions"]["geometry"] == 10.0  # Compact cavity gets max geometry score


def test_severity_transient_discount():
    """Verify single-frame transient video detection receives -2.0 penalty."""
    det = {"box": [300, 400, 500, 600], "class_code": "D10", "confidence": 0.75}
    res = estimate_severity(det, (1280, 720), context={"track_hits": 1, "is_video": True})

    assert res["contributions"]["temporal"] == -2.0
    assert res["persistence_adjustment"] == -2.0
