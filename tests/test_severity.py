"""
Unit Tests: Severity Estimation Engine
File: tests/test_severity.py
"""

import pytest
from core.severity_engine import (
    estimate_severity, classify_severity_score,
    CLASS_BASE_HAZARD, SEVERITY_THRESHOLDS
)


def test_classify_severity_score():
    assert classify_severity_score(95.0) == "CRITICAL"
    assert classify_severity_score(80.0) == "CRITICAL"
    assert classify_severity_score(79.9) == "HIGH"
    assert classify_severity_score(60.0) == "HIGH"
    assert classify_severity_score(59.9) == "MODERATE"
    assert classify_severity_score(30.0) == "MODERATE"
    assert classify_severity_score(29.9) == "LOW"
    assert classify_severity_score(5.0) == "LOW"


def test_estimate_severity_critical_pothole():
    frame_size = (1920, 1080)
    # Large pothole in near-field lane directly ahead
    detection = {
        "box": [800, 750, 1200, 1020],
        "class_code": "D40",
        "confidence": 0.95
    }
    result = estimate_severity(detection, frame_size)
    assert result["severity_level"] == "CRITICAL"
    assert result["severity_score"] >= 80.0
    assert "D40" in result["reason"]
    assert "contributing_factors" in result
    assert result["contributing_factors"]["class_code"] == "D40"


def test_estimate_severity_minor_crack():
    frame_size = (1920, 1080)
    # Minor hairline longitudinal crack at horizon
    detection = {
        "box": [900, 300, 930, 420],
        "class_code": "D00",
        "confidence": 0.60
    }
    result = estimate_severity(detection, frame_size)
    assert result["severity_level"] == "LOW"
    assert result["severity_score"] < 30.0


def test_estimate_severity_moderate_transverse_crack():
    frame_size = (1920, 1080)
    detection = {
        "box": [700, 600, 1200, 680],
        "class_code": "D10",
        "confidence": 0.78
    }
    result = estimate_severity(detection, frame_size)
    assert result["severity_level"] in ("MODERATE", "HIGH")
    assert result["severity_score"] >= 30.0


def test_severity_persistence_context():
    frame_size = (1920, 1080)
    detection = {
        "box": [700, 600, 950, 750],
        "class_code": "D20",
        "confidence": 0.80
    }
    res_single = estimate_severity(detection, frame_size, context={"track_hits": 1})
    res_persistent = estimate_severity(detection, frame_size, context={"track_hits": 6})
    # Persistent observation should yield higher severity than transient single-frame blip
    assert res_persistent["severity_score"] > res_single["severity_score"]


def test_zero_dimensions_edge_case():
    detection = {"box": [0, 0, 0, 0], "class_code": "OTHER", "confidence": 0.5}
    result = estimate_severity(detection, (0, 0))
    assert 0.0 <= result["severity_score"] <= 100.0
    assert result["severity_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
