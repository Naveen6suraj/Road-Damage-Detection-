"""
Unit Tests: Road Damage Detector & Normalization
File: tests/test_detector.py
"""

import os
import numpy as np
import pytest
from core.detector import RoadDamageDetector
import config


def test_detector_initialization():
    detector = RoadDamageDetector()
    status = detector.get_status()
    assert "status" in status
    assert "device" in status
    assert status["status"] in ("READY_CUSTOM", "READY_FALLBACK", "WEIGHTS_MISSING")


def test_class_name_normalization():
    detector = RoadDamageDetector()

    # Test handling of plural 'Potholes' vs singular 'Pothole'
    code1, name1 = detector._normalize_class(3, "Potholes")
    assert code1 == "D40"
    assert name1 == "Pothole"

    code2, name2 = detector._normalize_class(3, "Pothole")
    assert code2 == "D40"
    assert name2 == "Pothole"

    code3, name3 = detector._normalize_class(0, "Longitudinal Crack")
    assert code3 == "D00"
    assert name3 == "Longitudinal Crack"

    code4, name4 = detector._normalize_class(999, "unknown_distress")
    assert code4 == "OTHER"


def test_detection_on_synthetic_frame():
    detector = RoadDamageDetector()
    # Create black frame 640x480
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    annotated, detections, counts = detector.detect(frame, conf=0.25)
    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
    assert isinstance(detections, list)
    assert isinstance(counts, dict)
    assert "D00" in counts and "D40" in counts
