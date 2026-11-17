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
    assert status["status"] in ("READY_CUSTOM", "CUSTOM_MODEL_MISSING", "WEIGHTS_MISSING")


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

    # Valid other corruption
    code_other, name_other = detector._normalize_class(4, "Other Corruption")
    assert code_other == "OTHER"
    assert name_other == "Other Corruption"

    # Unknown distress must return None, None (never mapped to arbitrary road distress)
    code4, name4 = detector._normalize_class(999, "unknown_distress")
    assert code4 is None and name4 is None

    # Never map by class ID alone: ID 0 with name 'person' must NOT become D00
    code_coco, name_coco = detector._normalize_class(0, "person")
    assert code_coco is None and name_coco is None


def test_reject_generic_coco_model():
    detector = RoadDamageDetector()

    # Standard COCO subset
    coco_subset = {0: "person", 1: "bicycle", 2: "car"}
    assert detector._is_valid_road_damage_model(coco_subset) is False

    # Generic 80-class dictionary
    coco_80 = {i: f"class_{i}" for i in range(80)}
    assert detector._is_valid_road_damage_model(coco_80) is False


def test_reject_partially_compatible_model():
    detector = RoadDamageDetector()

    # Only pothole (missing D00, D10, D20)
    partial_pothole_only = {0: "pothole"}
    assert detector._is_valid_road_damage_model(partial_pothole_only) is False

    # Missing D40 pothole
    missing_pothole = {0: "D00", 1: "D10", 2: "D20"}
    assert detector._is_valid_road_damage_model(missing_pothole) is False


def test_reject_incompatible_classes_model():
    detector = RoadDamageDetector()

    # Contains required classes but also incompatible non-road class 'airplane'
    mixed_incompatible = {0: "D00", 1: "D10", 2: "D20", 3: "D40", 4: "airplane"}
    assert detector._is_valid_road_damage_model(mixed_incompatible) is False


def test_accept_complete_valid_road_damage_model():
    detector = RoadDamageDetector()

    # Complete 4-class taxonomy
    valid_4_class = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
    assert detector._is_valid_road_damage_model(valid_4_class) is True

    # Complete 5-class taxonomy with full names
    valid_5_class = {
        0: "Longitudinal Crack",
        1: "Transverse Crack",
        2: "Alligator Crack",
        3: "Pothole",
        4: "Other Corruption"
    }
    assert detector._is_valid_road_damage_model(valid_5_class) is True


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
