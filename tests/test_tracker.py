"""
Unit Tests: Multi-Object Defect Tracker
File: tests/test_tracker.py
"""

import pytest
from core.object_tracker import compute_iou, compute_centroid_distance, RoadDefectTracker


def test_compute_iou():
    boxA = [0, 0, 10, 10]
    boxB = [0, 0, 10, 10]
    assert compute_iou(boxA, boxB) == 1.0

    boxC = [20, 20, 30, 30]
    assert compute_iou(boxA, boxC) == 0.0

    boxD = [5, 0, 15, 10]
    # Intersection is 5x10=50. Union is 100+100-50=150. IoU = 50/150 = 1/3
    assert abs(compute_iou(boxA, boxD) - (1.0 / 3.0)) < 1e-4


def test_tracker_association_across_frames():
    tracker = RoadDefectTracker(iou_threshold=0.3, max_age=5, min_hits=2)

    # Frame 1: Defect appears
    det1 = [{
        "box": [100, 100, 200, 200],
        "class_code": "D40",
        "class_name": "Pothole",
        "confidence": 0.85,
        "severity_score": 75.0,
        "severity_level": "HIGH"
    }]
    f1_tracks = tracker.update(det1, frame_index=1, timestamp_sec=0.04)
    assert len(f1_tracks) == 1
    track_id = f1_tracks[0]["track_id"]

    # Frame 2: Same defect shifts slightly forward as vehicle advances
    det2 = [{
        "box": [102, 105, 205, 208],
        "class_code": "D40",
        "class_name": "Pothole",
        "confidence": 0.88,
        "severity_score": 78.0,
        "severity_level": "HIGH"
    }]
    f2_tracks = tracker.update(det2, frame_index=2, timestamp_sec=0.08)
    assert len(f2_tracks) == 1
    # Track ID should remain identical
    assert f2_tracks[0]["track_id"] == track_id

    # Frame 3: Same defect continues
    det3 = [{
        "box": [105, 110, 210, 215],
        "class_code": "D40",
        "class_name": "Pothole",
        "confidence": 0.90,
        "severity_score": 82.0,
        "severity_level": "CRITICAL"
    }]
    tracker.update(det3, frame_index=3, timestamp_sec=0.12)

    # Finalize tracking session
    unique_defects = tracker.finalize()
    assert len(unique_defects) == 1
    assert unique_defects[0]["track_id"] == track_id
    assert unique_defects[0]["hits"] == 3
    assert unique_defects[0]["peak_confidence"] == 0.90
    assert unique_defects[0]["peak_severity_level"] == "CRITICAL"


def test_tracker_deduplication():
    """Verify that multiple consecutive frames of the same pothole are counted as 1 unique defect."""
    tracker = RoadDefectTracker(iou_threshold=0.25, min_hits=2)

    # Simulate 15 consecutive frames of the same defect
    for i in range(15):
        dets = [{
            "box": [500 + i * 2, 600 + i * 3, 600 + i * 2, 700 + i * 3],
            "class_code": "D40",
            "class_name": "Pothole",
            "confidence": 0.80 + (i * 0.01),
            "severity_score": 70.0 + i,
            "severity_level": "HIGH" if (70 + i) < 80 else "CRITICAL"
        }]
        tracker.update(dets, frame_index=i + 1, timestamp_sec=(i + 1) * 0.033)

    unique = tracker.finalize()
    assert len(unique) == 1, "Should deduplicate 15 frame detections into 1 unique defect"
    assert unique[0]["hits"] == 15


def test_tracker_transient_vs_confirmed():
    tracker = RoadDefectTracker(iou_threshold=0.3, min_hits=2)

    # Defect A appears for 1 frame only (transient)
    tracker.update([{
        "box": [100, 100, 200, 200],
        "class_code": "D00",
        "class_name": "Longitudinal Crack",
        "confidence": 0.65,
        "raw_severity_score": 25.0
    }], frame_index=1, timestamp_sec=0.033)

    # Defect B appears for 6 consecutive frames (confirmed and persistent)
    for i in range(6):
        tracker.update([{
            "box": [500 + i * 2, 500 + i * 2, 650 + i * 2, 650 + i * 2],
            "class_code": "D40",
            "class_name": "Pothole",
            "confidence": 0.90,
            "raw_severity_score": 85.0
        }], frame_index=i + 1, timestamp_sec=(i + 1) * 0.033)

    results = tracker.finalize()
    assert len(results) == 2, "Both transient and confirmed defects must be preserved"

    transient_item = [d for d in results if d["class_code"] == "D00"][0]
    confirmed_item = [d for d in results if d["class_code"] == "D40"][0]

    assert transient_item["is_confirmed"] is False
    assert transient_item["status"] == "transient"
    assert transient_item["persistence_adjustment"] == -2.0

    assert confirmed_item["is_confirmed"] is True
    assert confirmed_item["status"] == "confirmed"
    assert confirmed_item["persistence_adjustment"] == 3.0


def test_tracker_different_classes_no_association():
    tracker = RoadDefectTracker(iou_threshold=0.3, min_hits=2)

    # Frame 1: Pothole (D40) at [200, 200, 300, 300]
    tracker.update([{
        "box": [200, 200, 300, 300],
        "class_code": "D40",
        "class_name": "Pothole",
        "confidence": 0.85
    }], frame_index=1, timestamp_sec=0.033)

    # Frame 2: Crack (D00) at exact same coordinates
    tracker.update([{
        "box": [200, 200, 300, 300],
        "class_code": "D00",
        "class_name": "Longitudinal Crack",
        "confidence": 0.85
    }], frame_index=2, timestamp_sec=0.066)

    results = tracker.finalize()
    assert len(results) == 2, "Different defect classes must NOT be associated into same track"

