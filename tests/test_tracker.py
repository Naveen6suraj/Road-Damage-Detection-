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
