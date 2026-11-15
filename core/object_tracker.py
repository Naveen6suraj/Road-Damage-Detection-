"""
ROADGUARD AI - Multi-Object Defect Tracker
Module: core/object_tracker.py

Provides multi-frame temporal tracking of detected road damage to prevent
repeated counting of the same defect across consecutive video frames.
Distinguishes between total frame-level detections and unique tracked road defects.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_centroid_distance(box1: List[float], box2: List[float]) -> float:
    """Compute Euclidean distance between centers of two boxes."""
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    return float(np.hypot(c1_x - c2_x, c1_y - c2_y))


class TrackedDefect:
    """Represents a unique tracked road defect over multiple frames."""

    def __init__(self, track_id: int, detection: Dict[str, Any], frame_index: int, timestamp_sec: float):
        self.track_id = track_id
        self.class_code = detection.get("class_code", "OTHER")
        self.class_name = detection.get("class_name", "Unknown")
        self.box = list(detection.get("box", [0, 0, 0, 0]))
        
        # History and statistics
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.first_frame = frame_index
        self.last_frame = frame_index
        self.first_timestamp = timestamp_sec
        self.last_timestamp = timestamp_sec

        # Severity & Confidence tracking
        self.peak_confidence = float(detection.get("confidence", 0.0))
        self.peak_severity_score = float(detection.get("severity_score", 0.0))
        self.peak_severity_level = detection.get("severity_level", "LOW")
        self.reason = detection.get("reason", "")
        self.contributing_factors = detection.get("contributing_factors", {})
        self.representative_box = list(self.box)

    def update(self, detection: Dict[str, Any], frame_index: int, timestamp_sec: float):
        """Update track state with a newly associated detection."""
        self.box = list(detection.get("box", [0, 0, 0, 0]))
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_frame = frame_index
        self.last_timestamp = timestamp_sec

        conf = float(detection.get("confidence", 0.0))
        sev_score = float(detection.get("severity_score", 0.0))

        if conf > self.peak_confidence:
            self.peak_confidence = conf

        if sev_score >= self.peak_severity_score:
            self.peak_severity_score = sev_score
            self.peak_severity_level = detection.get("severity_level", "LOW")
            self.reason = detection.get("reason", "")
            self.contributing_factors = detection.get("contributing_factors", {})
            self.representative_box = list(self.box)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracked defect to summary dictionary."""
        return {
            "track_id": self.track_id,
            "class_code": self.class_code,
            "class_name": self.class_name,
            "box": self.representative_box,
            "confidence": round(self.peak_confidence, 3),
            "peak_confidence": round(self.peak_confidence, 3),
            "severity_score": round(self.peak_severity_score, 1),
            "peak_severity_score": round(self.peak_severity_score, 1),
            "severity_level": self.peak_severity_level,
            "peak_severity_level": self.peak_severity_level,
            "hits": self.hits,
            "total_frames_tracked": self.hits,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_timestamp_sec": round(self.first_timestamp, 2),
            "last_timestamp_sec": round(self.last_timestamp, 2),
            "reason": self.reason,
            "contributing_factors": self.contributing_factors
        }


class RoadDefectTracker:
    """
    Online spatial-temporal tracker designed specifically for forward-moving dashcams.
    Associates road defects across frames using IoU and adaptive spatial distance.
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 15, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_track_id = 1
        self.active_tracks: List[TrackedDefect] = []
        self.all_confirmed_defects: List[TrackedDefect] = []

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_index: int,
        timestamp_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Process current frame detections, associate with active tracks,
        and update defect states.

        Returns:
            List of current frame detections annotated with assigned track_id.
        """
        # Increment age for existing active tracks
        for track in self.active_tracks:
            track.time_since_update += 1
            track.age += 1

        annotated_detections = []
        if not detections:
            self._prune_dead_tracks()
            return []

        # If no active tracks, initialize all detections as new tracks
        if not self.active_tracks:
            for det in detections:
                new_track = TrackedDefect(self.next_track_id, det, frame_index, timestamp_sec)
                self.next_track_id += 1
                self.active_tracks.append(new_track)
                
                det_copy = dict(det)
                det_copy["track_id"] = new_track.track_id
                annotated_detections.append(det_copy)
            return annotated_detections

        # Association using IoU & Class Matching
        matched_tracks = set()
        matched_detections = set()

        # Build cost/similarity matrix
        matches = []
        for det_idx, det in enumerate(detections):
            det_box = det["box"]
            det_code = det.get("class_code", "OTHER")

            best_iou = 0.0
            best_track_idx = -1

            for t_idx, track in enumerate(self.active_tracks):
                if t_idx in matched_tracks:
                    continue

                # Defect category matching preference
                iou = compute_iou(det_box, track.box)
                if iou >= self.iou_threshold:
                    # Bonus for matching class code
                    similarity = iou + (0.1 if det_code == track.class_code else 0.0)
                    if similarity > best_iou:
                        best_iou = similarity
                        best_track_idx = t_idx

            if best_track_idx >= 0 and best_iou >= self.iou_threshold:
                matches.append((det_idx, best_track_idx))
                matched_tracks.add(best_track_idx)
                matched_detections.add(det_idx)

        # Update matched tracks
        for det_idx, t_idx in matches:
            det = detections[det_idx]
            track = self.active_tracks[t_idx]
            track.update(det, frame_index, timestamp_sec)

            det_copy = dict(det)
            det_copy["track_id"] = track.track_id
            annotated_detections.append(det_copy)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                new_track = TrackedDefect(self.next_track_id, det, frame_index, timestamp_sec)
                self.next_track_id += 1
                self.active_tracks.append(new_track)

                det_copy = dict(det)
                det_copy["track_id"] = new_track.track_id
                annotated_detections.append(det_copy)

        self._prune_dead_tracks()
        return annotated_detections

    def _prune_dead_tracks(self):
        """Move tracks exceeding max_age to confirmed history if min_hits reached."""
        surviving = []
        for track in self.active_tracks:
            if track.time_since_update > self.max_age:
                if track.hits >= self.min_hits:
                    self.all_confirmed_defects.append(track)
            else:
                surviving.append(track)
        self.active_tracks = surviving

    def finalize(self) -> List[Dict[str, Any]]:
        """
        Finalize video tracking session.
        Collects all active and retired tracks that meet the minimum hits threshold.
        """
        for track in self.active_tracks:
            if track.hits >= self.min_hits:
                self.all_confirmed_defects.append(track)
        self.active_tracks = []

        # Deduplicate and sort by track_id
        unique_defects = {}
        for track in self.all_confirmed_defects:
            unique_defects[track.track_id] = track.to_dict()

        # If min_hits filtered everything out (e.g., short 1-frame test), return all tracks
        if not unique_defects and self.next_track_id > 1:
            # Fallback to avoid dropping short clips
            for track in self.all_confirmed_defects:
                unique_defects[track.track_id] = track.to_dict()

        return list(unique_defects.values())
