"""
ROADGUARD AI - Multi-Object Defect Tracker
Module: core/object_tracker.py
Version: 2.1.1

Provides multi-frame temporal tracking of detected road damage to prevent
repeated counting of the same defect across consecutive video frames.
Differentiates between Confirmed Tracks (observed across >= min_hits frames)
and Transient Detections (1-frame observations), preserving both in the final record
without discarding unconfirmed detections or inflating defect tallies.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

import config
from core.severity_engine import classify_severity_score


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
    """
    Represents a unique tracked road defect across consecutive video frames.
    Maintains raw visual spatial severity, temporal persistence adjustments,
    and confirmation status.
    """

    def __init__(
        self,
        track_id: int,
        detection: Dict[str, Any],
        frame_index: int,
        timestamp_sec: float,
        min_hits: int = config.TRACKER_MIN_HITS
    ):
        self.track_id = track_id
        self.min_hits = min_hits
        self.class_code = detection.get("class_code", "OTHER")
        self.class_name = detection.get("class_name", "Unknown")
        self.box = list(detection.get("box", [0, 0, 0, 0]))
        self.representative_box = list(self.box)

        # History and statistics
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.first_frame = frame_index
        self.last_frame = frame_index
        self.first_timestamp = timestamp_sec
        self.last_timestamp = timestamp_sec

        # Raw Spatial Severity (without temporal modifier)
        raw_sev = float(detection.get("raw_severity_score", detection.get("severity_score", 0.0)))
        self.peak_raw_severity_score = raw_sev
        self.peak_raw_severity_level = classify_severity_score(raw_sev)
        self.peak_confidence = float(detection.get("confidence", 0.0))
        self.reason = detection.get("reason", "")
        self.contributing_factors = dict(detection.get("contributing_factors", {}))

    def update(self, detection: Dict[str, Any], frame_index: int, timestamp_sec: float):
        """Update track state with a newly associated detection."""
        self.box = list(detection.get("box", [0, 0, 0, 0]))
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_frame = frame_index
        self.last_timestamp = timestamp_sec

        conf = float(detection.get("confidence", 0.0))
        raw_sev = float(detection.get("raw_severity_score", detection.get("severity_score", 0.0)))

        # Update peak representative state
        if conf > self.peak_confidence:
            self.peak_confidence = conf
            self.representative_box = list(self.box)

        if raw_sev >= self.peak_raw_severity_score:
            self.peak_raw_severity_score = raw_sev
            self.peak_raw_severity_level = classify_severity_score(raw_sev)
            self.reason = detection.get("reason", "")
            self.contributing_factors = dict(detection.get("contributing_factors", {}))

    @property
    def is_confirmed(self) -> bool:
        """Confirmed if observed in >= min_hits frames."""
        return self.hits >= self.min_hits

    def compute_adjusted_severity(self) -> Tuple[float, str, float]:
        """
        Compute final severity with temporal persistence modifier.
        Returns: (adjusted_score, adjusted_level, persistence_adjustment)
        """
        if self.hits >= config.PERSISTENCE_THRESHOLD_HITS:
            # Verified persistent defect across >= 5 frames
            adjustment = config.PERSISTENCE_BONUS  # +3.0 pts
        elif self.hits == 1:
            # Single-frame transient observation in video
            adjustment = config.TRANSIENT_DISCOUNT  # -2.0 pts
        else:
            adjustment = 0.0

        final_score = min(100.0, max(5.0, self.peak_raw_severity_score + adjustment))
        final_level = classify_severity_score(final_score)
        return round(final_score, 1), final_level, round(adjustment, 1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracked defect to complete summary dictionary."""
        adj_score, adj_level, adj_pts = self.compute_adjusted_severity()
        status_str = "confirmed" if self.is_confirmed else "transient"

        return {
            "track_id": self.track_id,
            "class_code": self.class_code,
            "class_name": self.class_name,
            "box": self.representative_box,
            "confidence": round(self.peak_confidence, 3),
            "peak_confidence": round(self.peak_confidence, 3),
            "raw_severity_score": round(self.peak_raw_severity_score, 1),
            "raw_severity_level": self.peak_raw_severity_level,
            "adjusted_severity_score": adj_score,
            "adjusted_severity_level": adj_level,
            "severity_score": adj_score,
            "severity_level": adj_level,
            "peak_severity_score": adj_score,
            "peak_severity_level": adj_level,
            "persistence_adjustment": adj_pts,
            "is_confirmed": self.is_confirmed,
            "status": status_str,
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
    Preserves both Confirmed Tracks and Transient Detections.
    """

    def __init__(
        self,
        iou_threshold: float = config.TRACKER_IOU_THRESHOLD,
        max_age: int = config.TRACKER_MAX_AGE,
        min_hits: int = config.TRACKER_MIN_HITS
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_track_id = 1
        self.active_tracks: List[TrackedDefect] = []
        self.confirmed_history: List[TrackedDefect] = []
        self.transient_history: List[TrackedDefect] = []

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_index: int,
        timestamp_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Process current frame detections, associate with active tracks,
        and return annotated detections with track metadata.
        """
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
                new_track = TrackedDefect(
                    self.next_track_id, det, frame_index, timestamp_sec, self.min_hits
                )
                self.next_track_id += 1
                self.active_tracks.append(new_track)

                det_copy = dict(det)
                det_copy["track_id"] = new_track.track_id
                det_copy["is_confirmed"] = new_track.is_confirmed
                annotated_detections.append(det_copy)
            return annotated_detections

        # Association using IoU & Class Matching
        matched_tracks = set()
        matched_detections = set()
        matches = []

        for det_idx, det in enumerate(detections):
            det_box = det["box"]
            det_code = det.get("class_code", "OTHER")

            best_iou = 0.0
            best_track_idx = -1

            for t_idx, track in enumerate(self.active_tracks):
                if t_idx in matched_tracks:
                    continue
                # Enforce strict class consistency across frames
                if det_code != track.class_code:
                    continue

                iou = compute_iou(det_box, track.box)
                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
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
            det_copy["is_confirmed"] = track.is_confirmed
            annotated_detections.append(det_copy)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                new_track = TrackedDefect(
                    self.next_track_id, det, frame_index, timestamp_sec, self.min_hits
                )
                self.next_track_id += 1
                self.active_tracks.append(new_track)

                det_copy = dict(det)
                det_copy["track_id"] = new_track.track_id
                det_copy["is_confirmed"] = new_track.is_confirmed
                annotated_detections.append(det_copy)

        self._prune_dead_tracks()
        return annotated_detections

    def _prune_dead_tracks(self):
        """Move tracks exceeding max_age to appropriate history list without losing data."""
        surviving = []
        for track in self.active_tracks:
            if track.time_since_update > self.max_age:
                if track.is_confirmed:
                    self.confirmed_history.append(track)
                else:
                    self.transient_history.append(track)
            else:
                surviving.append(track)
        self.active_tracks = surviving

    def finalize(self) -> List[Dict[str, Any]]:
        """
        Finalize video tracking session.
        Returns deduplicated unique defect records, placing Confirmed Tracks first,
        followed by Transient Detections.
        """
        for track in self.active_tracks:
            if track.is_confirmed:
                self.confirmed_history.append(track)
            else:
                self.transient_history.append(track)
        self.active_tracks = []

        all_records = []
        # Confirmed unique defects
        for t in self.confirmed_history:
            all_records.append(t.to_dict())

        # Transient unconfirmed detections
        for t in self.transient_history:
            all_records.append(t.to_dict())

        return all_records

    def get_confirmed_defects(self) -> List[Dict[str, Any]]:
        """Return only verified confirmed unique tracks."""
        return [t.to_dict() for t in self.confirmed_history if t.is_confirmed]

    def get_transient_defects(self) -> List[Dict[str, Any]]:
        """Return only transient unconfirmed detections."""
        return [t.to_dict() for t in self.transient_history if not t.is_confirmed]
