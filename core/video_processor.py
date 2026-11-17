"""
ROADGUARD AI - Video Processing Engine
Module: core/video_processor.py
Version: 2.1.1

Executes end-to-end dashcam video analysis:
- Frame extraction with configurable stride
- YOLO defect detection with video context
- Multi-frame object tracking and deduplication
- Temporal severity adjustment (distinguishing confirmed vs. transient)
- High-contrast HUD annotation
- Real-time progress monitoring
- Safe OpenCV resource release
"""

import os
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple, List

import config
from core.detector import RoadDamageDetector
from core.object_tracker import RoadDefectTracker


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Extract metadata from video file using OpenCV."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = total_frames / fps if fps > 0 else 0.0
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        return {
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "fourcc": fourcc_str
        }
    finally:
        cap.release()


class VideoProcessor:
    """
    Manages video processing lifecycle, inference, tracking, and progress reporting.
    """

    def __init__(self, detector: Optional[RoadDamageDetector] = None):
        self.detector = detector or RoadDamageDetector()
        self.progress_state: Dict[str, Any] = {
            "status": "idle",
            "percent": 0.0,
            "current_frame": 0,
            "total_frames": 0,
            "fps_processing": 0.0,
            "time_elapsed_sec": 0.0,
            "estimated_remaining_sec": 0.0,
            "total_frame_detections": 0,
            "unique_defects_count": 0,
            "transient_defects_count": 0,
            "error": None
        }

    def process_video(
        self,
        input_path: str,
        output_path: str,
        conf_threshold: float = config.DEFAULT_CONFIDENCE,
        frame_stride: int = 1,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Process dashcam video through detection and tracking pipeline.

        Parameters:
        -----------
        input_path: str
            Path to uploaded video.
        output_path: str
            Destination path for annotated video.
        conf_threshold: float
            Detection confidence threshold.
        frame_stride: int
            Process every Nth frame (default 1 = every frame).
        progress_callback: callable
            Optional function receiving updated progress dict.

        Returns:
        --------
        dict containing video analysis summary and unique defect registry.
        """
        metadata = get_video_metadata(input_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video stream at: {input_path}")

        width = metadata["width"]
        height = metadata["height"]
        video_fps = metadata["fps"]
        total_frames = metadata["total_frames"]

        # Ensure parent directory for output exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Setup VideoWriter with best compatible codec (mp4v prioritized on Windows)
        writer = None
        codecs_to_try = [
            ("mp4v", ".mp4"),
            ("avc1", ".mp4"),
            ("XVID", ".avi")
        ]

        actual_output_path = output_path
        for codec_str, ext in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                test_path = os.path.splitext(output_path)[0] + ext
                w = cv2.VideoWriter(test_path, fourcc, video_fps, (width, height))
                if w.isOpened():
                    writer = w
                    actual_output_path = test_path
                    break
            except Exception:
                continue

        if writer is None or not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            actual_output_path = os.path.splitext(output_path)[0] + ".mp4"
            writer = cv2.VideoWriter(actual_output_path, fourcc, video_fps, (width, height))

        # Initialize tracker
        tracker = RoadDefectTracker(
            iou_threshold=config.TRACKER_IOU_THRESHOLD,
            max_age=config.TRACKER_MAX_AGE,
            min_hits=config.TRACKER_MIN_HITS
        )

        total_frame_detections = 0
        frame_idx = 0
        processed_frames_count = 0
        start_time = time.time()

        self.progress_state["status"] = "processing"
        self.progress_state["total_frames"] = total_frames
        self.progress_state["error"] = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                timestamp_sec = frame_idx / video_fps if video_fps > 0 else 0.0

                # Frame stride sampling
                if frame_stride > 1 and (frame_idx % frame_stride != 0):
                    writer.write(frame)
                    continue

                processed_frames_count += 1

                # 1. Run YOLO inference with video context
                _, raw_detections, _ = self.detector.detect(
                    frame,
                    conf=conf_threshold,
                    imgsz=config.DEFAULT_IMGSZ,
                    context={"is_video": True}
                )
                total_frame_detections += len(raw_detections)

                # 2. Update multi-object tracker
                tracked_detections = tracker.update(raw_detections, frame_idx, timestamp_sec)

                # 3. Draw annotations on frame
                annotated_frame = frame.copy()
                for det in tracked_detections:
                    self.detector._draw_detection_overlay(annotated_frame, det)

                # Draw top HUD Status Banner
                confirmed_active = sum(1 for t in tracker.active_tracks if t.is_confirmed)
                self._draw_hud_banner(
                    annotated_frame,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    fps=video_fps,
                    confirmed_tracks_count=confirmed_active,
                    total_frame_detections=total_frame_detections
                )

                # 4. Write frame
                writer.write(annotated_frame)

                # 5. Update progress metrics
                elapsed = time.time() - start_time
                fps_proc = processed_frames_count / elapsed if elapsed > 0 else 0.0
                remaining_frames = max(0, total_frames - frame_idx)
                est_remaining = remaining_frames / (fps_proc * frame_stride) if fps_proc > 0 else 0.0
                percent = min(100.0, (frame_idx / total_frames) * 100.0) if total_frames > 0 else 0.0

                confirmed_count = len(tracker.confirmed_history) + confirmed_active
                transient_count = len(tracker.transient_history) + (len(tracker.active_tracks) - confirmed_active)

                self.progress_state.update({
                    "percent": round(percent, 1),
                    "current_frame": frame_idx,
                    "fps_processing": round(fps_proc, 1),
                    "time_elapsed_sec": round(elapsed, 1),
                    "estimated_remaining_sec": round(est_remaining, 1),
                    "total_frame_detections": total_frame_detections,
                    "unique_defects_count": confirmed_count,
                    "transient_defects_count": transient_count
                })

                if progress_callback and (frame_idx % 10 == 0 or frame_idx == total_frames):
                    progress_callback(self.progress_state)

        except Exception as e:
            self.progress_state["status"] = "failed"
            self.progress_state["error"] = str(e)
            raise
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        # Finalize tracking session (returns confirmed and transient defects)
        all_unique_defects = tracker.finalize()
        total_duration = time.time() - start_time

        confirmed_defects = [d for d in all_unique_defects if d.get("is_confirmed", False)]
        transient_defects = [d for d in all_unique_defects if not d.get("is_confirmed", False)]

        # Calculate summary distributions across confirmed defects (or all if none confirmed)
        eval_set = confirmed_defects if confirmed_defects else all_unique_defects

        class_dist = {code: 0 for code in config.CLASS_CODES.values()}
        sev_dist = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
        highest_score = 0.0
        highest_level = "LOW"
        conf_sum = 0.0

        for d in eval_set:
            code = d.get("class_code", "OTHER")
            class_dist[code] = class_dist.get(code, 0) + 1

            level = d.get("severity_level", "LOW")
            sev_dist[level] = sev_dist.get(level, 0) + 1

            score = float(d.get("severity_score", 0.0))
            if score > highest_score:
                highest_score = score
                highest_level = level

            conf_sum += float(d.get("confidence", 0.0))

        avg_conf = conf_sum / len(eval_set) if eval_set else 0.0

        self.progress_state["status"] = "completed"
        self.progress_state["percent"] = 100.0

        summary = {
            "output_path": actual_output_path,
            "total_frames_processed": frame_idx,
            "total_frame_detections": total_frame_detections,
            "unique_defects_count": len(confirmed_defects),
            "transient_defects_count": len(transient_defects),
            "total_unique_records": len(all_unique_defects),
            "unique_defects": all_unique_defects,
            "confirmed_defects": confirmed_defects,
            "transient_defects": transient_defects,
            "damage_type_distribution": class_dist,
            "severity_distribution": sev_dist,
            "highest_severity_score": round(highest_score, 1),
            "highest_severity_level": highest_level,
            "avg_confidence": round(avg_conf, 3),
            "processing_duration_sec": round(total_duration, 2),
            "video_duration_sec": metadata["duration_sec"],
            "video_fps": video_fps,
            "resolution": (width, height)
        }

        return summary

    def _draw_hud_banner(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        confirmed_tracks_count: int,
        total_frame_detections: int
    ):
        """Draw an unobtrusive analytics banner across the top of the video frame."""
        h, w = frame.shape[:2]
        banner_h = 36

        # Semi-transparent dark banner overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 17, 28), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Cyan accent line
        cv2.line(frame, (0, banner_h), (w, banner_h), (6, 182, 212), 1)

        # Header information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (240, 240, 240)

        t1 = f"ROADGUARD AI v2.1 | Frame: {frame_idx}/{total_frames} ({fps:.0f} FPS)"
        t2 = f"Confirmed Tracks: {confirmed_tracks_count} | Detections: {total_frame_detections}"

        cv2.putText(frame, t1, (15, 23), font, font_scale, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(t2, font, font_scale, 1)
        cv2.putText(frame, t2, (max(15, w - tw - 15), 23), font, font_scale, (6, 182, 212), 1, cv2.LINE_AA)
