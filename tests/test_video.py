"""
Unit Tests: Video Processing & Metadata
File: tests/test_video.py
"""

import os
import cv2
import numpy as np
import pytest
from core.video_processor import get_video_metadata, VideoProcessor


@pytest.fixture
def sample_video(tmp_path):
    """Generate a valid synthetic 15-frame MP4 video clip for testing."""
    vid_path = str(tmp_path / "test_dashcam.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    w, h = 640, 360
    out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    for i in range(15):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        # Draw roadway lane markings
        cv2.line(frame, (w // 2, 0), (w // 2, h), (200, 200, 200), 2)
        out.write(frame)
    out.release()
    return vid_path


def test_get_video_metadata(sample_video):
    meta = get_video_metadata(sample_video)
    assert meta["width"] == 640
    assert meta["height"] == 360
    assert meta["fps"] == 30.0
    assert meta["total_frames"] == 15
    assert meta["duration_sec"] > 0.0


def test_video_processor_pipeline(sample_video, tmp_path):
    out_path = str(tmp_path / "annotated_test.mp4")
    processor = VideoProcessor()

    summary = processor.process_video(
        input_path=sample_video,
        output_path=out_path,
        conf_threshold=0.25
    )

    assert os.path.exists(summary["output_path"])
    assert summary["total_frames_processed"] == 15
    assert "unique_defects_count" in summary
    assert "damage_type_distribution" in summary
    assert "severity_distribution" in summary
    assert summary["video_fps"] == 30.0


def test_corrupt_video_handling():
    with pytest.raises((FileNotFoundError, ValueError)):
        get_video_metadata("non_existent_video_file.mp4")
