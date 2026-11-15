"""
Unit Tests: Report Generation (PDF & CSV)
File: tests/test_reports.py
"""

import os
import csv
import pytest
from core.report_generator import generate_csv_report, generate_pdf_report
import config


def test_generate_csv_report(tmp_path):
    out_csv = str(tmp_path / "test_report.csv")
    analysis_data = {
        "id": "TEST-001",
        "created_at": "2026-11-15T10:00:00",
        "media_type": "image",
        "filename": "road_sample.jpg"
    }
    detections = [
        {
            "track_id": 1,
            "class_code": "D40",
            "class_name": "Pothole",
            "confidence": 0.88,
            "severity_score": 85.0,
            "severity_level": "CRITICAL",
            "box": [100, 200, 300, 400],
            "reason": "Large pothole in wheel path"
        }
    ]

    res_path = generate_csv_report(analysis_data, detections, out_csv)
    assert os.path.exists(res_path)

    with open(res_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["Analysis_ID"] == "TEST-001"
        assert reader[0]["Class_Code"] == "D40"
        assert reader[0]["Severity_Level"] == "CRITICAL"


def test_generate_pdf_report(tmp_path):
    out_pdf = str(tmp_path / "test_report.pdf")
    analysis_data = {
        "id": "TEST-002",
        "created_at": "2026-11-15T10:00:00",
        "media_type": "video",
        "filename": "dashcam_clip.mp4",
        "unique_defects": 2,
        "max_severity_level": "HIGH",
        "avg_confidence": 0.82,
        "processing_time_sec": 3.45,
        "critical_count": 0,
        "high_count": 1,
        "moderate_count": 1,
        "low_count": 0,
        "metadata": {
            "damage_type_distribution": {"D40": 1, "D20": 1}
        }
    }
    detections = [
        {
            "track_id": 1,
            "class_code": "D40",
            "class_name": "Pothole",
            "confidence": 0.85,
            "severity_score": 75.0,
            "severity_level": "HIGH",
            "reason": "Noticeable depression"
        }
    ]

    res_path = generate_pdf_report(analysis_data, detections, out_pdf)
    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000
