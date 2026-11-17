"""
ROADGUARD AI - Demonstration Data Seeder
Module: seed_demo_data.py
Version: 2.1.1

Populates synthetic, realistic road condition inspection records for academic demonstration.
Explicitly tags all entries with is_demo = 1 and [DEMO / SYNTHETIC] prefixes.

Usage:
    python seed_demo_data.py                  # Seed demo data into roadguard_demo.db (default)
    python seed_demo_data.py --target production # Seed demo data into roadguard.db
    python seed_demo_data.py --db-path <path> # Target custom database path
"""

import sys
import argparse
import os
from datetime import datetime, timedelta
import config
import database


def seed_demo_analyses(target_db: str = None):
    target_path = target_db or config.DEMO_DATABASE_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    database.init_db(target_path)

    # -------------------------------------------------------------------------
    # Analysis 1: Dashcam Video Analysis with Confirmed and Transient Defects
    # -------------------------------------------------------------------------
    id_1 = "VID-20261115-DEMO-HWY101"
    detections_1 = [
        {
            "track_id": 1,
            "class_code": "D40",
            "class_name": "Pothole",
            "confidence": 0.92,
            "raw_severity_score": 90.1,
            "raw_severity_level": "CRITICAL",
            "severity_score": 93.1,
            "severity_level": "CRITICAL",
            "is_confirmed": True,
            "status": "confirmed",
            "box": [620, 680, 850, 890],
            "frame_index": 42,
            "timestamp_sec": 1.40,
            "reason": "Classified as CRITICAL (Score 93.1/100) based on: Pothole (D40) baseline risk of 35 pts; relative visible area 4.12% (35.0 pts); lane proximity index 0.73 (10.6 pts); aspect ratio 1.10 (10.0 pts); temporal persistence bonus (+3.0 pts)."
        },
        {
            "track_id": 2,
            "class_code": "D20",
            "class_name": "Alligator Crack",
            "confidence": 0.85,
            "raw_severity_score": 79.0,
            "raw_severity_level": "HIGH",
            "severity_score": 79.0,
            "severity_level": "HIGH",
            "is_confirmed": True,
            "status": "confirmed",
            "box": [450, 580, 780, 790],
            "frame_index": 78,
            "timestamp_sec": 2.60,
            "reason": "Classified as HIGH (Score 79.0/100) based on: Alligator Crack (D20) baseline risk of 25 pts; relative visible area 2.45% (27.2 pts); lane proximity index 0.63 (8.0 pts); aspect ratio 1.57 (6.0 pts)."
        },
        {
            "track_id": 3,
            "class_code": "D10",
            "class_name": "Transverse Crack",
            "confidence": 0.62,
            "raw_severity_score": 42.0,
            "raw_severity_level": "MODERATE",
            "severity_score": 40.0,
            "severity_level": "MODERATE",
            "is_confirmed": False,
            "status": "transient",
            "box": [300, 320, 480, 360],
            "frame_index": 120,
            "timestamp_sec": 4.00,
            "reason": "Classified as MODERATE (Score 40.0/100) based on: Transverse Crack (D10) baseline risk of 16 pts; single-frame transient discount (-2.0 pts)."
        }
    ]

    record_1 = {
        "id": id_1,
        "created_at": (datetime.now() - timedelta(minutes=45)).isoformat(),
        "media_type": "video",
        "filename": "[DEMO / SYNTHETIC] Highway 101 Dashcam Survey - Sector B",
        "source_path": "/uploads/highway101_dashcam_segment4.mp4",
        "output_path": "/outputs/annotated_sample_dashcam.mp4",
        "total_defects": 42,
        "unique_defects": 2,
        "transient_defects": 1,
        "critical_count": 1,
        "high_count": 1,
        "moderate_count": 1,
        "low_count": 0,
        "max_severity_score": 93.1,
        "max_severity_level": "CRITICAL",
        "avg_confidence": 0.797,
        "processing_time_sec": 4.12,
        "status": "completed",
        "is_demo": 1,
        "metadata": {
            "is_demo": True,
            "video_fps": 30.0,
            "video_duration_sec": 8.5,
            "total_frames_processed": 255,
            "damage_type_distribution": {"D00": 0, "D10": 1, "D20": 1, "D40": 1, "OTHER": 0},
            "severity_distribution": {"CRITICAL": 1, "HIGH": 1, "MODERATE": 1, "LOW": 0}
        }
    }
    database.save_analysis(record_1, detections_1, db_path=target_path)

    # -------------------------------------------------------------------------
    # Analysis 2: Static Road Image with Longitudinal & Transverse Cracks
    # -------------------------------------------------------------------------
    id_2 = "IMG-20261115-DEMO-SECTOR7"
    detections_2 = [
        {
            "track_id": None,
            "class_code": "D00",
            "class_name": "Longitudinal Crack",
            "confidence": 0.76,
            "raw_severity_score": 26.4,
            "raw_severity_level": "LOW",
            "severity_score": 26.4,
            "severity_level": "LOW",
            "is_confirmed": True,
            "status": "confirmed",
            "box": [820, 410, 860, 580],
            "frame_index": None,
            "timestamp_sec": None,
            "reason": "Classified as LOW (Score 26.4/100) based on: Longitudinal Crack (D00) baseline risk of 10 pts; relative visible area 0.35% (8.2 pts); lane proximity index 0.46 (4.2 pts); aspect ratio 4.25 (10.0 pts)."
        },
        {
            "track_id": None,
            "class_code": "D10",
            "class_name": "Transverse Crack",
            "confidence": 0.81,
            "raw_severity_score": 54.2,
            "raw_severity_level": "MODERATE",
            "severity_score": 54.2,
            "severity_level": "MODERATE",
            "is_confirmed": True,
            "status": "confirmed",
            "box": [350, 500, 750, 560],
            "frame_index": None,
            "timestamp_sec": None,
            "reason": "Classified as MODERATE (Score 54.2/100) based on: Transverse Crack (D10) baseline risk of 16 pts; relative visible area 1.05% (17.2 pts); lane proximity index 0.52 (5.5 pts); aspect ratio 6.67 (10.0 pts)."
        }
    ]

    record_2 = {
        "id": id_2,
        "created_at": (datetime.now() - timedelta(minutes=15)).isoformat(),
        "media_type": "image",
        "filename": "[DEMO / SYNTHETIC] Sector 7 Arterial Pavement Survey",
        "source_path": "/uploads/sector7_arterial_road.jpg",
        "output_path": "/outputs/annotated_sector7.jpg",
        "total_defects": 2,
        "unique_defects": 2,
        "transient_defects": 0,
        "critical_count": 0,
        "high_count": 0,
        "moderate_count": 1,
        "low_count": 1,
        "max_severity_score": 54.2,
        "max_severity_level": "MODERATE",
        "avg_confidence": 0.785,
        "processing_time_sec": 0.08,
        "status": "completed",
        "is_demo": 1,
        "metadata": {
            "is_demo": True,
            "confidence_threshold": 0.25,
            "damage_type_distribution": {"D00": 1, "D10": 1, "D20": 0, "D40": 0, "OTHER": 0},
            "severity_distribution": {"CRITICAL": 0, "HIGH": 0, "MODERATE": 1, "LOW": 1}
        }
    }
    database.save_analysis(record_2, detections_2, db_path=target_path)

    print(f"[ROADGUARD AI] Successfully seeded demo analyses into: {target_path}")


def main():
    parser = argparse.ArgumentParser(description="ROADGUARD AI Demo Data Seeder")
    parser.add_argument(
        "--target",
        choices=["demo", "production"],
        default="demo",
        help="Target database: 'demo' (roadguard_demo.db) or 'production' (roadguard.db)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Custom target database path"
    )

    args = parser.parse_args()
    if args.db_path:
        target = args.db_path
    elif args.target == "production":
        target = config.DATABASE_PATH
    else:
        target = config.DEMO_DATABASE_PATH

    seed_demo_analyses(target)


if __name__ == "__main__":
    main()
