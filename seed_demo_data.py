"""
ROADGUARD AI - Initial Demonstration Seeder
Populates initial realistic road inspection entries into roadguard.db for university demonstration.
"""

from datetime import datetime, timedelta
import database
import config


def seed_demo_analyses():
    database.init_db()

    # Analysis 1: Dashcam Video Analysis with multiple tracked defects
    id_1 = "VID-20261115-Highway101"
    detections_1 = [
        {
            "track_id": 1,
            "class_code": "D40",
            "class_name": "Pothole",
            "confidence": 0.91,
            "severity_score": 93.1,
            "severity_level": "CRITICAL",
            "box": [620, 680, 850, 890],
            "frame_index": 42,
            "timestamp_sec": 1.40,
            "reason": "Classified as CRITICAL (Score 93.1/100) based on: Pothole (D40) baseline risk of 35 pts; relative visible area 4.12% (35.0 pts); lane proximity index 0.73 (10.6 pts); aspect ratio 1.10 (10.0 pts); temporal persistence (+3.0 pts)."
        },
        {
            "track_id": 2,
            "class_code": "D20",
            "class_name": "Alligator Crack",
            "confidence": 0.84,
            "severity_score": 79.0,
            "severity_level": "HIGH",
            "box": [450, 580, 780, 790],
            "frame_index": 78,
            "timestamp_sec": 2.60,
            "reason": "Classified as HIGH (Score 79.0/100) based on: Alligator Crack (D20) baseline risk of 25 pts; relative visible area 2.45% (27.2 pts); lane proximity index 0.63 (8.0 pts); aspect ratio 1.57 (6.0 pts)."
        },
        {
            "track_id": 3,
            "class_code": "D10",
            "class_name": "Transverse Crack",
            "confidence": 0.79,
            "severity_score": 57.4,
            "severity_level": "MODERATE",
            "box": [400, 520, 880, 590],
            "frame_index": 115,
            "timestamp_sec": 3.83,
            "reason": "Classified as MODERATE (Score 57.4/100) based on: Transverse Crack (D10) baseline risk of 16 pts; relative visible area 1.12% (17.7 pts); lane proximity index 0.51 (5.3 pts); aspect ratio 6.85 (10.0 pts)."
        }
    ]

    record_1 = {
        "id": id_1,
        "created_at": (datetime.now() - timedelta(minutes=45)).isoformat(),
        "media_type": "video",
        "filename": "highway101_dashcam_segment4.mp4",
        "source_path": "/uploads/highway101_dashcam_segment4.mp4",
        "output_path": "/outputs/annotated_sample_dashcam.mp4",
        "total_defects": 38,
        "unique_defects": 3,
        "critical_count": 1,
        "high_count": 1,
        "moderate_count": 1,
        "low_count": 0,
        "max_severity_score": 93.1,
        "max_severity_level": "CRITICAL",
        "avg_confidence": 0.847,
        "processing_time_sec": 4.12,
        "status": "completed",
        "metadata": {
            "video_fps": 30.0,
            "video_duration_sec": 8.5,
            "total_frames_processed": 255,
            "damage_type_distribution": {"D00": 0, "D10": 1, "D20": 1, "D40": 1, "OTHER": 0},
            "severity_distribution": {"CRITICAL": 1, "HIGH": 1, "MODERATE": 1, "LOW": 0}
        }
    }
    database.save_analysis(record_1, detections_1)

    # Analysis 2: Road Image with longitudinal cracking
    id_2 = "IMG-20261115-Sector7"
    detections_2 = [
        {
            "track_id": None,
            "class_code": "D00",
            "class_name": "Longitudinal Crack",
            "confidence": 0.72,
            "severity_score": 26.4,
            "severity_level": "LOW",
            "box": [820, 410, 860, 580],
            "frame_index": None,
            "timestamp_sec": None,
            "reason": "Classified as LOW (Score 26.4/100) based on: Longitudinal Crack (D00) baseline risk of 10 pts; relative visible area 0.35% (8.2 pts); lane proximity index 0.46 (4.2 pts); aspect ratio 4.25 (10.0 pts)."
        }
    ]

    record_2 = {
        "id": id_2,
        "created_at": (datetime.now() - timedelta(minutes=15)).isoformat(),
        "media_type": "image",
        "filename": "sector7_arterial_road.jpg",
        "source_path": "/uploads/sector7_arterial_road.jpg",
        "output_path": "/outputs/annotated_sector7.jpg",
        "total_defects": 1,
        "unique_defects": 1,
        "critical_count": 0,
        "high_count": 0,
        "moderate_count": 0,
        "low_count": 1,
        "max_severity_score": 26.4,
        "max_severity_level": "LOW",
        "avg_confidence": 0.72,
        "processing_time_sec": 0.08,
        "status": "completed",
        "metadata": {
            "confidence_threshold": 0.25,
            "damage_type_distribution": {"D00": 1, "D10": 0, "D20": 0, "D40": 0, "OTHER": 0},
            "severity_distribution": {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 1}
        }
    }
    database.save_analysis(record_2, detections_2)
    print("Database successfully seeded with realistic sample records.")


if __name__ == "__main__":
    seed_demo_analyses()
