"""
ROADGUARD AI - Database Management Module
SQLite persistent storage for analyses and defect detections.
"""
import sqlite3
import json
import os
from datetime import datetime
from config import DATABASE_PATH

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Analyses master table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            media_type TEXT NOT NULL,
            filename TEXT NOT NULL,
            source_path TEXT,
            output_path TEXT,
            total_defects INTEGER DEFAULT 0,
            unique_defects INTEGER DEFAULT 0,
            critical_count INTEGER DEFAULT 0,
            high_count INTEGER DEFAULT 0,
            moderate_count INTEGER DEFAULT 0,
            low_count INTEGER DEFAULT 0,
            max_severity_score REAL DEFAULT 0.0,
            max_severity_level TEXT DEFAULT 'NONE',
            avg_confidence REAL DEFAULT 0.0,
            processing_time_sec REAL DEFAULT 0.0,
            status TEXT DEFAULT 'completed',
            metadata_json TEXT
        )
    """)

    # Defect records table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            track_id INTEGER,
            class_code TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity_score REAL NOT NULL,
            severity_level TEXT NOT NULL,
            box_x1 REAL,
            box_y1 REAL,
            box_x2 REAL,
            box_y2 REAL,
            frame_index INTEGER,
            timestamp_sec REAL,
            reason TEXT,
            contributing_factors TEXT,
            FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()

def save_analysis(analysis_data, detections=None):
    """Save an analysis run and its associated defect detections."""
    conn = get_db_connection()
    cursor = conn.cursor()

    metadata_str = json.dumps(analysis_data.get("metadata", {}))

    cursor.execute("""
        INSERT OR REPLACE INTO analyses (
            id, created_at, media_type, filename, source_path, output_path,
            total_defects, unique_defects, critical_count, high_count,
            moderate_count, low_count, max_severity_score, max_severity_level,
            avg_confidence, processing_time_sec, status, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        analysis_data["id"],
        analysis_data.get("created_at", datetime.now().isoformat()),
        analysis_data["media_type"],
        analysis_data["filename"],
        analysis_data.get("source_path", ""),
        analysis_data.get("output_path", ""),
        analysis_data.get("total_defects", 0),
        analysis_data.get("unique_defects", 0),
        analysis_data.get("critical_count", 0),
        analysis_data.get("high_count", 0),
        analysis_data.get("moderate_count", 0),
        analysis_data.get("low_count", 0),
        analysis_data.get("max_severity_score", 0.0),
        analysis_data.get("max_severity_level", "NONE"),
        analysis_data.get("avg_confidence", 0.0),
        analysis_data.get("processing_time_sec", 0.0),
        analysis_data.get("status", "completed"),
        metadata_str
    ))

    if detections:
        for d in detections:
            factors_str = json.dumps(d.get("contributing_factors", {}))
            cursor.execute("""
                INSERT INTO detections (
                    analysis_id, track_id, class_code, class_name, confidence,
                    severity_score, severity_level, box_x1, box_y1, box_x2, box_y2,
                    frame_index, timestamp_sec, reason, contributing_factors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_data["id"],
                d.get("track_id"),
                d.get("class_code", "OTHER"),
                d.get("class_name", "Unknown"),
                float(d.get("confidence", 0.0)),
                float(d.get("severity_score", 0.0)),
                d.get("severity_level", "LOW"),
                d.get("box", [0, 0, 0, 0])[0] if "box" in d else d.get("box_x1"),
                d.get("box", [0, 0, 0, 0])[1] if "box" in d else d.get("box_y1"),
                d.get("box", [0, 0, 0, 0])[2] if "box" in d else d.get("box_x2"),
                d.get("box", [0, 0, 0, 0])[3] if "box" in d else d.get("box_y2"),
                d.get("frame_index"),
                d.get("timestamp_sec"),
                d.get("reason", ""),
                factors_str
            ))

    conn.commit()
    conn.close()

def get_analysis(analysis_id):
    """Retrieve single analysis with metadata."""
    conn = get_db_connection()
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    if d.get("metadata_json"):
        try:
            d["metadata"] = json.loads(d["metadata_json"])
        except Exception:
            d["metadata"] = {}
    return d

def get_analysis_detections(analysis_id):
    """Retrieve all detections associated with an analysis."""
    conn = get_db_connection()
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT * FROM detections WHERE analysis_id = ? ORDER BY id ASC", 
        (analysis_id,)
    ).fetchall()
    conn.close()
    detections = []
    for r in rows:
        item = dict(r)
        if item.get("contributing_factors"):
            try:
                item["contributing_factors"] = json.loads(item["contributing_factors"])
            except Exception:
                item["contributing_factors"] = {}
        detections.append(item)
    return detections

def get_all_analyses(limit=50):
    """Retrieve recent analyses list."""
    conn = get_db_connection()
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", 
        (limit,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        item = dict(r)
        if item.get("metadata_json"):
            try:
                item["metadata"] = json.loads(item["metadata_json"])
            except Exception:
                item["metadata"] = {}
        result.append(item)
    return result

def get_dashboard_stats():
    """Compute aggregate dashboard metrics from real database records."""
    conn = get_db_connection()
    cursor = conn.cursor()

    row = cursor.execute("""
        SELECT 
            COUNT(*) as total_analyses,
            COALESCE(SUM(CASE WHEN media_type = 'video' THEN 1 ELSE 0 END), 0) as total_videos,
            COALESCE(SUM(CASE WHEN media_type = 'image' THEN 1 ELSE 0 END), 0) as total_images,
            COALESCE(SUM(unique_defects), 0) as total_defects,
            COALESCE(SUM(critical_count), 0) as critical_defects,
            COALESCE(SUM(high_count), 0) as high_defects,
            COALESCE(SUM(moderate_count), 0) as moderate_defects,
            COALESCE(SUM(low_count), 0) as low_defects,
            COALESCE(AVG(avg_confidence), 0.0) as overall_avg_confidence
        FROM analyses
    """).fetchone()

    # Class distribution from all detections
    class_rows = cursor.execute("""
        SELECT class_code, class_name, COUNT(*) as count 
        FROM detections 
        GROUP BY class_code
    """).fetchall()

    class_distribution = {
        "D00": 0, "D10": 0, "D20": 0, "D40": 0, "OTHER": 0
    }
    for cr in class_rows:
        code = cr["class_code"]
        if code in class_distribution:
            class_distribution[code] = cr["count"]
        else:
            class_distribution["OTHER"] += cr["count"]

    conn.close()

    stats = dict(row) if row else {}
    stats["class_distribution"] = class_distribution
    stats["severity_distribution"] = {
        "CRITICAL": stats.get("critical_defects", 0),
        "HIGH": stats.get("high_defects", 0),
        "MODERATE": stats.get("moderate_defects", 0),
        "LOW": stats.get("low_defects", 0)
    }
    return stats

# Initialize database schema immediately on import
init_db()
