"""
ROADGUARD AI - Database Management Module
Module: database.py
Version: 2.1.0

SQLite persistent storage for analyses and defect detections.
Includes WAL mode for concurrency, explicit demo data tracking,
and schema initialization routines.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import config


def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """Return thread-safe SQLite connection with WAL journal mode."""
    target_path = db_path or config.DATABASE_PATH
    conn = sqlite3.connect(target_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Configure SQLite for robust concurrent reads/writes
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: str = None):
    """Initialize relational database tables if they do not exist."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Analyses Master Table
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
            transient_defects INTEGER DEFAULT 0,
            critical_count INTEGER DEFAULT 0,
            high_count INTEGER DEFAULT 0,
            moderate_count INTEGER DEFAULT 0,
            low_count INTEGER DEFAULT 0,
            max_severity_score REAL DEFAULT 0.0,
            max_severity_level TEXT DEFAULT 'NONE',
            avg_confidence REAL DEFAULT 0.0,
            processing_time_sec REAL DEFAULT 0.0,
            status TEXT DEFAULT 'completed',
            is_demo INTEGER DEFAULT 0,
            metadata_json TEXT
        )
    """)

    # Defect Detections & Tracking Records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            track_id INTEGER,
            class_code TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            raw_severity_score REAL DEFAULT 0.0,
            raw_severity_level TEXT DEFAULT 'LOW',
            severity_score REAL NOT NULL,
            severity_level TEXT NOT NULL,
            is_confirmed INTEGER DEFAULT 1,
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

    # Ensure all v2.1.0 columns exist if table was initialized with previous schema
    analyses_cols = [c[1] for c in cursor.execute("PRAGMA table_info(analyses)").fetchall()]
    if "is_demo" not in analyses_cols:
        cursor.execute("ALTER TABLE analyses ADD COLUMN is_demo INTEGER DEFAULT 0")
    if "transient_defects" not in analyses_cols:
        cursor.execute("ALTER TABLE analyses ADD COLUMN transient_defects INTEGER DEFAULT 0")

    det_cols = [c[1] for c in cursor.execute("PRAGMA table_info(detections)").fetchall()]
    if "raw_severity_score" not in det_cols:
        cursor.execute("ALTER TABLE detections ADD COLUMN raw_severity_score REAL DEFAULT 0.0")
    if "raw_severity_level" not in det_cols:
        cursor.execute("ALTER TABLE detections ADD COLUMN raw_severity_level TEXT DEFAULT 'LOW'")
    if "is_confirmed" not in det_cols:
        cursor.execute("ALTER TABLE detections ADD COLUMN is_confirmed INTEGER DEFAULT 1")

    conn.commit()
    conn.close()


def clear_all_records(db_path: str = None):
    """Wipe all records for a pristine production deployment."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    cursor.execute("DELETE FROM analyses")
    conn.commit()
    conn.close()


def save_analysis(analysis_data: dict, detections: list = None, db_path: str = None):
    """Save an analysis session and its associated defect detections."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    metadata_str = json.dumps(analysis_data.get("metadata", {}))

    cursor.execute("""
        INSERT OR REPLACE INTO analyses (
            id, created_at, media_type, filename, source_path, output_path,
            total_defects, unique_defects, transient_defects, critical_count, high_count,
            moderate_count, low_count, max_severity_score, max_severity_level,
            avg_confidence, processing_time_sec, status, is_demo, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        analysis_data["id"],
        analysis_data.get("created_at", datetime.now().isoformat()),
        analysis_data["media_type"],
        analysis_data["filename"],
        analysis_data.get("source_path", ""),
        analysis_data.get("output_path", ""),
        analysis_data.get("total_defects", 0),
        analysis_data.get("unique_defects", 0),
        analysis_data.get("transient_defects", 0),
        analysis_data.get("critical_count", 0),
        analysis_data.get("high_count", 0),
        analysis_data.get("moderate_count", 0),
        analysis_data.get("low_count", 0),
        analysis_data.get("max_severity_score", 0.0),
        analysis_data.get("max_severity_level", "NONE"),
        analysis_data.get("avg_confidence", 0.0),
        analysis_data.get("processing_time_sec", 0.0),
        analysis_data.get("status", "completed"),
        1 if analysis_data.get("is_demo", False) else 0,
        metadata_str
    ))

    if detections:
        for d in detections:
            factors_str = json.dumps(d.get("contributing_factors", {}))
            box = d.get("box", [0, 0, 0, 0])
            if not box and "box_x1" in d:
                box = [d.get("box_x1", 0), d.get("box_y1", 0), d.get("box_x2", 0), d.get("box_y2", 0)]

            is_conf = 1 if d.get("is_confirmed", True) else 0
            raw_score = float(d.get("raw_severity_score", d.get("severity_score", 0.0)))
            raw_lvl = d.get("raw_severity_level", config.get_severity_level(raw_score))
            adj_score = float(d.get("severity_score", 0.0))
            adj_lvl = d.get("severity_level", config.get_severity_level(adj_score))

            cursor.execute("""
                INSERT INTO detections (
                    analysis_id, track_id, class_code, class_name, confidence,
                    raw_severity_score, raw_severity_level, severity_score, severity_level, is_confirmed,
                    box_x1, box_y1, box_x2, box_y2, frame_index, timestamp_sec,
                    reason, contributing_factors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_data["id"],
                d.get("track_id"),
                d.get("class_code", "OTHER"),
                d.get("class_name", "Unknown"),
                float(d.get("confidence", 0.0)),
                raw_score,
                raw_lvl,
                adj_score,
                adj_lvl,
                is_conf,
                box[0], box[1], box[2], box[3],
                d.get("frame_index"),
                d.get("timestamp_sec", d.get("first_timestamp_sec")),
                d.get("reason", ""),
                factors_str
            ))

    conn.commit()
    conn.close()


def get_analysis(analysis_id: str, db_path: str = None) -> Optional[dict]:
    """Retrieve single analysis with parsed metadata."""
    conn = get_db_connection(db_path)
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


def get_analysis_detections(analysis_id: str, db_path: str = None) -> list:
    """Retrieve all defect detections associated with an analysis session."""
    conn = get_db_connection(db_path)
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
        item["box"] = [item["box_x1"], item["box_y1"], item["box_x2"], item["box_y2"]]
        item["is_confirmed"] = bool(item.get("is_confirmed", 1))
        detections.append(item)
    return detections


def get_all_analyses(limit: int = 50, include_demo: bool = True, db_path: str = None) -> list:
    """Retrieve list of recent analyses."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    query = "SELECT * FROM analyses"
    if not include_demo:
        query += " WHERE is_demo = 0"
    query += " ORDER BY created_at DESC LIMIT ?"

    rows = cursor.execute(query, (limit,)).fetchall()
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


def get_dashboard_stats(include_demo: bool = True, db_path: str = None) -> dict:
    """Compute aggregate dashboard metrics from real database records."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    where_clause = "" if include_demo else "WHERE is_demo = 0"

    row = cursor.execute(f"""
        SELECT 
            COUNT(*) as total_analyses,
            COALESCE(SUM(CASE WHEN media_type = 'video' THEN 1 ELSE 0 END), 0) as total_videos,
            COALESCE(SUM(CASE WHEN media_type = 'image' THEN 1 ELSE 0 END), 0) as total_images,
            COALESCE(SUM(unique_defects), 0) as total_defects,
            COALESCE(SUM(transient_defects), 0) as total_transient,
            COALESCE(SUM(critical_count), 0) as critical_defects,
            COALESCE(SUM(high_count), 0) as high_defects,
            COALESCE(SUM(moderate_count), 0) as moderate_defects,
            COALESCE(SUM(low_count), 0) as low_defects,
            COALESCE(AVG(avg_confidence), 0.0) as overall_avg_confidence
        FROM analyses
        {where_clause}
    """).fetchone()

    # Class distribution query
    join_clause = "" if include_demo else "JOIN analyses a ON d.analysis_id = a.id WHERE a.is_demo = 0"
    class_rows = cursor.execute(f"""
        SELECT d.class_code, COUNT(*) as count 
        FROM detections d
        {join_clause}
        GROUP BY d.class_code
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


# Initialize schema upon module import
init_db()
