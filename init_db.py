"""
ROADGUARD AI - Database Initialization Script
Module: init_db.py
Version: 2.1.0

Initializes the SQLite schema for ROADGUARD AI.
Supports fresh database creation, migration checks, and optional database wipe.

Usage:
    python init_db.py              # Initialize schema if tables don't exist
    python init_db.py --reset      # Wipe existing tables and create fresh schema
    python init_db.py --db-path <path>  # Target custom database path
"""

import sys
import argparse
import os
import config
import database


def main():
    parser = argparse.ArgumentParser(description="ROADGUARD AI Database Initializer")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe all existing records and rebuild empty schema"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database file (defaults to config.DATABASE_PATH)"
    )

    args = parser.parse_args()
    target_db = args.db_path or config.DATABASE_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_db)), exist_ok=True)

    print(f"[ROADGUARD AI] Target Database: {target_db}")

    if args.reset:
        print("[ROADGUARD AI] Reset requested. Wiping existing records...")
        database.init_db(db_path=target_db)
        database.clear_all_records(db_path=target_db)
        print("[ROADGUARD AI] All tables cleared successfully.")
    else:
        database.init_db(db_path=target_db)
        print("[ROADGUARD AI] Schema verification / initialization complete.")

    conn = database.get_db_connection(target_db)
    cursor = conn.cursor()
    analyses_count = cursor.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    detections_count = cursor.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    conn.close()

    print(f"[ROADGUARD AI] Database Status: READY | Analyses: {analyses_count} | Detections: {detections_count}")


if __name__ == "__main__":
    main()
