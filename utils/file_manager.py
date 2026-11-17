"""
ROADGUARD AI - File Manager Utilities
Module: utils/file_manager.py
"""

import os
import uuid
import time
from werkzeug.utils import secure_filename
import config


def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    """Generate a collision-free unique filename preserving original extension."""
    safe_name = secure_filename(original_filename)
    name, ext = os.path.splitext(safe_name)
    if not ext:
        ext = ".bin"
    uid = uuid.uuid4().hex[:12]
    ts = int(time.time())
    pfx = f"{prefix}_" if prefix else ""
    return f"{pfx}{ts}_{uid}{ext}"


def get_upload_path(filename: str) -> str:
    """Return absolute path inside uploads directory."""
    return os.path.join(config.UPLOAD_FOLDER, filename)


def get_output_path(filename: str) -> str:
    """Return absolute path inside outputs directory."""
    return os.path.join(config.OUTPUT_FOLDER, filename)


def get_report_path(filename: str) -> str:
    """Return absolute path inside reports directory."""
    return os.path.join(config.REPORT_FOLDER, filename)


def cleanup_old_files(folder: str, max_age_seconds: int = 86400):
    """Safely purge stale files older than max_age_seconds."""
    if not os.path.exists(folder):
        return
    now = time.time()
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            try:
                if now - os.path.getmtime(fpath) > max_age_seconds:
                    os.remove(fpath)
            except Exception:
                pass
