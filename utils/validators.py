"""
ROADGUARD AI - Validation Utilities
Module: utils/validators.py
"""

import os
from typing import Any
from werkzeug.datastructures import FileStorage
import config


def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if filename has a permitted extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions


def validate_image_upload(file: FileStorage) -> str:
    """Validate uploaded image file."""
    if not file or not file.filename:
        raise ValueError("No file uploaded or empty filename.")
    if not is_allowed_file(file.filename, config.ALLOWED_IMAGE_EXTENSIONS):
        exts = ", ".join(sorted(config.ALLOWED_IMAGE_EXTENSIONS))
        raise ValueError(f"Invalid image format. Allowed formats: {exts}")
    return file.filename


def validate_video_upload(file: FileStorage) -> str:
    """Validate uploaded video file."""
    if not file or not file.filename:
        raise ValueError("No video file uploaded or empty filename.")
    if not is_allowed_file(file.filename, config.ALLOWED_VIDEO_EXTENSIONS):
        exts = ", ".join(sorted(config.ALLOWED_VIDEO_EXTENSIONS))
        raise ValueError(f"Invalid video format. Allowed formats: {exts}")
    return file.filename


def parse_confidence(val: Any, default: float = config.DEFAULT_CONFIDENCE) -> float:
    """Parse and clamp confidence threshold between 0.05 and 0.95."""
    try:
        conf = float(val)
        return max(0.05, min(0.95, conf))
    except (TypeError, ValueError):
        return default
