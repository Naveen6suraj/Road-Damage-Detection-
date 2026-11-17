"""
ROADGUARD AI - Configuration Module
Intelligent Road Damage Detection & Severity Assessment
Version: 2.1.1 (Audited Release)
"""
import os
import torch
import secrets

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Application Metadata
APP_NAME = "ROADGUARD AI"
APP_SUBTITLE = "FIELD INTELLIGENCE — Road Damage & Severity Platform"
APP_VERSION = "3.1.0"

# Environment and Secret Key Security
ENV = os.environ.get("ROADGUARD_ENV", os.environ.get("FLASK_ENV", "development")).strip().lower()
_env_secret = os.environ.get("ROADGUARD_SECRET_KEY")

if ENV == "production":
    if not _env_secret:
        raise RuntimeError(
            "CRITICAL SECURITY CONFIGURATION ERROR: ROADGUARD_SECRET_KEY environment variable "
            "must be explicitly set when running in production mode."
        )
    SECRET_KEY = _env_secret
else:
    SECRET_KEY = _env_secret if _env_secret else secrets.token_hex(32)

# Storage Directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
DATABASE_PATH = os.path.join(BASE_DIR, "roadguard.db")
DEMO_DATABASE_PATH = os.path.join(BASE_DIR, "roadguard_demo.db")

# Ensure required directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, REPORT_FOLDER, MODEL_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# File Validation & Security
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH = 150 * 1024 * 1024  # 150 MB

# Hardware Acceleration
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"

# YOLO Model Configuration
MODEL_WEIGHTS_PATH = os.path.join(MODEL_FOLDER, "best.pt")
FALLBACK_MODEL_NAME = "yolo11n.pt"  # Pretrained base for transfer learning ONLY
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640

# Strict Road Damage Classes
CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Other Corruption"
}

CLASS_CODES = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
    4: "OTHER"
}

# Accepted Class Aliases for Custom Models
CLASS_ALIAS_MAP = {
    "longitudinal crack": 0,
    "longitudinal": 0,
    "d00": 0,
    "transverse crack": 1,
    "transverse": 1,
    "d10": 1,
    "alligator crack": 2,
    "alligator": 2,
    "d20": 2,
    "pothole": 3,
    "potholes": 3,
    "d40": 3,
    "other corruption": 4,
    "other": 4,
    "d43": 4,
    "d44": 4,
    "d50": 4
}

# Standard Road Damage Class Names for Verification
VALID_ROAD_DAMAGE_CLASSES = set(CLASS_ALIAS_MAP.keys())
# Mandatory classes required for a model to be validated as a legitimate road damage detector
REQUIRED_ROAD_DAMAGE_CODES = {"D00", "D10", "D20", "D40"}

# Severity Engine Configuration (Mathematical Formula S in [5, 100])
SEVERITY_LEVELS = {
    "LOW": {"min": 0.0, "max": 29.9, "color": "#10B981", "badge": "success"},
    "MODERATE": {"min": 30.0, "max": 59.9, "color": "#F59E0B", "badge": "warning"},
    "HIGH": {"min": 60.0, "max": 79.9, "color": "#F97316", "badge": "orange"},
    "CRITICAL": {"min": 80.0, "max": 100.0, "color": "#EF4444", "badge": "danger"}
}

# Category Base Hazard Points (W_cat in [8.0, 35.0])
CLASS_BASE_HAZARD = {
    "D40": 35.0,   # Pothole: Direct tire/suspension hazard
    "D20": 25.0,   # Alligator Crack: Pavement base course fatigue
    "D10": 16.0,   # Transverse Crack: Thermal stress, perpendicular jolt
    "D00": 10.0,   # Longitudinal Crack: Longitudinal joint wear
    "OTHER": 8.0   # Surface corruption, ravelling, weathering
}

# Object Tracking Parameters
TRACKER_MAX_AGE = 15          # Consecutive missing frames before track deactivation
TRACKER_MIN_HITS = 2          # Threshold to promote track to Confirmed status
TRACKER_IOU_THRESHOLD = 0.25   # IoU threshold for spatial association

# Temporal Persistence Modifiers
PERSISTENCE_THRESHOLD_HITS = 5  # Min hits for persistent defect bonus
PERSISTENCE_BONUS = 3.0         # Added to score for confirmed persistent defects
TRANSIENT_DISCOUNT = -2.0       # Subtracted for 1-frame transient observations in video


def get_severity_level(score: float) -> str:
    """Classify numerical severity score into standard tier."""
    if score >= 80.0:
        return "CRITICAL"
    elif score >= 60.0:
        return "HIGH"
    elif score >= 30.0:
        return "MODERATE"
    return "LOW"

