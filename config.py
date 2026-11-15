"""
ROADGUARD AI - Configuration Module
Intelligent Road Damage Detection & Severity Assessment
"""
import os
import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Application Metadata
APP_NAME = "ROADGUARD AI"
APP_SUBTITLE = "Intelligent Road Damage Detection & Severity Assessment"
APP_VERSION = "2.0.0"

# Storage Directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
DATABASE_PATH = os.path.join(BASE_DIR, "roadguard.db")

# Ensure required directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, REPORT_FOLDER, MODEL_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# File Validation
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH = 150 * 1024 * 1024  # 150 MB

# Hardware Acceleration
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"

# YOLO Model Configuration
MODEL_WEIGHTS_PATH = os.path.join(MODEL_FOLDER, "best.pt")
FALLBACK_MODEL_NAME = "yolo11n.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640

# Road Damage Classes (Standardized across dataset, training & inference)
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

# Alias map to handle historical/variant names safely (e.g. "Potholes" vs "Pothole")
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

# Severity Engine Configuration
SEVERITY_LEVELS = {
    "LOW": {"min": 0, "max": 29, "color": "#10B981", "badge": "success"},
    "MODERATE": {"min": 30, "max": 59, "color": "#F59E0B", "badge": "warning"},
    "HIGH": {"min": 60, "max": 79, "color": "#F97316", "badge": "orange"},
    "CRITICAL": {"min": 80, "max": 100, "color": "#EF4444", "badge": "danger"}
}

# Category Base Impact (0-40 pts of the 100-pt severity score)
CLASS_BASE_SEVERITY = {
    "D40": 40,   # Potholes pose immediate tire/suspension hazard
    "D20": 32,   # Alligator cracks indicate structural roadbed failure
    "D10": 22,   # Transverse cracks cause vehicle jolt & water intrusion
    "D00": 18,   # Longitudinal cracks represent early surface fatigue
    "OTHER": 15  # Miscellaneous road distress
}

# Object Tracking Parameters
TRACKER_MAX_AGE = 15          # Max consecutive frames to keep missing defect active
TRACKER_MIN_HITS = 2          # Min hits before confirming unique defect
TRACKER_IOU_THRESHOLD = 0.3   # IoU threshold for spatial association
