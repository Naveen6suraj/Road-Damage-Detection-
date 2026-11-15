"""
ROADGUARD AI - Severity Estimation Engine
Module: core/severity_engine.py

Provides an empirical, visual risk scoring methodology for road defects detected
in single-camera dashcam imagery and video streams.

ACADEMIC DISCLAIMER & METHODOLOGY NOTE:
---------------------------------------
Physical road defect metrics (such as true depth in millimeters or calibrated
width in centimeters) cannot be directly measured from monocular 2D images without
calibrated stereo camera geometry, LiDAR, or structured light sensors.
This severity engine computes an empirical "Visual Severity Risk Index" (0 - 100),
derived from:
1. Defect Category Base Hazard (e.g. pothole vs. superficial crack)
2. Normalized Relative Damage Area (defect area relative to frame resolution)
3. Perspective Road Proximity (dashcam horizon vs. near-field vehicle lane zone)
4. Defect Aspect Ratio / Geometric Distortion
5. Multi-Frame Temporal Persistence (when processing video streams)

Detection confidence is maintained as a strictly separate measure representing
model classification certainty, and is not equated to physical defect severity.
"""

from typing import Dict, Any, Optional, Tuple
import math

# Category Base Hazard Mapping (0 - 35 points)
CLASS_BASE_HAZARD = {
    "D40": 35.0,   # Pothole: Direct tire, wheel rim, and suspension impact risk
    "D20": 25.0,   # Alligator Crack: Extensive fatigue cracking, structural base failure
    "D10": 16.0,   # Transverse Crack: Thermal stress, perpendicular vehicle jolting
    "D00": 10.0,   # Longitudinal Crack: Longitudinal joint wear, lane-tracking fatigue
    "OTHER": 8.0   # Other surface corruption, ravelling, or weathering
}

# Human-readable category labels
CLASS_LABELS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "OTHER": "Other Corruption"
}

# Standardized Severity Classification Boundaries
SEVERITY_THRESHOLDS = {
    "CRITICAL": 80.0,
    "HIGH": 60.0,
    "MODERATE": 30.0,
    "LOW": 0.0
}


def classify_severity_score(score: float) -> str:
    """Map a 0-100 severity score to one of the four standardized severity levels."""
    if score >= SEVERITY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif score >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif score >= SEVERITY_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    else:
        return "LOW"


def estimate_severity(
    detection: Dict[str, Any],
    frame_dimensions: Tuple[int, int],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Estimate the empirical severity of a detected road defect.

    Parameters:
    -----------
    detection: dict
        Contains detection information:
        - 'box': [x1, y1, x2, y2] in pixels
        - 'class_code': 'D00', 'D10', 'D20', 'D40', or 'OTHER'
        - 'confidence': float (0.0 to 1.0)
    frame_dimensions: tuple (width, height)
        Dimensions of the video or image frame in pixels.
    context: dict, optional
        Additional contextual indicators (e.g. 'track_hits': int).

    Returns:
    --------
    dict containing:
        - severity_score: float (0.0 to 100.0)
        - severity_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
        - reason: str (human-readable justification)
        - contributing_factors: dict (detailed breakdown of scoring components)
    """
    frame_w, frame_h = frame_dimensions
    frame_w = max(1, frame_w)
    frame_h = max(1, frame_h)
    frame_area = frame_w * frame_h

    # Extract bounding box coordinates
    box = detection.get("box", [0, 0, 0, 0])
    x1, y1, x2, y2 = box
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    box_area = box_w * box_h

    # Extract class code & confidence
    class_code = detection.get("class_code", "OTHER").upper()
    confidence = float(detection.get("confidence", 0.0))

    # 1. CATEGORY HAZARD COMPONENT (0 to 35 pts)
    category_base = CLASS_BASE_HAZARD.get(class_code, 8.0)

    # 2. RELATIVE DAMAGED AREA COMPONENT (0 to 35 pts)
    area_ratio = (box_area / frame_area) * 100.0  # Percentage of frame area
    if area_ratio >= 4.0:
        area_score = 35.0
    elif area_ratio >= 2.0:
        area_score = 25.0 + ((area_ratio - 2.0) / 2.0) * 10.0
    elif area_ratio >= 0.8:
        area_score = 15.0 + ((area_ratio - 0.8) / 1.2) * 10.0
    elif area_ratio >= 0.2:
        area_score = 6.0 + ((area_ratio - 0.2) / 0.6) * 9.0
    else:
        area_score = max(1.0, (area_ratio / 0.2) * 5.0)

    # 3. PERSPECTIVE ROAD PROXIMITY COMPONENT (0 to 20 pts)
    # y near frame_h indicates immediate vehicle path (near-field)
    y_center = (y1 + y2) / 2.0
    norm_y = min(1.0, max(0.0, y_center / frame_h))
    
    # Quadratic proximity weighting favoring lower road half
    proximity_factor = math.pow(norm_y, 2.0)
    proximity_score = 20.0 * proximity_factor

    # 4. GEOMETRIC ASPECT RATIO COMPONENT (0 to 10 pts)
    aspect_ratio = max(box_w / box_h, box_h / box_w)
    if class_code in ("D00", "D10"):
        # Severe cracks propagate linearly: high aspect ratio indicates extended crack path
        if aspect_ratio >= 4.0:
            geometry_score = 10.0
        elif aspect_ratio >= 2.0:
            geometry_score = 6.0
        else:
            geometry_score = 3.0
    elif class_code in ("D40", "D20"):
        # Severe potholes & alligator cracking expand in both dimensions (broad depression)
        if aspect_ratio <= 1.8:
            geometry_score = 10.0
        elif aspect_ratio <= 3.0:
            geometry_score = 6.0
        else:
            geometry_score = 3.0
    else:
        geometry_score = 4.0

    # Composite Score
    raw_score = category_base + area_score + proximity_score + geometry_score

    # 5. TEMPORAL VIDEO PERSISTENCE ADJUSTMENT (when available)
    persistence_adjustment = 0.0
    track_hits = 0
    if context and "track_hits" in context:
        track_hits = int(context["track_hits"])
        if track_hits >= 5:
            persistence_adjustment = 3.0
        elif track_hits == 1:
            persistence_adjustment = -2.0

    final_score = min(100.0, max(5.0, raw_score + persistence_adjustment))
    severity_level = classify_severity_score(final_score)

    # Human-readable justification
    class_label = CLASS_LABELS.get(class_code, "Road Defect")
    reason_parts = [
        f"{class_label} ({class_code}) baseline risk of {category_base:.0f} pts",
        f"relative visible area {area_ratio:.2f}% ({area_score:.1f} pts)",
        f"lane proximity index {norm_y:.2f} ({proximity_score:.1f} pts)",
        f"aspect ratio {aspect_ratio:.2f} ({geometry_score:.1f} pts)"
    ]
    if persistence_adjustment != 0.0:
        reason_parts.append(f"temporal persistence ({persistence_adjustment:+.1f} pts)")
    reason = f"Classified as {severity_level} (Score {final_score:.1f}/100) based on: " + "; ".join(reason_parts) + "."

    return {
        "severity_score": round(final_score, 1),
        "severity_level": severity_level,
        "reason": reason,
        "contributing_factors": {
            "class_code": class_code,
            "category_hazard_pts": round(category_base, 1),
            "area_ratio_percent": round(area_ratio, 3),
            "area_score_pts": round(area_score, 1),
            "proximity_norm_y": round(norm_y, 3),
            "proximity_score_pts": round(proximity_score, 1),
            "aspect_ratio": round(aspect_ratio, 2),
            "geometry_score_pts": round(geometry_score, 1),
            "persistence_adjustment_pts": round(persistence_adjustment, 1),
            "detection_confidence": round(confidence, 3),
            "track_hits": track_hits
        }
    }
