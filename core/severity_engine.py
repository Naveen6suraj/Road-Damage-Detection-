"""
ROADGUARD AI - Severity Estimation Engine
Module: core/severity_engine.py
Version: 2.1.1

Provides an empirical, visual risk scoring methodology for road defects detected
in single-camera dashcam imagery and video streams.

Mathematical Formulation:
-------------------------
S = min(100.0, max(5.0, W_cat + W_area + W_pos + W_geom + Delta_pers))

Components:
1. Category Hazard (W_cat in [8.0, 35.0]):
   - Pothole (D40): 35.0
   - Alligator Crack (D20): 25.0
   - Transverse Crack (D10): 16.0
   - Longitudinal Crack (D00): 10.0
   - Other Corruption (OTHER): 8.0
2. Normalized Relative Area (W_area in [1.0, 35.0]):
   - Percentage of total frame area
3. Perspective Road Proximity (W_pos in [0.0, 20.0]):
   - Quadratic near-field weighting: 20.0 * (y_norm)^2
4. Geometric Aspect Ratio (W_geom in [3.0, 10.0]):
   - Crack elongation vs. pothole/alligator compactness
5. Temporal Persistence Modifier (Delta_pers):
   - +3.0 for verified persistent defects (hits >= 5)
   - -2.0 for single-frame transient detections in video context (hits == 1)
   - 0.0 otherwise

Detection confidence is maintained as a strictly separate measure representing
model classification certainty, and is not equated to physical defect severity.
"""

from typing import Dict, Any, Optional, Tuple
import math
import config


# Category Base Hazard Mapping (0 - 35 points)
CLASS_BASE_HAZARD = config.CLASS_BASE_HAZARD

# Human-readable category labels
CLASS_LABELS = config.CLASS_NAMES

# Severity Classification Boundaries
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
        Additional contextual indicators:
        - 'track_hits': int (number of frames observed)
        - 'is_video': bool (indicates video stream context)

    Returns:
    --------
    dict containing:
        - raw_severity_score: float (pure spatial visual score, 5.0 - 100.0)
        - raw_severity_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
        - adjusted_severity_score: float (temporally adjusted score, 5.0 - 100.0)
        - adjusted_severity_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
        - severity_score: float (alias for adjusted_severity_score)
        - severity_level: str (alias for adjusted_severity_level)
        - persistence_adjustment: float (-2.0, 0.0, or +3.0)
        - reason: str (human-readable justification)
        - contributing_factors: dict (breakdown of scoring components)
    """
    # Defensive frame dimension handling
    raw_w, raw_h = frame_dimensions if frame_dimensions and len(frame_dimensions) == 2 else (640, 640)
    frame_w = max(1.0, float(raw_w))
    frame_h = max(1.0, float(raw_h))
    frame_area = frame_w * frame_h

    # Extract and defensively clamp bounding box coordinates
    raw_box = detection.get("box", [0, 0, 0, 0])
    if len(raw_box) != 4:
        raw_box = [0, 0, 0, 0]

    rx1, ry1, rx2, ry2 = raw_box
    # Handle inverted or negative coordinates
    xmin = min(rx1, rx2)
    xmax = max(rx1, rx2)
    ymin = min(ry1, ry2)
    ymax = max(ry1, ry2)

    # Clamp strictly within image boundaries
    x1 = max(0.0, min(frame_w, float(xmin)))
    x2 = max(0.0, min(frame_w, float(xmax)))
    y1 = max(0.0, min(frame_h, float(ymin)))
    y2 = max(0.0, min(frame_h, float(ymax)))

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    box_area = box_w * box_h

    # Extract class code & confidence
    class_code = str(detection.get("class_code", "OTHER")).upper().strip()
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
    # y near frame_h indicates near-field roadway directly in front of vehicle wheels
    y_center = (y1 + y2) / 2.0
    norm_y = min(1.0, max(0.0, y_center / frame_h))
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

    # Raw Spatial Score (Pure visual observation without temporal context)
    raw_composite = category_base + area_score + proximity_score + geometry_score
    raw_score = min(100.0, max(5.0, raw_composite))
    raw_level = classify_severity_score(raw_score)

    # 5. TEMPORAL PERSISTENCE MODIFIER (Delta_pers)
    persistence_adjustment = 0.0
    track_hits = 0
    is_video = False

    if context:
        is_video = context.get("is_video", False)
        if "track_hits" in context:
            track_hits = int(context["track_hits"])
            if track_hits >= config.PERSISTENCE_THRESHOLD_HITS:
                # Confirmed persistent defect across >= 5 frames
                persistence_adjustment = config.PERSISTENCE_BONUS  # +3.0 pts
            elif track_hits == 1 and is_video:
                # 1-frame transient detection in video stream
                persistence_adjustment = config.TRANSIENT_DISCOUNT  # -2.0 pts

    final_score = min(100.0, max(5.0, raw_score + persistence_adjustment))
    final_level = classify_severity_score(final_score)

    # Formulate clear justification
    class_name = config.CLASS_NAMES.get(
        config.CLASS_ALIAS_MAP.get(class_code.lower(), 4),
        "Road Defect"
    )
    reason_parts = [
        f"{class_name} ({class_code}) baseline risk of {category_base:.0f} pts",
        f"relative visible area {area_ratio:.2f}% ({area_score:.1f} pts)",
        f"lane proximity index {norm_y:.2f} ({proximity_score:.1f} pts)",
        f"aspect ratio {aspect_ratio:.2f} ({geometry_score:.1f} pts)"
    ]
    if persistence_adjustment != 0.0:
        reason_parts.append(f"temporal persistence modifier ({persistence_adjustment:+.1f} pts)")

    reason = f"Classified as {final_level} (Score {final_score:.1f}/100) based on: " + "; ".join(reason_parts) + "."

    return {
        "raw_severity_score": round(raw_score, 1),
        "raw_severity_level": raw_level,
        "adjusted_severity_score": round(final_score, 1),
        "adjusted_severity_level": final_level,
        "severity_score": round(final_score, 1),
        "severity_level": final_level,
        "persistence_adjustment": round(persistence_adjustment, 1),
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
            "track_hits": track_hits,
            "is_video": is_video
        }
    }
