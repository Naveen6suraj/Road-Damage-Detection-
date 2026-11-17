"""
ROADGUARD AI - Analytics & Metrics Utilities
Module: utils/metrics.py
"""

from typing import List, Dict, Any
import config


def aggregate_severity_counts(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count detections falling into each severity category."""
    counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    for d in detections:
        level = d.get("severity_level", "LOW")
        if level in counts:
            counts[level] += 1
        else:
            counts["LOW"] += 1
    return counts


def aggregate_class_counts(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count detections falling into each standardized defect class."""
    counts = {code: 0 for code in config.CLASS_CODES.values()}
    for d in detections:
        code = d.get("class_code", "OTHER")
        if code in counts:
            counts[code] += 1
        else:
            counts["OTHER"] += 1
    return counts


def compute_analysis_summary(
    detections: List[Dict[str, Any]],
    media_type: str = "image"
) -> Dict[str, Any]:
    """Compute complete statistical summary for an analysis."""
    sev_counts = aggregate_severity_counts(detections)
    class_counts = aggregate_class_counts(detections)

    highest_score = 0.0
    highest_level = "LOW"
    conf_sum = 0.0

    for d in detections:
        score = float(d.get("severity_score", 0.0))
        if score > highest_score:
            highest_score = score
            highest_level = d.get("severity_level", "LOW")
        conf_sum += float(d.get("confidence", 0.0))

    avg_conf = conf_sum / len(detections) if detections else 0.0

    return {
        "total_defects": len(detections),
        "unique_defects": len(detections),
        "severity_distribution": sev_counts,
        "class_distribution": class_counts,
        "critical_count": sev_counts["CRITICAL"],
        "high_count": sev_counts["HIGH"],
        "moderate_count": sev_counts["MODERATE"],
        "low_count": sev_counts["LOW"],
        "max_severity_score": round(highest_score, 1),
        "max_severity_level": highest_level,
        "avg_confidence": round(avg_conf, 3),
        "media_type": media_type
    }
