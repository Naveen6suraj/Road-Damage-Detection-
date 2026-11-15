"""
ROADGUARD AI - Image Analysis Routes
Module: routes/image_routes.py
"""

import os
import time
import cv2
from datetime import datetime
from flask import Blueprint, request, jsonify, url_for

import config
import database
from core.detector import RoadDamageDetector
from utils.validators import validate_image_upload, parse_confidence
from utils.file_manager import generate_unique_filename, get_upload_path, get_output_path
from utils.metrics import compute_analysis_summary

image_bp = Blueprint("image_routes", __name__)
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = RoadDamageDetector()
    return _detector


@image_bp.route("/predict_image", methods=["POST"])
@image_bp.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    """Process single road image for pothole and crack detection with severity."""
    start_time = time.time()
    
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in upload request."}), 400

    file = request.files["file"]
    try:
        orig_filename = validate_image_upload(file)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    conf_thresh = parse_confidence(request.form.get("confidence", config.DEFAULT_CONFIDENCE))

    # Save uploaded image safely
    stored_name = generate_unique_filename(orig_filename, prefix="img")
    upload_path = get_upload_path(stored_name)
    file.save(upload_path)

    # Perform detection & severity estimation
    detector = get_detector()
    try:
        annotated_bgr, detections, class_counts = detector.detect(
            upload_path,
            conf=conf_thresh,
            imgsz=config.DEFAULT_IMGSZ
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"Inference error: {str(e)}"}), 500

    # Save annotated result image
    output_filename = f"annotated_{stored_name}"
    output_path = get_output_path(output_filename)
    cv2.imwrite(output_path, annotated_bgr)

    proc_time = round(time.time() - start_time, 3)

    # Compute analysis summary
    analysis_id = f"IMG-{stored_name.split('.')[0]}"
    summary = compute_analysis_summary(detections, media_type="image")

    analysis_record = {
        "id": analysis_id,
        "created_at": datetime.now().isoformat(),
        "media_type": "image",
        "filename": orig_filename,
        "source_path": f"/uploads/{stored_name}",
        "output_path": f"/outputs/{output_filename}",
        "total_defects": summary["total_defects"],
        "unique_defects": summary["unique_defects"],
        "critical_count": summary["critical_count"],
        "high_count": summary["high_count"],
        "moderate_count": summary["moderate_count"],
        "low_count": summary["low_count"],
        "max_severity_score": summary["max_severity_score"],
        "max_severity_level": summary["max_severity_level"],
        "avg_confidence": summary["avg_confidence"],
        "processing_time_sec": proc_time,
        "status": "completed",
        "metadata": {
            "confidence_threshold": conf_thresh,
            "damage_type_distribution": class_counts,
            "severity_distribution": summary["severity_distribution"],
            "image_dimensions": [annotated_bgr.shape[1], annotated_bgr.shape[0]]
        }
    }

    # Save to database
    database.save_analysis(analysis_record, detections)

    # Return response formatted for both legacy frontend and modern dashboard
    return jsonify({
        "success": True,
        "analysis_id": analysis_id,
        "result_image": f"/outputs/{output_filename}",
        "source_image": f"/uploads/{stored_name}",
        "counts": class_counts,
        "summary": analysis_record,
        "detections": detections,
        "processing_time_sec": proc_time
    })
