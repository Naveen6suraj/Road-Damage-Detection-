"""
ROADGUARD AI - Real-Time Camera Stream Routes
Module: routes/camera_routes.py
"""

import base64
import cv2
import numpy as np
from flask import Blueprint, request, jsonify

import config
from core.detector import RoadDamageDetector
from utils.validators import parse_confidence

camera_bp = Blueprint("camera_routes", __name__)
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = RoadDamageDetector()
    return _detector


@camera_bp.route("/api/camera/process_frame", methods=["POST"])
def process_camera_frame():
    """Process a single live webcam frame transmitted from frontend via WebRTC/Canvas."""
    detector = get_detector()
    if not detector.inference_enabled:
        return jsonify({
            "success": False,
            "error": "Custom road damage detection model is not available.",
            "message": "Custom YOLO model trained on road damage (RDD2020) was not found at models/best.pt or lacks required road damage classes.",
            "model_status": detector.get_model_status()
        }), 503

    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image") or request.form.get("image")

    if not image_b64:
        return jsonify({"success": False, "error": "No image payload received."}), 400

    conf_thresh = parse_confidence(payload.get("confidence", config.DEFAULT_CONFIDENCE))

    # Clean base64 header if present (data:image/jpeg;base64,...)
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image bytes into valid frame.")
    except Exception as e:
        return jsonify({"success": False, "error": f"Frame decode error: {str(e)}"}), 400

    # Run detection & severity evaluation
    try:
        annotated_bgr, detections, class_counts = detector.detect(
            frame,
            conf=conf_thresh,
            imgsz=config.DEFAULT_IMGSZ
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"Inference failure: {str(e)}"}), 500

    # Encode annotated frame back to base64 JPEG
    _, buffer = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    encoded_annotated = base64.b64encode(buffer).decode("utf-8")
    annotated_uri = f"data:image/jpeg;base64,{encoded_annotated}"

    highest_score = max([d.get("severity_score", 0.0) for d in detections], default=0.0)
    highest_level = "LOW"
    for d in detections:
        if d.get("severity_score", 0.0) == highest_score:
            highest_level = d.get("severity_level", "LOW")
            break

    return jsonify({
        "success": True,
        "annotated_frame": annotated_uri,
        "detections": detections,
        "counts": class_counts,
        "highest_severity_score": highest_score,
        "highest_severity_level": highest_level,
        "defect_count": len(detections)
    })
