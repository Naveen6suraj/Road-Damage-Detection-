"""
ROADGUARD AI - Video Analysis Routes
Module: routes/video_routes.py
"""

import os
import threading
from datetime import datetime
from typing import Dict, Any
from flask import Blueprint, request, jsonify

import config
import database
from core.video_processor import VideoProcessor
from utils.validators import validate_video_upload, parse_confidence
from utils.file_manager import generate_unique_filename, get_upload_path, get_output_path

video_bp = Blueprint("video_routes", __name__)

# Active background video processing jobs registry
# job_id -> { "status": ..., "percent": ..., "result": ..., "error": ... }
_video_jobs: Dict[str, Dict[str, Any]] = {}
_video_processor = None

def get_video_processor():
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor


def _execute_video_pipeline(job_id: str, upload_path: str, output_path: str, orig_filename: str, conf: float):
    """Background worker executing video processing, tracking, and database persistence."""
    processor = VideoProcessor()

    def on_progress(state):
        if job_id in _video_jobs:
            _video_jobs[job_id].update({
                "status": "processing",
                "percent": state.get("percent", 0.0),
                "current_frame": state.get("current_frame", 0),
                "total_frames": state.get("total_frames", 0),
                "fps_processing": state.get("fps_processing", 0.0),
                "time_elapsed_sec": state.get("time_elapsed_sec", 0.0),
                "estimated_remaining_sec": state.get("estimated_remaining_sec", 0.0),
                "total_frame_detections": state.get("total_frame_detections", 0),
                "unique_defects_count": state.get("unique_defects_count", 0)
            })

    try:
        summary = processor.process_video(
            input_path=upload_path,
            output_path=output_path,
            conf_threshold=conf,
            progress_callback=on_progress
        )

        output_rel_path = f"/outputs/{os.path.basename(summary['output_path'])}"
        source_rel_path = f"/uploads/{os.path.basename(upload_path)}"

        analysis_record = {
            "id": job_id,
            "created_at": datetime.now().isoformat(),
            "media_type": "video",
            "filename": orig_filename,
            "source_path": source_rel_path,
            "output_path": output_rel_path,
            "total_defects": summary["total_frame_detections"],
            "unique_defects": summary["unique_defects_count"],
            "critical_count": summary["severity_distribution"]["CRITICAL"],
            "high_count": summary["severity_distribution"]["HIGH"],
            "moderate_count": summary["severity_distribution"]["MODERATE"],
            "low_count": summary["severity_distribution"]["LOW"],
            "max_severity_score": summary["highest_severity_score"],
            "max_severity_level": summary["highest_severity_level"],
            "avg_confidence": summary["avg_confidence"],
            "processing_time_sec": summary["processing_duration_sec"],
            "status": "completed",
            "metadata": {
                "confidence_threshold": conf,
                "video_fps": summary["video_fps"],
                "video_duration_sec": summary["video_duration_sec"],
                "total_frames_processed": summary["total_frames_processed"],
                "damage_type_distribution": summary["damage_type_distribution"],
                "severity_distribution": summary["severity_distribution"],
                "resolution": summary["resolution"]
            }
        }

        # Save to database with deduplicated unique defect records
        database.save_analysis(analysis_record, summary["unique_defects"])

        _video_jobs[job_id].update({
            "status": "completed",
            "percent": 100.0,
            "result": {
                "analysis_id": job_id,
                "result_video": output_rel_path,
                "source_video": source_rel_path,
                "counts": summary["damage_type_distribution"],
                "summary": analysis_record,
                "unique_defects": summary["unique_defects"]
            }
        })

    except Exception as e:
        _video_jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })


@video_bp.route("/api/analyze/video", methods=["POST"])
def api_analyze_video():
    """Async video upload and analysis initiation."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        orig_filename = validate_video_upload(file)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    conf_thresh = parse_confidence(request.form.get("confidence", config.DEFAULT_CONFIDENCE))
    is_sync = request.form.get("sync", "false").lower() in ("true", "1")

    stored_name = generate_unique_filename(orig_filename, prefix="vid")
    upload_path = get_upload_path(stored_name)
    file.save(upload_path)

    out_name = f"annotated_{stored_name}"
    output_path = get_output_path(out_name)
    job_id = f"VID-{stored_name.split('.')[0]}"

    _video_jobs[job_id] = {
        "status": "queued",
        "percent": 0.0,
        "job_id": job_id,
        "filename": orig_filename
    }

    if is_sync:
        _execute_video_pipeline(job_id, upload_path, output_path, orig_filename, conf_thresh)
        job_info = _video_jobs.get(job_id, {})
        if job_info.get("status") == "completed":
            return jsonify({
                "success": True,
                "job_id": job_id,
                **job_info.get("result", {})
            })
        else:
            return jsonify({"success": False, "error": job_info.get("error", "Processing failed")}), 500

    # Asynchronous thread processing
    t = threading.Thread(
        target=_execute_video_pipeline,
        args=(job_id, upload_path, output_path, orig_filename, conf_thresh),
        daemon=True
    )
    t.start()

    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": "Video analysis queued for processing."
    })


@video_bp.route("/api/video/progress/<job_id>", methods=["GET"])
def api_video_progress(job_id):
    """Poll processing status of a video analysis job."""
    if job_id not in _video_jobs:
        return jsonify({"success": False, "error": "Job ID not found"}), 404
    return jsonify({
        "success": True,
        "job": _video_jobs[job_id]
    })


# Legacy route for backward compatibility
@video_bp.route("/predict_video", methods=["POST"])
def legacy_predict_video():
    """Legacy route providing synchronous response matching original app.py."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        orig_filename = validate_video_upload(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    conf_thresh = parse_confidence(request.form.get("confidence", config.DEFAULT_CONFIDENCE))
    stored_name = generate_unique_filename(orig_filename, prefix="vid")
    upload_path = get_upload_path(stored_name)
    file.save(upload_path)

    out_name = f"annotated_{stored_name}"
    output_path = get_output_path(out_name)
    job_id = f"VID-{stored_name.split('.')[0]}"

    _video_jobs[job_id] = {"status": "processing", "percent": 0.0}
    _execute_video_pipeline(job_id, upload_path, output_path, orig_filename, conf_thresh)
    job_info = _video_jobs.get(job_id, {})

    if job_info.get("status") == "completed":
        res = job_info.get("result", {})
        return jsonify({
            "result_video": res.get("result_video"),
            "counts": res.get("counts")
        })
    return jsonify({"error": job_info.get("error", "Video processing failed")}), 500
