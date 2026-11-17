"""
ROADGUARD AI - Dashboard & Page Navigation Routes
Module: routes/dashboard_routes.py
"""

from flask import Blueprint, render_template, jsonify, request
import database
import config
from core.detector import RoadDamageDetector

dashboard_bp = Blueprint("dashboard", __name__)
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = RoadDamageDetector()
    return _detector


@dashboard_bp.route("/")
def index():
    """Main application landing and overview."""
    stats = database.get_dashboard_stats()
    recent = database.get_all_analyses(limit=10)
    sys_status = get_detector().get_status()
    return render_template(
        "index.html",
        stats=stats,
        recent=recent,
        sys_status=sys_status,
        app_name=config.APP_NAME,
        app_subtitle=config.APP_SUBTITLE
    )


@dashboard_bp.route("/dashboard")
def dashboard_view():
    """Dedicated analytics dashboard view."""
    stats = database.get_dashboard_stats()
    recent = database.get_all_analyses(limit=25)
    sys_status = get_detector().get_status()
    return render_template(
        "dashboard.html",
        stats=stats,
        recent=recent,
        sys_status=sys_status,
        app_name=config.APP_NAME
    )


@dashboard_bp.route("/analysis")
def analysis_view():
    """Upload and media analysis workspace."""
    sys_status = get_detector().get_status()
    return render_template(
        "analysis.html",
        sys_status=sys_status,
        app_name=config.APP_NAME
    )


@dashboard_bp.route("/results/<analysis_id>")
def results_view(analysis_id):
    """Detailed inspection results view for a specific analysis."""
    analysis = database.get_analysis(analysis_id)
    if not analysis:
        return render_template("404.html", message="Analysis record not found."), 404
    
    detections = database.get_analysis_detections(analysis_id)
    return render_template(
        "results.html",
        analysis=analysis,
        detections=detections,
        app_name=config.APP_NAME
    )


@dashboard_bp.route("/history")
def history_view():
    """Dedicated inspection history archive."""
    analyses = database.get_all_analyses(limit=100)
    sys_status = get_detector().get_status()
    return render_template(
        "history.html",
        analyses=analyses,
        sys_status=sys_status,
        app_name=config.APP_NAME
    )


@dashboard_bp.route("/reports")
def reports_view():
    """Dedicated inspection reports management portal."""
    analyses = database.get_all_analyses(limit=100)
    sys_status = get_detector().get_status()
    return render_template(
        "reports.html",
        analyses=analyses,
        sys_status=sys_status,
        app_name=config.APP_NAME
    )


@dashboard_bp.route("/api/analyses/<analysis_id>", methods=["DELETE"])
def api_delete_analysis(analysis_id):
    """Delete an analysis session from the registry."""
    success = database.delete_analysis(analysis_id)
    if success:
        return jsonify({"success": True, "message": f"Analysis {analysis_id} deleted."})
    return jsonify({"success": False, "error": "Analysis ID not found"}), 404


@dashboard_bp.route("/api/dashboard/stats")
def api_dashboard_stats():
    """Return JSON metrics for dashboard charts and counters."""
    stats = database.get_dashboard_stats()
    return jsonify({
        "success": True,
        "stats": stats
    })


@dashboard_bp.route("/api/analyses")
def api_analyses_list():
    """Return list of recent analyses."""
    limit = request.args.get("limit", default=25, type=int)
    analyses = database.get_all_analyses(limit=limit)
    return jsonify({
        "success": True,
        "analyses": analyses
    })


@dashboard_bp.route("/api/system/status")
def api_system_status():
    """Return hardware accelerator and YOLO model status."""
    status = get_detector().get_status()
    return jsonify({
        "success": True,
        "system": status
    })


@dashboard_bp.route("/api/model/status")
def api_model_status():
    """Return complete YOLO road damage model status and class verification info."""
    detector = get_detector()
    return jsonify({
        "success": True,
        "model": detector.get_model_status(),
        "system": detector.get_status()
    })

