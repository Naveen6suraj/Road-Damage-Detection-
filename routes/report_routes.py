"""
ROADGUARD AI - Report Export & Static Media Routes
Module: routes/report_routes.py
"""

import os
from flask import Blueprint, send_file, jsonify, abort

import config
import database
from core.report_generator import generate_pdf_report, generate_csv_report

report_bp = Blueprint("report_routes", __name__)


@report_bp.route("/api/reports/download/<analysis_id>/pdf", methods=["GET"])
def download_pdf_report(analysis_id):
    """Generate and download executive PDF inspection report."""
    analysis = database.get_analysis(analysis_id)
    if not analysis:
        return jsonify({"success": False, "error": "Analysis ID not found"}), 404

    detections = database.get_analysis_detections(analysis_id)
    pdf_filename = f"RoadGuard_Report_{analysis_id}.pdf"
    pdf_path = os.path.join(config.REPORT_FOLDER, pdf_filename)

    try:
        generate_pdf_report(analysis, detections, pdf_path)
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=pdf_filename,
            mimetype="application/pdf"
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"PDF generation error: {str(e)}"}), 500


@report_bp.route("/api/reports/download/<analysis_id>/csv", methods=["GET"])
def download_csv_report(analysis_id):
    """Generate and download structured CSV defect registry."""
    analysis = database.get_analysis(analysis_id)
    if not analysis:
        return jsonify({"success": False, "error": "Analysis ID not found"}), 404

    detections = database.get_analysis_detections(analysis_id)
    csv_filename = f"RoadGuard_Registry_{analysis_id}.csv"
    csv_path = os.path.join(config.REPORT_FOLDER, csv_filename)

    try:
        generate_csv_report(analysis, detections, csv_path)
        return send_file(
            csv_path,
            as_attachment=True,
            download_name=csv_filename,
            mimetype="text/csv"
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"CSV generation error: {str(e)}"}), 500


@report_bp.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serve generated annotated media files."""
    file_path = os.path.join(config.OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        abort(404)
    return send_file(file_path)


@report_bp.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serve uploaded input media files."""
    file_path = os.path.join(config.UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        abort(404)
    return send_file(file_path)


@report_bp.route("/reports/<path:filename>")
def serve_report(filename):
    """Serve saved report files."""
    file_path = os.path.join(config.REPORT_FOLDER, filename)
    if not os.path.exists(file_path):
        abort(404)
    return send_file(file_path)
