"""
ROADGUARD AI - Application Entry Point
Intelligent Road Damage Detection & Severity Assessment
"""

import os
import argparse
from flask import Flask, render_template, jsonify, request

import config
import database
from routes.dashboard_routes import dashboard_bp
from routes.image_routes import image_bp
from routes.video_routes import video_bp
from routes.camera_routes import camera_bp
from routes.report_routes import report_bp


def create_app():
    """Application factory for RoadGuard AI."""
    app = Flask(
        __name__,
        static_folder=config.STATIC_FOLDER,
        template_folder=os.path.join(config.BASE_DIR, "templates")
    )

    # Configuration
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "roadguard-ai-secret-key-2026")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # Initialize SQLite schema
    database.init_db()

    # Register Route Blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(image_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(camera_bp)
    app.register_blueprint(report_bp)

    # Error Handlers
    @app.errorhandler(404)
    def handle_not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Endpoint not found"}), 404
        return render_template("404.html", message="Requested page does not exist."), 404

    @app.errorhandler(413)
    def handle_file_too_large(e):
        return jsonify({
            "success": False,
            "error": f"Uploaded file exceeds maximum limit of {config.MAX_CONTENT_LENGTH // (1024*1024)} MB."
        }), 413

    @app.errorhandler(500)
    def handle_server_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "An internal server error occurred."}), 500
        return render_template("500.html", message="Internal application error."), 500

    # Custom Jinja Template Filters
    @app.template_filter("datetime_format")
    def format_datetime(value):
        if not value:
            return ""
        try:
            return str(value)[:19].replace("T", " ")
        except Exception:
            return str(value)

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROADGUARD AI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", default=5000, type=int, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  {config.APP_NAME} v{config.APP_VERSION}")
    print(f"  {config.APP_SUBTITLE}")
    print(f"  Hardware: {config.DEVICE_NAME} (CUDA={config.CUDA_AVAILABLE})")
    print(f"  Listening at: http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug)
