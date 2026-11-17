"""
Integration Tests: Flask API Routes & Page Views
File: tests/test_routes.py
"""

import io
import cv2
import numpy as np
import pytest
from app import create_app
import database


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"ROADGUARD" in res.data


def test_dashboard_page(client):
    res = client.get("/dashboard")
    assert res.status_code == 200
    assert b"Road Condition Analytics Dashboard" in res.data


def test_analysis_page(client):
    res = client.get("/analysis")
    assert res.status_code == 200
    assert b"Road Defect Analysis Studio" in res.data


def test_api_dashboard_stats(client):
    res = client.get("/api/dashboard/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "stats" in data
    assert "class_distribution" in data["stats"]
    assert "severity_distribution" in data["stats"]


def test_api_system_status(client):
    res = client.get("/api/system/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "device" in data["system"]


def test_image_upload_without_file(client):
    res = client.post("/api/analyze/image", data={})
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False


def test_image_upload_invalid_extension(client):
    data = {"file": (io.BytesIO(b"fake content"), "test.exe")}
    res = client.post("/api/analyze/image", data=data, content_type="multipart/form-data")
    assert res.status_code == 400


def test_api_model_status(client):
    res = client.get("/api/model/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "model" in data
    assert "system" in data


def test_image_upload_missing_custom_model(client):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    data = {
        "file": (io.BytesIO(buf.tobytes()), "synthetic_road.jpg"),
        "confidence": "0.25"
    }
    res = client.post("/api/analyze/image", data=data, content_type="multipart/form-data")
    assert res.status_code == 503
    resp_data = res.get_json()
    assert resp_data["success"] is False
    assert "Custom road damage detection model is not available" in resp_data["error"]


def test_video_upload_missing_custom_model(client):
    data = {
        "file": (io.BytesIO(b"fake video data"), "test.mp4")
    }
    res = client.post("/api/analyze/video", data=data, content_type="multipart/form-data")
    assert res.status_code == 503
    resp_data = res.get_json()
    assert resp_data["success"] is False


def test_image_upload_valid_synthetic(client, monkeypatch):
    import routes.image_routes as img_rt
    det = img_rt.get_detector()
    monkeypatch.setattr(det, "inference_enabled", True)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    data = {
        "file": (io.BytesIO(buf.tobytes()), "synthetic_road.jpg"),
        "confidence": "0.25"
    }
    res = client.post("/api/analyze/image", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    resp_data = res.get_json()
    assert resp_data["success"] is True
    assert "analysis_id" in resp_data
    assert "result_image" in resp_data
    assert "counts" in resp_data


def test_legacy_predict_image_route(client, monkeypatch):
    import routes.image_routes as img_rt
    det = img_rt.get_detector()
    monkeypatch.setattr(det, "inference_enabled", True)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    data = {
        "file": (io.BytesIO(buf.tobytes()), "legacy_test.jpg")
    }
    res = client.post("/predict_image", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    resp_data = res.get_json()
    assert "result_image" in resp_data
    assert "counts" in resp_data


def test_camera_process_frame_missing_payload(client):
    res = client.post("/api/camera/process_frame", json={})
    # Returns 503 when custom model is missing or 400 when model is loaded and payload is missing
    assert res.status_code in (400, 503)


def test_404_handler(client):
    res = client.get("/non_existent_page_route")
    assert res.status_code == 404

