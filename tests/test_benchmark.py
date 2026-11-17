"""
Unit Tests: Hardware Benchmarking Suite & Diagnostics API
File: tests/test_benchmark.py
"""

import os
import json
import pytest
from benchmark import collect_system_telemetry
import config
from app import app


def test_collect_system_telemetry():
    """Verify system telemetry gathers accurate host hardware specifications."""
    telemetry = collect_system_telemetry()

    assert "os" in telemetry
    assert "python_version" in telemetry
    assert "cpu" in telemetry
    assert "system_ram_gb" in telemetry
    assert telemetry["system_ram_gb"] > 0
    assert "cuda_available" in telemetry
    assert "gpu_name" in telemetry


def test_benchmark_reports_exist_and_conform():
    """Verify generated hardware benchmark files exist and contain required schema."""
    json_path = os.path.join(config.REPORT_FOLDER, "hardware_benchmark.json")
    assert os.path.exists(json_path), "hardware_benchmark.json must be generated"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "timestamp" in data
    assert "system" in data
    assert "latencies_ms" in data
    assert "model_load" in data["latencies_ms"]
    assert "cold_start" in data["latencies_ms"]
    assert "warm_mean" in data["latencies_ms"]
    assert "throughput" in data
    assert "fps" in data["throughput"]
    assert "memory" in data

    csv_path = os.path.join(config.REPORT_FOLDER, "hardware_benchmark.csv")
    assert os.path.exists(csv_path), "hardware_benchmark.csv must be generated"
    assert os.path.getsize(csv_path) > 100


def test_api_benchmark_endpoint():
    """Verify Flask /api/benchmark/status endpoint returns valid benchmark payload."""
    client = app.test_client()
    resp = client.get("/api/benchmark/status")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert "benchmark" in payload
    assert payload["benchmark"]["version"] == config.APP_VERSION
