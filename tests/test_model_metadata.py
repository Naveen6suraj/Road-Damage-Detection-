"""
Unit Tests: Model Metadata & MLOps Audit Specification
File: tests/test_model_metadata.py
"""

import os
import json
import pytest
from train import generate_model_metadata
import config


def test_model_metadata_file_schema():
    """Verify models/model_metadata.json conforms to academic audit schema."""
    meta_path = os.path.join(config.MODEL_FOLDER, "model_metadata.json")
    assert os.path.exists(meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "model_name" in meta
    assert "architecture" in meta
    assert "version" in meta
    assert "taxonomy_mapping" in meta
    assert "required_codes" in meta
    assert "input_resolution" in meta
    assert "target_hardware" in meta
    assert "metrics_summary" in meta

    # Verify road damage classes
    codes = [v["code"] for v in meta["taxonomy_mapping"].values()]
    assert "D00" in codes
    assert "D10" in codes
    assert "D20" in codes
    assert "D40" in codes


def test_generate_model_metadata_function(tmp_path):
    """Test runtime generation of model metadata."""
    out_file = str(tmp_path / "test_meta.json")
    meta = generate_model_metadata(
        model_path="models/best.pt",
        architecture="YOLO11n",
        hyperparameters={"epochs": 50, "batch": 8},
        metrics_summary={"mAP50": 0.72},
        output_path=out_file
    )

    assert os.path.exists(out_file)
    assert meta["architecture"] == "YOLO11n"
    assert meta["hyperparameters"]["epochs"] == 50
    assert meta["metrics_summary"]["mAP50"] == 0.72
