"""
Unit Tests: YOLO Dataset Integrity & Leakage Validator
File: tests/test_dataset_validator.py
"""

import os
import yaml
import pytest
from PIL import Image
import numpy as np

from utils.dataset_validator import DatasetValidator, validate_yolo_dataset


@pytest.fixture
def mock_dataset_env(tmp_path):
    """Create a temporary synthetic YOLO dataset environment."""
    base_dir = tmp_path / "dataset"
    img_train = base_dir / "images" / "train"
    img_val = base_dir / "images" / "val"
    lbl_train = base_dir / "labels" / "train"
    lbl_val = base_dir / "labels" / "val"

    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    # Create dummy images
    im1 = Image.fromarray(np.full((100, 100, 3), 120, dtype=np.uint8))
    im1.save(str(img_train / "img1.jpg"))

    im2 = Image.fromarray(np.full((100, 100, 3), 180, dtype=np.uint8))
    im2.save(str(img_val / "img2.jpg"))

    # Create matching labels: class_id x_center y_center width height
    with open(str(lbl_train / "img1.txt"), "w") as f:
        f.write("3 0.5 0.5 0.2 0.2\n")

    with open(str(lbl_val / "img2.txt"), "w") as f:
        f.write("0 0.4 0.6 0.1 0.3\n")

    # Create data.yaml
    data_yaml = tmp_path / "data.yaml"
    cfg = {
        "path": str(base_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "D00",
            1: "D10",
            2: "D20",
            3: "D40"
        }
    }
    with open(str(data_yaml), "w") as f:
        yaml.dump(cfg, f)

    return {
        "yaml": str(data_yaml),
        "base_dir": str(base_dir),
        "img_train": str(img_train),
        "img_val": str(img_val),
        "lbl_train": str(lbl_train),
        "lbl_val": str(lbl_val)
    }


def test_validator_valid_dataset(mock_dataset_env):
    """Test validation on a correctly structured synthetic dataset."""
    validator = DatasetValidator(data_yaml_path=mock_dataset_env["yaml"], verbose=False)
    results = validator.validate()

    assert results["valid"] is True
    assert results["yaml_parsed"] is True
    assert results["taxonomy_valid"] is True
    assert results["total_images"] == 2
    assert results["total_labels"] == 2
    assert results["total_boxes"] == 2
    assert results["leakage_count"] == 0
    assert len(results["errors"]) == 0


def test_validator_detects_data_leakage(mock_dataset_env):
    """Test validator flags exact image leakage between train and val."""
    # Write identical image to both train and val
    shared_img = Image.fromarray(np.full((100, 100, 3), 99, dtype=np.uint8))
    shared_img.save(os.path.join(mock_dataset_env["img_train"], "shared.jpg"))
    shared_img.save(os.path.join(mock_dataset_env["img_val"], "shared_leak.jpg"))

    # Give labels
    with open(os.path.join(mock_dataset_env["lbl_train"], "shared.txt"), "w") as f:
        f.write("3 0.5 0.5 0.2 0.2\n")
    with open(os.path.join(mock_dataset_env["lbl_val"], "shared_leak.txt"), "w") as f:
        f.write("3 0.5 0.5 0.2 0.2\n")

    validator = DatasetValidator(data_yaml_path=mock_dataset_env["yaml"], verbose=False)
    results = validator.validate()

    assert results["leakage_count"] >= 1
    assert results["valid"] is False


def test_validator_detects_out_of_bounds_bbox(mock_dataset_env):
    """Test validator flags coordinates outside [0.0, 1.0]."""
    with open(os.path.join(mock_dataset_env["lbl_train"], "img1.txt"), "w") as f:
        f.write("3 1.50 0.50 0.20 0.20\n")  # x_center > 1.0

    validator = DatasetValidator(data_yaml_path=mock_dataset_env["yaml"], verbose=False)
    results = validator.validate()

    assert len(results["invalid_boxes"]) >= 1
    assert results["valid"] is False


def test_validator_detects_missing_label(mock_dataset_env):
    """Test validator detects image missing its corresponding label file."""
    os.remove(os.path.join(mock_dataset_env["lbl_train"], "img1.txt"))

    validator = DatasetValidator(data_yaml_path=mock_dataset_env["yaml"], verbose=False)
    results = validator.validate()

    assert len(results["missing_labels"]) >= 1
