"""
Unit Tests: Dataset Acquisition & Verification Protocol
File: tests/test_dataset_acquisition.py
"""

import os
import pytest
from utils.dataset_acquisition import RDD2022_METADATA, RDD2020_METADATA, verify_archive


def test_rdd2022_metadata_completeness():
    """Verify official RDD2022 DOI, expected MD5, and category taxonomy."""
    assert RDD2022_METADATA["doi"] == "10.6084/m9.figshare.21431547"
    assert RDD2022_METADATA["expected_md5"] == "b62bd51d2ffcfaa76c60f234f0cc2bb3"
    assert "D00" in RDD2022_METADATA["taxonomy"]
    assert "D10" in RDD2022_METADATA["taxonomy"]
    assert "D20" in RDD2022_METADATA["taxonomy"]
    assert "D40" in RDD2022_METADATA["taxonomy"]
    assert RDD2022_METADATA["total_images"] == 47420
    assert RDD2022_METADATA["labeled_training_images"] == 38385


def test_verify_archive_nonexistent_file():
    """Verify verify_archive gracefully returns False when file is missing."""
    assert verify_archive("nonexistent_path/fake_rdd.zip") is False


def test_verify_archive_checksum_validation(tmp_path):
    """Verify MD5 checksum calculation against mock file."""
    mock_file = tmp_path / "mock.zip"
    with open(mock_file, "wb") as f:
        f.write(b"ROADGUARD_DATASET_MOCK_PAYLOAD")

    # Should calculate checksum and return False against RDD2022 MD5
    assert verify_archive(str(mock_file)) is False
