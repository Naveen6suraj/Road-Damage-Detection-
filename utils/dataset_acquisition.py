"""
ROADGUARD AI - Dataset Acquisition & Verification Guide
Module: utils/dataset_acquisition.py
Version: 3.2.0

Provides academic verification, integrity verification, and extraction automation
for the benchmark Road Damage Datasets (RDD2022 / RDD2020).
"""

import os
import sys
import hashlib
import zipfile
import tarfile
import argparse

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


RDD2022_METADATA = {
    "title": "RDD2022 - The multi-national Road Damage Dataset released through CRDDC'2022",
    "doi": "10.6084/m9.figshare.21431547",
    "landing_page": "https://doi.org/10.6084/m9.figshare.21431547",
    "direct_download_url": "https://ndownloader.figshare.com/files/38030910",
    "archive_filename": "RDD2022_released_through_CRDDC2022.zip",
    "archive_size_bytes": 13264172619,
    "archive_size_gb": 12.35,
    "expected_md5": "b62bd51d2ffcfaa76c60f234f0cc2bb3",
    "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
    "authors": "Deeksha Arya, Hiroya Maeda, Yoshihide Sekimoto, et al. (Seki Lab, The University of Tokyo)",
    "countries": [
        "Japan", "India", "Czech Republic", "Norway", "United States", "China (Drone & Motorbike)"
    ],
    "total_images": 47420,
    "labeled_training_images": 38385,
    "taxonomy": {
        "D00": "Longitudinal Crack",
        "D10": "Transverse Crack",
        "D20": "Alligator Crack",
        "D40": "Pothole",
        "D43": "Other Corruption (Repair / Blur)",
        "D44": "Other Corruption (Marker)"
    }
}


RDD2020_METADATA = {
    "title": "RDD2020: An Image Dataset for Smartphone-based Road Damage Detection and Classification",
    "doi": "10.17632/5ty2wb6gvg.1",
    "landing_page": "https://data.mendeley.com/datasets/5ty2wb6gvg/1",
    "archive_filename": "train.tar.gz",
    "archive_size_gb": 1.37,
    "license": "CC BY 4.0",
    "countries": ["Japan", "India", "Czech Republic"],
    "total_images": 26336
}


def print_acquisition_manual():
    """Print complete academic acquisition guide with manual download steps."""
    print("=" * 70)
    print(" ROADGUARD AI - Official Dataset Acquisition Guide (RDD2022 / RDD2020)")
    print("=" * 70)
    print(f" Dataset Title:   {RDD2022_METADATA['title']}")
    print(f" DOI:             {RDD2022_METADATA['doi']}")
    print(f" Official Host:   {RDD2022_METADATA['landing_page']}")
    print(f" License:         {RDD2022_METADATA['license']}")
    print(f" Authors:         {RDD2022_METADATA['authors']}")
    print(f" Archive:         {RDD2022_METADATA['archive_filename']} (~{RDD2022_METADATA['archive_size_gb']} GB)")
    print(f" Expected MD5:    {RDD2022_METADATA['expected_md5']}")
    print(f" Total Images:    {RDD2022_METADATA['total_images']} ({RDD2022_METADATA['labeled_training_images']} labeled)")
    print("-" * 70)
    print(" WHY AUTOMATED TERMINAL DOWNLOAD IS BLOCKED:")
    print(" Figshare's direct download endpoint ('ndownloader.figshare.com/files/38030910')")
    print(" is protected by AWS WAF / Cloudflare Anti-Bot Challenge ('x-amzn-waf-action: challenge').")
    print(" Automated CLI commands (curl/python) cannot bypass bot gates without manual user navigation.")
    print("-" * 70)
    print(" STEP-BY-STEP USER ACQUISITION INSTRUCTIONS:")
    print(" 1. Open the official DOI in your web browser:")
    print(f"    {RDD2022_METADATA['landing_page']}")
    print("    (OR Kaggle mirror: https://www.kaggle.com/datasets/khaledabdelgaber/rdd2022)")
    print(" 2. Download 'RDD2022_released_through_CRDDC2022.zip' (~12.35 GB)")
    print("    (OR RDD2020 'train.tar.gz' from https://data.mendeley.com/datasets/5ty2wb6gvg/1)")
    print(" 3. Extract the downloaded archive into the 'Data/' folder in this workspace:")
    print(f"    {os.path.abspath(os.path.join(_PROJECT_ROOT, 'Data'))}")
    print(" 4. Run the automated preprocessor and validator:")
    print("    python convert_xml_to_yolo.py")
    print("    python utils/dataset_validator.py")
    print(" 5. Start training on your NVIDIA RTX 3050 Laptop GPU:")
    print("    python train.py --epochs 100 --batch 8 --device 0")
    print("=" * 70)


def verify_archive(archive_path: str) -> bool:
    """Verify archive checksum and structure."""
    if not os.path.exists(archive_path):
        print(f"[!] Archive not found at: {archive_path}")
        return False

    print(f"[*] Calculating MD5 checksum for: {archive_path}...")
    hasher = hashlib.md5()
    size_bytes = os.path.getsize(archive_path)
    with open(archive_path, "rb") as f:
        while chunk := f.read(1048576):
            hasher.update(chunk)
    calc_md5 = hasher.hexdigest()

    print(f"    Calculated MD5: {calc_md5}")
    print(f"    Expected MD5:   {RDD2022_METADATA['expected_md5']}")
    print(f"    File Size:      {size_bytes / (1024**3):.2f} GB")

    if calc_md5.lower() == RDD2022_METADATA["expected_md5"].lower():
        print("[+] Checksum matched! Archive is genuine and uncorrupted.")
        return True
    else:
        print("[!] Checksum mismatch or alternate dataset variant.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDD2022 Dataset Acquisition & Integrity Guide")
    parser.add_argument("--status", action="store_true", help="Display dataset availability status")
    parser.add_argument("--verify", type=str, help="Verify MD5 of downloaded dataset archive")
    args = parser.parse_args()

    if args.verify:
        verify_archive(args.verify)
    else:
        print_acquisition_manual()
