"""
ROADGUARD AI - XML to YOLO Annotation Converter
Script: convert_xml_to_yolo.py
Version: 3.2.0

Converts Pascal VOC XML road damage annotations (from RDD2020 / RDD2022 datasets)
into standard YOLO bounding box format with normalized coordinates.
Supports:
- Multi-country directory structures from official RDD2022 Figshare releases
  (China_Drone, China_MotorBike, Czech, India, Japan, Norway, United_States).
- Automatic discovery of country train/test splits.
- Coordinate clamping to [0.0, 1.0], zero-area box rejection, and inversion handling.
- Deterministic 80/20 train/val split generation with fixed random seed (seed=42).
- Class distribution statistics and conversion summary reporting.
"""

import os
import sys
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple
from PIL import Image
from tqdm import tqdm

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config

# Standardized RoadGuard Class Mapping
CLASS_MAP = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    "D40": 3,  # Pothole
    "D43": 4,  # Other road distress / surface corruption
    "D44": 4,
    "D50": 4
}


def convert_box(size: Tuple[int, int], box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Normalize bounding box coordinates to (x_center, y_center, width, height) relative to image size."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    xmin, xmax, ymin, ymax = box

    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    return (x * dw, y * dh, w * dw, h * dh)


def convert_xml_file(xml_path: str, img_search_dirs: list, out_img_dir: str, out_lbl_dir: str, stats: dict) -> bool:
    """Parse a single Pascal VOC XML annotation file and write YOLO label file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_node = root.find("filename")
        img_name = filename_node.text.strip() if (filename_node is not None and filename_node.text) else ""
        img_path = None

        # Search across provided image directories
        if img_name:
            for search_dir in img_search_dirs:
                candidate = os.path.join(search_dir, img_name)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        # Fallback: search by stem name matching XML filename
        if not img_path:
            stem = os.path.splitext(os.path.basename(xml_path))[0]
            for search_dir in img_search_dirs:
                for ext in [".jpg", ".png", ".jpeg", ".JPG", ".JPEG"]:
                    candidate = os.path.join(search_dir, stem + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break

        if not img_path or not os.path.exists(img_path):
            stats["skipped_missing_image"] += 1
            return False

        with Image.open(img_path) as img:
            w, h = img.size

        if w <= 0 or h <= 0:
            stats["skipped_zero_dim_image"] += 1
            return False

        valid_boxes = []
        for obj in root.findall("object"):
            cls_name_node = obj.find("name")
            if cls_name_node is None or not cls_name_node.text:
                continue

            cls_name = cls_name_node.text.strip()
            if cls_name not in CLASS_MAP:
                stats["unknown_class_count"] += 1
                continue

            cls_id = CLASS_MAP[cls_name]
            xmlbox = obj.find("bndbox")
            if xmlbox is None:
                continue

            try:
                xmin = float(xmlbox.find("xmin").text)
                xmax = float(xmlbox.find("xmax").text)
                ymin = float(xmlbox.find("ymin").text)
                ymax = float(xmlbox.find("ymax").text)
            except (ValueError, TypeError, AttributeError):
                stats["invalid_bbox_format"] += 1
                continue

            # Coordinate clamping and inversion handling
            actual_xmin = min(xmin, xmax)
            actual_xmax = max(xmin, xmax)
            actual_ymin = min(ymin, ymax)
            actual_ymax = max(ymin, ymax)

            clamped_xmin = max(0.0, min(float(w), actual_xmin))
            clamped_xmax = max(0.0, min(float(w), actual_xmax))
            clamped_ymin = max(0.0, min(float(h), actual_ymin))
            clamped_ymax = max(0.0, min(float(h), actual_ymax))

            box_w = clamped_xmax - clamped_xmin
            box_h = clamped_ymax - clamped_ymin

            # Reject zero-area or sub-pixel bounding boxes
            if box_w <= 1.0 or box_h <= 1.0:
                stats["rejected_zero_area_boxes"] += 1
                continue

            bb = convert_box((w, h), (clamped_xmin, clamped_xmax, clamped_ymin, clamped_ymax))
            valid_boxes.append((cls_id, bb))
            stats["class_boxes"][cls_id] += 1

        if not valid_boxes:
            stats["skipped_no_valid_boxes"] += 1
            return False

        # Write label file
        label_name = os.path.basename(xml_path).rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(out_lbl_dir, label_name)

        with open(label_path, "w", encoding="utf-8") as f:
            for cls_id, bb in valid_boxes:
                f.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

        # Copy image to target directory
        dest_img_path = os.path.join(out_img_dir, os.path.basename(img_path))
        if not os.path.exists(dest_img_path):
            shutil.copy2(img_path, dest_img_path)

        stats["converted_images"] += 1
        return True

    except Exception as e:
        stats["xml_parse_errors"] += 1
        return False


def discover_rdd_sources(data_dir: str) -> List[Tuple[str, str, str]]:
    """
    Recursively discover all XML annotation directories and corresponding image directories
    under data_dir across all countries and split conventions.
    Returns: list of (country_or_split_name, xml_dir, images_dir)
    """
    pairs = []
    if not os.path.exists(data_dir):
        return pairs

    for root, dirs, files in os.walk(data_dir):
        # Look for XML annotation directories (e.g., annotations/xmls or annotations)
        if any(f.lower().endswith(".xml") for f in files):
            xml_dir = root
            # Candidate image directories: look in parent/images, sibling images, or root
            parent_dir = os.path.dirname(root)
            candidate_img_dirs = [
                os.path.join(parent_dir, "images"),
                os.path.join(parent_dir, "JPEGImages"),
                os.path.join(os.path.dirname(parent_dir), "images"),
                root
            ]
            matched_img_dir = None
            for cid in candidate_img_dirs:
                if os.path.exists(cid) and any(f.lower().endswith((".jpg", ".png", ".jpeg")) for f in os.listdir(cid)):
                    matched_img_dir = cid
                    break

            if matched_img_dir:
                # Determine clean display label (e.g. Japan_train, Czech, etc.)
                rel = os.path.relpath(xml_dir, data_dir).replace("\\", "/")
                parts = [p for p in rel.split("/") if p.lower() not in ("annotations", "xmls", "xml")]
                label = "_".join(parts) if parts else "source"
                pairs.append((label, xml_dir, matched_img_dir))

    return pairs


def run_dataset_conversion(data_dir: str = "Data", output_dir: str = "dataset", val_split: float = 0.2):
    """Execute complete dataset conversion across all discovered splits and countries."""
    print("=" * 65)
    print(" ROADGUARD AI - Dataset Annotation Converter v3.2")
    print(f" Source Directory:      {os.path.abspath(data_dir)}")
    print(f" Output Directory:      {os.path.abspath(output_dir)}")
    print("=" * 65)

    if not os.path.exists(data_dir):
        print(f"\n[!] Dataset source folder '{data_dir}' was not found.")
        print("    Instructions for acquiring the benchmark Road Damage Dataset:")
        print("    1. Download RDD2022 from https://doi.org/10.6084/m9.figshare.21431547")
        print("    2. Extract it into a folder named 'Data/' (e.g., Data/RDD2022/Japan/train/...)")
        print("    3. Re-run: python convert_xml_to_yolo.py")
        return

    # Discover XML/Image pairs across all country folders
    sources = discover_rdd_sources(data_dir)

    if not sources:
        print(f"\n[!] No Pascal VOC XML annotations found inside '{data_dir}'.")
        print("    Please ensure extracted XML files (*.xml) and JPEG images (*.jpg) are in 'Data/'.")
        return

    print(f"\n[*] Discovered {len(sources)} dataset source partition(s):")
    for name, xdir, idir in sources:
        print(f"    - [{name}]: {xdir}")

    out_train_img = os.path.join(output_dir, "images", "train")
    out_train_lbl = os.path.join(output_dir, "labels", "train")
    os.makedirs(out_train_img, exist_ok=True)
    os.makedirs(out_train_lbl, exist_ok=True)

    overall_stats = {
        "converted_images": 0,
        "class_boxes": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        "rejected_zero_area_boxes": 0,
        "skipped_missing_image": 0,
        "skipped_missing_filename": 0,
        "skipped_zero_dim_image": 0,
        "skipped_no_valid_boxes": 0,
        "unknown_class_count": 0,
        "invalid_bbox_format": 0,
        "xml_parse_errors": 0
    }

    for name, xml_dir, img_dir in sources:
        xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.lower().endswith(".xml")]
        print(f"\n[*] Processing partition '{name}' ({len(xml_files)} XML files)...")

        for xf in tqdm(xml_files, desc=f"Converting {name}"):
            convert_xml_file(xf, [img_dir], out_train_img, out_train_lbl, overall_stats)

    print(f"\n[+] Total successfully converted images: {overall_stats['converted_images']}")
    print(f"    Total annotations: {sum(overall_stats['class_boxes'].values())}")

    # Generate 80/20 train/val split if sufficient images converted
    val_out_dir = os.path.join(output_dir, "images", "val")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")
    os.makedirs(val_out_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    all_train_imgs = [f for f in os.listdir(out_train_img) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if len(all_train_imgs) >= 10 and val_split > 0:
        print(f"\n[*] Partitioning validation split ({int(val_split * 100)}%) with seed 42...")
        random.seed(42)
        val_count = max(1, int(len(all_train_imgs) * val_split))
        val_selection = random.sample(all_train_imgs, val_count)

        for img_f in val_selection:
            stem = os.path.splitext(img_f)[0]
            lbl_f = stem + ".txt"
            shutil.move(os.path.join(out_train_img, img_f), os.path.join(val_out_dir, img_f))
            src_lbl = os.path.join(out_train_lbl, lbl_f)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, os.path.join(val_lbl_dir, lbl_f))

        print(f"[+] Partitioned {val_count} images into 'dataset/images/val'.")

    # Final summary
    print("\n" + "=" * 65)
    print(" DATASET CONVERSION REPORT SUMMARY")
    print("=" * 65)
    print(f" Images Successfully Converted: {overall_stats['converted_images']}")
    print(f" D00 (Longitudinal Cracks):     {overall_stats['class_boxes'][0]}")
    print(f" D10 (Transverse Cracks):       {overall_stats['class_boxes'][1]}")
    print(f" D20 (Alligator Cracks):        {overall_stats['class_boxes'][2]}")
    print(f" D40 (Potholes):                {overall_stats['class_boxes'][3]}")
    print(f" OTHER (Surface Distress):      {overall_stats['class_boxes'][4]}")
    if overall_stats['rejected_zero_area_boxes'] > 0:
        print(f" Rejected Zero-Area Bboxes:     {overall_stats['rejected_zero_area_boxes']}")
    if overall_stats['skipped_missing_image'] > 0:
        print(f" Skipped Missing Image Files:   {overall_stats['skipped_missing_image']}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML dataset to YOLO format (v3.2)")
    parser.add_argument("--data-dir", default="Data", help="Path to raw dataset folder")
    parser.add_argument("--output-dir", default="dataset", help="Output directory for YOLO dataset")
    parser.add_argument("--val-split", default=0.2, type=float, help="Validation split ratio")
    args = parser.parse_args()

    run_dataset_conversion(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_split=args.val_split
    )
