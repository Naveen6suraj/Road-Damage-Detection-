"""
ROADGUARD AI - XML to YOLO Annotation Converter
Script: convert_xml_to_yolo.py
Version: 2.1.1

Converts Pascal VOC XML road damage annotations (from RDD2020 / RDD2022 datasets)
into standard YOLO bounding box format with normalized coordinates.
Supports train, val, and test splits with automatic validation splitting,
coordinate clamping, zero-area rejection, and conversion summary statistics.
"""

import os
import argparse
import random
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import shutil

# Standardized RoadGuard Class Mapping
CLASS_MAP = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    "D40": 3,  # Pothole
    "D43": 4,  # Other road distress / corruption
    "D44": 4,
    "D50": 4
}


def convert_box(size, box):
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
        if filename_node is None or not filename_node.text:
            stats["skipped_missing_filename"] += 1
            return False

        img_name = filename_node.text.strip()
        img_path = None

        # Search across provided image directories
        for search_dir in img_search_dirs:
            candidate = os.path.join(search_dir, img_name)
            if os.path.exists(candidate):
                img_path = candidate
                break

            # Try alternate extension matching if exact match not found
            stem = os.path.splitext(os.path.basename(xml_path))[0]
            for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
                alt_candidate = os.path.join(search_dir, stem + ext)
                if os.path.exists(alt_candidate):
                    img_path = alt_candidate
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

            # Defensive coordinate clamping and inversion handling
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


def run_dataset_conversion(data_dir: str = "Data", output_dir: str = "dataset", val_split: float = 0.2):
    """Execute complete dataset conversion across available splits."""
    print("=" * 65)
    print(" ROADGUARD AI - Dataset Annotation Converter v2.1.1")
    print(f" Source Directory:      {os.path.abspath(data_dir)}")
    print(f" Output Directory:      {os.path.abspath(output_dir)}")
    print("=" * 65)

    if not os.path.exists(data_dir):
        print(f"\n[!] Dataset source folder '{data_dir}' was not found.")
        print("    Instructions for acquiring the benchmark Road Damage Dataset:")
        print("    1. Download RDD2022 from https://crddc2022.sekilab.global/")
        print("    2. Extract it into a folder named 'Data/' containing train/ and optionally val/ or test/")
        print("    3. Re-run: python convert_xml_to_yolo.py")
        return

    # Check for split structures
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    splits_to_process = []
    if os.path.exists(train_dir):
        splits_to_process.append(("train", train_dir))
    if os.path.exists(val_dir):
        splits_to_process.append(("val", val_dir))
    if os.path.exists(test_dir):
        splits_to_process.append(("test", test_dir))

    # If no standard subfolders, check if XML files are directly in data_dir
    if not splits_to_process:
        splits_to_process.append(("train", data_dir))

    overall_stats = {}

    for split_name, split_path in splits_to_process:
        print(f"\n[*] Processing split: '{split_name}' from {split_path}")
        out_img_dir = os.path.join(output_dir, "images", split_name)
        out_lbl_dir = os.path.join(output_dir, "labels", split_name)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        stats = {
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

        # Search for XML files recursively
        xml_files = []
        img_search_dirs = [
            split_path,
            os.path.join(split_path, "images"),
            os.path.join(split_path, "JPEGImages")
        ]

        for root_d, _, files in os.walk(split_path):
            if "annotations" in root_d.lower() or any(f.endswith(".xml") for f in files):
                for f in files:
                    if f.endswith(".xml"):
                        xml_files.append(os.path.join(root_d, f))
            # Track any image folders found
            if any(f.lower().endswith((".jpg", ".png", ".jpeg")) for f in files):
                if root_d not in img_search_dirs:
                    img_search_dirs.append(root_d)

        print(f"    Found {len(xml_files)} XML annotation files.")
        for xml_path in tqdm(xml_files, desc=f"Converting {split_name.upper()}"):
            convert_xml_file(xml_path, img_search_dirs, out_img_dir, out_lbl_dir, stats)

        overall_stats[split_name] = stats
        print(f"    Converted: {stats['converted_images']} images | Total boxes: {sum(stats['class_boxes'].values())}")

    # Auto-generate val split from train if val split did not exist
    val_out_dir = os.path.join(output_dir, "images", "val")
    train_out_img = os.path.join(output_dir, "images", "train")
    train_out_lbl = os.path.join(output_dir, "labels", "train")

    if not os.path.exists(val_out_dir) or len(os.listdir(val_out_dir)) == 0:
        if os.path.exists(train_out_img) and len(os.listdir(train_out_img)) > 10 and val_split > 0:
            print(f"\n[*] Creating validation split ({int(val_split*100)}%) from training data...")
            val_out_lbl = os.path.join(output_dir, "labels", "val")
            os.makedirs(val_out_dir, exist_ok=True)
            os.makedirs(val_out_lbl, exist_ok=True)

            img_files = os.listdir(train_out_img)
            random.seed(42)
            val_count = max(1, int(len(img_files) * val_split))
            val_selection = random.sample(img_files, val_count)

            for img_f in val_selection:
                stem = os.path.splitext(img_f)[0]
                lbl_f = stem + ".txt"
                shutil.move(os.path.join(train_out_img, img_f), os.path.join(val_out_dir, img_f))
                src_lbl = os.path.join(train_out_lbl, lbl_f)
                if os.path.exists(src_lbl):
                    shutil.move(src_lbl, os.path.join(val_out_lbl, lbl_f))

            print(f"[+] Partitioned {val_count} images into 'images/val'.")

    # Summary report
    print("\n" + "=" * 65)
    print(" DATASET CONVERSION REPORT SUMMARY")
    print("=" * 65)
    for s_name, s_data in overall_stats.items():
        print(f" Split [{s_name.upper()}]:")
        print(f"   - Images Successfully Converted: {s_data['converted_images']}")
        print(f"   - D00 (Longitudinal Cracks):     {s_data['class_boxes'][0]}")
        print(f"   - D10 (Transverse Cracks):       {s_data['class_boxes'][1]}")
        print(f"   - D20 (Alligator Cracks):        {s_data['class_boxes'][2]}")
        print(f"   - D40 (Potholes):                {s_data['class_boxes'][3]}")
        print(f"   - OTHER (Surface Distress):      {s_data['class_boxes'][4]}")
        if s_data['rejected_zero_area_boxes'] > 0:
            print(f"   - Rejected Zero-Area Bboxes:     {s_data['rejected_zero_area_boxes']}")
        if s_data['skipped_missing_image'] > 0:
            print(f"   - Skipped Missing Image Files:   {s_data['skipped_missing_image']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML dataset to YOLO format (v2.1.1)")
    parser.add_argument("--data-dir", default="Data", help="Path to raw dataset folder")
    parser.add_argument("--output-dir", default="dataset", help="Output directory for YOLO dataset")
    parser.add_argument("--val-split", default=0.2, type=float, help="Validation split ratio if val folder absent")
    args = parser.parse_args()

    run_dataset_conversion(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_split=args.val_split
    )
