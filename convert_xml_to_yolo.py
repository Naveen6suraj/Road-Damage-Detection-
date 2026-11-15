"""
ROADGUARD AI - XML to YOLO Annotation Converter
Script: convert_xml_to_yolo.py

Converts Pascal VOC XML road damage annotations (from RDD2020 / RDD2022 datasets)
into standard YOLO bounding box format with normalized coordinates.
"""

import os
import argparse
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


def convert_split(data_dir: str, output_dir: str, split_name: str = "train"):
    """Convert XML annotations and images for a specific split (train or val)."""
    img_dir = os.path.join(data_dir, split_name, "images")
    ann_root = os.path.join(data_dir, split_name, "annotations")

    out_img_dir = os.path.join(output_dir, "images", split_name)
    out_lbl_dir = os.path.join(output_dir, "labels", split_name)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    if not os.path.exists(ann_root):
        print(f"[!] Annotations directory not found at: {ann_root}")
        return

    xml_files = []
    for root_dir, _, files in os.walk(ann_root):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root_dir, file))

    print(f"[*] Found {len(xml_files)} XML annotation files in {split_name} split.")
    class_box_counts = {cls_id: 0 for cls_id in range(5)}
    converted_count = 0

    for xml_path in tqdm(xml_files, desc=f"Converting {split_name.upper()}"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename_node = root.find("filename")
            if filename_node is None or not filename_node.text:
                continue
            img_name = filename_node.text
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                # Try finding image with same stem in case extension differs
                stem = os.path.splitext(os.path.basename(xml_path))[0]
                for ext in [".jpg", ".png", ".jpeg"]:
                    alt_path = os.path.join(img_dir, stem + ext)
                    if os.path.exists(alt_path):
                        img_path = alt_path
                        break

            if not os.path.exists(img_path):
                continue

            with Image.open(img_path) as img:
                w, h = img.size

            label_name = os.path.basename(xml_path).rsplit(".", 1)[0] + ".txt"
            label_path = os.path.join(out_lbl_dir, label_name)

            has_valid_boxes = False
            with open(label_path, "w", encoding="utf-8") as f:
                for obj in root.findall("object"):
                    cls_name_node = obj.find("name")
                    if cls_name_node is None:
                        continue
                    cls_name = cls_name_node.text.strip()

                    if cls_name not in CLASS_MAP:
                        continue

                    cls_id = CLASS_MAP[cls_name]
                    xmlbox = obj.find("bndbox")
                    if xmlbox is None:
                        continue

                    xmin = float(xmlbox.find("xmin").text)
                    xmax = float(xmlbox.find("xmax").text)
                    ymin = float(xmlbox.find("ymin").text)
                    ymax = float(xmlbox.find("ymax").text)

                    # Clamp box to image bounds
                    xmin = max(0.0, min(float(w), xmin))
                    xmax = max(0.0, min(float(w), xmax))
                    ymin = max(0.0, min(float(h), ymin))
                    ymax = max(0.0, min(float(h), ymax))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    bb = convert_box((w, h), (xmin, xmax, ymin, ymax))
                    f.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
                    class_box_counts[cls_id] += 1
                    has_valid_boxes = True

            # Copy image if valid annotations exist
            if has_valid_boxes and not os.path.exists(os.path.join(out_img_dir, os.path.basename(img_path))):
                shutil.copy2(img_path, out_img_dir)
                converted_count += 1

        except Exception as e:
            print(f"[!] Error processing {xml_path}: {e}")

    print(f"\n[+] Conversion summary for {split_name}:")
    print(f"    Images converted: {converted_count}")
    print(f"    Class boxes:")
    print(f"      - D00 (Longitudinal): {class_box_counts[0]}")
    print(f"      - D10 (Transverse):   {class_box_counts[1]}")
    print(f"      - D20 (Alligator):    {class_box_counts[2]}")
    print(f"      - D40 (Pothole):      {class_box_counts[3]}")
    print(f"      - OTHER:              {class_box_counts[4]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML dataset to YOLO format")
    parser.add_argument("--data-dir", default="Data", help="Path to raw dataset folder containing train/ and test/")
    parser.add_argument("--output-dir", default="dataset", help="Output directory for YOLO formatted dataset")
    args = parser.parse_args()

    print("=" * 60)
    print(" ROADGUARD AI - Dataset Annotation Converter")
    print(f" Source: {args.data_dir} -> Destination: {args.output_dir}")
    print("=" * 60)

    if not os.path.exists(args.data_dir):
        print(f"[!] Source data directory '{args.data_dir}' not found.")
        print("    Please place the RDD dataset in 'Data/' or specify --data-dir <path>")
    else:
        convert_split(args.data_dir, args.output_dir, "train")
        print("\n[+] Dataset conversion complete.")
