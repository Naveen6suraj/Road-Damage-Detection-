"""
ROADGUARD AI - YOLO Dataset Integrity & Leakage Validator
Module: utils/dataset_validator.py
Version: 3.1.0

Comprehensive dataset verification suite for computer vision and MLOps:
1. Directory structure and YAML specification integrity.
2. Road damage taxonomy alignment (D00, D10, D20, D40, OTHER).
3. Bounding box validity (YOLO format: class x_center y_center width height in [0, 1]).
4. Corrupted or unreadable image identification.
5. Image-to-label pairing and orphaned file detection.
6. Cross-split data leakage detection via SHA-256 content hashing.
7. Class distribution and box geometry analytics.
"""

import os
import sys
import yaml
import hashlib
from typing import Dict, Any, List, Set, Tuple, Optional
from PIL import Image

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class DatasetValidationError(Exception):
    """Raised when an irrecoverable dataset corruption or misconfiguration is found."""
    pass


class DatasetValidator:
    """
    Validates YOLO dataset structure, annotation integrity, class distribution,
    and cross-split data leakage.
    """

    def __init__(self, data_yaml_path: str = "data.yaml", verbose: bool = True):
        self.data_yaml_path = os.path.abspath(data_yaml_path)
        self.verbose = verbose
        self.base_dir = os.path.dirname(self.data_yaml_path)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def compute_file_hash(self, filepath: str) -> str:
        """Compute SHA-256 hash of a file for exact duplicate/leakage detection."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate(self) -> Dict[str, Any]:
        """
        Execute full dataset audit and return structured diagnostic dictionary.
        """
        results: Dict[str, Any] = {
            "valid": False,
            "data_yaml": self.data_yaml_path,
            "yaml_parsed": False,
            "taxonomy_valid": False,
            "splits": {},
            "total_images": 0,
            "total_labels": 0,
            "total_boxes": 0,
            "class_distribution": {},
            "leakage_count": 0,
            "leakage_pairs": [],
            "corrupt_images": [],
            "corrupt_labels": [],
            "orphaned_labels": [],
            "missing_labels": [],
            "invalid_boxes": [],
            "errors": [],
            "warnings": []
        }

        # 1. Parse YAML
        if not os.path.exists(self.data_yaml_path):
            results["errors"].append(f"data.yaml not found at '{self.data_yaml_path}'")
            return results

        try:
            with open(self.data_yaml_path, "r", encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f)
            results["yaml_parsed"] = True
        except Exception as e:
            results["errors"].append(f"Failed to parse YAML file: {e}")
            return results

        if not isinstance(yaml_cfg, dict):
            results["errors"].append("data.yaml must contain a YAML dictionary mapping.")
            return results

        # 2. Check taxonomy
        names = yaml_cfg.get("names", {})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        elif not isinstance(names, dict):
            names = {}

        # Check required road damage codes
        found_codes = set()
        for idx, name in names.items():
            name_clean = str(name).strip().lower()
            if name_clean in config.CLASS_ALIAS_MAP:
                std_id = config.CLASS_ALIAS_MAP[name_clean]
                found_codes.add(config.CLASS_CODES[std_id])

        if config.REQUIRED_ROAD_DAMAGE_CODES.issubset(found_codes):
            results["taxonomy_valid"] = True
        else:
            missing_codes = config.REQUIRED_ROAD_DAMAGE_CODES - found_codes
            results["warnings"].append(
                f"Dataset names miss required road damage codes: {sorted(missing_codes)}"
            )

        # 3. Resolve split paths
        base_path = yaml_cfg.get("path", "")
        if base_path and not os.path.isabs(base_path):
            base_path = os.path.normpath(os.path.join(self.base_dir, base_path))
        elif not base_path:
            base_path = self.base_dir

        split_dirs: Dict[str, Dict[str, str]] = {}
        for split_key in ["train", "val", "test"]:
            if split_key in yaml_cfg and yaml_cfg[split_key]:
                rel_img_dir = yaml_cfg[split_key]
                abs_img_dir = rel_img_dir if os.path.isabs(rel_img_dir) else os.path.normpath(os.path.join(base_path, rel_img_dir))
                
                # Derive corresponding labels directory
                # Common YOLO convention: dataset/images/<split> -> dataset/labels/<split>
                if "images" in abs_img_dir:
                    abs_label_dir = abs_img_dir.replace("images", "labels")
                else:
                    abs_label_dir = os.path.normpath(os.path.join(os.path.dirname(abs_img_dir), "labels", split_key))

                split_dirs[split_key] = {
                    "images": abs_img_dir,
                    "labels": abs_label_dir
                }

        if not split_dirs:
            results["errors"].append("No valid splits (train/val/test) configured in data.yaml.")
            return results

        # 4. Inspect each split
        split_image_hashes: Dict[str, Dict[str, str]] = {}  # split -> hash -> filepath

        for split_name, paths in split_dirs.items():
            img_dir = paths["images"]
            lbl_dir = paths["labels"]

            split_stat = {
                "split": split_name,
                "images_dir": img_dir,
                "labels_dir": lbl_dir,
                "images_exist": os.path.exists(img_dir),
                "labels_exist": os.path.exists(lbl_dir),
                "image_count": 0,
                "label_count": 0,
                "box_count": 0,
                "class_counts": {i: 0 for i in names.keys()}
            }

            if not os.path.exists(img_dir):
                results["warnings"].append(f"Image directory for '{split_name}' does not exist: {img_dir}")
                results["splits"][split_name] = split_stat
                continue

            # Gather images
            img_files = [
                f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            ]
            split_stat["image_count"] = len(img_files)
            results["total_images"] += len(img_files)

            split_image_hashes[split_name] = {}

            # Process images and labels
            for img_name in img_files:
                img_path = os.path.join(img_dir, img_name)
                
                # Check image readability and compute hash
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                    im_hash = self.compute_file_hash(img_path)
                    split_image_hashes[split_name][im_hash] = img_path
                except Exception as e:
                    results["corrupt_images"].append({"file": img_path, "error": str(e)})
                    continue

                # Check corresponding label
                base_stem = os.path.splitext(img_name)[0]
                lbl_file = f"{base_stem}.txt"
                lbl_path = os.path.join(lbl_dir, lbl_file)

                if not os.path.exists(lbl_path):
                    results["missing_labels"].append({"image": img_path, "expected_label": lbl_path})
                    continue

                split_stat["label_count"] += 1
                results["total_labels"] += 1

                # Parse label file
                try:
                    with open(lbl_path, "r", encoding="utf-8") as lf:
                        lines = [ln.strip() for ln in lf.readlines() if ln.strip()]

                    for line_idx, line in enumerate(lines, 1):
                        parts = line.split()
                        if len(parts) != 5:
                            results["invalid_boxes"].append({
                                "file": lbl_path,
                                "line": line_idx,
                                "raw": line,
                                "reason": f"Expected 5 tokens, got {len(parts)}"
                            })
                            continue

                        try:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = [float(p) for p in parts[1:5]]
                        except ValueError as ve:
                            results["invalid_boxes"].append({
                                "file": lbl_path,
                                "line": line_idx,
                                "raw": line,
                                "reason": f"Value parsing error: {ve}"
                            })
                            continue

                        # Bounds validation [0.0, 1.0]
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                            results["invalid_boxes"].append({
                                "file": lbl_path,
                                "line": line_idx,
                                "raw": line,
                                "reason": "Normalized coordinate out of bounds (must be in [0.0, 1.0], width/height > 0)"
                            })
                            continue

                        if cls_id not in names:
                            results["invalid_boxes"].append({
                                "file": lbl_path,
                                "line": line_idx,
                                "raw": line,
                                "reason": f"Class ID {cls_id} not defined in YAML names"
                            })
                            continue

                        # Box is valid
                        split_stat["box_count"] += 1
                        results["total_boxes"] += 1
                        split_stat["class_counts"][cls_id] = split_stat["class_counts"].get(cls_id, 0) + 1
                        results["class_distribution"][cls_id] = results["class_distribution"].get(cls_id, 0) + 1

                except Exception as e:
                    results["corrupt_labels"].append({"file": lbl_path, "error": str(e)})

            # Check for orphaned labels (labels without corresponding images)
            if os.path.exists(lbl_dir):
                label_files = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")]
                img_stems = {os.path.splitext(f)[0] for f in img_files}
                for lf in label_files:
                    stem = os.path.splitext(lf)[0]
                    if stem not in img_stems:
                        results["orphaned_labels"].append(os.path.join(lbl_dir, lf))

            results["splits"][split_name] = split_stat

        # 5. Check for Cross-Split Leakage
        split_keys = list(split_image_hashes.keys())
        for i in range(len(split_keys)):
            for j in range(i + 1, len(split_keys)):
                s1, s2 = split_keys[i], split_keys[j]
                common_hashes = set(split_image_hashes[s1].keys()) & set(split_image_hashes[s2].keys())
                for ch in common_hashes:
                    pair = {
                        "split_a": s1,
                        "file_a": split_image_hashes[s1][ch],
                        "split_b": s2,
                        "file_b": split_image_hashes[s2][ch],
                        "hash": ch
                    }
                    results["leakage_pairs"].append(pair)

        results["leakage_count"] = len(results["leakage_pairs"])
        if results["leakage_count"] > 0:
            results["errors"].append(
                f"Data leakage detected: {results['leakage_count']} identical image(s) shared between splits."
            )

        # 6. Overall Validity Determination
        has_critical_errors = (
            len(results["errors"]) > 0 or
            len(results["corrupt_images"]) > 0 or
            len(results["invalid_boxes"]) > 0 or
            results["leakage_count"] > 0
        )
        has_data = results["total_images"] > 0 and results["total_boxes"] > 0
        results["valid"] = (not has_critical_errors) and has_data

        if not has_data and not results["errors"]:
            results["warnings"].append(
                "Dataset contains zero images or annotations. Preparation is complete, but training data is missing."
            )

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print formatted terminal summary of validation results."""
        print("=" * 65)
        print(" ROADGUARD AI - YOLO Dataset Integrity & Leakage Report")
        print("=" * 65)
        print(f" Dataset Config:    {results['data_yaml']}")
        print(f" YAML Valid:        {'YES' if results['yaml_parsed'] else 'NO'}")
        print(f" Taxonomy Valid:    {'YES' if results['taxonomy_valid'] else 'NO'}")
        print(f" Total Images:      {results['total_images']}")
        print(f" Total Labels:      {results['total_labels']}")
        print(f" Total Annotations: {results['total_boxes']}")
        print(f" Data Leakage:      {results['leakage_count']} duplicate pairs")
        print(f" Corrupt Images:    {len(results['corrupt_images'])}")
        print(f" Invalid Boxes:     {len(results['invalid_boxes'])}")
        print(f" Missing Labels:    {len(results['missing_labels'])}")
        print(f" Orphaned Labels:   {len(results['orphaned_labels'])}")
        print("-" * 65)

        for s_name, s_data in results.get("splits", {}).items():
            print(f" Split [{s_name.upper():<5}]: {s_data['image_count']} images, "
                  f"{s_data['label_count']} labels, {s_data['box_count']} boxes")

        if results.get("class_distribution"):
            print("-" * 65)
            print(" Class Distribution:")
            for cid, count in sorted(results["class_distribution"].items()):
                cname = config.CLASS_NAMES.get(cid, f"Class {cid}")
                ccode = config.CLASS_CODES.get(cid, "UNK")
                print(f"   [{ccode}] {cname:<24}: {count:>6} annotations")

        if results.get("errors"):
            print("-" * 65)
            print(" [!] ERRORS DETECTED:")
            for err in results["errors"]:
                print(f"     * {err}")

        if results.get("warnings"):
            print("-" * 65)
            print(" [*] WARNINGS:")
            for w in results["warnings"]:
                print(f"     * {w}")

        print("=" * 65)
        status_label = "VALID & READY" if results["valid"] else "BLOCKED / INCOMPLETE"
        print(f" STATUS: {status_label}")
        print("=" * 65)


def validate_yolo_dataset(data_yaml_path: str = "data.yaml", verbose: bool = True) -> Dict[str, Any]:
    """Convenience function to run validation and optionally display report."""
    validator = DatasetValidator(data_yaml_path=data_yaml_path, verbose=verbose)
    results = validator.validate()
    if verbose:
        validator.print_summary(results)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate YOLO dataset structure, boxes, and cross-split leakage.")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose stdout output")
    args = parser.parse_args()

    res = validate_yolo_dataset(args.data, verbose=not args.quiet)
    # Exit code: 0 if valid or clean empty dataset without errors, 1 if critical errors present
    if len(res["errors"]) > 0 or len(res["corrupt_images"]) > 0 or len(res["invalid_boxes"]) > 0:
        sys.exit(1)
    sys.exit(0)
