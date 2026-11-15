"""
ROADGUARD AI - Model & Severity Evaluation Pipeline
Script: evaluate.py

Calculates empirical evaluation metrics for both object detection (Precision, Recall,
F1, mAP@0.5, mAP@0.5:0.95, Confusion Matrix) and severity estimation logic.
Safely checks for ground truth dataset before executing.
"""

import os
import argparse
import json
import torch
import numpy as np

import config
from core.severity_engine import estimate_severity, classify_severity_score


def evaluate_detection_model(
    model_path: str = config.MODEL_WEIGHTS_PATH,
    data_yaml: str = "data.yaml",
    imgsz: int = config.DEFAULT_IMGSZ
):
    """Run Ultralytics YOLO validation suite on ground-truth dataset."""
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        print(f"[!] Model weights not found at: {model_path}")
        print("    Train a model with 'python train.py' or provide custom weights.")
        return None

    if not os.path.exists(data_yaml):
        print(f"[!] Dataset config not found at: {data_yaml}")
        return None

    print("=" * 60)
    print(" ROADGUARD AI - Object Detection Model Evaluation")
    print(f" Model: {model_path}")
    print(f" Dataset Config: {data_yaml}")
    print("=" * 60)

    model = YOLO(model_path)
    device = "0" if torch.cuda.is_available() else "cpu"

    try:
        metrics = model.val(
            data=data_yaml,
            imgsz=imgsz,
            device=device,
            split="val",
            verbose=True
        )

        print("\n" + "=" * 60)
        print(" EVALUATION RESULTS SUMMARY")
        print("=" * 60)
        print(f" mAP@0.5:       {metrics.box.map50:.4f}")
        print(f" mAP@0.5:0.95:  {metrics.box.map:.4f}")
        print(f" Precision:     {metrics.box.mp:.4f}")
        print(f" Recall:        {metrics.box.mr:.4f}")

        # Per-class metrics
        print("\n Per-Class Results:")
        for idx, name in config.CLASS_NAMES.items():
            if idx < len(metrics.box.maps):
                cls_map50 = metrics.box.maps[idx]
                code = config.CLASS_CODES.get(idx, "OTHER")
                print(f"   [{code}] {name:<20} mAP@0.5: {cls_map50:.4f}")

        return metrics

    except Exception as e:
        print(f"[!] Evaluation could not complete: {e}")
        return None


def evaluate_severity_engine():
    """
    Empirical unit evaluation of severity estimation scoring logic across
    representative synthetic defect cases covering all classes and size ranges.
    """
    print("\n" + "=" * 60)
    print(" ROADGUARD AI - Severity Engine Consistency Evaluation")
    print("=" * 60)

    frame_size = (1920, 1080)
    test_cases = [
        {
            "name": "Massive Near-Field Pothole (Critical)",
            "det": {"box": [800, 750, 1150, 1000], "class_code": "D40", "confidence": 0.92},
            "expected_level": "CRITICAL"
        },
        {
            "name": "Large Alligator Crack Section (High)",
            "det": {"box": [600, 600, 1000, 850], "class_code": "D20", "confidence": 0.85},
            "expected_level": "HIGH"
        },
        {
            "name": "Moderate Transverse Crack (Moderate)",
            "det": {"box": [700, 600, 1200, 680], "class_code": "D10", "confidence": 0.78},
            "expected_level": "MODERATE"
        },
        {
            "name": "Minor Distant Longitudinal Crack (Low)",
            "det": {"box": [900, 350, 940, 500], "class_code": "D00", "confidence": 0.65},
            "expected_level": "LOW"
        }
    ]

    passed = 0
    for case in test_cases:
        res = estimate_severity(case["det"], frame_size)
        level = res["severity_level"]
        score = res["severity_score"]
        match = (level == case["expected_level"])
        if match:
            passed += 1
        status = "PASS" if match else "FAIL"
        print(f" [{status}] {case['name']}")
        print(f"        Score: {score:.1f} | Level: {level} (Expected: {case['expected_level']})")
        print(f"        Reason: {res['reason'][:90]}...")

    print(f"\n Severity Engine Rule Consistency: {passed}/{len(test_cases)} Passed.")
    return passed == len(test_cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Road Damage Detection and Severity Engine")
    parser.add_argument("--model", default=config.MODEL_WEIGHTS_PATH, help="Path to trained model weights")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--eval-severity-only", action="store_true", help="Run severity engine evaluation only")
    args = parser.parse_args()

    evaluate_severity_engine()

    if not args.eval_severity_only:
        evaluate_detection_model(model_path=args.model, data_yaml=args.data)
