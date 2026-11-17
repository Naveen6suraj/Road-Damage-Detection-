"""
ROADGUARD AI - YOLO11 Road Damage Model Training Pipeline
Script: train.py
Version: 3.1.0

Automates custom YOLO11 model training with hardware-aware device selection
(NVIDIA RTX 3050 Laptop GPU / CPU fallback), comprehensive dataset validation,
reproducible seed control, automatic mixed precision (AMP), and standardized
model metadata export (models/model_metadata.json).
"""

import os
import sys
import argparse
import json
import datetime
import torch
import shutil
from typing import Dict, Any, Optional

import config
from utils.dataset_validator import validate_yolo_dataset


def generate_model_metadata(
    model_path: str,
    architecture: str = "YOLO11n",
    dataset_name: str = "RDD2022-RoadDamage",
    dataset_version: str = "1.0",
    hyperparameters: Optional[Dict[str, Any]] = None,
    metrics_summary: Optional[Dict[str, Any]] = None,
    output_path: str = "models/model_metadata.json"
) -> Dict[str, Any]:
    """
    Generate and save standardized model metadata for auditability and MLOps tracking.
    """
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB VRAM)"

    metadata = {
        "model_name": os.path.basename(model_path),
        "model_path": model_path,
        "architecture": architecture,
        "version": config.APP_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "taxonomy_mapping": {
            str(k): {
                "code": config.CLASS_CODES.get(k, "OTHER"),
                "name": config.CLASS_NAMES.get(k, "Unknown"),
                "base_hazard": config.CLASS_BASE_HAZARD.get(config.CLASS_CODES.get(k, "OTHER"), 8.0)
            }
            for k in sorted(config.CLASS_NAMES.keys())
        },
        "required_codes": sorted(list(config.REQUIRED_ROAD_DAMAGE_CODES)),
        "input_resolution": [config.DEFAULT_IMGSZ, config.DEFAULT_IMGSZ],
        "precision": "FP16 (AMP)" if torch.cuda.is_available() else "FP32",
        "device_name": device_name,
        "cuda_available": torch.cuda.is_available(),
        "hyperparameters": hyperparameters or {},
        "metrics_summary": metrics_summary or {
            "status": "UNVERIFIED",
            "mAP50": None,
            "mAP50_95": None,
            "precision": None,
            "recall": None
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Saved model metadata specification to: {output_path}")
    return metadata


def run_training(
    data_yaml: str = "data.yaml",
    epochs: int = 100,
    batch_size: int = 8,
    imgsz: int = config.DEFAULT_IMGSZ,
    device: str = "auto",
    model_name: str = "yolo11n.pt",
    workers: int = 2,
    patience: int = 15,
    seed: int = 42
):
    """
    Execute hardware-optimized YOLO11 transfer learning pipeline.
    """
    # 1. Device configuration
    if device == "auto":
        target_device = "0" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    print("=" * 65)
    print(" ROADGUARD AI - YOLO11 Training Pipeline v3.1")
    print(f" Target Device:    {target_device} (CUDA: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f" GPU Model:        {gpu_name}")
        print(f" VRAM Capacity:    {gpu_vram:.2f} GB")
        # Ensure memory allocation is safe for 4GB RTX 3050 Laptop
        if gpu_vram <= 4.5 and batch_size > 8:
            print(f" [*] Notice: Adjusting batch size from {batch_size} to 8 for 4GB VRAM stability.")
            batch_size = 8
    print(f" Base Model:       {model_name}")
    print(f" Epochs:           {epochs} | Batch: {batch_size} | Resolution: {imgsz}")
    print(f" Early Stopping:   patience={patience} | Random Seed: {seed}")
    print("=" * 65)

    # 2. Comprehensive dataset validation
    print("\n[*] Validating dataset integrity and structure...")
    val_report = validate_yolo_dataset(data_yaml, verbose=True)

    if not val_report["valid"]:
        print("\n[!] Training aborted: Dataset validation failed or dataset is empty.")
        print("    ROADGUARD AI requires a verified dataset (e.g., RDD2022) with categories:")
        print("    D00 (Longitudinal Crack), D10 (Transverse Crack),")
        print("    D20 (Alligator Crack), D40 (Pothole).")
        print("    Status: BLOCKED BY MISSING RESOURCE.")
        sys.exit(1)

    # 3. Initialize base model for transfer learning
    from ultralytics import YOLO
    print(f"\n[*] Initializing {model_name} for transfer learning...")
    model = YOLO(model_name)

    # 4. Execute training loop
    print("\n[*] Starting training loop with automatic mixed precision and early stopping...")
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": target_device,
        "workers": workers,
        "patience": patience,
        "seed": seed,
        "cache": False,
        "pretrained": True,
        "amp": torch.cuda.is_available(),
        "project": "runs/detect",
        "name": "roadguard_yolo11",
        "verbose": True
    }
    results = model.train(**train_args)

    # 5. Export trained weights to models/best.pt
    best_weights = os.path.join("runs", "detect", "roadguard_yolo11", "weights", "best.pt")
    if os.path.exists(best_weights):
        os.makedirs(config.MODEL_FOLDER, exist_ok=True)
        shutil.copy2(best_weights, config.MODEL_WEIGHTS_PATH)
        print(f"\n[+] Successfully exported best model weights to: {config.MODEL_WEIGHTS_PATH}")

        # Extract validation metrics if available
        metrics_summary = {}
        if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            metrics_summary = {
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
                "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0.0))
            }

        # Save metadata
        generate_model_metadata(
            model_path=config.MODEL_WEIGHTS_PATH,
            architecture="YOLO11n",
            hyperparameters=train_args,
            metrics_summary=metrics_summary,
            output_path="models/model_metadata.json"
        )
    else:
        print("\n[!] Training finished, but best.pt was not found in runs/detect/roadguard_yolo11/weights.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLO11 Road Damage Detection Model")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--batch", default=8, type=int, help="Batch size (default 8 for 4GB RTX 3050 safety)")
    parser.add_argument("--imgsz", default=config.DEFAULT_IMGSZ, type=int, help="Image resolution")
    parser.add_argument("--device", default="auto", help="Device to use ('auto', '0', or 'cpu')")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained YOLO base model")
    parser.add_argument("--workers", default=2, type=int, help="DataLoader workers")
    parser.add_argument("--patience", default=15, type=int, help="Early stopping patience")
    parser.add_argument("--seed", default=42, type=int, help="Reproducible random seed")
    args = parser.parse_args()

    run_training(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        model_name=args.model,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed
    )
