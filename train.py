"""
ROADGUARD AI - YOLO11 Road Damage Model Training Pipeline
Script: train.py

Automates custom YOLO11 model training with hardware-aware device selection
(NVIDIA RTX 3050 Laptop GPU / CPU fallback), dataset validation, and model export.
"""

import os
import sys
import argparse
import yaml
import torch
import shutil

import config


def validate_dataset(data_yaml_path: str) -> bool:
    """Validate dataset configuration file and verify that images and labels exist."""
    if not os.path.exists(data_yaml_path):
        print(f"[!] Dataset configuration file not found at: {data_yaml_path}")
        return False

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except Exception as e:
            print(f"[!] Error parsing {data_yaml_path}: {e}")
            return False

    base_path = cfg.get("path", "")
    train_rel = cfg.get("train", "")
    val_rel = cfg.get("val", "")

    train_dir = os.path.join(base_path, train_rel) if not os.path.isabs(train_rel) else train_rel
    val_dir = os.path.join(base_path, val_rel) if not os.path.isabs(val_rel) else val_rel

    if not os.path.exists(train_dir):
        print(f"[!] Training images directory not found: {train_dir}")
        print("    Please run 'convert_xml_to_yolo.py' after placing raw data in 'Data/'.")
        return False

    # Check for image files
    img_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if len(img_files) == 0:
        print(f"[!] No training images found in: {train_dir}")
        return False

    if not os.path.exists(val_dir) or len([f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]) == 0:
        print(f"[!] Validation images directory missing or empty at: {val_dir}")
        print("    Run 'python convert_xml_to_yolo.py --val-split 0.2' to create train and validation splits.")
        return False

    print(f"[+] Dataset validation passed: Found {len(img_files)} train images in {train_dir}.")
    return True


def run_training(
    data_yaml: str = "data.yaml",
    epochs: int = 100,
    batch_size: int = 4,
    imgsz: int = 640,
    device: str = "auto",
    model_name: str = "yolo11n.pt",
    workers: int = 2
):
    """Execute YOLO11 transfer learning pipeline."""
    from ultralytics import YOLO

    # 1. Device configuration
    if device == "auto":
        target_device = "0" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    print("=" * 60)
    print(" ROADGUARD AI - YOLO11 Training Pipeline")
    print(f" Target Device: {target_device} (CUDA: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f" GPU Model: {torch.cuda.get_device_name(0)}")
        print(f" VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f" Base Model: {model_name}")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | Img Size: {imgsz}")
    print("=" * 60)

    # 2. Dataset verification
    if not validate_dataset(data_yaml):
        print("\n[!] Training aborted: Dataset validation failed.")
        print("    To train a road damage detection model:")
        print("    1. Download the RDD2022 dataset from https://crddc2022.sekilab.global/")
        print("    2. Extract it into the 'Data/' folder.")
        print("    3. Run: python convert_xml_to_yolo.py")
        print("    4. Then re-run: python train.py")
        sys.exit(1)

    # 3. Load base model for transfer learning
    print(f"\n[*] Initializing {model_name} for transfer learning...")
    model = YOLO(model_name)

    # 4. Train model
    print("\n[*] Starting training loop...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=target_device,
        workers=workers,
        cache=False,
        pretrained=True,
        optimizer="SGD" if target_device == "cpu" else "auto",
        verbose=True,
        project="runs/detect",
        name="roadguard_yolo11"
    )

    # 5. Export trained weights to models/best.pt
    best_weights = os.path.join("runs", "detect", "roadguard_yolo11", "weights", "best.pt")
    if os.path.exists(best_weights):
        os.makedirs(config.MODEL_FOLDER, exist_ok=True)
        shutil.copy2(best_weights, config.MODEL_WEIGHTS_PATH)
        print(f"\n[+] Successfully exported best model weights to: {config.MODEL_WEIGHTS_PATH}")
    else:
        print("\n[!] Training finished, but best.pt was not located in default runs directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLO11 Road Damage Detection Model")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--batch", default=4, type=int, help="Batch size (default 4 for 4GB RTX 3050 safety)")
    parser.add_argument("--imgsz", default=640, type=int, help="Image resolution for training")
    parser.add_argument("--device", default="auto", help="Device to use ('auto', '0', or 'cpu')")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained YOLO base model")
    parser.add_argument("--workers", default=2, type=int, help="DataLoader workers")
    args = parser.parse_args()

    run_training(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        model_name=args.model,
        workers=args.workers
    )
