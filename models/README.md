# ROADGUARD AI: Model Weights Directory
Version: 2.1.1

This directory stores deep learning model weights for the ROADGUARD AI road distress assessment system.

---

## 1. Primary Model: `best.pt` (Custom Road-Damage Weights)
- **Role**: Primary production model for road damage detection and severity assessment.
- **Taxonomy Requirement**: The custom model **must** contain the complete standardized road damage taxonomy:
  - `0`: Longitudinal Crack (`D00`)
  - `1`: Transverse Crack (`D10`)
  - `2`: Alligator Crack (`D20`)
  - `3`: Pothole (`D40`)
  - `4`: Other Corruption (`OTHER`, optional / alias `D43`, `D44`, `D50`)
- **Generation**: Created automatically upon running `python train.py` on the preprocessed Road Damage Dataset (RDD2020 / RDD2022).
- **Validation**: When placed at `models/best.pt`, ROADGUARD AI inspects the model metadata on startup, verifies the class taxonomy, and unlocks inference across image, video, and camera pipelines.

---

## 2. Base Backbone: `yolo11n.pt` (Transfer-Learning Base Only)
- **Role**: Pretrained Ultralytics YOLO11 Nano weights trained on the 80-class Common Objects in Context (COCO) dataset.
- **Inference Restriction**: **Generic COCO weights cannot and will not be used for road-damage inference.**
- **Automatic Protection**: If `models/best.pt` is absent, the system detects `yolo11n.pt` as a generic transfer-learning base and explicitly disables road-damage inference with an informative UI message (`CUSTOM_MODEL_MISSING`). Generic classes (e.g., person, car, bicycle) are never mapped to road defects.
- **Purpose**: Serves exclusively as the initialization backbone when fine-tuning via `python train.py`.

---

## Instructions for Deploying Trained Weights
1. Train a model on the RDD2022 dataset:
   ```powershell
   python train.py --epochs 100 --batch 4 --imgsz 640 --device auto
   ```
2. Or copy your custom-trained YOLO `.pt` checkpoint with matching taxonomy to:
   `models/best.pt`
3. Launch the application:
   ```powershell
   python app.py
   ```
   ROADGUARD AI will verify the taxonomy and report `READY_CUSTOM` status.
