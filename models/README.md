# ROADGUARD AI: Model Weights Directory

This directory stores trained deep learning weights for road damage detection.

## Supported Model Formats:
1. **`best.pt` (Custom Model Weights)**:
   - When present, ROADGUARD AI automatically loads this file as the primary custom-trained road damage detector.
   - Standard Road Damage classes:
     - `0`: Longitudinal Crack (`D00`)
     - `1`: Transverse Crack (`D10`)
     - `2`: Alligator Crack (`D20`)
     - `3`: Pothole (`D40`)
     - `4`: Other Corruption (`OTHER`)
   - Placed automatically here upon completing training with `python train.py`.

2. **`yolo11n.pt` (Base / Fallback Model)**:
   - Ultralytics YOLO11 Nano base weights used for transfer learning in `train.py` or fallback verification when custom `best.pt` has not yet been trained.
   - Operating status is transparently reported in the application UI as "Evaluation / Fallback Mode".

## Instructions for Supplying Trained Weights:
If you have trained weights from an existing training run:
1. Copy your `.pt` checkpoint file into this folder:
   `models/best.pt`
2. Restart the application or refresh the dashboard. ROADGUARD AI will automatically detect and load `best.pt` on the active GPU/CPU device.
