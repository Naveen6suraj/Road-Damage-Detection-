"""
ROADGUARD AI - YOLO11 Road Damage Detector
Module: core/detector.py

Handles model initialization, hardware acceleration (NVIDIA RTX GPU / CPU),
inference execution, class name normalization, and visual annotation.
Gracefully handles missing weights without crashing.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image

import config
from core.severity_engine import estimate_severity, SEVERITY_THRESHOLDS


# Color palette for severity levels in BGR (OpenCV format)
SEVERITY_COLORS_BGR = {
    "CRITICAL": (50, 50, 235),    # Vibrant Red
    "HIGH":     (30, 115, 245),   # Deep Orange
    "MODERATE": (15, 180, 245),   # Amber / Yellow
    "LOW":      (80, 185, 20),    # Emerald Green
    "DEFAULT":  (200, 200, 200)   # Neutral Gray
}


class RoadDamageDetector:
    """
    Robust road damage detector wrapping YOLO11 with automatic device selection,
    class normalization, severity integration, and visual annotation.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or config.MODEL_WEIGHTS_PATH
        self.device = device or config.DEVICE
        self.model = None
        self.is_custom_trained = False
        self.status = "UNINITIALIZED"
        self.status_message = ""
        self._load_model()

    def _load_model(self):
        """Safely load YOLO model weights with hardware acceleration."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self.status = "DEPENDENCY_ERROR"
            self.status_message = "Ultralytics package is not installed."
            return

        # Check for trained custom model weights
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000:
            try:
                self.model = YOLO(self.model_path)
                self.is_custom_trained = True
                self.status = "READY_CUSTOM"
                self.status_message = f"Loaded custom weights from {os.path.basename(self.model_path)} on {self.device}."
                print(f"[ROADGUARD AI] {self.status_message}")
                return
            except Exception as e:
                print(f"[ROADGUARD AI] Error loading custom weights: {e}")

        # Check for base model fallback (e.g. yolo11n.pt)
        fallback_path = os.path.join(config.MODEL_FOLDER, config.FALLBACK_MODEL_NAME)
        if not os.path.exists(fallback_path):
            fallback_path = config.FALLBACK_MODEL_NAME

        try:
            # Attempt to initialize fallback YOLO11n
            self.model = YOLO(fallback_path)
            self.is_custom_trained = False
            self.status = "READY_FALLBACK"
            self.status_message = (
                f"Trained weights 'models/best.pt' not found. "
                f"Operating in fallback mode using {config.FALLBACK_MODEL_NAME}."
            )
            print(f"[ROADGUARD AI] {self.status_message}")
        except Exception as e:
            self.model = None
            self.status = "WEIGHTS_MISSING"
            self.status_message = (
                f"No usable model weights found at {self.model_path}. "
                f"Please place trained 'best.pt' inside the models/ folder."
            )
            print(f"[ROADGUARD AI] {self.status_message}")

    def get_status(self) -> Dict[str, Any]:
        """Return operational status and hardware info."""
        return {
            "status": self.status,
            "status_message": self.status_message,
            "is_custom_trained": self.is_custom_trained,
            "model_path": self.model_path,
            "device": self.device,
            "device_name": config.DEVICE_NAME,
            "cuda_available": config.CUDA_AVAILABLE
        }

    def _normalize_class(self, cls_id: int, cls_name: str) -> Tuple[str, str]:
        """
        Normalize detected class ID and name to standardized RoadGuard classes.
        Handles both custom model output and variant class labels (e.g., Pothole vs Potholes).
        """
        lower_name = cls_name.lower().strip()
        
        # Check alias map
        if lower_name in config.CLASS_ALIAS_MAP:
            std_id = config.CLASS_ALIAS_MAP[lower_name]
            return config.CLASS_CODES[std_id], config.CLASS_NAMES[std_id]
        
        # Check numeric ID if custom trained
        if self.is_custom_trained and cls_id in config.CLASS_CODES:
            return config.CLASS_CODES[cls_id], config.CLASS_NAMES[cls_id]

        # For base model detections, heuristically map or classify as OTHER
        return "OTHER", "Other Corruption"

    def detect(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        conf: float = config.DEFAULT_CONFIDENCE,
        imgsz: int = config.DEFAULT_IMGSZ,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
        """
        Run road defect detection on an image frame.

        Parameters:
        -----------
        image_input: str path, BGR numpy array, or PIL Image.
        conf: float confidence threshold (0.0 to 1.0).
        imgsz: int inference size.
        context: dict optional context for severity estimation.

        Returns:
        --------
        (annotated_bgr_image, detections_list, class_counts)
        """
        # Load frame into BGR numpy array
        if isinstance(image_input, str):
            frame = cv2.imread(image_input)
            if frame is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        elif isinstance(image_input, Image.Image):
            frame = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            frame = image_input.copy()
        else:
            raise TypeError("Unsupported image input type.")

        h, w = frame.shape[:2]
        class_counts = {code: 0 for code in config.CLASS_CODES.values()}
        detections = []

        if self.model is None:
            # Model weights not available - annotate warning banner
            cv2.putText(
                frame, "ROADGUARD AI: Model weights not found in models/best.pt",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            return frame, detections, class_counts

        # Clear GPU cache if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run inference
        results = self.model(frame, conf=conf, imgsz=imgsz, device=self.device, verbose=False)
        annotated = frame.copy()

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                box = [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])]
                score = float(confs[i])
                cls_idx = int(clss[i])
                raw_name = self.model.names.get(cls_idx, "Unknown")

                class_code, class_name = self._normalize_class(cls_idx, raw_name)
                class_counts[class_code] = class_counts.get(class_code, 0) + 1

                det_payload = {
                    "box": box,
                    "confidence": score,
                    "class_code": class_code,
                    "class_name": class_name
                }

                # Run Severity Engine
                severity_result = estimate_severity(det_payload, (w, h), context=context)
                
                full_detection = {
                    **det_payload,
                    "severity_score": severity_result["severity_score"],
                    "severity_level": severity_result["severity_level"],
                    "reason": severity_result["reason"],
                    "contributing_factors": severity_result["contributing_factors"]
                }
                detections.append(full_detection)

                # Draw sophisticated HUD bounding box on annotated frame
                self._draw_detection_overlay(annotated, full_detection)

        return annotated, detections, class_counts

    def _draw_detection_overlay(self, image: np.ndarray, detection: Dict[str, Any]):
        """Render high-contrast, professional bounding box and severity HUD badge."""
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        sev_level = detection.get("severity_level", "LOW")
        color = SEVERITY_COLORS_BGR.get(sev_level, SEVERITY_COLORS_BGR["DEFAULT"])
        
        # Bounding Box with rounded corners appearance
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Corner accent markers
        accent_len = max(8, min(20, (x2 - x1) // 5))
        cv2.line(image, (x1, y1), (x1 + accent_len, y1), color, 3)
        cv2.line(image, (x1, y1), (x1, y1 + accent_len), color, 3)
        cv2.line(image, (x2, y1), (x2 - accent_len, y1), color, 3)
        cv2.line(image, (x2, y1), (x2, y1 + accent_len), color, 3)
        cv2.line(image, (x1, y2), (x1 + accent_len, y2), color, 3)
        cv2.line(image, (x1, y2), (x1, y2 - accent_len), color, 3)
        cv2.line(image, (x2, y2), (x2 - accent_len, y2), color, 3)
        cv2.line(image, (x2, y2), (x2, y2 - accent_len), color, 3)

        # Label Header: "D40 Pothole | 88% | HIGH (74)"
        code = detection.get("class_code", "DEFECT")
        name = detection.get("class_name", "")
        conf = int(detection.get("confidence", 0.0) * 100)
        sev_score = detection.get("severity_score", 0.0)
        track_id = detection.get("track_id")

        id_str = f"#{track_id} " if track_id else ""
        label = f"{id_str}{code} {name} {conf}% | {sev_level} [{sev_score:.0f}]"

        # Badge background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        bg_y1 = max(0, y1 - txt_h - 8)
        bg_y2 = y1
        bg_x1 = x1
        bg_x2 = min(image.shape[1], x1 + txt_w + 10)

        # Dark background backing with colored border for readability
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 25, 35), -1)
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)

        # Label text
        cv2.putText(
            image, label, (bg_x1 + 5, bg_y2 - 4),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
