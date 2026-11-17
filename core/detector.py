"""
ROADGUARD AI - YOLO11 Road Damage Detector
Module: core/detector.py
Version: 2.1.0

Handles custom model initialization, strict road-damage class validation,
hardware acceleration (CUDA / CPU), and HUD overlay rendering.
Explicitly disables inference when custom road-damage weights are missing,
preventing arbitrary COCO predictions from being misrepresented as road defects.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image

import config
from core.severity_engine import estimate_severity, SEVERITY_THRESHOLDS


class ModelNotAvailableError(Exception):
    """Raised when road-damage inference is attempted without valid custom weights."""
    pass


# Color palette for severity levels in BGR (OpenCV format)
SEVERITY_COLORS_BGR = {
    "CRITICAL": (50, 50, 235),    # Vibrant Red
    "HIGH":     (30, 115, 245),   # Deep Orange
    "MODERATE": (15, 180, 245),   # Amber / Yellow
    "LOW":      (80, 185, 20),    # Emerald Green
    "DEFAULT":  (160, 160, 160)   # Neutral Gray
}


class RoadDamageDetector:
    """
    Road damage detector wrapping YOLO11 with strict class verification,
    automatic device selection, and empirical severity integration.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or config.MODEL_WEIGHTS_PATH
        self.device = device or config.DEVICE
        self.model = None
        self.is_custom_trained = False
        self.inference_enabled = False
        self.status = "UNINITIALIZED"
        self.status_message = ""
        self.active_classes: Dict[int, str] = {}
        self._load_model()

    def _is_valid_road_damage_model(self, names_dict: Dict[int, str]) -> bool:
        """
        Verify that model classes represent legitimate road damage distress types,
        rather than standard 80-class COCO objects (e.g., person, car, bicycle).
        """
        if not names_dict:
            return False

        # If it has 80 classes including 'person', it is a standard COCO model
        if len(names_dict) == 80 and any(n.lower() in ("person", "bicycle", "car") for n in names_dict.values()):
            return False

        # Check if any class matches the road damage taxonomy
        recognized = 0
        for name in names_dict.values():
            clean_name = str(name).strip().lower()
            if clean_name in config.CLASS_ALIAS_MAP:
                recognized += 1

        # Must recognize at least one road damage category
        return recognized >= 1

    def _load_model(self):
        """Safely load and validate custom YOLO road damage model weights."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self.status = "DEPENDENCY_ERROR"
            self.status_message = "Ultralytics package is not installed."
            self.inference_enabled = False
            return

        # Check if custom model file exists
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000:
            try:
                candidate_model = YOLO(self.model_path)
                names = getattr(candidate_model, "names", {})

                if self._is_valid_road_damage_model(names):
                    self.model = candidate_model
                    self.is_custom_trained = True
                    self.inference_enabled = True
                    self.active_classes = names
                    self.status = "READY_CUSTOM"
                    self.status_message = (
                        f"Custom road-damage weights loaded from '{os.path.basename(self.model_path)}' "
                        f"on {self.device}."
                    )
                    print(f"[ROADGUARD AI] {self.status_message}")
                    return
                else:
                    self.model = None
                    self.is_custom_trained = False
                    self.inference_enabled = False
                    self.status = "INCOMPATIBLE_WEIGHTS"
                    self.status_message = (
                        f"Model weights at '{self.model_path}' do not contain required road-damage classes. "
                        "Generic COCO inference is disabled to prevent false road distress reports."
                    )
                    print(f"[ROADGUARD AI] Warning: {self.status_message}")
                    return
            except Exception as e:
                self.model = None
                self.is_custom_trained = False
                self.inference_enabled = False
                self.status = "LOAD_ERROR"
                self.status_message = f"Failed to load weights at '{self.model_path}': {str(e)}"
                print(f"[ROADGUARD AI] Error: {self.status_message}")
                return

        # Custom weights not present on disk
        self.model = None
        self.is_custom_trained = False
        self.inference_enabled = False
        self.status = "CUSTOM_MODEL_MISSING"
        self.status_message = (
            f"Custom road-damage model ('{os.path.basename(self.model_path)}') is not installed. "
            "Road-damage inference is disabled. Please train a model with 'python train.py' or "
            "place trained weights in 'models/best.pt' to enable live inference."
        )
        print(f"[ROADGUARD AI] {self.status_message}")

    def get_status(self) -> Dict[str, Any]:
        """Return operational status, active model info, and hardware configuration."""
        return {
            "status": self.status,
            "status_message": self.status_message,
            "inference_enabled": self.inference_enabled,
            "is_custom_trained": self.is_custom_trained,
            "model_path": self.model_path,
            "active_classes": self.active_classes,
            "device": self.device,
            "device_name": config.DEVICE_NAME,
            "cuda_available": config.CUDA_AVAILABLE
        }

    def get_model_status(self) -> Dict[str, Any]:
        """Return model loading state and class compatibility information."""
        return self.get_status()

    def _normalize_class(self, cls_id: int, cls_name: str) -> Tuple[str, str]:
        """
        Normalize detected class ID and name to standardized RoadGuard classes:
        D00, D10, D20, D40, OTHER.
        """
        lower_name = str(cls_name).lower().strip()
        if lower_name in config.CLASS_ALIAS_MAP:
            std_id = config.CLASS_ALIAS_MAP[lower_name]
            return config.CLASS_CODES[std_id], config.CLASS_NAMES[std_id]

        if cls_id in config.CLASS_CODES:
            return config.CLASS_CODES[cls_id], config.CLASS_NAMES[cls_id]

        return "OTHER", "Other Corruption"

    def detect(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        conf: float = config.DEFAULT_CONFIDENCE,
        imgsz: int = config.DEFAULT_IMGSZ,
        context: Optional[Dict[str, Any]] = None,
        mock_detections: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
        """
        Execute road defect detection on an image frame.

        Parameters:
        -----------
        image_input: str path, BGR numpy array, or PIL Image.
        conf: float confidence threshold (0.0 to 1.0).
        imgsz: int inference size.
        context: dict optional context for severity estimation.
        mock_detections: optional list for unit tests to verify downstream pipeline.

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
        annotated = frame.copy()

        # Handle Mock Detections (Used exclusively in automated tests)
        if mock_detections is not None:
            for raw_det in mock_detections:
                code, name = self._normalize_class(
                    raw_det.get("class_id", 3),
                    raw_det.get("class_name", "Pothole")
                )
                class_counts[code] = class_counts.get(code, 0) + 1
                det_payload = {
                    "box": raw_det["box"],
                    "confidence": float(raw_det.get("confidence", 0.85)),
                    "class_code": code,
                    "class_name": name
                }
                sev = estimate_severity(det_payload, (w, h), context=context)
                full_det = {
                    **det_payload,
                    "severity_score": sev["severity_score"],
                    "severity_level": sev["severity_level"],
                    "reason": sev["reason"],
                    "contributing_factors": sev["contributing_factors"]
                }
                detections.append(full_det)
                self._draw_detection_overlay(annotated, full_det)
            return annotated, detections, class_counts

        # If custom model is not available, do not run fake or COCO inference
        if not self.inference_enabled or self.model is None:
            # Stamp notice on frame
            cv2.putText(
                annotated, "ROADGUARD AI: Custom road-damage model unavailable",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                annotated, "Inference disabled. Place best.pt in models/ to enable detection.",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
            )
            return annotated, detections, class_counts

        # Clear GPU cache if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Execute YOLO Inference
        results = self.model(frame, conf=conf, imgsz=imgsz, device=self.device, verbose=False)

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
                self._draw_detection_overlay(annotated, full_detection)

        return annotated, detections, class_counts

    def _draw_detection_overlay(self, image: np.ndarray, detection: Dict[str, Any]):
        """Render high-contrast, professional bounding box and severity HUD badge."""
        box = detection.get("box", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in box]
        sev_level = detection.get("severity_level", "LOW")
        color = SEVERITY_COLORS_BGR.get(sev_level, SEVERITY_COLORS_BGR["DEFAULT"])

        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        accent_len = max(6, min(18, (x2 - x1) // 5))
        cv2.line(image, (x1, y1), (x1 + accent_len, y1), color, 3)
        cv2.line(image, (x1, y1), (x1, y1 + accent_len), color, 3)
        cv2.line(image, (x2, y1), (x2 - accent_len, y1), color, 3)
        cv2.line(image, (x2, y1), (x2, y1 + accent_len), color, 3)
        cv2.line(image, (x1, y2), (x1 + accent_len, y2), color, 3)
        cv2.line(image, (x1, y2), (x1, y2 - accent_len), color, 3)
        cv2.line(image, (x2, y2), (x2 - accent_len, y2), color, 3)
        cv2.line(image, (x2, y2), (x2, y2 - accent_len), color, 3)

        # Badge Label: "[#ID] D40 Pothole 88% | HIGH [74]"
        code = detection.get("class_code", "DEFECT")
        name = detection.get("class_name", "")
        conf = int(detection.get("confidence", 0.0) * 100)
        sev_score = detection.get("severity_score", 0.0)
        track_id = detection.get("track_id")

        id_str = f"#{track_id} " if track_id is not None else ""
        label = f"{id_str}{code} {name} {conf}% | {sev_level} [{sev_score:.0f}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        bg_y1 = max(0, y1 - txt_h - 8)
        bg_y2 = y1
        bg_x1 = x1
        bg_x2 = min(image.shape[1], x1 + txt_w + 10)

        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 25, 35), -1)
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)

        cv2.putText(
            image, label, (bg_x1 + 5, bg_y2 - 4),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
