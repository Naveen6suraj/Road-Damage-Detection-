# ACADEMIC PROJECT REPORT
# ROADGUARD AI: Intelligent Pothole & Road-Crack Severity Estimator from Dashcam Video

**Author / Candidate**: Senior AI/ML Engineering & Computer Vision Research Team  
**Institution / Department**: Department of Computer Science & Engineering, Advanced Machine Learning Division  
**Academic Year**: 2026–2027  
**Software Artifact**: ROADGUARD AI (Version 2.0.0)

---

## Abstract

Road surface deterioration poses significant threats to vehicular safety, transportation efficiency, and civil infrastructure longevity. Traditional manual road inspection methods are time-consuming, hazardous, and logistically unscalable, while conventional deep-learning object detectors merely detect defects without evaluating their prospective physical risk.

This project presents **ROADGUARD AI**, an end-to-end intelligent road condition monitoring system. ROADGUARD AI processes forward-facing dashcam video streams and static imagery using an edge-optimized **YOLO11 Nano** neural architecture accelerated via **PyTorch CUDA 12.4 on an NVIDIA GeForce RTX 3050 Laptop GPU**. To eliminate duplicate counting across consecutive frames, a multi-object **spatial-temporal tracking algorithm** based on Intersection-over-Union (IoU) and centroid association is implemented. Furthermore, the system incorporates a novel **Empirical Severity Estimation Engine**, synthesizing defect categorization, normalized relative surface area, perspective road proximity, aspect ratio, and multi-frame persistence into a standardized 0–100 Visual Severity Risk Index categorized into four maintenance urgency tiers: **CRITICAL**, **HIGH**, **MODERATE**, and **LOW**.

The system is deployed as a full-stack Flask web application providing interactive analytics dashboards, WebRTC live camera inspection, automated PDF executive report compilation, and persistent SQLite defect logging. Comprehensive automated testing with 28 verification suites confirms the numerical stability, tracking fidelity, and operational robustness of the entire architecture.

---

## 1. Introduction & Problem Definition

Civil transportation networks represent substantial economic investments requiring continuous maintenance. However, surface distress phenomena—predominantly potholes and cracking—inevitably develop due to dynamic traffic loading, thermal expansion cycles, and water ingress. When pavement fissures go unrepaired, hydraulic pressure from passing vehicular tires erodes the granular base, rapidly creating hazardous potholes.

Municipal roadway surveys currently face three primary technical bottlenecks:
1. **Manual Visual Surveys**: Dependent on field inspectors manually driving and recording road defects, exposing workers to traffic hazards and introducing subjective human bias.
2. **High Capital Profiler Vehicles**: Specialized laser and LiDAR survey vans cost upwards of \$250,000 per unit, precluding frequent surveys across secondary and rural roads.
3. **Flaws in Prior AI Implementations**: Existing computer vision demonstrators often naively accumulate frame-level bounding boxes in video streams, multiplying defect counts 30–50x, while failing to provide severity assessments necessary for triage.

---

## 2. Objectives

The primary research and engineering objectives of ROADGUARD AI comprise:
1. **Real-Time Defect Detection**: Deploy a lightweight YOLO11 Nano model classifying road damage into standardized RDD categories: Longitudinal Cracks (`D00`), Transverse Cracks (`D10`), Alligator Cracks (`D20`), Potholes (`D40`), and Surface Distress (`OTHER`).
2. **Hardware Acceleration**: Implement automatic hardware dispatch optimizing tensor operations on consumer-grade laptop GPUs (NVIDIA RTX 3050 4GB) with seamless CPU fallback.
3. **Temporal Object Tracking & Deduplication**: Eliminate consecutive frame overcounting using spatial-temporal track association, reporting both raw frame detections and true unique defect counts.
4. **Empirical Severity Risk Formulation**: Engineer a transparent scoring methodology ($0 - 100$) combining spatial defect geometry, lane perspective proximity, and temporal persistence.
5. **Full-Stack Industrial Dashboard**: Create an intuitive web interface with interactive visualizations, live camera inference, and downloadable executive reports.

---

## 3. System Design & Architectural Implementation

The architecture is structured into decoupled modules:
- **`app.py` & Blueprints (`routes/`)**: Modular REST API handling dashboard analytics, image processing, asynchronous video jobs, live camera WebRTC canvas streaming, and report downloads.
- **`core/detector.py`**: YOLO11 model wrapper with automatic device selection (`cuda` vs `cpu`), class alias normalization, and high-contrast HUD bounding box rendering.
- **`core/severity_engine.py`**: Mathematical severity formulation computing empirical risk indices without making unsubstantiated monocular depth claims.
- **`core/object_tracker.py`**: Multi-frame IoU-based tracking maintaining persistent track IDs and filtering out transient false alarms.
- **`core/video_processor.py`**: Complete video decoding, frame-stepping, tracking, and OpenCV video writer pipeline with real-time polling progress feedback.
- **`core/report_generator.py`**: Cryptographic-free PDF compilation using ReportLab Flowables and CSV registry generation.
- **`database.py`**: SQLite relational schema (`analyses` and `detections` tables) preserving analysis history.

---

## 4. Verification & Testing

The system underwent rigorous automated testing using `pytest`:
- **Severity Rule Testing**: Verified boundary conditions, persistence weighting, and edge cases.
- **Tracker Deduplication**: Verified that a defect tracked across 15 consecutive frames resolves to exactly 1 unique confirmed defect.
- **Video Pipeline**: Verified frame extraction, OpenCV codecs (`avc1`, `mp4v`), and corrupt file exception handling.
- **API Integration**: Verified HTTP response codes, file upload validation (rejecting non-image/video formats), and JSON schema integrity.

**Result**: All 28 automated tests passed in 9.80 seconds on the NVIDIA RTX 3050 GPU.

---

## 5. Conclusion & Future Scope

ROADGUARD AI demonstrates that affordable consumer hardware combined with edge-optimized deep learning and spatial-temporal tracking can provide accurate, automated road distress inspection.

Future enhancements include:
- Integrating multi-camera stereo disparity for calibrated millimeter pothole depth estimation.
- Incorporating GPS NMEA telemetry logging to automatically plot defect coordinates onto interactive GIS maps (OpenStreetMap/Mapbox).
- Quantizing YOLO11 models to ONNX / TensorRT INT8 for embedded deployment on edge devices such as Raspberry Pi 5 or NVIDIA Jetson Orin Nano.

---

## References

1. Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y. (2020). Global Road Damage Detection: State-of-the-Art Solutions. *IEEE Transactions on Intelligent Transportation Systems*.
2. Jocher, G., & Chaurasia, A. (2024). Ultralytics YOLO11: Real-Time Object Detection and Segmentation. Ultralytics.
3. Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking (SORT). *IEEE International Conference on Image Processing (ICIP)*.
4. Federal Highway Administration (FHWA). (2014). Distress Identification Manual for the Long-Term Pavement Performance Program. U.S. Department of Transportation.
