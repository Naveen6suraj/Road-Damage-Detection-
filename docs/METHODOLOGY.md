# ROADGUARD AI: Research & Engineering Methodology

## 1. Research Context & Problem Statement

Pavement distress detection is a vital component of transportation asset management and public roadway safety. Deteriorating pavement produces potholes, longitudinal fractures, transverse thermal cracks, and alligator fatigue cracking. If left unaddressed, these defects accelerate structural degradation under repeated vehicular loads, increase vehicle maintenance costs, and present significant collision hazards.

Traditional pavement management relies on manual road surveys, specialized laser profiler vans, or crowdsourced complaint logging—all of which are labor-intensive, logistically slow, and cost-prohibitive for municipal governments.

ROADGUARD AI implements an automated, real-time computer vision system capable of transforming standard forward-facing dashcam video into a continuous, spatially-deduplicated defect monitoring pipeline.

---

## 2. Dataset Pipeline & Annotation Architecture

### 2.1 The Road Damage Dataset (RDD)
ROADGUARD AI supports the international benchmark Road Damage Dataset (RDD2020/RDD2022) established by Seki Lab:
- **D00 (Longitudinal Crack)**: Fractures propagating parallel to the direction of vehicular travel, typically situated along wheel paths or paving joints.
- **D10 (Transverse Crack)**: Fractures traversing perpendicular to traffic flow, primarily driven by thermal contraction and subgrade settlement.
- **D20 (Alligator Crack)**: Interconnected fatigue cracking resembling reptilian scales, signaling structural base course exhaustion.
- **D40 (Pothole)**: Bowl-shaped surface depressions formed when water intrusion weakens the roadbed and traffic dislodges pavement chunks.
- **OTHER (Surface Corruption)**: Combined distress including rutting, bump distress, and surface weathering (`D43`, `D44`, `D50`).

### 2.2 XML-to-YOLO Conversion Protocol
The raw dataset provides bounding box annotations in Pascal VOC XML format:
$$\text{BndBox} = [x_{\min}, y_{\min}, x_{\max}, y_{\max}]$$

The script `convert_xml_to_yolo.py` normalizes spatial bounds relative to image dimensions $(W, H)$:
$$x_{\text{center}} = \frac{x_{\min} + x_{\max}}{2 \cdot W}, \quad y_{\text{center}} = \frac{y_{\min} + y_{\max}}{2 \cdot H}$$
$$w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}$$

---

## 3. Deep Learning Architecture: YOLO11 Nano

ROADGUARD AI utilizes Ultralytics YOLO11 Nano (`yolo11n.pt`) as its core object detector:
1. **Backbone**: Modified CSPDarknet featuring C3k2 building blocks and SPPF (Spatial Pyramid Pooling Fast) for multi-scale feature representation.
2. **Neck**: Path Aggregation Network (PANet) fusing shallow high-resolution spatial features with deep semantic representations.
3. **Head**: Decoupled, anchor-free detection head optimizing classification and bounding box regression losses independently.
4. **Hardware Dispatch**: Leverages PyTorch 2.6 with NVIDIA CUDA 12.4 for accelerated GPU tensor operations on laptop hardware (NVIDIA RTX 3050), processing frames at 25–40 FPS.

---

## 4. Multi-Frame Spatial-Temporal Tracking

Dashcam video processing at 25–30 FPS captures the same road defect across dozens of consecutive frames. A naive detection counter sums every observation, resulting in inflated counts (e.g., counting one pothole 45 times over 1.5 seconds).

ROADGUARD AI resolves this using the `RoadDefectTracker`:
1. **Spatial Association**: Associates detections across frame $t-1$ and frame $t$ using Intersection-over-Union (IoU) matching with spatial threshold $\tau_{\text{IoU}} = 0.25$.
2. **Track Identity Maintenance**: Assigns persistent `track_id` values to individual defects as the vehicle advances.
3. **Temporal Filtering**: Enforces a minimum hits threshold ($N_{\text{hits}} \ge 2$) to discard transient visual false alarms caused by vehicle vibrations or transient shadows.
4. **Session Deduplication**: Aggregates peak confidence, peak visual severity, and spatial trajectory into a unified defect record.

---

## 5. Empirical Severity Estimation Pipeline

The core analytical innovation is the empirical Visual Severity Risk Index ($S \in [0, 100]$):
- Combines defect category baseline risk, normalized relative surface area, perspective proximity weighting, and geometric aspect ratio.
- Categorizes defects into **CRITICAL**, **HIGH**, **MODERATE**, and **LOW** risk tiers.
- Formulates transparent, human-readable rationales stored alongside bounding boxes and exported in formal reports.
