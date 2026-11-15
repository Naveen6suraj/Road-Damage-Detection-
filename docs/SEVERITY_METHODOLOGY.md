# ROADGUARD AI: Damage Severity Estimation Methodology

## 1. Academic Rationale & Context

Traditional computer vision applications for pavement inspection typically restrict output to binary or multi-class detection (e.g., bounding boxes labeled "pothole" or "crack"). However, in civil infrastructure management and municipal road maintenance, detection alone is insufficient. Road maintenance crews must prioritize repairs based on **hazard risk**, operational urgency, and potential vehicular damage.

ROADGUARD AI introduces an empirical **Visual Severity Risk Index ($S \in [0, 100]$)**, translating 2D visual observation metrics into actionable maintenance tiers.

---

## 2. Monocular Visual Estimation vs. Physical Calibration

> [!IMPORTANT]
> **Scientific Integrity & Limitations Notice:**
> Physical defect dimensions—specifically **calibrated depth in millimeters** or **true ground surface area in square meters**—cannot be determined from monocular 2D dashcam imagery without calibrated multi-camera stereo geometry, structured LiDAR depth sensors, or high-precision IMU/odometry telemetry.
> 
> Therefore, ROADGUARD AI makes **no false claims** of measuring physical pothole depth in centimeters. Instead, it computes an **Empirical Visual Risk Score** reflecting relative visual severity, prospective tire impact risk, and pavement deterioration scale.
> 
> Furthermore, **Model Detection Confidence** ($C \in [0.0, 1.0]$) is maintained as a strictly separate variable representing classification certainty, and is never conflated with physical damage severity.

---

## 3. Mathematical Formulation of Severity Score

The composite Visual Severity Score $S$ is computed as the sum of four primary spatial components and an optional temporal persistence modifier:

$$S = \min\left(100.0, \max\left(5.0, W_{\text{cat}} + W_{\text{area}} + W_{\text{pos}} + W_{\text{geom}} + \Delta_{\text{pers}}\right)\right)$$

Where:

### A. Defect Category Base Hazard ($W_{\text{cat}} \in [8.0, 35.0]$)
Different roadway defect classes pose fundamentally distinct mechanical risks to vehicular stability, tire integrity, and structural pavement life:

| Class Code | Category Description | Base Hazard Points ($W_{\text{cat}}$) | Civil Engineering Justification |
| :--- | :--- | :---: | :--- |
| **D40** | Pothole | **35.0** | High immediate mechanical hazard (rim bending, tire blowout, suspension failure, loss of control). |
| **D20** | Alligator Crack | **25.0** | Structural base failure indicating localized fatigue of the pavement foundation under load. |
| **D10** | Transverse Crack | **16.0** | Thermal contraction or subgrade shifting perpendicular to traffic; vehicle jolting and water ingress. |
| **D00** | Longitudinal Crack | **10.0** | Longitudinal joint fatigue along wheel paths; early-stage distress. |
| **OTHER** | Surface Corruption | **8.0** | Surface spalling, ravelling, or minor weathering distress. |

### B. Normalized Damaged Area ($W_{\text{area}} \in [1.0, 35.0]$)
Defect bounding box area is normalized relative to total camera frame area:

$$A_{\text{ratio}} = \frac{W_{\text{bbox}} \times H_{\text{bbox}}}{W_{\text{frame}} \times H_{\text{frame}}} \times 100\%$$

The area contribution $W_{\text{area}}$ is computed via piecewise linear scaling:
- If $A_{\text{ratio}} \ge 4.0\%$: $W_{\text{area}} = 35.0$ pts (Massive defect covering significant portion of the lane).
- If $2.0\% \le A_{\text{ratio}} < 4.0\%$: $W_{\text{area}} = 25.0 + \left(\frac{A_{\text{ratio}} - 2.0}{2.0}\right) \times 10.0$ pts.
- If $0.8\% \le A_{\text{ratio}} < 2.0\%$: $W_{\text{area}} = 15.0 + \left(\frac{A_{\text{ratio}} - 0.8}{1.2}\right) \times 10.0$ pts.
- If $0.2\% \le A_{\text{ratio}} < 0.8\%$: $W_{\text{area}} = 6.0 + \left(\frac{A_{\text{ratio}} - 0.2}{0.6}\right) \times 9.0$ pts.
- If $A_{\text{ratio}} < 0.2\%$: $W_{\text{area}} = \max\left(1.0, \frac{A_{\text{ratio}}}{0.2} \times 5.0\right)$ pts.

### C. Perspective Road Proximity ($W_{\text{pos}} \in [0.0, 20.0]$)
In forward-facing dashcam perspective, the road plane exhibits perspective compression:
- The bottom third of the frame ($y \to H_{\text{frame}}$) represents the **near field** directly in the vehicle's immediate trajectory.
- The upper half represents distant roadway near the horizon.

$$y_{\text{norm}} = \frac{y_1 + y_2}{2 \cdot H_{\text{frame}}} \in [0.0, 1.0]$$

To reflect increased hazard as the vehicle approaches the defect:
$$W_{\text{pos}} = 20.0 \times \left(y_{\text{norm}}\right)^{2.0}$$

### D. Geometric Aspect Ratio Factor ($W_{\text{geom}} \in [3.0, 10.0]$)
The geometric aspect ratio $\text{AR} = \max\left(\frac{W_{\text{bbox}}}{H_{\text{bbox}}}, \frac{H_{\text{bbox}}}{W_{\text{bbox}}}\right)$:
- For **cracks** (`D00`, `D10`): Severe cracks propagate linearly. High elongation ($\text{AR} \ge 4.0$) yields $10.0$ pts.
- For **potholes and alligator cracking** (`D40`, `D20`): Severe instances expand in both width and height, forming broad depressions. Compact aspect ratio ($\text{AR} \le 1.8$) combined with large area indicates a deep widespread crater, yielding $10.0$ pts.

### E. Multi-Frame Temporal Persistence Adjustment ($\Delta_{\text{pers}} \in [-2.0, +3.0]$)
In video processing, individual 1-frame detections may occasionally arise from transient shadows, road debris, or camera sensor noise.
- When tracked across $\ge 5$ consecutive frames: $\Delta_{\text{pers}} = +3.0$ pts (Confirmed persistent physical defect).
- When observed in only 1 isolated frame: $\Delta_{\text{pers}} = -2.0$ pts (Transient discount).

---

## 4. Severity Risk Tiers

The calculated score maps to four standardized maintenance action levels:

| Tier | Score Range | Display Color | Maintenance Urgency |
| :--- | :---: | :---: | :--- |
| **CRITICAL** | **80.0 – 100.0** | Red (`#EF4444`) | Immediate structural hazard; urgent on-site inspection and emergency patch within 24–48 hours. |
| **HIGH** | **60.0 – 79.9** | Orange (`#F97316`) | Substantial roadbed distress; prioritized for maintenance within 7 days. |
| **MODERATE**| **30.0 – 59.9** | Amber (`#F59E0B`) | Noticeable surface cracking/wear; scheduled for routine maintenance queue. |
| **LOW** | **0.0 – 29.9** | Emerald (`#10B981`) | Superficial hairline cracking; monitor during periodic road surveys. |

---

## 5. Configuration Knobs

All scoring weights and threshold boundaries are exposed in `config.py` and `core/severity_engine.py`, permitting municipal road authorities to calibrate scoring to local road conditions and pavement standards.
