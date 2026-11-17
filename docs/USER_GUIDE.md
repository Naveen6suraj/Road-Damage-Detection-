# ROADGUARD AI: Comprehensive User & Demonstration Guide
Version: 3.0.0 (Field Intelligence Release)

## 1. System Requirements & Setup

### Hardware:
- **Processor**: Intel Core i5 / AMD Ryzen 5 or better.
- **GPU (Recommended)**: NVIDIA GeForce RTX GPU (e.g. RTX 3050 Laptop GPU 4GB) with CUDA support. Transparent CPU fallback is supported.
- **RAM**: 8 GB minimum (16 GB recommended).
- **Disk**: 1 GB free storage.

### Software Environment:
- **Python**: 3.10, 3.11, or 3.12.
- **Dependencies**: Listed in `requirements.txt`.

### Quick Installation:
```powershell
# Navigate to project directory
cd C:\Users\navee\.gemini\antigravity\scratch\RoadGuard-AI

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Launching the Web Application

Run the application using:
```powershell
python app.py
```

Console startup banner:
```text
============================================================
  ROADGUARD AI v3.0.0
  FIELD INTELLIGENCE — Road Damage & Severity Platform
  Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA=True)
  Listening at: http://127.0.0.1:5000
============================================================
```

Open your browser and navigate to:
**`http://127.0.0.1:5000`**

---

## 3. Navigation & Interface Architecture

ROADGUARD AI v3.0 introduces an industrial **Field Intelligence** interface structured around ergonomic vertical sidebar navigation:
- **OVERVIEW**: Command Center (`/`) — High-impact split-hero workstation with live roadway HUD simulation preview, hardware telemetry strip, asymmetric KPI metrics, and recent inspections.
- **INSPECTION**: Analysis Studio (`/analysis`) — 3-mode workstation for Road Image, Dashcam Video, and WebRTC Camera feeds.
- **INTELLIGENCE**:
  - Road Intelligence (`/dashboard`) — Aggregate pavement metrics, RDD2022 distress specifications table with base weights ($W_{cat}$) and SLAs, and Chart.js visualizations.
  - Inspection History (`/history`) — Filterable registry with live client-side search, media filter, severity tier filter, and record deletion.
  - Report Management (`/reports`) — Centralized document hub with metadata cards (ReportLab PDF Flowables, RFC 4180 CSV, Seki Lab Standard) and export actions.
- **SYSTEM**: Hardware Diagnostics Modal with real-time PyTorch accelerator telemetry and device runtime metrics.

---

## 4. Academic Evaluation Demonstration Protocol

Follow this exact sequence to conduct a demonstration for reviewers or evaluators:

### Step 1: Open the Command Center
- Navigate to `http://127.0.0.1:5000/`.
- Review the **Editorial Split-Hero**: active roadway HUD bounding box simulation viewport.
- Observe the **Hardware Readiness Strip** (`CUDA 12.4 ACTIVE • YOLO11 NANO`) confirming GPU acceleration.
- Point out the **Asymmetric KPI Tiles** (Deduplicated Defects, Critical Hazards, High Severity, Moderate, and Low).
- Review the **Damage Composition Breakdown** and **Severity Risk Ratio** bars.

### Step 2: Road Image Detection & Severity Assessment
- Click on **Analysis Studio** in the sidebar (`/analysis`).
- Ensure the **📸 Road Image Analysis** tab is selected.
- Drag & drop or click to select a road photo (`.jpg` or `.png`).
- Adjust the **Confidence Threshold Slider** (e.g. 25% or 30%).
- Click **🔍 Run Detection & Severity Assessment**.
- Observe:
  1. Annotated road image displayed with high-contrast HUD bounding boxes and severity color tags.
  2. The class counter pills updating across `D00`, `D10`, `D20`, `D40`, and `OTHER`.
  3. The **Defect Registry Table** listing each defect, its confidence percentage, visual severity score (0–100), severity level badge, and assessment rationale.

### Step 3: Executive Report Generation & Download
- Click **PDF Report** in the top-right of the inspection results.
- Review the generated official PDF document containing:
  - Header branding and Inspection ID.
  - Executive defect and severity breakdown tables.
  - Detailed defect registry.
  - Academic methodology and monocular vision disclaimer.
- Click **CSV Data** to download the raw structured dataset for GIS integration.

### Step 4: Dashcam Video Processing & Multi-Object Tracking
- In Analysis Studio, click the **🎥 Dashcam Video Analysis** tab.
- Upload a roadway dashcam video (`.mp4`, `.avi`, or `.mov`).
- Set the confidence threshold.
- Click **🚀 Process Dashcam Video & Track Defects**.
- Point out the **Real-Time Progress Bar**:
  - Percentage completed.
  - Current frame vs. total frames.
  - Processing throughput in frames-per-second (FPS).
  - Real-time active defect track counter.

### Step 5: Video Analytics & Tracking Deduplication
- Upon completion, the annotated video player displays the processed footage with HUD overlays.
- Highlight the **Tracking Deduplication Statistics**:
  - Compare **Unique Defects** (e.g., 2 confirmed road hazards) against **Total Frame Observations** (e.g., 48 frame detections).
  - Emphasize that the system **does not** falsely inflate defects by counting the same pothole on every frame.
- Inspect the **Unique Tracked Defect Registry** detailing each defect's `track_id`, duration in seconds, and peak severity score.

### Step 6: Live Webcam Inspection
- Click the **📹 Live Camera Feed** tab.
- Click **📹 Start Live Camera**.
- Point the camera at a road image or roadway scene.
- Show live frames streaming to the backend YOLO detector with low-latency overlays, live FPS counter, and real-time hazard severity badges.
- Click **⏹ Stop Camera** to safely release the camera device.

### Step 7: Historical Inspection Registry & Reports Center
- Navigate to **History** (`/history`) to demonstrate real-time search, filtering by media type or severity tier, and atomic deletion.
- Navigate to **Reports** (`/reports`) to demonstrate centralized PDF Flowable and CSV compilation for municipal maintenance workflows.
