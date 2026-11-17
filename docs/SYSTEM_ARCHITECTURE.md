# ROADGUARD AI: System Architecture Specification
Version: 3.0.0 (Field Intelligence Release)

## 1. Executive Architecture Overview

ROADGUARD AI is an edge-optimized, modular computer vision system designed to detect roadway defects (potholes and structural cracking), track defect occurrences across consecutive video frames, compute empirical visual risk severity indices, and generate executive inspection reports.

The application adheres to clean separation of concerns:
- **Presentation Layer**: Field Intelligence Design System featuring modular CSS tokens (`design-system.css`, `layout.css`, `components.css`, `dashboard.css`, `analysis.css`, `responsive.css`), fixed vertical sidebar navigation with Lucide-style SVGs, Space Grotesk + Inter typography, high-contrast severity tiers, responsive desktop-to-mobile layouts, and interactive Chart.js visualizations.
- **Routing & Service Layer**: Flask modular blueprints providing RESTful endpoints, input validation, and asynchronous job monitoring across Command Center (`/`), Analysis Studio (`/analysis`), Road Intelligence (`/dashboard`), History (`/history`), and Reports (`/reports`).
- **Computer Vision & AI/ML Layer**: YOLO11 Nano object detection with automated hardware dispatch (NVIDIA CUDA GPU acceleration with CPU fallback) and strict road-damage taxonomy validation.
- **Severity Estimation Engine**: Empirical visual risk scoring module evaluating defect type, normalized relative bounding box area, lane perspective proximity, aspect ratio, and multi-frame persistence.
- **Multi-Object Tracking Engine**: Spatial-temporal defect association utilizing Intersection-over-Union (IoU) and centroid proximity to eliminate duplicate counting in video streams.
- **Reporting & Persistence Layer**: SQLite relational database for historical analyses and ReportLab engine for cryptographic-free, professional PDF and CSV generation.

---

## 2. End-to-End System Flowchart

```mermaid
flowchart TD
    subgraph Client ["Client Interface (Browser)"]
        UI_Dash["Dashboard & Analytics"]
        UI_Upload["Upload & Analysis Studio"]
        UI_Cam["WebRTC Live Camera Feed"]
        UI_Reports["Report Export Interface"]
    end

    subgraph Server ["Flask Application Layer (app.py)"]
        BP_Dash["Dashboard Routes"]
        BP_Img["Image Analysis Routes"]
        BP_Vid["Video Analysis & Async Worker"]
        BP_Cam["Camera Frame Processor"]
        BP_Rep["Report Generation Routes"]
    end

    subgraph AI_Core ["Core AI/ML & Analytics Engine"]
        YOLO_Engine["YOLO11 Nano Detector\n(NVIDIA RTX 3050 CUDA / CPU)"]
        Sev_Engine["Severity Estimation Engine\n(Visual Risk Index 0-100)"]
        Track_Engine["Spatial-Temporal Defect Tracker\n(SORT / IoU Deduplication)"]
        Vid_Proc["Video Pipeline & OpenCV HUD Renderer"]
    end

    subgraph Data_Layer ["Storage & Database Layer"]
        DB[(SQLite roadguard.db\nanalyses & detections)]
        Disk_Uploads["uploads/ Directory"]
        Disk_Outputs["outputs/ Directory"]
        Disk_Reports["reports/ (PDF & CSV)"]
        Disk_Models["models/ (best.pt / yolo11n.pt)"]
    end

    UI_Upload -->|HTTP POST Image| BP_Img
    UI_Upload -->|HTTP POST Video| BP_Vid
    UI_Cam -->|Base64 JPEG Frame| BP_Cam
    UI_Dash -->|HTTP GET Stats| BP_Dash
    UI_Reports -->|HTTP GET Download| BP_Rep

    BP_Img --> Disk_Uploads
    BP_Img --> YOLO_Engine
    BP_Vid --> Disk_Uploads
    BP_Vid --> Vid_Proc

    Vid_Proc --> YOLO_Engine
    Vid_Proc --> Sev_Engine
    Vid_Proc --> Track_Engine
    Vid_Proc --> Disk_Outputs

    YOLO_Engine --> Sev_Engine
    BP_Cam --> YOLO_Engine

    BP_Img --> DB
    BP_Vid --> DB
    BP_Rep --> DB
    BP_Rep --> Disk_Reports
```

---

## 3. Video Processing & Defect Tracking Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User as Field Inspector / User
    participant Browser as Client UI (Browser)
    participant Flask as Flask Video Route
    participant Worker as Background Thread
    participant Detector as YOLO11 Detector
    participant Severity as Severity Engine
    participant Tracker as Defect Tracker
    participant DB as SQLite Database

    User->>Browser: Selects Dashcam MP4 & Clicks "Process Video"
    Browser->>Flask: POST /api/analyze/video (file, conf=0.25)
    Flask->>Worker: Spawn Thread (_execute_video_pipeline)
    Flask-->>Browser: Return JSON { success: true, job_id: "VID-..." }

    loop Progress Polling (Every 800ms)
        Browser->>Flask: GET /api/video/progress/VID-...
        Flask-->>Browser: Return Progress { percent: 45%, fps: 28.5, tracks: 3 }
        Browser->>Browser: Update Progress Bar & Real-Time Stats
    end

    loop Frame by Frame Video Loop
        Worker->>Detector: Run YOLO11 on Frame (Tensor CUDA)
        Detector-->>Worker: Raw Bounding Boxes & Confidence
        Worker->>Severity: Compute Visual Severity Index (0-100)
        Severity-->>Worker: Score, Level (CRITICAL/HIGH/MODERATE/LOW), Reason
        Worker->>Tracker: Associate Detections with Active Tracks (IoU >= 0.25)
        Tracker-->>Worker: Annotated Tracks (track_id, hits)
        Worker->>Worker: Draw HUD Bounding Boxes & Top Status Banner
    end

    Worker->>Tracker: Finalize Session (Deduplicate Unique Defects)
    Tracker-->>Worker: Confirmed Unique Road Defects Registry
    Worker->>DB: Save Analysis Record & Unique Detections
    Worker-->>Flask: Job Status: COMPLETED

    Browser->>Flask: GET /api/video/progress/VID-...
    Flask-->>Browser: Return Final Summary & Result Video URL
    Browser->>Browser: Render Video Player & Unique Defects Table
```

---

## 4. Hardware Optimization & Memory Architecture

The system is configured specifically for standard workstation hardware:
- **Primary GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MB VRAM).
- **Host CPU**: AMD Ryzen 5 5600H (6 Cores, 12 Threads).
- **RAM**: 16 GB DDR4.
- **Operating System**: Windows 11.

### Memory Optimization Safeguards:
1. **CUDA Cache Management**: Explicit calls to `torch.cuda.empty_cache()` before multi-frame batch inference prevent CUDA Out-of-Memory (OOM) conditions within 4GB VRAM.
2. **Batch Pacing**: Video processing streams frames sequentially through OpenCV buffers rather than buffering entire uncompressed videos in system RAM.
3. **Graceful Fallback**: If CUDA drivers or GPU hardware are absent, `core/detector.py` automatically initializes inference on CPU without throwing uncaught exceptions.
