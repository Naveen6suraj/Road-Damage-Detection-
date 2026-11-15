# ROADGUARD AI: Quality Assurance & Testing Protocol

## 1. Automated Test Suite Overview

ROADGUARD AI includes a comprehensive, multi-layer automated test suite implemented with `pytest`. The test suite validates core AI/ML inference, mathematical severity scoring consistency, spatial-temporal multi-object tracking, video decoding/encoding pipelines, executive PDF and CSV report compilation, and RESTful API route endpoints.

### Summary of Executed Test Suites:
- **`tests/test_severity.py`**: Severity scoring logic, boundary values, category hazard base points, perspective proximity, and temporal persistence modifiers.
- **`tests/test_tracker.py`**: Bounding box IoU calculation, spatial track association across consecutive frames, track identity persistence, and multi-frame deduplication.
- **`tests/test_detector.py`**: Detector initialization, hardware accelerator reporting (RTX 3050 CUDA vs CPU), class name normalization (handling plural vs singular "Pothole"), and synthetic frame inference.
- **`tests/test_video.py`**: Video metadata extraction, OpenCV `VideoCapture` and `VideoWriter` resource lifecycle, frame-stride sampling, and corrupt video error handling.
- **`tests/test_reports.py`**: ReportLab PDF generation, table formatting, and CSV defect registry compilation.
- **`tests/test_routes.py`**: Flask application factory, HTTP status codes, JSON response schemas, file size and extension validation, legacy `/predict_image` route compatibility, and 404 error handler.

---

## 2. Test Execution Command

To execute the entire automated test suite:

```powershell
python -m pytest tests/ -v
```

### Verified Test Run Results:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\navee\.gemini\antigravity\scratch\RoadGuard-AI

tests/test_detector.py::test_detector_initialization PASSED              [  3%]
tests/test_detector.py::test_class_name_normalization PASSED             [  7%]
tests/test_detector.py::test_detection_on_synthetic_frame PASSED         [ 10%]
tests/test_reports.py::test_generate_csv_report PASSED                   [ 14%]
tests/test_reports.py::test_generate_pdf_report PASSED                   [ 17%]
tests/test_routes.py::test_index_page PASSED                             [ 21%]
tests/test_routes.py::test_dashboard_page PASSED                         [ 25%]
tests/test_routes.py::test_analysis_page PASSED                          [ 28%]
tests/test_routes.py::test_api_dashboard_stats PASSED                    [ 32%]
tests/test_routes.py::test_api_system_status PASSED                      [ 35%]
tests/test_routes.py::test_image_upload_without_file PASSED              [ 39%]
tests/test_routes.py::test_image_upload_invalid_extension PASSED         [ 42%]
tests/test_routes.py::test_image_upload_valid_synthetic PASSED           [ 46%]
tests/test_routes.py::test_legacy_predict_image_route PASSED             [ 50%]
tests/test_routes.py::test_camera_process_frame_missing_payload PASSED   [ 53%]
tests/test_routes.py::test_404_handler PASSED                            [ 57%]
tests/test_severity.py::test_classify_severity_score PASSED              [ 60%]
tests/test_severity.py::test_estimate_severity_critical_pothole PASSED   [ 64%]
tests/test_severity.py::test_estimate_severity_minor_crack PASSED        [ 67%]
tests/test_severity.py::test_estimate_severity_moderate_transverse_crack PASSED [ 71%]
tests/test_severity.py::test_severity_persistence_context PASSED         [ 75%]
tests/test_severity.py::test_zero_dimensions_edge_case PASSED            [ 78%]
tests/test_tracker.py::test_compute_iou PASSED                           [ 82%]
tests/test_tracker.py::test_tracker_association_across_frames PASSED     [ 85%]
tests/test_tracker.py::test_tracker_deduplication PASSED                 [ 89%]
tests/test_video.py::test_get_video_metadata PASSED                      [ 92%]
tests/test_video.py::test_video_processor_pipeline PASSED                [ 96%]
tests/test_video.py::test_corrupt_video_handling PASSED                  [100%]

============================= 28 passed in 9.80s ==============================
```

---

## 3. Severity Consistency Verification

To run standalone verification of the severity estimation scoring rules:

```powershell
python evaluate.py --eval-severity-only
```

### Verified Evaluation Output:
```text
============================================================
 ROADGUARD AI - Severity Engine Consistency Evaluation
============================================================
 [PASS] Massive Near-Field Pothole (Critical)
        Score: 93.1 | Level: CRITICAL (Expected: CRITICAL)
 [PASS] Large Alligator Crack Section (High)
        Score: 79.0 | Level: HIGH (Expected: HIGH)
 [PASS] Moderate Transverse Crack (Moderate)
        Score: 57.4 | Level: MODERATE (Expected: MODERATE)
 [PASS] Minor Distant Longitudinal Crack (Low)
        Score: 26.4 | Level: LOW (Expected: LOW)

 Severity Engine Rule Consistency: 4/4 Passed.
```
