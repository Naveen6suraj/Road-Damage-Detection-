# ROADGUARD AI: Quality Assurance & Testing Protocol

## 1. Automated Test Suite Overview

ROADGUARD AI includes a comprehensive, multi-layer automated test suite implemented with `pytest`. The test suite validates core AI/ML inference, mathematical severity scoring consistency, spatial-temporal multi-object tracking, video decoding/encoding pipelines, executive PDF and CSV report compilation, and RESTful API route endpoints.

### Summary of Executed Test Suites:
- **`tests/test_severity.py`**: Severity scoring logic, boundary values, category hazard base points, perspective proximity, out-of-bounds/inverted bounding box clamping, and raw vs adjusted score distinction.
- **`tests/test_tracker.py`**: Bounding box IoU calculation, spatial track association across consecutive frames, track identity persistence, multi-frame deduplication, transient vs confirmed classification, and class-consistent tracking.
- **`tests/test_detector.py`**: Detector initialization, hardware accelerator reporting (RTX 3050 CUDA vs CPU), strict custom model validation (rejection of generic COCO), class name normalization, and synthetic frame inference.
- **`tests/test_video.py`**: Video metadata extraction, OpenCV `VideoCapture` and `VideoWriter` resource lifecycle, frame-stride sampling, and corrupt video error handling.
- **`tests/test_reports.py`**: ReportLab PDF generation with Status and Raw/Adjusted severity scores, table formatting, and CSV defect registry compilation.
- **`tests/test_routes.py`**: Flask application factory, HTTP status codes, JSON response schemas, file size and extension validation, model availability 503 rejection, `/api/model/status` endpoint, legacy `/predict_image` route compatibility, and 404 error handler.

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

tests/test_detector.py::test_detector_initialization PASSED              [  2%]
tests/test_detector.py::test_class_name_normalization PASSED             [  5%]
tests/test_detector.py::test_detection_on_synthetic_frame PASSED         [  8%]
tests/test_reports.py::test_generate_csv_report PASSED                   [ 11%]
tests/test_reports.py::test_generate_pdf_report PASSED                   [ 14%]
tests/test_routes.py::test_index_page PASSED                             [ 17%]
tests/test_routes.py::test_dashboard_page PASSED                         [ 20%]
tests/test_routes.py::test_analysis_page PASSED                          [ 22%]
tests/test_routes.py::test_api_dashboard_stats PASSED                    [ 25%]
tests/test_routes.py::test_api_system_status PASSED                      [ 28%]
tests/test_routes.py::test_image_upload_without_file PASSED              [ 31%]
tests/test_routes.py::test_image_upload_invalid_extension PASSED         [ 34%]
tests/test_routes.py::test_api_model_status PASSED                       [ 37%]
tests/test_routes.py::test_image_upload_missing_custom_model PASSED      [ 40%]
tests/test_routes.py::test_video_upload_missing_custom_model PASSED      [ 42%]
tests/test_routes.py::test_image_upload_valid_synthetic PASSED           [ 45%]
tests/test_routes.py::test_legacy_predict_image_route PASSED             [ 48%]
tests/test_routes.py::test_camera_process_frame_missing_payload PASSED   [ 51%]
tests/test_routes.py::test_404_handler PASSED                            [ 54%]
tests/test_severity.py::test_classify_severity_score PASSED              [ 57%]
tests/test_severity.py::test_estimate_severity_critical_pothole PASSED   [ 60%]
tests/test_severity.py::test_estimate_severity_minor_crack PASSED        [ 62%]
tests/test_severity.py::test_estimate_severity_moderate_transverse_crack PASSED [ 65%]
tests/test_severity.py::test_severity_persistence_context PASSED         [ 68%]
tests/test_severity.py::test_zero_dimensions_edge_case PASSED            [ 71%]
tests/test_severity.py::test_out_of_bounds_and_inverted_bbox PASSED      [ 74%]
tests/test_severity.py::test_raw_and_adjusted_severity_distinction PASSED [ 77%]
tests/test_tracker.py::test_compute_iou PASSED                           [ 80%]
tests/test_tracker.py::test_tracker_association_across_frames PASSED     [ 82%]
tests/test_tracker.py::test_tracker_deduplication PASSED                 [ 85%]
tests/test_tracker.py::test_tracker_transient_vs_confirmed PASSED        [ 88%]
tests/test_tracker.py::test_tracker_different_classes_no_association PASSED [ 91%]
tests/test_video.py::test_get_video_metadata PASSED                      [ 94%]
tests/test_video.py::test_video_processor_pipeline PASSED                [ 97%]
tests/test_video.py::test_corrupt_video_handling PASSED                  [100%]

============================= 35 passed in 8.63s ==============================
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
