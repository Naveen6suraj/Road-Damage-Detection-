# ROADGUARD AI: Quality Assurance & Testing Protocol
Version: 2.1.1 (Audited Release)

## 1. Automated Test Suite Overview

ROADGUARD AI includes a comprehensive, multi-layer automated test suite implemented with `pytest`. The test suite validates core AI/ML inference, mathematical severity scoring consistency, spatial-temporal multi-object tracking, video decoding/encoding pipelines, executive PDF and CSV report compilation, and RESTful API route endpoints.

### Summary of Executed Test Suites:
- **`tests/test_severity.py`**: Severity scoring logic, boundary values, category hazard base points, perspective proximity, out-of-bounds/inverted bounding box clamping, and raw vs adjusted score distinction.
- **`tests/test_tracker.py`**: Bounding box IoU calculation, spatial track association across consecutive frames, track identity persistence, multi-frame deduplication, transient vs confirmed classification, and class-consistent tracking.
- **`tests/test_detector.py`**: Detector initialization, hardware accelerator reporting (RTX GPU CUDA vs CPU), strict custom model validation (rejection of generic COCO, rejection of partially compatible models missing required classes, rejection of incompatible mappings), class name normalization, and synthetic frame inference.
- **`tests/test_video.py`**: Video metadata extraction, OpenCV `VideoCapture` and `VideoWriter` resource lifecycle, frame-stride sampling, and corrupt video error handling.
- **`tests/test_reports.py`**: ReportLab PDF generation with Status and Raw/Adjusted severity scores, table formatting, and CSV defect registry compilation.
- **`tests/test_routes.py`**: Flask application factory, HTTP status codes, JSON response schemas, file size and extension validation, model availability 503 rejection, `/api/model/status` endpoint, directory traversal protection (`/outputs/../`, `/reports/../../`), safe analysis ID regex enforcement, production secret key requirement, legacy `/predict_image` route compatibility, and 404 error handler.

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
tests/test_detector.py::test_class_name_normalization PASSED             [  4%]
tests/test_detector.py::test_reject_generic_coco_model PASSED            [  7%]
tests/test_detector.py::test_reject_partially_compatible_model PASSED    [  9%]
tests/test_detector.py::test_reject_incompatible_classes_model PASSED    [ 11%]
tests/test_detector.py::test_accept_complete_valid_road_damage_model PASSED [ 14%]
tests/test_detector.py::test_detection_on_synthetic_frame PASSED         [ 16%]
tests/test_reports.py::test_generate_csv_report PASSED                   [ 19%]
tests/test_reports.py::test_generate_pdf_report PASSED                   [ 21%]
tests/test_routes.py::test_index_page PASSED                             [ 23%]
tests/test_routes.py::test_dashboard_page PASSED                         [ 26%]
tests/test_routes.py::test_analysis_page PASSED                          [ 28%]
tests/test_routes.py::test_api_dashboard_stats PASSED                    [ 30%]
tests/test_routes.py::test_api_system_status PASSED                      [ 33%]
tests/test_routes.py::test_image_upload_without_file PASSED              [ 35%]
tests/test_routes.py::test_image_upload_invalid_extension PASSED         [ 38%]
tests/test_routes.py::test_api_model_status PASSED                       [ 40%]
tests/test_routes.py::test_image_upload_missing_custom_model PASSED      [ 42%]
tests/test_routes.py::test_video_upload_missing_custom_model PASSED      [ 45%]
tests/test_routes.py::test_image_upload_valid_synthetic PASSED           [ 47%]
tests/test_routes.py::test_legacy_predict_image_route PASSED             [ 50%]
tests/test_routes.py::test_camera_process_frame_missing_payload PASSED   [ 52%]
tests/test_routes.py::test_404_handler PASSED                            [ 54%]
tests/test_routes.py::test_security_path_traversal_protection PASSED     [ 57%]
tests/test_routes.py::test_security_invalid_analysis_id PASSED           [ 59%]
tests/test_routes.py::test_security_production_secret_key_enforcement PASSED [ 61%]
tests/test_severity.py::test_classify_severity_score PASSED              [ 64%]
tests/test_severity.py::test_estimate_severity_critical_pothole PASSED   [ 66%]
tests/test_severity.py::test_estimate_severity_minor_crack PASSED        [ 69%]
tests/test_severity.py::test_estimate_severity_moderate_transverse_crack PASSED [ 71%]
tests/test_severity.py::test_severity_persistence_context PASSED         [ 73%]
tests/test_severity.py::test_zero_dimensions_edge_case PASSED            [ 76%]
tests/test_severity.py::test_out_of_bounds_and_inverted_bbox PASSED      [ 78%]
tests/test_severity.py::test_raw_and_adjusted_severity_distinction PASSED [ 80%]
tests/test_tracker.py::test_compute_iou PASSED                           [ 83%]
tests/test_tracker.py::test_tracker_association_across_frames PASSED     [ 85%]
tests/test_tracker.py::test_tracker_deduplication PASSED                 [ 88%]
tests/test_tracker.py::test_tracker_transient_vs_confirmed PASSED        [ 90%]
tests/test_tracker.py::test_tracker_different_classes_no_association PASSED [ 92%]
tests/test_video.py::test_get_video_metadata PASSED                      [ 95%]
tests/test_video.py::test_video_processor_pipeline PASSED                [ 97%]
tests/test_video.py::test_corrupt_video_handling PASSED                  [100%]

============================= 42 passed in 5.84s ==============================
```

---

## 3. Severity Rule Consistency Evaluation

Executed via `evaluate.py --eval-severity-only`:

```text
======================================================================
ROADGUARD AI - Severity Engine Consistency Evaluation
======================================================================
1. Massive Near-Field Pothole:
   Score: 93.1 -> Level: CRITICAL (Expected: CRITICAL) -> [PASS]
2. Large Alligator Crack Section:
   Score: 79.0 -> Level: HIGH (Expected: HIGH) -> [PASS]
3. Moderate Transverse Crack:
   Score: 57.4 -> Level: MODERATE (Expected: MODERATE) -> [PASS]
4. Minor Distant Longitudinal Crack:
   Score: 26.4 -> Level: LOW (Expected: LOW) -> [PASS]

Rule Consistency Score: 4/4 Passed (100.0%)
======================================================================
```
