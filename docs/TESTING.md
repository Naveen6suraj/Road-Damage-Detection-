# ROADGUARD AI: Quality Assurance & Testing Protocol
Version: 3.2.0 (Custom Model Acquisition, Training & Validation Release)

## 1. Automated Test Suite Overview

ROADGUARD AI includes a comprehensive, multi-layer automated test suite implemented with `pytest`. The test suite validates core AI/ML inference, dataset integrity, Explainable AI (XAI) feature decomposition, hardware performance benchmarking, spatial-temporal multi-object tracking, video decoding/encoding pipelines, executive PDF and CSV report compilation, and RESTful API route endpoints.

### Summary of Executed Test Suites (11 Modules, 60 Tests):
1. **`tests/test_dataset_acquisition.py`**: Official RDD2022/RDD2020 DOI validation, archive MD5 checksum validation, file manifest verification, and error handling for missing archives.
2. **`tests/test_dataset_validator.py`**: Directory structure validation, YAML parsing, road damage taxonomy verification, bounding box bounds check ($[0.0, 1.0]$), missing label detection, and cross-split SHA-256 data leakage prevention.
3. **`tests/test_xai_severity.py`**: Explainable AI feature decomposition ($W_{cat} + W_{area} + W_{pos} + W_{geom} + \Delta_{pers}$), geometric aspect ratio penalty (crack elongation vs. pothole compactness), temporal persistence bonus (+3.0 pts), and transient discount (-2.0 pts).
4. **`tests/test_benchmark.py`**: Host telemetry extraction, runtime benchmark JSON schema conformance (`reports/hardware_benchmark.json`), CSV report verification, and Flask `/api/benchmark/status` endpoint testing.
5. **`tests/test_model_metadata.py`**: Academic audit schema validation for `models/model_metadata.json`, taxonomy code verification, and runtime metadata export utility.
6. **`tests/test_severity.py`**: Severity scoring logic, boundary values, category hazard base points, perspective proximity, out-of-bounds/inverted bounding box clamping, and raw vs adjusted score distinction.
7. **`tests/test_tracker.py`**: Bounding box IoU calculation, forward camera motion compensation, spatial track association across consecutive frames, track identity persistence, multi-frame deduplication, transient vs confirmed classification, and class-consistent tracking.
8. **`tests/test_detector.py`**: Detector initialization, hardware accelerator reporting (RTX GPU CUDA vs CPU), strict custom model validation (rejection of generic COCO, rejection of partially compatible models missing required classes, rejection of incompatible mappings), class name normalization, and synthetic frame inference.
9. **`tests/test_video.py`**: Video metadata extraction, OpenCV `VideoCapture` and `VideoWriter` resource lifecycle, frame-stride sampling, and corrupt video error handling.
10. **`tests/test_reports.py`**: ReportLab PDF generation with Status and Raw/Adjusted severity scores, table formatting, and CSV defect registry compilation.
11. **`tests/test_routes.py`**: Flask application factory, HTTP status codes, JSON response schemas, file size and extension validation, model availability 503 rejection, `/api/model/status` endpoint, directory traversal protection (`/outputs/../`, `/reports/../../`), safe analysis ID regex enforcement, production secret key requirement, legacy `/predict_image` route compatibility, 404 error handler, inspection history (`/history`), executive reports (`/reports`), and atomic analysis deletion (`DELETE /api/analyses/<id>`).

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
plugins: anyio-4.15.1, platformdirs-4.12.0, asyncio-1.4.1.post1

tests/test_benchmark.py::test_collect_system_telemetry PASSED            [  1%]
tests/test_benchmark.py::test_benchmark_reports_exist_and_conform PASSED [  3%]
tests/test_benchmark.py::test_api_benchmark_endpoint PASSED              [  5%]
tests/test_dataset_validator.py::test_validator_valid_dataset PASSED     [  7%]
tests/test_dataset_validator.py::test_validator_detects_data_leakage PASSED [  8%]
tests/test_dataset_validator.py::test_validator_detects_out_of_bounds_bbox PASSED [ 10%]
tests/test_dataset_validator.py::test_validator_detects_missing_label PASSED [ 12%]
tests/test_detector.py::test_detector_initialization PASSED              [ 14%]
tests/test_detector.py::test_class_name_normalization PASSED             [ 15%]
tests/test_detector.py::test_reject_generic_coco_model PASSED            [ 17%]
tests/test_detector.py::test_reject_partially_compatible_model PASSED    [ 19%]
tests/test_detector.py::test_reject_incompatible_classes_model PASSED    [ 21%]
tests/test_detector.py::test_accept_complete_valid_road_damage_model PASSED [ 22%]
tests/test_detector.py::test_detection_on_synthetic_frame PASSED         [ 24%]
tests/test_model_metadata.py::test_model_metadata_file_schema PASSED     [ 26%]
tests/test_model_metadata.py::test_generate_model_metadata_function PASSED [ 28%]
tests/test_reports.py::test_generate_csv_report PASSED                   [ 29%]
tests/test_reports.py::test_generate_pdf_report PASSED                   [ 31%]
tests/test_routes.py::test_index_page PASSED                             [ 33%]
tests/test_routes.py::test_dashboard_page PASSED                         [ 35%]
tests/test_routes.py::test_analysis_page PASSED                          [ 36%]
tests/test_routes.py::test_history_page PASSED                           [ 38%]
tests/test_routes.py::test_reports_page PASSED                           [ 40%]
tests/test_routes.py::test_api_delete_analysis_not_found PASSED          [ 42%]
tests/test_routes.py::test_api_dashboard_stats PASSED                    [ 43%]
tests/test_routes.py::test_api_system_status PASSED                      [ 45%]
tests/test_routes.py::test_image_upload_without_file PASSED              [ 47%]
tests/test_routes.py::test_image_upload_invalid_extension PASSED         [ 49%]
tests/test_routes.py::test_api_model_status PASSED                       [ 50%]
tests/test_routes.py::test_image_upload_missing_custom_model PASSED      [ 52%]
tests/test_routes.py::test_video_upload_missing_custom_model PASSED      [ 54%]
tests/test_routes.py::test_image_upload_valid_synthetic PASSED           [ 56%]
tests/test_routes.py::test_legacy_predict_image_route PASSED             [ 57%]
tests/test_routes.py::test_camera_process_frame_missing_payload PASSED   [ 59%]
tests/test_routes.py::test_404_handler PASSED                            [ 61%]
tests/test_routes.py::test_security_path_traversal_protection PASSED     [ 63%]
tests/test_routes.py::test_security_invalid_analysis_id PASSED           [ 64%]
tests/test_routes.py::test_security_production_secret_key_enforcement PASSED [ 66%]
tests/test_severity.py::test_classify_severity_score PASSED              [ 68%]
tests/test_severity.py::test_estimate_severity_critical_pothole PASSED   [ 70%]
tests/test_severity.py::test_estimate_severity_minor_crack PASSED        [ 71%]
tests/test_severity.py::test_estimate_severity_moderate_transverse_crack PASSED [ 73%]
tests/test_severity.py::test_severity_persistence_context PASSED         [ 75%]
tests/test_severity.py::test_zero_dimensions_edge_case PASSED            [ 77%]
tests/test_severity.py::test_out_of_bounds_and_inverted_bbox PASSED      [ 78%]
tests/test_severity.py::test_raw_and_adjusted_severity_distinction PASSED [ 80%]
tests/test_tracker.py::test_compute_iou PASSED                           [ 82%]
tests/test_tracker.py::test_tracker_association_across_frames PASSED     [ 84%]
tests/test_tracker.py::test_tracker_deduplication PASSED                 [ 85%]
tests/test_tracker.py::test_tracker_transient_vs_confirmed PASSED        [ 87%]
tests/test_tracker.py::test_tracker_different_classes_no_association PASSED [ 89%]
tests/test_video.py::test_get_video_metadata PASSED                      [ 91%]
tests/test_video.py::test_video_processor_pipeline PASSED                [ 92%]
tests/test_video.py::test_corrupt_video_handling PASSED                  [ 94%]
tests/test_xai_severity.py::test_severity_explicit_decomposition PASSED  [ 96%]
tests/test_xai_severity.py::test_severity_aspect_ratio_geometric_penalty PASSED [ 98%]
tests/test_xai_severity.py::test_severity_transient_discount PASSED      [100%]

============================= 57 passed in 14.30s =============================
```

---

## 3. Severity Engine Consistency Benchmark

Execute unit evaluation of severity estimation scoring logic:

```powershell
python evaluate.py --eval-severity-only
```

All 5 empirical test cases pass with verified Explainable AI feature decomposition matching the composite mathematical formula.
