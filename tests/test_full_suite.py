"""
Smart Traffic Monitoring System - Comprehensive Final Verification Suite
B.Tech Computer Science Minor Project

Automated Test Suite covering:
1. Model loading & fallback verification (YOLO26n / YOLO11n)
2. Target vehicle class mapping verification (Cars=2, Motorcycles=3, Buses=5, Trucks=7)
3. Static image inference & detection of ALL 4 vehicle categories
4. Rule-based traffic density classification based on active road load
5. ByteTrack & virtual-line counting duplicate-prevention logic
6. Full video processing pipeline & CPU FPS benchmark
7. Negative validations & error handling (empty, unsupported, blank frame)
8. Flask REST API integration (Synchronous image + Asynchronous video jobs)
"""

import os
import sys
import io
import time
import cv2
import numpy as np

# Ensure project root in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from app import app, SHARED_DETECTOR, SHARED_PROCESSOR
from modules.detector import VehicleDetector
from modules.tracker_counter import VehicleCounter
from modules.density_classifier import TrafficDensityClassifier
from modules.video_processor import VideoTrafficProcessor


def run_full_verification():
    print("=" * 75)
    print(" SMART TRAFFIC MONITORING SYSTEM - FULL VERIFICATION TEST SUITE")
    print("=" * 75)
    
    passed_tests = 0
    total_tests = 8

    client = app.test_client()

    # -------------------------------------------------------------
    # 1. MODEL LOADING & FALLBACK VERIFICATION
    # -------------------------------------------------------------
    print("\n[TEST 1/8] Verifying YOLO Model Initialization & Target Class Mapping...")
    try:
        assert SHARED_DETECTOR is not None, "SHARED_DETECTOR not initialized in app.py"
        print(f" -> Active Detection Model: {SHARED_DETECTOR.active_model_name}")
        assert "yolo" in SHARED_DETECTOR.active_model_name.lower()
        
        # Verify fallback logic by requesting non-existent primary model
        fallback_detector = VehicleDetector(
            primary_model=os.path.join(PROJECT_ROOT, "missing_primary_model.pt"),
            fallback_model=config.FALLBACK_MODEL
        )
        print(f" -> Fallback test loaded: {fallback_detector.active_model_name}")
        assert fallback_detector.model is not None, "Fallback model failed to load!"

        # Verify target vehicle classes
        target_classes = SHARED_DETECTOR.target_class_map
        print(f" -> Target vehicle class mapping: {target_classes}")
        assert 2 in target_classes and target_classes[2] == "car"
        assert 3 in target_classes and target_classes[3] == "motorcycle"
        assert 5 in target_classes and target_classes[5] == "bus"
        assert 7 in target_classes and target_classes[7] == "truck"
        print(" [PASS] Test 1: Model loading and class mappings verified.")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 1 failed: {e}")
        raise

    # -------------------------------------------------------------
    # 2. STATIC IMAGE DETECTION ON ALL 4 TARGET CLASSES
    # -------------------------------------------------------------
    print("\n[TEST 2/8] Testing Image Detection on All 4 COCO Vehicle Categories...")
    test_samples = [
        ("sample_car.jpg", "car", "Car"),
        ("sample_motorcycle.jpg", "motorcycle", "Motorcycle"),
        ("sample_bus.jpg", "bus", "Bus"),
        ("sample_truck.jpg", "truck", "Truck")
    ]

    detected_classes = set()
    for fname, expected_class, label in test_samples:
        img_path = os.path.join(config.SAMPLE_IMAGE_FOLDER, fname)
        assert os.path.exists(img_path), f"Sample image {fname} missing from {config.SAMPLE_IMAGE_FOLDER}!"
        
        out_path = os.path.join(config.OUTPUT_FOLDER, f"test_annotated_{fname}")
        res = SHARED_PROCESSOR.process_image(img_path, out_path)
        
        assert res["status"] == "success"
        assert os.path.exists(out_path), f"Annotated image not created: {out_path}"
        
        # Check detected objects
        found_expected = any(d["class_name"] == expected_class for d in res["detections"])
        print(f" -> {label} Image ({fname}): Detected {res['cumulative_counts']['total_vehicles']} total vehicles.")
        for d in res["detections"]:
            print(f"    - Class: {d['class_name'].upper()} | Confidence: {d['confidence']:.2f} | Box: {d['box']}")
            detected_classes.add(d["class_name"])
            
        assert found_expected, f"Failed to detect expected {expected_class} in {fname}!"

    print(f" -> Confirmed detection of classes: {detected_classes}")
    assert {"car", "motorcycle", "bus", "truck"}.issubset(detected_classes), "Not all 4 classes were detected!"
    print(" [PASS] Test 2: All 4 vehicle categories (Car, Motorcycle, Bus, Truck) verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # 3. TRAFFIC DENSITY CLASSIFIER LOGIC
    # -------------------------------------------------------------
    print("\n[TEST 3/8] Testing Rule-Based Traffic Density Classifier...")
    classifier = TrafficDensityClassifier(low_max=2, medium_max=5)
    
    r_low = classifier.classify_active(active_count=1)
    r_low_boundary = classifier.classify_active(active_count=2)
    r_med = classifier.classify_active(active_count=4)
    r_med_boundary = classifier.classify_active(active_count=5)
    r_high = classifier.classify_active(active_count=6)
    r_high_heavy = classifier.classify_active(active_count=15)

    print(f" -> Active=1:  {r_low['level']} ({r_low['code']}) - Color: {r_low['color']}")
    print(f" -> Active=2:  {r_low_boundary['level']} ({r_low_boundary['code']}) [Boundary]")
    print(f" -> Active=4:  {r_med['level']} ({r_med['code']}) - Color: {r_med['color']}")
    print(f" -> Active=5:  {r_med_boundary['level']} ({r_med_boundary['code']}) [Boundary]")
    print(f" -> Active=6:  {r_high['level']} ({r_high['code']}) - Color: {r_high['color']}")
    print(f" -> Active=15: {r_high_heavy['level']} ({r_high_heavy['code']})")

    assert r_low["code"] == "LOW"
    assert r_low_boundary["code"] == "LOW"
    assert r_med["code"] == "MEDIUM"
    assert r_med_boundary["code"] == "MEDIUM"
    assert r_high["code"] == "HIGH"
    assert r_high_heavy["code"] == "HIGH"
    print(" [PASS] Test 3: Rule-based density thresholds verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # 4. DUPLICATE COUNTING PREVENTION & TEMPORAL TRACKING
    # -------------------------------------------------------------
    print("\n[TEST 4/8] Testing Duplicate Prevention and Bidirectional Tracking...")
    counter = VehicleCounter(use_counting_line=True, line_position=0.5)
    counter.reset()

    # Track 101 moving top to bottom across line (y=100 in 200h)
    counter.update([{"box": [50, 40, 90, 80], "track_id": 101, "class_name": "car", "confidence": 0.85}], 200, 200)
    counter.update([{"box": [50, 95, 90, 135], "track_id": 101, "class_name": "car", "confidence": 0.87}], 200, 200)
    counter.update([{"box": [50, 140, 90, 180], "track_id": 101, "class_name": "car", "confidence": 0.89}], 200, 200)
    counter.update([{"box": [50, 150, 90, 190], "track_id": 101, "class_name": "car", "confidence": 0.88}], 200, 200)

    # Track 102 moving bottom to top across line
    counter.update([{"box": [120, 160, 160, 190], "track_id": 102, "class_name": "truck", "confidence": 0.90}], 200, 200)
    counter.update([{"box": [120, 60, 160, 90], "track_id": 102, "class_name": "truck", "confidence": 0.92}], 200, 200)

    summary = counter.get_summary()
    print(f" -> After 6 simulated frames across 2 vehicles:")
    print(f"    Total Unique Count: {summary['total_vehicles']} (Expected: 2)")
    print(f"    Cars: {summary['cars']} (Expected: 1)")
    print(f"    Trucks: {summary['trucks']} (Expected: 1)")
    print(f"    Line crossings: {summary['line_crossings']} (Expected: 2)")
    
    assert summary["total_vehicles"] == 2, f"Expected 2 unique vehicles, got {summary['total_vehicles']}"
    assert summary["cars"] == 1
    assert summary["trucks"] == 1
    assert summary["line_crossings"] == 2
    print(" [PASS] Test 4: Duplicate counting prevention and bidirectional crossing verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # 5. FULL VIDEO PIPELINE VERIFICATION & CPU FPS BENCHMARK
    # -------------------------------------------------------------
    print("\n[TEST 5/8] Testing Full Video Processing Pipeline & Measuring CPU Inference Speed...")
    sample_video = os.path.join(config.SAMPLE_FOLDER, "sample_traffic_1.mp4")
    out_video = os.path.join(config.OUTPUT_FOLDER, "test_verification_out.mp4")
    
    progress_records = []
    def progress_monitor(data):
        if data["frame"] % 50 == 0 or data["progress"] == 100:
            progress_records.append(data)
            print(f"    Progress: {data['progress']}% | Frame: {data['frame']}/{data['total_frames']} | "
                  f"Active: {data['active_vehicles']} | Density: {data['current_density']['level']}")

    t_start = time.time()
    vid_res = SHARED_PROCESSOR.process_video(
        input_path=sample_video,
        output_path=out_video,
        progress_callback=progress_monitor,
        frame_stride=1
    )
    t_total = time.time() - t_start

    perf = vid_res["performance"]
    counts = vid_res["cumulative_counts"]
    active = vid_res["active_vehicles"]
    density = vid_res["density"]

    print(f" -> Video Processing Completed in {t_total:.2f}s:")
    print(f"    Resolution: {perf['video_resolution']}")
    print(f"    Frames Processed: {perf['total_frames_processed']}")
    print(f"    Measured Speed: {perf['processing_fps']:.1f} FPS (CPU)")
    print(f"    Cumulative Vehicles: {counts['total_vehicles']} (Cars: {counts['cars']}, Buses: {counts['buses']}, Motorcycles: {counts['motorcycles']}, Trucks: {counts['trucks']})")
    print(f"    Average Active in Road ROI: {active['average']} veh/frame (Peak: {active['peak']})")
    print(f"    Classified Density: {density['level']} ({density['code']})")
    
    assert os.path.exists(out_video), "Output video file missing!"
    assert counts["total_vehicles"] > 0, "No vehicles counted in sample video!"
    assert perf["processing_fps"] > 0, "Invalid FPS measurement!"
    print(" [PASS] Test 5: Full video pipeline and CPU benchmark verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # 6. NEGATIVE VALIDATIONS & ERROR HANDLING
    # -------------------------------------------------------------
    print("\n[TEST 6/8] Testing Negative Validations & Edge Case Handling...")
    # 6a. Empty POST /api/upload
    r_empty = client.post("/api/upload")
    assert r_empty.status_code == 400
    assert "file provided" in r_empty.get_json()["error"]
    print(" -> Empty upload correctly rejected with HTTP 400.")

    # 6b. Unsupported format
    txt_data = {"video": (io.BytesIO(b"Not a video"), "test.txt")}
    r_txt = client.post("/api/upload", data=txt_data, content_type="multipart/form-data")
    assert r_txt.status_code == 400
    assert "Unsupported format" in r_txt.get_json()["error"]
    print(" -> Invalid file extension rejected with HTTP 400.")

    # 6c. Blank / black image (0 vehicles)
    blank_img_path = os.path.join(config.STATIC_DIR, "test_blank.jpg")
    blank_mat = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(blank_img_path, blank_mat)
    blank_out = os.path.join(config.OUTPUT_FOLDER, "test_blank_out.jpg")
    blank_res = SHARED_PROCESSOR.process_image(blank_img_path, blank_out)
    assert blank_res["cumulative_counts"]["total_vehicles"] == 0
    assert blank_res["density"]["code"] == "LOW"
    print(" -> Blank image (0 detections) handled cleanly without errors.")
    if os.path.exists(blank_img_path): os.remove(blank_img_path)
    if os.path.exists(blank_out): os.remove(blank_out)

    print(" [PASS] Test 6: Negative validations and edge cases verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # 7. FLASK REST API - SYNCHRONOUS IMAGE WORKFLOW
    # -------------------------------------------------------------
    print("\n[TEST 7/8] Testing Flask REST API: Image Input Workflow...")
    # Upload image
    with open(os.path.join(config.SAMPLE_IMAGE_FOLDER, "sample_car.jpg"), "rb") as f:
        img_bytes = f.read()
    upload_res = client.post("/api/upload", data={"video": (io.BytesIO(img_bytes), "upload_car.jpg")}, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    upload_json = upload_res.get_json()
    assert upload_json["success"] is True
    assert upload_json["file_type"] == "image"
    img_filename = upload_json["filename"]
    print(f" -> Uploaded test image: {img_filename} (Type: {upload_json['file_type']})")

    # Analyze image synchronously
    analyze_res = client.post("/api/analyze", json={"filename": img_filename, "low_threshold": 2, "medium_threshold": 5})
    assert analyze_res.status_code == 200
    analyze_json = analyze_res.get_json()
    assert analyze_json["success"] is True
    assert analyze_json["file_type"] == "image"
    assert analyze_json["status"] == "completed"
    assert "results" in analyze_json
    assert "output_url" in analyze_json
    print(f" -> Image analyzed synchronously: {analyze_json['results']['cumulative_counts']['total_vehicles']} vehicle(s) found.")
    print(" [PASS] Test 7: Synchronous image REST workflow verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # 8. FLASK REST API - ASYNCHRONOUS VIDEO WORKFLOW
    # -------------------------------------------------------------
    print("\n[TEST 8/8] Testing Flask REST API: Video Asynchronous Job Workflow...")
    # Create a small valid AVI video
    test_avi_path = os.path.join(config.STATIC_DIR, "api_test.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(test_avi_path, fourcc, 10.0, (320, 240))
    for i in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        vw.write(frame)
    vw.release()

    with open(test_avi_path, "rb") as f:
        avi_data = f.read()
    v_upload = client.post("/api/upload", data={"video": (io.BytesIO(avi_data), "api_test.avi")}, content_type="multipart/form-data")
    assert v_upload.status_code == 200
    v_json = v_upload.get_json()
    assert v_json["file_type"] == "video"
    v_filename = v_json["filename"]
    
    # Analyze video asynchronously
    v_analyze = client.post("/api/analyze", json={"filename": v_filename, "low_threshold": 2, "medium_threshold": 5})
    assert v_analyze.status_code == 200
    va_json = v_analyze.get_json()
    assert va_json["status"] == "processing"
    job_id = va_json["job_id"]
    print(f" -> Video job spawned: {job_id}")

    # Poll status until completed
    max_wait = 15
    start_poll = time.time()
    job_done = False
    while time.time() - start_poll < max_wait:
        poll_res = client.get(f"/api/status/{job_id}")
        assert poll_res.status_code == 200
        p_data = poll_res.get_json()["job"]
        if p_data["status"] == "completed":
            job_done = True
            print(f" -> Job {job_id} finished: status='completed', output={p_data['output_url']}")
            break
        elif p_data["status"] == "error":
            raise RuntimeError(f"Video job failed: {p_data['error_message']}")
        time.sleep(0.5)

    assert job_done, f"Job {job_id} did not complete within {max_wait}s!"
    if os.path.exists(test_avi_path): os.remove(test_avi_path)
    print(" [PASS] Test 8: Asynchronous video job REST workflow verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f" FINAL VERIFICATION SUMMARY: {passed_tests} / {total_tests} TESTS PASSED (100%)")
    print("=" * 75)


if __name__ == "__main__":
    run_full_verification()
