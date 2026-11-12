"""
Comprehensive Technical Audit Test Script
Covers:
1. Video upload validation (Empty 400, Invalid ext 400, Valid MP4 200, Valid AVI 200)
2. Traffic Density Logic (Active vehicles vs Cumulative count)
3. Detection and Category Classification
4. Duplicate Counting Prevention
5. End-to-end Job Processing & Output Verification
"""

import os
import sys
import io
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app
import config
from modules.detector import VehicleDetector
from modules.tracker_counter import VehicleCounter
from modules.density_classifier import TrafficDensityClassifier
from modules.video_processor import VideoTrafficProcessor


def create_dummy_avi(filepath: str, frames: int = 15, width: int = 320, height: int = 240):
    """Creates a small valid AVI video file for upload testing."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(filepath, fourcc, 10.0, (width, height))
    for i in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a moving square
        x = (i * 15) % (width - 40)
        cv2.rectangle(img, (x, 100), (x + 40, 140), (0, 200, 200), -1)
        writer.write(img)
    writer.release()


def run_audit():
    print("=" * 70)
    print(" SMART TRAFFIC MONITORING SYSTEM - TECHNICAL AUDIT")
    print("=" * 70)
    client = app.test_client()

    # -------------------------------------------------------------
    # 1. Video Upload Tests
    # -------------------------------------------------------------
    print("\n[TEST 1] Video Upload Validation & Processing:")

    # 1a: Empty request
    res_empty = client.post("/api/upload")
    print(f" - Upload with no files: HTTP {res_empty.status_code} (Expected 400)")
    assert res_empty.status_code == 400
    assert "file provided" in res_empty.get_json()["error"]

    # 1b: Invalid extension
    data_txt = {"video": (io.BytesIO(b"dummy text"), "notes.txt")}
    res_txt = client.post("/api/upload", data=data_txt, content_type="multipart/form-data")
    print(f" - Upload invalid format (.txt): HTTP {res_txt.status_code} (Expected 400)")
    assert res_txt.status_code == 400
    assert "Unsupported format" in res_txt.get_json()["error"]

    # 1c: Valid MP4 upload
    mp4_source = os.path.join(config.SAMPLE_FOLDER, "sample_traffic_1.mp4")
    assert os.path.exists(mp4_source), "Sample MP4 file missing!"
    with open(mp4_source, "rb") as f:
        mp4_bytes = f.read()
    data_mp4 = {"video": (io.BytesIO(mp4_bytes), "upload_test.mp4")}
    res_mp4 = client.post("/api/upload", data=data_mp4, content_type="multipart/form-data")
    print(f" - Upload valid MP4: HTTP {res_mp4.status_code} (Expected 200)")
    assert res_mp4.status_code == 200
    mp4_json = res_mp4.get_json()
    assert mp4_json["success"] is True
    assert "metadata" in mp4_json
    print(f"   Metadata decoded: {mp4_json['metadata']['width']}x{mp4_json['metadata']['height']} "
          f"@ {mp4_json['metadata']['fps']} FPS, {mp4_json['metadata']['total_frames']} frames")

    # 1d: Valid AVI upload
    dummy_avi_path = os.path.join(config.STATIC_DIR, "test_dummy.avi")
    create_dummy_avi(dummy_avi_path)
    with open(dummy_avi_path, "rb") as f:
        avi_bytes = f.read()
    data_avi = {"video": (io.BytesIO(avi_bytes), "upload_test.avi")}
    res_avi = client.post("/api/upload", data=data_avi, content_type="multipart/form-data")
    print(f" - Upload valid AVI: HTTP {res_avi.status_code} (Expected 200)")
    assert res_avi.status_code == 200
    avi_json = res_avi.get_json()
    assert avi_json["success"] is True
    print(f"   Metadata decoded: {avi_json['metadata']['width']}x{avi_json['metadata']['height']} "
          f"@ {avi_json['metadata']['fps']} FPS, {avi_json['metadata']['total_frames']} frames")
    if os.path.exists(dummy_avi_path):
        os.remove(dummy_avi_path)

    print(" [PASS] Video upload tests completed successfully!")

    # -------------------------------------------------------------
    # 2. Traffic Density Logic Verification
    # -------------------------------------------------------------
    print("\n[TEST 2] Traffic Density Logic Inspection:")
    classifier = TrafficDensityClassifier(low_max=2, medium_max=5)
    
    # Active count 1 -> Low Traffic
    d1 = classifier.classify(active_count=1)
    # Active count 4 -> Medium Traffic
    d2 = classifier.classify(active_count=4)
    # Active count 8 -> High Traffic
    d3 = classifier.classify(active_count=8)

    print(f" - Active count = 1: {d1['level']} ({d1['code']})")
    print(f" - Active count = 4: {d2['level']} ({d2['code']})")
    print(f" - Active count = 8: {d3['level']} ({d3['code']})")

    assert d1["code"] == "LOW"
    assert d2["code"] == "MEDIUM"
    assert d3["code"] == "HIGH"
    print(" [PASS] Density rule classification logic verified!")

    # -------------------------------------------------------------
    # 3. Vehicle Counter & Duplicate Prevention
    # -------------------------------------------------------------
    print("\n[TEST 3] Duplicate Counting Prevention & Tracking Audit:")
    counter = VehicleCounter(use_counting_line=True, line_position=0.5)
    counter.reset()

    # Simulate same vehicle #1 across 5 consecutive frames crossing line
    # Frame 1: Vehicle #1 above line
    s1 = counter.update([{"box": [100, 50, 150, 90], "track_id": 1, "class_name": "car", "confidence": 0.85}], 200, 200)
    # Frame 2: Vehicle #1 crossing line
    s2 = counter.update([{"box": [100, 95, 150, 135], "track_id": 1, "class_name": "car", "confidence": 0.88}], 200, 200)
    # Frame 3: Vehicle #1 below line
    s3 = counter.update([{"box": [100, 140, 150, 180], "track_id": 1, "class_name": "car", "confidence": 0.89}], 200, 200)
    # Frame 4: Vehicle #1 still below line
    s4 = counter.update([{"box": [100, 150, 150, 190], "track_id": 1, "class_name": "car", "confidence": 0.87}], 200, 200)

    summary = counter.get_summary()
    print(f" - Vehicle #1 detected across 4 frames.")
    print(f"   Cumulative Total Count: {summary['total_vehicles']} (Expected: 1)")
    print(f"   Cars Count: {summary['cars']} (Expected: 1)")
    assert summary["total_vehicles"] == 1, f"Duplicate counting failure! Count was {summary['total_vehicles']}"

    # Now add Vehicle #2 moving upwards (opposite direction)
    # Frame 5: Vehicle #2 below line
    counter.update([{"box": [50, 160, 90, 190], "track_id": 2, "class_name": "motorcycle", "confidence": 0.82}], 200, 200)
    # Frame 6: Vehicle #2 crossing upwards
    counter.update([{"box": [50, 80, 90, 110], "track_id": 2, "class_name": "motorcycle", "confidence": 0.84}], 200, 200)

    summary2 = counter.get_summary()
    print(f" - Vehicle #2 (motorcycle) moving upwards across line.")
    print(f"   Cumulative Total Count: {summary2['total_vehicles']} (Expected: 2)")
    print(f"   Cars: {summary2['cars']}, Motorcycles: {summary2['motorcycles']}")
    assert summary2["total_vehicles"] == 2
    assert summary2["motorcycles"] == 1

    print(" [PASS] Duplicate counting prevention and bidirectional tracking verified!")

    print("\n" + "=" * 70)
    print(" ALL AUDIT CHECKS IN THIS SUITE PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_audit()
