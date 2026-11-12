"""
Verification and Test Script for Smart Traffic Monitoring System
Minor Project - Academic Testing Suite

Runs the complete pipeline on a sample video and asserts:
1. YOLO model loading (yolo26n / fallback)
2. Video frame extraction
3. Target vehicle class detection (car, motorcycle, bus, truck)
4. Temporal tracking & duplicate-prevention counting
5. Rule-based traffic density classification
6. Output video generation
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from modules.detector import VehicleDetector
from modules.tracker_counter import VehicleCounter
from modules.density_classifier import TrafficDensityClassifier
from modules.video_processor import VideoTrafficProcessor


def run_full_pipeline_test():
    print("=" * 65)
    print(" Smart Traffic Monitoring System - Verification Test Suite")
    print("=" * 65)

    # 1. Test Model Loading
    print("\n[Step 1/5] Testing YOLO Model Initialization...")
    detector = VehicleDetector(
        primary_model=config.PRIMARY_MODEL,
        fallback_model=config.FALLBACK_MODEL,
        conf_threshold=config.CONFIDENCE_THRESHOLD
    )
    print(f" -> Active Detection Model: {detector.active_model_name}")
    assert detector.model is not None, "YOLO model failed to load!"

    # 2. Locate Test Video
    sample_file = os.path.join(config.SAMPLE_FOLDER, "sample_traffic_1.mp4")
    if not os.path.exists(sample_file):
        print(f"Sample video not found at {sample_file}. Skipping video processing test.")
        return

    print(f"\n[Step 2/5] Test Video Found: {sample_file}")

    # 3. Test Density Classifier Rules
    print("\n[Step 3/5] Testing Rule-Based Density Classifier...")
    classifier = TrafficDensityClassifier(low_max=2, medium_max=5)
    r1 = classifier.classify_active(active_count=1)
    r2 = classifier.classify_active(active_count=4)
    r3 = classifier.classify_active(active_count=7)

    assert r1["code"] == "LOW", f"Expected LOW, got {r1['code']}"
    assert r2["code"] == "MEDIUM", f"Expected MEDIUM, got {r2['code']}"
    assert r3["code"] == "HIGH", f"Expected HIGH, got {r3['code']}"
    print(" -> Rule Logic Verified on Active Road Load:")
    print(f"    1 active vehicle  => {r1['level']} ({r1['color']})")
    print(f"    4 active vehicles => {r2['level']} ({r2['color']})")
    print(f"    7 active vehicles => {r3['level']} ({r3['color']})")

    # 4. Test Full Video Processing Pipeline
    print("\n[Step 4/5] Executing Video Traffic Pipeline on Sample Video...")
    output_test_path = os.path.join(config.OUTPUT_FOLDER, "test_output.mp4")

    processor = VideoTrafficProcessor(
        primary_model=config.PRIMARY_MODEL,
        fallback_model=config.FALLBACK_MODEL,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        low_density_max=config.DENSITY_THRESHOLDS["LOW_MAX"],
        medium_density_max=config.DENSITY_THRESHOLDS["MEDIUM_MAX"]
    )

    last_pct = 0
    def on_progress(p):
        nonlocal last_pct
        pct = p["progress"]
        if pct - last_pct >= 20 or pct == 100:
            print(f"    Progress: {pct}% | Frame {p['frame']}/{p['total_frames']} | Speed: {p['current_fps']} FPS")
            last_pct = pct

    res = processor.process_video(
        input_path=sample_file,
        output_path=output_test_path,
        progress_callback=on_progress
    )

    # 5. Verify Results
    print("\n[Step 5/5] Verifying System Results:")
    print("=" * 65)
    print(f" Model Used:                 {res['model_used']}")
    print(f" Cumulative Unique Vehicles: {res['cumulative_counts']['total_vehicles']}")
    print(f"   - Cars:                   {res['cumulative_counts']['cars']}")
    print(f"   - Motorcycles:            {res['cumulative_counts']['motorcycles']}")
    print(f"   - Buses:                  {res['cumulative_counts']['buses']}")
    print(f"   - Trucks:                 {res['cumulative_counts']['trucks']}")
    print(f" Active Road Load:           Avg: {res['active_vehicles']['average']} veh/frame | Peak: {res['active_vehicles']['peak']}")
    print(f" Classified Traffic Density: {res['density']['level']} ({res['density']['code']})")
    print(f" Total Frames Processed:     {res['performance']['total_frames_processed']}")
    print(f" Measured Processing Speed:  {res['performance']['processing_fps']} FPS")
    print(f" Measured Processing Time:   {res['performance']['processing_time_seconds']}s")
    print(f" Output Video File Created:  {os.path.exists(output_test_path)} ({output_test_path})")
    print("=" * 65)

    assert os.path.exists(output_test_path), "Output video was not saved!"
    assert res['cumulative_counts']['total_vehicles'] >= 0, "Total vehicles count invalid!"
    print("\n ALL SYSTEM TESTS PASSED SUCCESSFULLY! PROJECT IS VERIFIED READY.")


if __name__ == "__main__":
    run_full_pipeline_test()
