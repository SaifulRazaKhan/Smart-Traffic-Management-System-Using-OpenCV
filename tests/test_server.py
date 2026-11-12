"""
Complete unit and integration test for Flask web application and REST API endpoints.
Tests:
1. GET / - Dashboard interface rendering
2. GET /api/samples - Preloaded sample videos discovery
3. POST /api/upload - Empty file error (400 validation)
4. POST /api/upload - Invalid file type error (400 validation)
5. POST /api/upload - Valid MP4 upload (200 success + metadata)
6. POST /api/upload - Valid AVI upload (200 success + metadata)
7. POST /api/analyze - Job initiation
8. GET /api/status/<job_id> - Progress polling and completed results structure
"""

import sys
import os
import io
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app
import config


def create_tiny_avi(filepath: str, frames: int = 10, width: int = 320, height: int = 240):
    """Creates a tiny valid AVI video file for upload testing."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(filepath, fourcc, 10.0, (width, height))
    for i in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(img, (20 + i * 10, 80), (70 + i * 10, 130), (0, 255, 0), -1)
        writer.write(img)
    writer.release()


def test_flask_endpoints():
    print("=" * 65)
    print(" FLASK REST API ENDPOINTS & INTEGRATION AUDIT")
    print("=" * 65)
    client = app.test_client()

    # 1. GET /
    res_index = client.get("/")
    assert res_index.status_code == 200, f"Expected 200, got {res_index.status_code}"
    assert b"Smart Traffic Monitoring System" in res_index.data
    assert b"Traffic Monitoring Dashboard" in res_index.data
    assert b"Cumulative Vehicles" in res_index.data
    assert b"Active in Road Zone" in res_index.data
    print(" [PASS] 1. GET / -> HTTP 200 OK (Dashboard HTML with separated metrics rendered)")

    # 2. GET /api/samples
    res_samples = client.get("/api/samples")
    assert res_samples.status_code == 200
    data_samples = res_samples.get_json()
    assert data_samples["success"] is True
    print(f" [PASS] 2. GET /api/samples -> HTTP 200 OK ({len(data_samples['samples'])} samples available)")

    # 3. POST /api/upload - Empty upload (Expected 400 validation error)
    res_empty = client.post("/api/upload")
    assert res_empty.status_code == 400
    assert "file provided" in res_empty.get_json()["error"]
    print(" [PASS] 3. POST /api/upload (Empty) -> HTTP 400 Bad Request (Validation confirmed)")

    # 4. POST /api/upload - Invalid extension (Expected 400 validation error)
    data_txt = {"video": (io.BytesIO(b"Hello text"), "report.pdf")}
    res_invalid = client.post("/api/upload", data=data_txt, content_type="multipart/form-data")
    assert res_invalid.status_code == 400
    assert "Unsupported format" in res_invalid.get_json()["error"]
    print(" [PASS] 4. POST /api/upload (Invalid format) -> HTTP 400 Bad Request (Validation confirmed)")

    # 5. POST /api/upload - Valid MP4
    mp4_file = os.path.join(config.SAMPLE_FOLDER, "sample_traffic_1.mp4")
    with open(mp4_file, "rb") as f:
        mp4_bytes = f.read()
    data_mp4 = {"video": (io.BytesIO(mp4_bytes), "sample_upload.mp4")}
    res_mp4 = client.post("/api/upload", data=data_mp4, content_type="multipart/form-data")
    assert res_mp4.status_code == 200
    mp4_data = res_mp4.get_json()
    assert mp4_data["success"] is True
    assert mp4_data["metadata"]["width"] == 768
    uploaded_filename = mp4_data["filename"]
    print(f" [PASS] 5. POST /api/upload (Valid MP4) -> HTTP 200 OK (Saved as {uploaded_filename})")

    # 6. POST /api/upload - Valid AVI
    avi_temp = os.path.join(config.STATIC_DIR, "temp_test.avi")
    create_tiny_avi(avi_temp, frames=12)
    with open(avi_temp, "rb") as f:
        avi_bytes = f.read()
    data_avi = {"video": (io.BytesIO(avi_bytes), "tiny_test.avi")}
    res_avi = client.post("/api/upload", data=data_avi, content_type="multipart/form-data")
    assert res_avi.status_code == 200
    avi_data = res_avi.get_json()
    assert avi_data["success"] is True
    assert avi_data["metadata"]["total_frames"] == 12
    print(f" [PASS] 6. POST /api/upload (Valid AVI) -> HTTP 200 OK (Saved as {avi_data['filename']})")
    if os.path.exists(avi_temp):
        os.remove(avi_temp)

    # 7. POST /api/analyze on uploaded video (use tiny AVI for rapid test)
    res_analyze = client.post("/api/analyze", json={
        "filename": avi_data["filename"],
        "low_threshold": 2,
        "medium_threshold": 5
    })
    assert res_analyze.status_code == 200
    job_info = res_analyze.get_json()
    assert job_info["success"] is True
    job_id = job_info["job_id"]
    print(f" [PASS] 7. POST /api/analyze -> HTTP 200 OK (Analysis job initiated: {job_id})")

    # 8. GET /api/status/<job_id> polling
    print(" [INFO] Polling job status...")
    max_wait = 60
    start = time.time()
    final_job = None
    while time.time() - start < max_wait:
        res_status = client.get(f"/api/status/{job_id}")
        assert res_status.status_code == 200
        job_data = res_status.get_json()["job"]
        if job_data["status"] == "completed":
            final_job = job_data
            break
        elif job_data["status"] == "error":
            raise RuntimeError(f"Job failed: {job_data.get('error_message')}")
        time.sleep(1.0)

    assert final_job is not None, "Job timed out!"
    results = final_job["results"]
    assert "cumulative_counts" in results
    assert "active_vehicles" in results
    assert "density" in results
    print(f" [PASS] 8. GET /api/status/{job_id} -> Status 'completed'")
    print(f"        - Cumulative Vehicles: {results['cumulative_counts']['total_vehicles']}")
    print(f"        - Active Road Load: Avg {results['active_vehicles']['average']} (Peak {results['active_vehicles']['peak']})")
    print(f"        - Traffic Density: {results['density']['level']} ({results['density']['code']})")
    print(f"        - Processed Video URL: {final_job['output_url']}")

    print("\nALL FLASK ENDPOINT AND ASYNC JOB TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_flask_endpoints()
