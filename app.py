"""
Flask Backend Application for Smart Traffic Monitoring System
Academic Minor Project - B.Tech Computer Science

Routes:
- GET /                    : Renders main dashboard interface
- POST /api/upload         : Uploads traffic image or video, returns metadata & preview
- POST /api/analyze        : Analyzes traffic image (instant) or video (background worker)
- GET /api/status/<job_id> : Polls video processing progress and final output
- GET /api/samples         : Lists bundled sample videos and sample images for demonstration
- POST /api/use-sample     : Selects a sample video or image for immediate analysis
"""

import os
import uuid
import time
import shutil
import threading
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import cv2

import config
from modules.detector import VehicleDetector
from modules.video_processor import VideoTrafficProcessor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = config.OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Pre-load YOLO vehicle detector ONCE at startup to avoid re-instantiation overhead
logger.info("Initializing shared YOLO Vehicle Detector...")
SHARED_DETECTOR = VehicleDetector(
    primary_model=config.PRIMARY_MODEL,
    fallback_model=config.FALLBACK_MODEL,
    conf_threshold=config.CONFIDENCE_THRESHOLD,
    imgsz=config.INFERENCE_IMG_SIZE
)
logger.info(f"Shared detector ready. Active model: {SHARED_DETECTOR.active_model_name}")
SHARED_PROCESSOR = VideoTrafficProcessor(detector=SHARED_DETECTOR)

# In-memory job state store for background video processing
JOBS = {}
JOBS_LOCK = threading.Lock()


def get_file_type(filename: str) -> str:
    """Returns 'video', 'image', or 'unsupported' based on extension."""
    if "." not in filename:
        return "unsupported"
    ext = filename.rsplit(".", 1)[1].lower()
    if ext in config.VIDEO_EXTENSIONS:
        return "video"
    if ext in config.IMAGE_EXTENSIONS:
        return "image"
    return "unsupported"


def allowed_file(filename: str) -> bool:
    """Validates allowed media file extensions."""
    return get_file_type(filename) in ("video", "image")


def get_video_metadata(filepath: str) -> dict:
    """Reads basic video metadata using OpenCV."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return {}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    cap.release()
    return {
        "type": "video",
        "width": width,
        "height": height,
        "fps": round(fps, 1),
        "total_frames": frames,
        "duration_seconds": round(duration, 1)
    }


def get_image_metadata(filepath: str) -> dict:
    """Reads basic image metadata using OpenCV."""
    img = cv2.imread(filepath)
    if img is None:
        return {}
    h, w = img.shape[:2]
    return {
        "type": "image",
        "width": w,
        "height": h,
        "fps": 0,
        "total_frames": 1,
        "duration_seconds": 0
    }


def run_video_job(
    job_id: str,
    input_path: str,
    output_path: str,
    low_max: int,
    med_max: int,
    frame_stride: int
):
    """Background worker function executing video processing pipeline."""
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["progress"] = 0

    def progress_callback(update_dict: dict):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(update_dict)

    try:
        processor = VideoTrafficProcessor(
            detector=SHARED_DETECTOR,
            low_density_max=low_max,
            medium_density_max=med_max
        )

        results = processor.process_video(
            input_path=input_path,
            output_path=output_path,
            progress_callback=progress_callback,
            frame_stride=frame_stride
        )

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["results"] = results
            JOBS[job_id]["output_filename"] = os.path.basename(output_path)
            JOBS[job_id]["output_url"] = f"/static/outputs/{os.path.basename(output_path)}"

    except Exception as e:
        logger.error(f"Error processing video job {job_id}: {e}", exc_info=True)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error_message"] = str(e)


# -------------------------------------------------------------
# Web Routes
# -------------------------------------------------------------

@app.route("/")
def index():
    """Renders the main Academic Dashboard."""
    return render_template(
        "index.html",
        primary_model=SHARED_DETECTOR.active_model_name or config.PRIMARY_MODEL,
        fallback_model=config.FALLBACK_MODEL,
        default_low=config.DENSITY_THRESHOLDS["LOW_MAX"],
        default_med=config.DENSITY_THRESHOLDS["MEDIUM_MAX"]
    )


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    Handles media upload (image or video), validates format,
    stores in uploads/, and returns metadata for instant preview.
    """
    if "video" not in request.files and "media" not in request.files:
        return jsonify({"success": False, "error": "No media file provided"}), 400

    file = request.files.get("video") or request.files.get("media")
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    file_type = get_file_type(file.filename)
    if file_type == "unsupported":
        allowed_all = sorted(list(config.ALLOWED_EXTENSIONS))
        return jsonify({
            "success": False,
            "error": f"Unsupported format. Allowed: {', '.join(allowed_all)}"
        }), 400

    # Generate safe unique filename
    ext = file.filename.rsplit(".", 1)[1].lower()
    prefix = "img" if file_type == "image" else "traffic"
    unique_name = f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"
    saved_path = os.path.join(config.UPLOAD_FOLDER, unique_name)

    file.save(saved_path)

    # Read media properties
    if file_type == "image":
        meta = get_image_metadata(saved_path)
        if not meta or meta.get("width", 0) <= 0:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            return jsonify({
                "success": False,
                "error": "Uploaded image file could not be decoded by OpenCV. Please try another JPG/PNG."
            }), 400
    else:
        meta = get_video_metadata(saved_path)
        if not meta or meta.get("total_frames", 0) <= 0:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            return jsonify({
                "success": False,
                "error": "Uploaded video file could not be decoded by OpenCV. Please try another MP4/AVI file."
            }), 400

    return jsonify({
        "success": True,
        "filename": unique_name,
        "file_type": file_type,
        "media_type": file_type,
        "preview_url": f"/static/uploads/{unique_name}",
        "metadata": meta
    })


@app.route("/api/analyze", methods=["POST"])
def start_analysis():
    """
    Initiates analysis on an uploaded or sample image/video.
    Images are processed synchronously (~150ms).
    Videos are processed asynchronously with background worker.
    """
    data = request.get_json() or {}
    filename = data.get("filename")

    if not filename:
        return jsonify({"success": False, "error": "Filename required"}), 400

    # Locate media file
    input_path = os.path.join(config.UPLOAD_FOLDER, filename)
    if not os.path.exists(input_path):
        sample_vid = os.path.join(config.SAMPLE_FOLDER, filename)
        sample_img = os.path.join(config.SAMPLE_IMAGE_FOLDER, filename)
        if os.path.exists(sample_vid):
            input_path = sample_vid
        elif os.path.exists(sample_img):
            input_path = sample_img
        else:
            return jsonify({"success": False, "error": "Media file not found on server"}), 404

    file_type = get_file_type(filename)
    low_max = int(data.get("low_threshold", config.DENSITY_THRESHOLDS["LOW_MAX"]))
    med_max = int(data.get("medium_threshold", config.DENSITY_THRESHOLDS["MEDIUM_MAX"]))
    frame_stride = int(data.get("frame_stride", config.DEFAULT_FRAME_STRIDE))

    # Case A: Static Traffic Image (Instant Processing)
    if file_type == "image":
        output_filename = f"processed_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(config.OUTPUT_FOLDER, output_filename)

        processor = VideoTrafficProcessor(
            detector=SHARED_DETECTOR,
            low_density_max=low_max,
            medium_density_max=med_max
        )

        try:
            results = processor.process_image(input_path, output_path)
            img_job_id = f"img_{uuid.uuid4().hex[:8]}"
            with JOBS_LOCK:
                JOBS[img_job_id] = {
                    "job_id": img_job_id,
                    "status": "completed",
                    "progress": 100,
                    "filename": filename,
                    "file_type": "image",
                    "media_type": "image",
                    "created_at": time.time(),
                    "output_url": f"/static/outputs/{output_filename}",
                    "results": results
                }
            return jsonify({
                "success": True,
                "status": "completed",
                "file_type": "image",
                "media_type": "image",
                "job_id": img_job_id,
                "output_url": f"/static/outputs/{output_filename}",
                "results": results
            })
        except Exception as e:
            logger.error(f"Image analysis error: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    # Case B: Traffic Video (Background Worker Processing)
    job_id = str(uuid.uuid4())
    output_filename = f"processed_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(config.OUTPUT_FOLDER, output_filename)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "filename": filename,
            "media_type": "video",
            "created_at": time.time(),
            "results": None
        }

    thread = threading.Thread(
        target=run_video_job,
        args=(job_id, input_path, output_path, low_max, med_max, frame_stride),
        daemon=True
    )
    thread.start()

    return jsonify({
        "success": True,
        "status": "processing",
        "file_type": "video",
        "media_type": "video",
        "job_id": job_id,
        "message": "Traffic video analysis initiated."
    })


@app.route("/api/status/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Returns current status and results of a video processing job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    return jsonify({
        "success": True,
        "job": job
    })


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Returns list of bundled sample videos and sample images for demonstration."""
    sample_videos = []
    if os.path.exists(config.SAMPLE_FOLDER):
        for f in os.listdir(config.SAMPLE_FOLDER):
            if get_file_type(f) == "video":
                meta = get_video_metadata(os.path.join(config.SAMPLE_FOLDER, f))
                sample_videos.append({
                    "filename": f,
                    "title": f.replace("_", " ").replace(".mp4", "").replace(".avi", "").title(),
                    "preview_url": f"/static/sample_videos/{f}",
                    "media_type": "video",
                    "metadata": meta
                })

    sample_images = []
    if os.path.exists(config.SAMPLE_IMAGE_FOLDER):
        for f in os.listdir(config.SAMPLE_IMAGE_FOLDER):
            if get_file_type(f) == "image":
                meta = get_image_metadata(os.path.join(config.SAMPLE_IMAGE_FOLDER, f))
                sample_images.append({
                    "filename": f,
                    "title": f.replace("_", " ").replace(".jpg", "").replace(".png", "").title(),
                    "preview_url": f"/static/sample_images/{f}",
                    "media_type": "image",
                    "metadata": meta
                })

    return jsonify({
        "success": True,
        "samples": sample_videos,
        "sample_images": sample_images
    })


@app.route("/api/use-sample", methods=["POST"])
def use_sample():
    """Copies a sample video or sample image to uploads folder for analysis."""
    data = request.get_json() or {}
    sample_name = data.get("filename")
    if not sample_name:
        return jsonify({"success": False, "error": "Sample filename required"}), 400

    # Search in sample videos or sample images
    source = os.path.join(config.SAMPLE_FOLDER, sample_name)
    file_type = get_file_type(sample_name)

    if not os.path.exists(source):
        source = os.path.join(config.SAMPLE_IMAGE_FOLDER, sample_name)
        if not os.path.exists(source):
            return jsonify({"success": False, "error": "Sample file not found"}), 404

    ext = sample_name.rsplit(".", 1)[1].lower()
    prefix = "sample_img" if file_type == "image" else "sample_vid"
    target_name = f"{prefix}_{uuid.uuid4().hex[:6]}.{ext}"
    target = os.path.join(config.UPLOAD_FOLDER, target_name)
    shutil.copyfile(source, target)

    if file_type == "image":
        meta = get_image_metadata(target)
    else:
        meta = get_video_metadata(target)

    return jsonify({
        "success": True,
        "filename": target_name,
        "media_type": file_type,
        "preview_url": f"/static/uploads/{target_name}",
        "metadata": meta
    })


if __name__ == "__main__":
    print("=" * 65)
    print(" Smart Traffic Monitoring System - B.Tech Minor Project")
    print(f" Server starting on http://{config.HOST}:{config.PORT}")
    print("=" * 65)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
