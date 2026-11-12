"""
Configuration settings for the Smart Traffic Monitoring System.
Academic Minor Project - B.Tech Computer Science
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
OUTPUT_FOLDER = STATIC_DIR / "outputs"
SAMPLE_FOLDER = STATIC_DIR / "sample_videos"
SAMPLE_IMAGE_FOLDER = STATIC_DIR / "sample_images"

# Ensure directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, SAMPLE_FOLDER, SAMPLE_IMAGE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Allowed Media Extensions
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

# Maximum upload size in bytes (100MB)
MAX_CONTENT_LENGTH = 100 * 1024 * 1024

# Model Configuration
# Primary: yolo26n.pt, Fallback: yolo11n.pt, Secondary Fallback: yolov8n.pt
PRIMARY_MODEL = "yolo26n.pt"
FALLBACK_MODEL = "yolo11n.pt"
SECONDARY_FALLBACK_MODEL = "yolov8n.pt"

# COCO Dataset Vehicle Class IDs
# 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Detection Confidence Threshold (standard COCO sensitivity)
CONFIDENCE_THRESHOLD = 0.25

# Inference Image Dimension (height/width resized for YOLO forward pass)
INFERENCE_IMG_SIZE = 640

# Default video frame processing stride (1 = process every frame; 2 = process every 2nd frame)
DEFAULT_FRAME_STRIDE = 1

# Road Monitoring Region (fraction of frame height)
REGION_TOP = 0.10
REGION_BOTTOM = 0.95
LINE_POSITION = 0.50

# Traffic Density Classification Thresholds
# Evaluated on the number of vehicles simultaneously active in the monitored road area.
DENSITY_THRESHOLDS = {
    "LOW_MAX": 2,        # 0 to 2 active vehicles in monitored region => Low Traffic
    "MEDIUM_MAX": 5,     # 3 to 5 active vehicles in monitored region => Medium Traffic
    # > 5 active vehicles in monitored region => High Traffic
}

# Web Server Configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
