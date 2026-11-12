"""
Vehicle Detection Module using YOLO
Supports YOLO26n as primary model, with graceful fallback to YOLO11n / YOLOv8n.
Filters exclusively for vehicle classes: car, motorcycle, bus, truck.
Supports both single-image detection and multi-frame video tracking.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard COCO target class names
TARGET_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}


class VehicleDetector:
    """
    Wrapper for YOLO-based vehicle detection.
    Dynamically maps target vehicle classes from model.names.
    Supports both static image detection and temporal video tracking.
    """

    def __init__(
        self,
        primary_model: str = config.PRIMARY_MODEL,
        fallback_model: str = config.FALLBACK_MODEL,
        conf_threshold: float = config.CONFIDENCE_THRESHOLD,
        imgsz: int = config.INFERENCE_IMG_SIZE,
        preloaded_model=None
    ):
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.model = preloaded_model
        self.active_model_name = None
        self.target_class_map: Dict[int, str] = {}

        if self.model is None:
            self.load_model(primary_model, fallback_model)
        else:
            self.active_model_name = getattr(self.model, "model_name", "yolo_custom")
            self._build_class_mapping()

    def load_model(self, primary_model: str, fallback_model: str):
        """Loads YOLO model with fallback handling."""
        from ultralytics import YOLO

        names = [primary_model, fallback_model, config.SECONDARY_FALLBACK_MODEL]
        candidates = []
        for name in names:
            abs_path = os.path.join(config.BASE_DIR, name)
            if os.path.exists(abs_path):
                candidates.append(abs_path)
            candidates.append(name)

        last_error = None
        for model_path in candidates:
            try:
                logger.info(f"Attempting to load vehicle detection model: {model_path}")
                self.model = YOLO(model_path)
                self.active_model_name = os.path.basename(model_path)
                logger.info(f"Successfully loaded model: {self.active_model_name}")
                self._build_class_mapping()
                return
            except Exception as e:
                logger.warning(f"Could not load {model_path}: {e}. Trying fallback...")
                last_error = e

        raise RuntimeError(
            f"Failed to load any YOLO model ({candidates}). Last error: {last_error}"
        )

    def _build_class_mapping(self):
        """
        Dynamically inspects model.names to map vehicle class IDs.
        Confirms actual IDs for car, motorcycle, bus, and truck.
        """
        self.target_class_map = {}
        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            for cls_id, name in self.model.names.items():
                name_clean = str(name).lower().strip()
                if name_clean in TARGET_CLASS_NAMES:
                    self.target_class_map[int(cls_id)] = name_clean

        # Fallback to standard COCO if model.names was not populated
        if not self.target_class_map:
            self.target_class_map = dict(config.COCO_VEHICLE_CLASSES)

        logger.info(f"Verified vehicle class mapping: {self.target_class_map}")

    def get_target_class_ids(self) -> List[int]:
        """Returns the list of target vehicle class IDs."""
        return list(self.target_class_map.keys())

    def get_class_name(self, class_id: int) -> str:
        """Returns the human-readable vehicle class name for an ID."""
        return self.target_class_map.get(int(class_id), "vehicle")

    def detect_image(
        self,
        image_or_path,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLO prediction on a single static image without temporal tracking.

        Returns:
            List of parsed detections:
            [{"box": [x1, y1, x2, y2], "class_id": int, "class_name": str, "confidence": float}]
        """
        confidence = conf if conf is not None else self.conf_threshold
        image_size = imgsz if imgsz is not None else self.imgsz
        target_ids = self.get_target_class_ids()

        results = self.model.predict(
            source=image_or_path,
            classes=target_ids,
            conf=confidence,
            imgsz=image_size,
            verbose=False
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in self.target_class_map:
                    continue

                conf_val = float(box.conf[0].item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]

                detections.append({
                    "box": xyxy,
                    "class_id": cls_id,
                    "class_name": self.get_class_name(cls_id),
                    "confidence": round(conf_val, 2),
                    "track_id": None
                })

        return detections

    def detect_and_track(
        self,
        frame: np.ndarray,
        persist: bool = True,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None
    ):
        """
        Runs YOLO tracking on a single video frame with ByteTrack temporal tracking.
        Returns the raw Ultralytics Results object.
        """
        confidence = conf if conf is not None else self.conf_threshold
        image_size = imgsz if imgsz is not None else self.imgsz
        target_ids = self.get_target_class_ids()

        results = self.model.track(
            source=frame,
            persist=persist,
            classes=target_ids,
            conf=confidence,
            imgsz=image_size,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        return results[0] if results else None
