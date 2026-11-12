"""
Media Processing Pipeline for Smart Traffic Monitoring
Handles both static traffic images and full traffic videos.
- Uses YOLO detection & ByteTrack tracking
- Monitors active vehicles in road region for density classification
- Tracks cumulative unique vehicle counts
- Draws bounding boxes, category labels, confidence scores, and dual HUD
- Optimizes performance with configurable frame stride and model reuse
"""

import os
import time
import logging
from typing import Dict, Any, Callable, Optional, List
import cv2
import numpy as np

import config
from modules.detector import VehicleDetector
from modules.tracker_counter import VehicleCounter
from modules.density_classifier import TrafficDensityClassifier

logger = logging.getLogger(__name__)

# Category colors in BGR format
CLASS_COLORS = {
    "car": (255, 140, 0),        # Sky Blue
    "motorcycle": (0, 165, 255), # Orange
    "bus": (70, 180, 50),        # Green
    "truck": (180, 50, 200)      # Purple
}
DEFAULT_COLOR = (200, 200, 200)


class VideoTrafficProcessor:
    """
    Coordinates detection, tracking, counting, density classification,
    and visual rendering for traffic images and videos.
    """

    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        primary_model: str = config.PRIMARY_MODEL,
        fallback_model: str = config.FALLBACK_MODEL,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        low_density_max: int = config.DENSITY_THRESHOLDS["LOW_MAX"],
        medium_density_max: int = config.DENSITY_THRESHOLDS["MEDIUM_MAX"]
    ):
        if detector is not None:
            self.detector = detector
        else:
            self.detector = VehicleDetector(
                primary_model=primary_model,
                fallback_model=fallback_model,
                conf_threshold=confidence_threshold
            )

        self.classifier = TrafficDensityClassifier(
            low_max=low_density_max,
            medium_max=medium_density_max
        )

    def process_image(
        self,
        input_path: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Processes a single static traffic image.
        Detects vehicles, counts per-category vehicles, classifies traffic density,
        draws annotations with HUD banner, and saves the annotated image.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not decode image file: {input_path}")

        height, width = image.shape[:2]
        t0 = time.time()

        # Step 1: Detect vehicles using YOLO
        detections = self.detector.detect_image(image)

        # Step 2: Categorize detections & Count in Monitored Region
        reg_y1 = int(height * config.REGION_TOP)
        reg_y2 = int(height * config.REGION_BOTTOM)

        category_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        active_in_region = 0

        for det in detections:
            cname = det["class_name"]
            if cname in category_counts:
                category_counts[cname] += 1
            else:
                category_counts["car"] += 1

            cy = int((det["box"][1] + det["box"][3]) / 2)
            if reg_y1 <= cy <= reg_y2:
                active_in_region += 1

        total_vehicles = len(detections)

        # Step 3: Classify Traffic Density based on active road occupancy
        density_result = self.classifier.classify_active(active_in_region)

        # Step 4: Render Annotations
        annotated = image.copy()

        # Draw monitored region boundary
        cv2.line(annotated, (0, reg_y1), (width, reg_y1), (120, 120, 120), 1)
        cv2.putText(annotated, "MONITORED ROAD REGION", (10, reg_y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(annotated, (0, reg_y2), (width, reg_y2), (120, 120, 120), 1)

        # Draw bounding boxes and labels
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lbl_y1 = max(0, y1 - lbl_h - 6)
            cv2.rectangle(annotated, (x1, lbl_y1), (x1 + lbl_w + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Top HUD Banner
        banner_height = 46
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (15, 20, 32), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        density_code = density_result.get("code", "LOW")
        density_color = (80, 200, 80) if density_code == "LOW" else ((30, 160, 240) if density_code == "MEDIUM" else (60, 60, 240))
        cv2.putText(annotated, f"ACTIVE IN ZONE: {active_in_region}  |  DENSITY: {density_code}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, density_color, 2, cv2.LINE_AA)

        right_hud = (
            f"TOTAL: {total_vehicles} "
            f"(Cars:{category_counts['car']} Bike:{category_counts['motorcycle']} "
            f"Bus:{category_counts['bus']} Trk:{category_counts['truck']})"
        )
        (r_w, _), _ = cv2.getTextSize(right_hud, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(annotated, right_hud, (max(width - r_w - 12, 340), 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 225, 235), 1, cv2.LINE_AA)

        # Save annotated image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)

        elapsed = time.time() - t0

        return {
            "status": "success",
            "media_type": "image",
            "model_used": self.detector.active_model_name,
            "cumulative_counts": {
                "total_vehicles": total_vehicles,
                "cars": category_counts["car"],
                "motorcycles": category_counts["motorcycle"],
                "buses": category_counts["bus"],
                "trucks": category_counts["truck"],
                "line_crossings": 0,
                "peak_simultaneous_vehicles": active_in_region
            },
            "counts": {
                "total_vehicles": total_vehicles,
                "cars": category_counts["car"],
                "motorcycles": category_counts["motorcycle"],
                "buses": category_counts["bus"],
                "trucks": category_counts["truck"]
            },
            "active_vehicles": {
                "average": float(active_in_region),
                "peak": active_in_region,
                "final_frame_active": active_in_region
            },
            "density": density_result,
            "detections": detections,
            "performance": {
                "total_frames_processed": 1,
                "processing_time_seconds": round(elapsed, 3),
                "processing_fps": round(1.0 / elapsed, 1) if elapsed > 0 else 0,
                "video_resolution": f"{width}x{height}",
                "original_video_duration_seconds": 0,
                "original_video_fps": 0
            },
            "output_media_path": output_path
        }

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        frame_stride: int = config.DEFAULT_FRAME_STRIDE
    ) -> Dict[str, Any]:
        """
        Executes complete video processing pipeline.
        Supports frame_stride for performance acceleration.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            total_frames = 1

        duration_sec = total_frames / fps
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        counter = VehicleCounter(
            use_counting_line=True,
            line_position=config.LINE_POSITION,
            region_top=config.REGION_TOP,
            region_bottom=config.REGION_BOTTOM
        )
        counter.reset()

        frame_count = 0
        inferred_frames = 0
        start_time = time.time()
        active_counts_history = []
        last_parsed_detections: List[Dict[str, Any]] = []

        logger.info(
            f"Starting video processing: {width}x{height} @ {fps:.1f} FPS, "
            f"{total_frames} frames (stride={frame_stride})"
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                should_infer = (frame_stride <= 1) or (frame_count % frame_stride == 1) or (frame_count == 1)

                if should_infer:
                    inferred_frames += 1
                    # 1. Detect and Track vehicles with YOLO + ByteTrack
                    track_results = self.detector.detect_and_track(frame, persist=True)

                    parsed_detections = []
                    if track_results is not None and track_results.boxes is not None:
                        boxes = track_results.boxes
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            if cls_id not in self.detector.target_class_map:
                                continue

                            conf = float(box.conf[0].item())
                            track_id = int(box.id[0].item()) if box.id is not None else None
                            xyxy = [int(v) for v in box.xyxy[0].tolist()]

                            parsed_detections.append({
                                "box": xyxy,
                                "track_id": track_id,
                                "class_id": cls_id,
                                "class_name": self.detector.get_class_name(cls_id),
                                "confidence": conf
                            })
                    last_parsed_detections = parsed_detections
                else:
                    # On skipped frames, reuse previous detections for visual continuity
                    parsed_detections = last_parsed_detections

                # 2. Update Vehicle Counter & Monitored Road Area
                count_status = counter.update(parsed_detections, height, width)
                active_count = count_status["active_count"]
                active_counts_history.append(active_count)

                # 3. Classify Instantaneous Traffic Density for Current Frame
                current_density = self.classifier.classify_active(active_count)

                # 4. Render Annotations on Frame
                annotated_frame = self._render_annotations(
                    frame=frame,
                    detections=parsed_detections,
                    counter=counter,
                    count_status=count_status,
                    current_density=current_density,
                    width=width,
                    height=height
                )

                writer.write(annotated_frame)

                # 5. Live Progress Callback (streaming active vehicles and density)
                if progress_callback and (frame_count % 5 == 0 or frame_count == total_frames):
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    pct = min(100, int((frame_count / total_frames) * 100))
                    progress_callback({
                        "progress": pct,
                        "frame": frame_count,
                        "total_frames": total_frames,
                        "current_fps": round(current_fps, 1),
                        "elapsed_seconds": round(elapsed, 1),
                        "active_vehicles": active_count,
                        "current_density": current_density,
                        "running_counts": counter.get_summary()
                    })

        finally:
            cap.release()
            writer.release()

        total_time = time.time() - start_time
        avg_processing_fps = frame_count / total_time if total_time > 0 else 0
        summary_counts = counter.get_summary()

        # 6. Overall Representative Density Classification
        avg_active = (
            sum(active_counts_history) / len(active_counts_history)
            if active_counts_history else 0.0
        )
        peak_active = max(active_counts_history) if active_counts_history else 0
        overall_density = self.classifier.classify_summary(
            avg_active=avg_active,
            peak_active=peak_active
        )

        logger.info(
            f"Finished processing in {total_time:.2f}s ({avg_processing_fps:.1f} FPS, {inferred_frames} inferred). "
            f"Cumulative Unique: {summary_counts['total_vehicles']}, "
            f"Avg Active: {avg_active:.1f}, Density: {overall_density['level']}"
        )

        return {
            "status": "success",
            "media_type": "video",
            "model_used": self.detector.active_model_name,
            "cumulative_counts": summary_counts,
            "counts": summary_counts,  # backwards compatibility
            "active_vehicles": {
                "average": round(avg_active, 1),
                "peak": peak_active,
                "final_frame_active": active_counts_history[-1] if active_counts_history else 0
            },
            "density": overall_density,
            "performance": {
                "total_frames_processed": frame_count,
                "inferred_frames": inferred_frames,
                "frame_stride": frame_stride,
                "processing_time_seconds": round(total_time, 2),
                "processing_fps": round(avg_processing_fps, 1),
                "video_resolution": f"{width}x{height}",
                "original_video_duration_seconds": round(duration_sec, 2),
                "original_video_fps": round(fps, 1)
            },
            "output_video_path": output_path
        }

    def _render_annotations(
        self,
        frame: np.ndarray,
        detections: list,
        counter: VehicleCounter,
        count_status: dict,
        current_density: dict,
        width: int,
        height: int
    ) -> np.ndarray:
        """Draws bounding boxes, labels, monitored region bounds, line, and dual HUD."""
        annotated = frame.copy()
        line_y = count_status["line_y"]
        reg_y1 = count_status["region_y1"]
        reg_y2 = count_status["region_y2"]

        # 1. Draw Monitored Road Region Boundaries
        cv2.line(annotated, (0, reg_y1), (width, reg_y1), (120, 120, 120), 1)
        cv2.putText(annotated, "MONITORED ROAD REGION", (10, reg_y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(annotated, (0, reg_y2), (width, reg_y2), (120, 120, 120), 1)

        # 2. Draw Virtual Counting Line
        cv2.line(annotated, (0, line_y), (width, line_y), (0, 215, 255), 2)  # Gold line
        cv2.putText(
            annotated,
            "COUNTING REFERENCE LINE",
            (10, line_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 215, 255),
            1,
            cv2.LINE_AA
        )

        # 3. Draw Vehicle Detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            track_id = det.get("track_id")
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

            # Label banner
            id_str = f" #{track_id}" if track_id is not None else ""
            label_text = f"{cls_name}{id_str} {conf:.2f}"
            (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lbl_y1 = max(0, y1 - lbl_h - 6)
            cv2.rectangle(annotated, (x1, lbl_y1), (x1 + lbl_w + 6, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Draw Top Dual HUD Banner
        banner_height = 46
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (15, 20, 32), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        density_code = current_density.get("code", "LOW")
        density_color = (80, 200, 80) if density_code == "LOW" else ((30, 160, 240) if density_code == "MEDIUM" else (60, 60, 240))

        # Left HUD: Active in zone & Instantaneous Density
        left_hud = f"ACTIVE IN ZONE: {count_status['active_count']}  |  DENSITY: {density_code}"
        cv2.putText(annotated, left_hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, density_color, 2, cv2.LINE_AA)

        # Right HUD: Cumulative Unique Vehicles
        cats = count_status["category_counts"]
        right_hud = (
            f"CUMULATIVE: {count_status['total_count']} "
            f"(Cars:{cats.get('car', 0)} Bike:{cats.get('motorcycle', 0)} Bus:{cats.get('bus', 0)} Trk:{cats.get('truck', 0)})"
        )
        (r_w, _), _ = cv2.getTextSize(right_hud, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(annotated, right_hud, (max(width - r_w - 12, 340), 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 225, 235), 1, cv2.LINE_AA)

        return annotated
