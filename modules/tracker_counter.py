"""
Vehicle Counting and Tracking Module
Tracks unique vehicles across video frames using ByteTrack track IDs to prevent duplicate counting.
Separates active vehicles in the monitored road region from cumulative unique vehicle counts.
Categorizes cumulative counts into: cars, motorcycles, buses, trucks, and total.
"""

from typing import Dict, Set, Tuple, List, Optional
import numpy as np


class VehicleCounter:
    """
    Maintains vehicle counts by tracking unique vehicle IDs across frames.
    - Monitors active vehicle count inside a defined road region for density estimation.
    - Tracks cumulative unique vehicles without double-counting.
    - Supports bidirectional virtual-line crossing.
    """

    def __init__(
        self,
        use_counting_line: bool = True,
        line_position: float = 0.50,
        region_top: float = 0.10,
        region_bottom: float = 0.95
    ):
        """
        Args:
            use_counting_line: If True, tracks line crossings across a reference line.
            line_position: Vertical position of counting line as fraction of height (0.0 to 1.0).
            region_top: Top boundary of monitored road region as fraction of height.
            region_bottom: Bottom boundary of monitored road region as fraction of height.
        """
        self.use_counting_line = use_counting_line
        self.line_position = line_position
        self.region_top = region_top
        self.region_bottom = region_bottom

        # Track history: track_id -> previous centroid (cx, cy)
        self.track_history: Dict[int, Tuple[int, int]] = {}

        # Set of vehicle IDs registered in cumulative count (prevents duplicate counting)
        self.registered_vehicle_ids: Set[int] = set()

        # Set of vehicle IDs that specifically crossed the virtual line
        self.line_crossed_ids: Set[int] = set()

        # Cumulative category-wise counts
        self.category_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }
        self.total_cumulative_count = 0

        # Peak instantaneous active vehicles in any single frame
        self.peak_active_count = 0

        # Frame counter
        self.frame_index = 0

    def reset(self):
        """Reset all counters and tracking state for a new video."""
        self.track_history.clear()
        self.registered_vehicle_ids.clear()
        self.line_crossed_ids.clear()
        self.category_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }
        self.total_cumulative_count = 0
        self.peak_active_count = 0
        self.frame_index = 0

    def update(self, detections: list, frame_height: int, frame_width: int) -> Dict[str, any]:
        """
        Process detections for the current frame.

        detections: list of dicts with keys:
            - 'box': (x1, y1, x2, y2)
            - 'track_id': int or None
            - 'class_name': str ('car', 'motorcycle', 'bus', 'truck')
            - 'confidence': float

        Returns:
            dict containing:
                - 'active_count': number of vehicles currently inside the monitored road region
                - 'total_count': cumulative unique vehicles counted so far
                - 'category_counts': dict with cumulative counts per vehicle category
                - 'line_crossings': count of vehicles that crossed the reference line
                - 'line_y': y-coordinate of the counting line
                - 'region_y1': top boundary y-coordinate of the monitored region
                - 'region_y2': bottom boundary y-coordinate of the monitored region
        """
        self.frame_index += 1
        line_y = int(frame_height * self.line_position)
        reg_y1 = int(frame_height * self.region_top)
        reg_y2 = int(frame_height * self.region_bottom)

        active_in_region = 0
        newly_registered = []

        for det in detections:
            track_id = det.get("track_id")
            class_name = det.get("class_name", "car")
            x1, y1, x2, y2 = det["box"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # If tracker temporarily has no ID, assign deterministic spatial ID
            if track_id is None:
                track_id = hash((cx // 50, cy // 50, class_name))

            prev_pos = self.track_history.get(track_id)
            self.track_history[track_id] = (cx, cy)

            # Check if centroid is within the defined monitored road area
            is_in_monitored_region = (reg_y1 <= cy <= reg_y2)
            if is_in_monitored_region:
                active_in_region += 1

            # 1. Check Bidirectional Line Crossing
            if self.use_counting_line and prev_pos is not None:
                prev_y = prev_pos[1]
                # Downward crossing: moved from above line to at/below line
                crossed_down = (prev_y < line_y and cy >= line_y)
                # Upward crossing: moved from below line to at/above line
                crossed_up = (prev_y > line_y and cy <= line_y)

                if (crossed_down or crossed_up) and (track_id not in self.line_crossed_ids):
                    self.line_crossed_ids.add(track_id)

            # 2. Cumulative Unique Registration
            # Register vehicle when it is tracked in the monitored zone or crosses the line
            if (is_in_monitored_region or track_id in self.line_crossed_ids) and (track_id not in self.registered_vehicle_ids):
                self.registered_vehicle_ids.add(track_id)
                newly_registered.append(track_id)

                if class_name in self.category_counts:
                    self.category_counts[class_name] += 1
                else:
                    self.category_counts["car"] += 1

                self.total_cumulative_count += 1

        # Track peak simultaneous vehicles in monitored region
        if active_in_region > self.peak_active_count:
            self.peak_active_count = active_in_region

        return {
            "active_count": active_in_region,
            "total_count": self.total_cumulative_count,
            "category_counts": dict(self.category_counts),
            "line_crossings": len(self.line_crossed_ids),
            "line_y": line_y,
            "region_y1": reg_y1,
            "region_y2": reg_y2,
            "newly_registered": newly_registered
        }

    def get_summary(self) -> Dict[str, any]:
        """Returns final cumulative summary statistics."""
        return {
            "total_vehicles": self.total_cumulative_count,
            "cars": self.category_counts.get("car", 0),
            "motorcycles": self.category_counts.get("motorcycle", 0),
            "buses": self.category_counts.get("bus", 0),
            "trucks": self.category_counts.get("truck", 0),
            "line_crossings": len(self.line_crossed_ids),
            "peak_simultaneous_vehicles": self.peak_active_count
        }
