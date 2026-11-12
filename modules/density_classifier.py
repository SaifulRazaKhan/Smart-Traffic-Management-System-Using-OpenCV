"""
Traffic Density Classification Module
Classifies traffic density into Low, Medium, or High using rule-based thresholds
evaluated strictly on the number of vehicles currently visible in the monitored road region.
Academic Minor Project - Explainable, student-friendly decision logic.
"""

from typing import Dict, Any, Optional


class TrafficDensityClassifier:
    """
    Rule-based traffic density classifier.
    Evaluates traffic density (Low, Medium, High) using the number of vehicles
    actively detected within the defined road monitoring zone.
    """

    def __init__(self, low_max: int = 2, medium_max: int = 5):
        """
        Args:
            low_max: Active vehicles upper limit for 'Low Traffic' (inclusive, e.g. 0 to 2)
            medium_max: Active vehicles upper limit for 'Medium Traffic' (inclusive, e.g. 3 to 5)
                        Above medium_max is classified as 'High Traffic' (congestion).
        """
        self.low_max = int(low_max)
        self.medium_max = int(medium_max)

    def classify_active(self, active_count: int) -> Dict[str, Any]:
        """
        Classifies instantaneous traffic density from the number of vehicles currently
        visible inside the monitored road area.

        Rules:
            active_count <= low_max               -> Low Traffic
            low_max < active_count <= medium_max  -> Medium Traffic
            active_count > medium_max             -> High Traffic
        """
        count = max(0, int(active_count))

        if count <= self.low_max:
            level = "Low Traffic"
            code = "LOW"
            color = "#10b981"  # Emerald green
            badge_class = "density-low"
            description = (
                f"{count} vehicle{'s' if count != 1 else ''} currently in monitored road region "
                f"(limit: <= {self.low_max}). Roadway is clear with free-flowing traffic."
            )
        elif count <= self.medium_max:
            level = "Medium Traffic"
            code = "MEDIUM"
            color = "#f59e0b"  # Amber / Yellow
            badge_class = "density-medium"
            description = (
                f"{count} vehicles currently in monitored road region "
                f"({self.low_max + 1} to {self.medium_max}). Steady traffic flow with normal density."
            )
        else:
            level = "High Traffic"
            code = "HIGH"
            color = "#ef4444"  # Red
            badge_class = "density-high"
            description = (
                f"{count} vehicles currently in monitored road region "
                f"(exceeds {self.medium_max}). High vehicle volume indicating traffic congestion or queueing."
            )

        return {
            "level": level,
            "code": code,
            "color": color,
            "badge_class": badge_class,
            "description": description,
            "active_count": count,
            "threshold_used": {
                "low_max": self.low_max,
                "medium_max": self.medium_max
            }
        }

    def classify_summary(self, avg_active: float, peak_active: int) -> Dict[str, Any]:
        """
        Classifies representative traffic density for an entire processed video
        based on the average active vehicles in the monitored region.
        """
        rounded_avg = round(avg_active, 1)
        base = self.classify_active(int(round(avg_active)))
        base["avg_active_vehicles"] = rounded_avg
        base["peak_active_vehicles"] = int(peak_active)
        base["description"] = (
            f"Average active vehicle load was {rounded_avg} vehicles/frame (Peak: {peak_active}). "
            f"Classified as {base['level']} based on active road area thresholds "
            f"(Low <= {self.low_max}, Medium <= {self.medium_max})."
        )
        return base

    def classify(
        self,
        active_count: Optional[int] = None,
        total_count: Optional[int] = None,
        avg_per_frame: float = 0.0
    ) -> Dict[str, Any]:
        """
        Unified classification interface.
        Prioritizes active_count if provided, or avg_per_frame if summarizing.
        """
        if active_count is not None:
            return self.classify_active(active_count)
        if avg_per_frame > 0:
            return self.classify_summary(avg_per_frame, total_count or 0)
        if total_count is not None:
            return self.classify_active(total_count)
        return self.classify_active(0)
