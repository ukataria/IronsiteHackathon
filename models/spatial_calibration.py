"""
Spatial Calibration Module

Calibrates pixel-to-physical-space conversion using known-dimension anchors.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from models.anchor_detection import Anchor


@dataclass
class DepthPlane:
    """A calibrated depth plane with spatial scale."""

    depth_value: float        # Representative depth value
    scale: float              # Pixels per inch at this depth
    confidence: float         # Calibration confidence (0-1)
    anchors: List[Anchor]     # Anchors used for calibration
    scale_std: float          # Standard deviation of scale estimates

    def measure_distance(self, pixel_distance: float) -> float:
        """
        Convert pixel distance to physical distance in inches.

        Args:
            pixel_distance: Distance in pixels

        Returns:
            Distance in inches
        """
        return pixel_distance / self.scale


class SpatialCalibrator:
    """Calibrates spatial measurements using anchor objects."""

    def __init__(self, depth_tolerance: float = 0.05):
        """
        Initialize calibrator.

        Args:
            depth_tolerance: Max depth difference to consider objects on same plane
        """
        self.depth_tolerance = depth_tolerance
        self.depth_planes: List[DepthPlane] = []

    def calibrate(
        self,
        anchors: List[Anchor],
        depth_map: np.ndarray
    ) -> List[DepthPlane]:
        """
        Calibrate spatial scale using detected anchors and depth map.

        Args:
            anchors: List of detected anchors with bounding boxes
            depth_map: Depth map from depth estimator

        Returns:
            List of calibrated depth planes
        """
        if not anchors:
            print("⚠️  No anchors detected - cannot calibrate")
            return []

        # Assign depth values to anchors
        for anchor in anchors:
            x1, y1, x2, y2 = anchor.bbox
            depth = self._get_depth_in_bbox(depth_map, (x1, y1, x2, y2))
            anchor.depth = depth

        # Group anchors by depth plane
        planes = self._group_anchors_by_depth(anchors)

        # Calibrate each plane
        calibrated_planes = []
        for plane_anchors in planes:
            plane = self._calibrate_plane(plane_anchors)
            if plane:
                calibrated_planes.append(plane)

        self.depth_planes = calibrated_planes
        return calibrated_planes

    def _get_depth_in_bbox(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """Get median depth value in bounding box."""
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape

        # Clip to bounds
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, x1 + 1, w))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, y1 + 1, h))

        region = depth_map[y1:y2, x1:x2]
        return float(np.median(region))

    def _group_anchors_by_depth(self, anchors: List[Anchor]) -> List[List[Anchor]]:
        """Group anchors into depth planes."""
        if not anchors:
            return []

        # Sort by depth
        sorted_anchors = sorted(anchors, key=lambda a: a.depth)

        # Group by depth similarity
        planes = []
        current_plane = [sorted_anchors[0]]

        for anchor in sorted_anchors[1:]:
            depth_diff = abs(anchor.depth - current_plane[0].depth)

            if depth_diff <= self.depth_tolerance:
                current_plane.append(anchor)
            else:
                planes.append(current_plane)
                current_plane = [anchor]

        planes.append(current_plane)
        return planes

    def _calibrate_plane(self, anchors: List[Anchor]) -> Optional[DepthPlane]:
        """
        Calibrate a single depth plane using multiple anchors.

        Uses cross-validation: compute scale from each anchor, take median.
        """
        if not anchors:
            return None

        # Compute scale estimate from each anchor
        scale_estimates = [a.scale_estimate for a in anchors]

        # Compute statistics
        scale_median = np.median(scale_estimates)
        scale_std = np.std(scale_estimates)
        scale_mean = np.mean(scale_estimates)

        # Confidence: lower std relative to mean = higher confidence
        confidence = 1.0 - min(1.0, scale_std / (scale_mean + 1e-8))

        # Average depth
        depth_value = np.mean([a.depth for a in anchors])

        plane = DepthPlane(
            depth_value=depth_value,
            scale=scale_median,
            confidence=confidence,
            anchors=anchors,
            scale_std=scale_std
        )

        return plane

    def get_plane_for_depth(self, depth: float) -> Optional[DepthPlane]:
        """
        Get the calibrated plane closest to a given depth value.

        Args:
            depth: Depth value

        Returns:
            Closest depth plane or None
        """
        if not self.depth_planes:
            return None

        # Find closest plane
        closest_plane = min(
            self.depth_planes,
            key=lambda p: abs(p.depth_value - depth)
        )

        return closest_plane

    def measure_distance_2d(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        depth_map: np.ndarray
    ) -> Optional[Dict]:
        """
        Measure 2D distance between two points on an image.

        Args:
            point1: (x, y) first point
            point2: (x, y) second point
            depth_map: Depth map

        Returns:
            Dict with measurement info or None if calibration failed
        """
        # Get depth at midpoint
        mid_x = (point1[0] + point2[0]) / 2
        mid_y = (point1[1] + point2[1]) / 2
        depth = self._get_depth_in_bbox(
            depth_map,
            (int(mid_x - 5), int(mid_y - 5), int(mid_x + 5), int(mid_y + 5))
        )

        # Get calibrated plane
        plane = self.get_plane_for_depth(depth)

        if not plane:
            return None

        # Compute pixel distance
        pixel_distance = np.sqrt(
            (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2
        )

        # Convert to inches
        distance_inches = plane.measure_distance(pixel_distance)

        return {
            "distance_inches": distance_inches,
            "distance_pixels": pixel_distance,
            "scale": plane.scale,
            "confidence": plane.confidence,
            "depth": depth,
            "num_anchors": len(plane.anchors)
        }

    def generate_measurement_report(self) -> str:
        """
        Generate a text report of calibration status.

        Returns:
            Human-readable calibration report
        """
        if not self.depth_planes:
            return "⚠️  No calibration data available"

        lines = ["SPATIAL CALIBRATION REPORT", "=" * 50]

        for i, plane in enumerate(self.depth_planes, 1):
            lines.append(f"\nDepth Plane {i}:")
            lines.append(f"  Depth: {plane.depth_value:.3f}")
            lines.append(f"  Scale: {plane.scale:.2f} pixels/inch")
            lines.append(f"  Confidence: {plane.confidence:.2f}")
            lines.append(f"  Anchors: {len(plane.anchors)}")

            # List anchor types
            anchor_types = {}
            for anchor in plane.anchors:
                anchor_types[anchor.class_name] = anchor_types.get(anchor.class_name, 0) + 1

            for anchor_type, count in anchor_types.items():
                lines.append(f"    - {count}x {anchor_type}")

        return "\n".join(lines)
