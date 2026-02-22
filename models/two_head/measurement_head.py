"""
Measurement Head: Calibrated Spatial Measurements

This module implements the calibration logic from Technical.md Stage 3.
Takes anchor detections + depth map and outputs calibrated measurements.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics

from .perception_head import AnchorDetection, DepthPlane


@dataclass
class Measurement:
    """A calibrated spatial measurement."""
    measurement_type: str  # "spacing", "height", "clearance", etc.
    value_inches: float
    confidence: float
    from_object: str  # Description of start object
    to_object: str  # Description of end object
    location: Tuple[float, float]  # (x, y) pixel location


class MeasurementHead:
    """
    Measurement head for spatial calibration.

    Implements the math from Technical.md Stage 3:
    - Single-anchor calibration
    - Multi-anchor cross-validation
    - Depth-adjusted measurement
    """

    def __init__(
        self,
        depth_tolerance: float = 0.5,  # meters - objects within this depth are on same plane
        min_anchors_for_validation: int = 2
    ):
        """
        Initialize measurement head.

        Args:
            depth_tolerance: Depth difference threshold for plane grouping
            min_anchors_for_validation: Minimum anchors needed for cross-validation
        """
        self.depth_tolerance = depth_tolerance
        self.min_anchors_for_validation = min_anchors_for_validation

    def group_anchors_by_plane(
        self,
        anchors: List[AnchorDetection]
    ) -> List[DepthPlane]:
        """
        Group anchors into depth planes.

        Anchors with similar depth values are on the same plane.

        Args:
            anchors: List of detected anchors

        Returns:
            List of depth planes with grouped anchors
        """
        if not anchors:
            return []

        # Filter anchors with valid depth
        valid_anchors = [a for a in anchors if a.depth_value is not None]
        if not valid_anchors:
            return []

        # Sort by depth
        sorted_anchors = sorted(valid_anchors, key=lambda a: a.depth_value)

        planes = []
        current_plane_anchors = [sorted_anchors[0]]
        current_depth = sorted_anchors[0].depth_value

        for anchor in sorted_anchors[1:]:
            if abs(anchor.depth_value - current_depth) <= self.depth_tolerance:
                # Same plane
                current_plane_anchors.append(anchor)
            else:
                # New plane - finalize current plane
                plane = self._calibrate_plane(current_plane_anchors)
                planes.append(plane)

                # Start new plane
                current_plane_anchors = [anchor]
                current_depth = anchor.depth_value

        # Add final plane
        if current_plane_anchors:
            plane = self._calibrate_plane(current_plane_anchors)
            planes.append(plane)

        return planes

    def _calibrate_plane(self, anchors: List[AnchorDetection]) -> DepthPlane:
        """
        Calibrate scale factor for a depth plane.

        Uses multi-anchor cross-validation from Technical.md.

        Args:
            anchors: Anchors on this plane

        Returns:
            Calibrated depth plane
        """
        # Compute scale estimate from each anchor
        # pixels_per_inch = pixel_width / known_width_inches
        scale_estimates = []

        for anchor in anchors:
            scale = anchor.pixel_width / anchor.known_width_inches
            scale_estimates.append(scale)

        # Use median for robustness
        calibrated_scale = statistics.median(scale_estimates)

        # Compute confidence based on consistency
        if len(scale_estimates) >= self.min_anchors_for_validation:
            mean_scale = statistics.mean(scale_estimates)
            std_scale = statistics.stdev(scale_estimates) if len(scale_estimates) > 1 else 0.0
            confidence = 1.0 - min(1.0, std_scale / mean_scale) if mean_scale > 0 else 0.5
        else:
            confidence = 0.5  # Lower confidence with few anchors

        # Average depth of plane
        avg_depth = statistics.mean([a.depth_value for a in anchors])

        return DepthPlane(
            depth_value=avg_depth,
            anchors=anchors,
            pixels_per_inch=calibrated_scale,
            confidence=confidence
        )

    def measure_spacing(
        self,
        anchor1: AnchorDetection,
        anchor2: AnchorDetection,
        plane: DepthPlane
    ) -> Measurement:
        """
        Measure center-to-center spacing between two anchors.

        Args:
            anchor1: First anchor
            anchor2: Second anchor
            plane: Calibrated depth plane

        Returns:
            Spacing measurement
        """
        # Pixel distance between centers
        dx = anchor2.center[0] - anchor1.center[0]
        dy = anchor2.center[1] - anchor1.center[1]
        pixel_distance = np.sqrt(dx**2 + dy**2)

        # Convert to inches using calibrated scale
        distance_inches = pixel_distance / plane.pixels_per_inch

        # Midpoint location
        mid_x = (anchor1.center[0] + anchor2.center[0]) / 2
        mid_y = (anchor1.center[1] + anchor2.center[1]) / 2

        return Measurement(
            measurement_type="spacing",
            value_inches=distance_inches,
            confidence=plane.confidence,
            from_object=f"{anchor1.class_name} at ({anchor1.center[0]:.0f}, {anchor1.center[1]:.0f})",
            to_object=f"{anchor2.class_name} at ({anchor2.center[0]:.0f}, {anchor2.center[1]:.0f})",
            location=(mid_x, mid_y)
        )

    def extract_measurements(
        self,
        planes: List[DepthPlane]
    ) -> List[Measurement]:
        """
        Extract all meaningful measurements from calibrated planes.

        Focuses on stud spacing (the most common construction measurement).

        Args:
            planes: Calibrated depth planes

        Returns:
            List of measurements
        """
        measurements = []

        for plane in planes:
            # Filter for studs (most common anchor type)
            studs = [a for a in plane.anchors if "stud" in a.class_name.lower()]

            if len(studs) < 2:
                continue

            # Sort studs by horizontal position (left to right)
            studs_sorted = sorted(studs, key=lambda a: a.center[0])

            # Measure spacing between consecutive studs
            for i in range(len(studs_sorted) - 1):
                measurement = self.measure_spacing(
                    studs_sorted[i],
                    studs_sorted[i + 1],
                    plane
                )
                measurements.append(measurement)

        return measurements

    def run(
        self,
        anchors: List[AnchorDetection]
    ) -> Tuple[List[DepthPlane], List[Measurement]]:
        """
        Run full measurement head pipeline.

        Args:
            anchors: Detected anchors from perception head

        Returns:
            Tuple of (depth_planes, measurements)
        """
        # Group anchors into depth planes
        planes = self.group_anchors_by_plane(anchors)

        # Extract measurements
        measurements = self.extract_measurements(planes)

        return planes, measurements
