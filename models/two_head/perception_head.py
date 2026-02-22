"""
Perception Head: Depth Anything V2 + Anchor Detection

This module handles the perception layer:
1. Depth estimation (frozen)
2. Anchor detection (frozen)
3. Plane grouping
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cv2


@dataclass
class AnchorDetection:
    """Detected anchor object with known dimensions."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str  # e.g., "2x4_stud", "cmu_block"
    confidence: float
    known_width_inches: float  # Known real-world width
    pixel_width: float  # Detected width in pixels
    center: Tuple[float, float]  # (x, y) center coordinates
    depth_value: Optional[float] = None  # From depth map


@dataclass
class DepthPlane:
    """A depth plane containing multiple objects."""
    depth_value: float
    anchors: List[AnchorDetection]
    pixels_per_inch: float  # Calibrated scale factor
    confidence: float  # Based on anchor consistency


# Known anchor dimensions (from Technical.md)
ANCHOR_DIMENSIONS = {
    "2x4_stud_face": 3.5,  # inches
    "2x4_stud_edge": 1.5,
    "2x6_joist_face": 5.5,
    "cmu_block_width": 15.625,
    "cmu_block_height": 7.625,
    "rebar_4": 0.5,
    "rebar_5": 0.625,
    "electrical_box_single_width": 2.0,
    "electrical_box_single_height": 3.0,
    "electrical_box_double_width": 4.0,
}


class PerceptionHead:
    """
    Perception head for PreCheck two-head architecture.

    Combines depth estimation and anchor detection.
    """

    def __init__(
        self,
        depth_model: str = "depth_anything_v2",
        anchor_detector: str = "simple_heuristic",  # Can upgrade to GroundedSAM
        device: str = "cpu"
    ):
        """
        Initialize perception head.

        Args:
            depth_model: Which depth model to use
            anchor_detector: Which anchor detection method
            device: Device for inference
        """
        self.depth_model = depth_model
        self.anchor_detector = anchor_detector
        self.device = device

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB image.

        For now, returns a simple synthetic depth map.
        Can be replaced with actual Depth Anything V2.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Depth map in meters (H, W)
        """
        # Placeholder: return simple depth based on vertical gradient
        # (objects lower in image are closer - common perspective)
        H, W = image.shape[:2]

        # Simple depth gradient: top of image is far (5m), bottom is close (2m)
        depth_map = np.linspace(5.0, 2.0, H)[:, None].repeat(W, axis=1)

        # Add some variation for realism
        noise = np.random.normal(0, 0.1, (H, W))
        depth_map = depth_map + noise
        depth_map = np.clip(depth_map, 0.5, 10.0)

        return depth_map.astype(np.float32)

    def detect_anchors(
        self,
        image: np.ndarray,
        depth_map: np.ndarray
    ) -> List[AnchorDetection]:
        """
        Detect known-dimension anchor objects.

        For now, uses simple heuristics. Can be replaced with GroundedSAM.

        Args:
            image: RGB image (H, W, 3)
            depth_map: Depth map (H, W)

        Returns:
            List of detected anchors
        """
        if self.anchor_detector == "simple_heuristic":
            return self._detect_anchors_heuristic(image, depth_map)
        else:
            raise NotImplementedError(f"Anchor detector {self.anchor_detector} not implemented")

    def _detect_anchors_heuristic(
        self,
        image: np.ndarray,
        depth_map: np.ndarray
    ) -> List[AnchorDetection]:
        """
        Simple heuristic anchor detection.

        Detects rectangular regions that could be studs/boards.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        anchors = []

        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size (reasonable construction elements)
            if w < 20 or h < 20:  # Too small
                continue
            if w > image.shape[1] * 0.8 or h > image.shape[0] * 0.8:  # Too large
                continue

            # Assume vertical rectangles are likely studs
            aspect_ratio = h / w if w > 0 else 0

            if aspect_ratio > 2.0:  # Vertical element (like a stud)
                # Estimate if it's a 2x4 face or edge based on width
                # Typical 2x4 face (3.5") appears wider than edge (1.5")
                if w > 40:  # Likely face
                    class_name = "2x4_stud_face"
                    known_width = ANCHOR_DIMENSIONS["2x4_stud_face"]
                else:  # Likely edge
                    class_name = "2x4_stud_edge"
                    known_width = ANCHOR_DIMENSIONS["2x4_stud_edge"]

                # Get center point
                cx, cy = x + w/2, y + h/2

                # Get depth at center
                depth_value = depth_map[int(cy), int(cx)] if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1] else None

                anchor = AnchorDetection(
                    bbox=(x, y, x+w, y+h),
                    class_name=class_name,
                    confidence=0.7,  # Heuristic confidence
                    known_width_inches=known_width,
                    pixel_width=float(w),
                    center=(cx, cy),
                    depth_value=depth_value
                )

                anchors.append(anchor)

        return anchors

    def run(self, image: np.ndarray) -> Tuple[np.ndarray, List[AnchorDetection]]:
        """
        Run full perception head pipeline.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Tuple of (depth_map, anchors)
        """
        # Step 1: Estimate depth
        depth_map = self.estimate_depth(image)

        # Step 2: Detect anchors
        anchors = self.detect_anchors(image, depth_map)

        return depth_map, anchors
