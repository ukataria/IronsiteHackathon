"""
Anchor Detection Module

Detects known-dimension construction objects (anchors) for spatial calibration.
Uses YOLOv8 for object detection.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


# Known anchor dimensions (in inches)
ANCHOR_DIMENSIONS = {
    "2x4_stud_face": 3.5,      # 2x4 lumber face width
    "2x4_stud_edge": 1.5,      # 2x4 lumber edge width
    "2x6_joist_face": 5.5,     # 2x6 lumber face width
    "cmu_block_width": 15.625, # CMU block width
    "cmu_block_height": 7.625, # CMU block height
    "rebar_4": 0.500,          # #4 rebar diameter
    "rebar_5": 0.625,          # #5 rebar diameter
    "electrical_box_single": 2.0,  # Single gang box width
    "electrical_box_double": 4.0,  # Double gang box width
    "door_opening_width": 38.5,    # Standard door rough opening
    "brick_length": 7.625,     # Standard brick length
    "brick_height": 2.25,      # Standard brick height
}


@dataclass
class Anchor:
    """Detected anchor object with known dimensions."""

    class_name: str           # Type of anchor (e.g., "2x4_stud_face")
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box
    confidence: float         # Detection confidence
    known_width: float        # Known real-world width in inches
    pixel_width: float        # Detected pixel width
    center: Tuple[float, float]  # (x, y) center point
    depth: float = None       # Depth value (set by calibration module)

    @property
    def scale_estimate(self) -> float:
        """Pixels per inch scale estimate from this anchor."""
        return self.pixel_width / self.known_width


class AnchorDetector:
    """Detects construction anchors using YOLO."""

    def __init__(self, model_path: str = None, device: str = "cuda"):
        """
        Initialize anchor detector.

        Args:
            model_path: Path to trained YOLO model (or None for pretrained COCO)
            device: Device to run on ("cuda" or "cpu")
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        self.device = device

        if model_path and Path(model_path).exists():
            # Load custom-trained model for construction objects
            self.model = YOLO(model_path)
        else:
            # Use pretrained YOLO for now (we'll detect generic objects)
            # In production, this would be a model fine-tuned on construction anchors
            self.model = YOLO("yolov8n.pt")  # Nano model for speed

        self.model.to(device)

    def detect(self, image_path: str, confidence_threshold: float = 0.25) -> List[Anchor]:
        """
        Detect anchor objects in an image.

        Args:
            image_path: Path to image
            confidence_threshold: Minimum detection confidence

        Returns:
            List of detected anchors
        """
        # Run YOLO detection
        try:
            results = self.model(image_path, conf=confidence_threshold, verbose=False)
        except Exception as e:
            print(f"  ⚠️  YOLO inference failed: {e}")
            return []

        anchors = []
        detected_objects = []  # For debugging
        total_detections = 0

        # For now, we'll use generic YOLO detections and map them to construction anchors
        # In production, this would use a construction-specific model
        for result in results:
            boxes = result.boxes
            total_detections += len(boxes)

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                # Track what YOLO detected (for debugging)
                detected_objects.append(class_name)

                # Map generic YOLO classes to construction anchors
                # This is a placeholder - in production, use construction-trained model
                anchor_type = self._map_to_anchor_type(class_name, x2 - x1, y2 - y1)

                if anchor_type:
                    pixel_width = x2 - x1
                    known_width = ANCHOR_DIMENSIONS[anchor_type]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    anchor = Anchor(
                        class_name=anchor_type,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        known_width=known_width,
                        pixel_width=pixel_width,
                        center=(center_x, center_y)
                    )
                    anchors.append(anchor)

        # Debug output
        if total_detections > 0:
            if detected_objects:
                unique_classes = set(detected_objects)
                print(f"  YOLO detected: {', '.join(sorted(unique_classes))} ({len(detected_objects)} total)")
                if not anchors:
                    print(f"  ⚠️  No objects mapped to known-dimension anchors")
                else:
                    anchor_types = [a.class_name for a in anchors]
                    print(f"  ✓ Found {len(anchors)} anchor(s): {', '.join(set(anchor_types))}")
            else:
                print(f"  ⚠️  YOLO found {total_detections} detection(s) but no objects?")
        else:
            print(f"  ⚠️  YOLO detected 0 objects (confidence threshold: {confidence_threshold})")

        return anchors

    def _map_to_anchor_type(self, yolo_class: str, width: float, height: float) -> str:
        """
        Map generic YOLO class to construction anchor type.

        Uses known dimensions of common objects for NYU dataset.
        In production, use a construction-specific model.

        Args:
            yolo_class: YOLO class name
            width: Detection width
            height: Detection height

        Returns:
            Anchor type string or None
        """
        # Map common COCO objects to approximate known dimensions
        # This enables calibration on indoor scenes (NYU dataset)

        class_lower = yolo_class.lower()

        # Common indoor objects with reasonably standard dimensions
        if "door" in class_lower:
            # Standard interior door is ~30-36 inches wide
            ANCHOR_DIMENSIONS["door_width"] = 32.0  # inches
            return "door_width"

        if "refrigerator" in class_lower or "fridge" in class_lower:
            # Standard fridge width ~30-36 inches
            ANCHOR_DIMENSIONS["refrigerator_width"] = 33.0
            return "refrigerator_width"

        if "oven" in class_lower or "stove" in class_lower:
            # Standard range width 30 inches
            ANCHOR_DIMENSIONS["oven_width"] = 30.0
            return "oven_width"

        if "microwave" in class_lower:
            # Typical microwave ~18-24 inches wide
            ANCHOR_DIMENSIONS["microwave_width"] = 20.0
            return "microwave_width"

        # Fall back to aspect ratio heuristics for generic objects
        aspect_ratio = height / width if width > 0 else 0

        if aspect_ratio > 2.0 and width > 20:  # Tall vertical object
            # Could be a door frame, window, or vertical structure
            ANCHOR_DIMENSIONS["vertical_element"] = 36.0  # Assume ~3ft wide
            return "vertical_element"

        # No reliable anchor found
        return None


class GroundTruthAnchorDetector:
    """
    Manual anchor annotation for ground truth / testing.

    For when automatic detection isn't available or for validation.
    """

    def __init__(self):
        self.annotations = {}

    def add_annotation(
        self,
        image_path: str,
        anchor_type: str,
        bbox: Tuple[int, int, int, int],
        confidence: float = 1.0
    ):
        """Add a manual annotation."""
        if image_path not in self.annotations:
            self.annotations[image_path] = []

        x1, y1, x2, y2 = bbox
        pixel_width = x2 - x1
        known_width = ANCHOR_DIMENSIONS[anchor_type]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        anchor = Anchor(
            class_name=anchor_type,
            bbox=bbox,
            confidence=confidence,
            known_width=known_width,
            pixel_width=pixel_width,
            center=(center_x, center_y)
        )

        self.annotations[image_path].append(anchor)

    def detect(self, image_path: str) -> List[Anchor]:
        """Return manual annotations for an image."""
        return self.annotations.get(image_path, [])
