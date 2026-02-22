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
            self.model = YOLO("yolov8x.pt")  # Extra Large model for best accuracy

        self.model.to(device)

    def detect(self, image_path: str, confidence_threshold: float = 0.01) -> List[Anchor]:
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
            results = self.model(image_path, conf=confidence_threshold, verbose=True)
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

        if "tv" in class_lower or "monitor" in class_lower:
            # Common TV/monitor width
            ANCHOR_DIMENSIONS["tv_width"] = 43.0  # inches (typical TV)
            return "tv_width"

        if "chair" in class_lower:
            # Standard dining chair width ~16-18 inches
            ANCHOR_DIMENSIONS["chair_width"] = 17.0
            return "chair_width"

        if "couch" in class_lower or "sofa" in class_lower:
            # Standard 3-seater sofa ~84 inches
            ANCHOR_DIMENSIONS["couch_width"] = 84.0
            return "couch_width"

        if "bed" in class_lower:
            # Queen bed width ~60 inches
            ANCHOR_DIMENSIONS["bed_width"] = 60.0
            return "bed_width"

        if "table" in class_lower or "desk" in class_lower:
            # Typical desk/dining table ~36-48 inches wide
            ANCHOR_DIMENSIONS["table_width"] = 42.0
            return "table_width"

        if "sink" in class_lower:
            # Standard kitchen sink ~22-33 inches
            ANCHOR_DIMENSIONS["sink_width"] = 27.0
            return "sink_width"

        if "toilet" in class_lower:
            # Standard toilet width ~14-16 inches
            ANCHOR_DIMENSIONS["toilet_width"] = 15.0
            return "toilet_width"

        if "bathtub" in class_lower:
            # Standard bathtub width ~30 inches
            ANCHOR_DIMENSIONS["bathtub_width"] = 30.0
            return "bathtub_width"

        if "cabinet" in class_lower:
            # Kitchen cabinet width varies, use standard base
            ANCHOR_DIMENSIONS["cabinet_width"] = 30.0
            return "cabinet_width"

        if "window" in class_lower:
            # Standard window ~36 inches wide
            ANCHOR_DIMENSIONS["window_width"] = 36.0
            return "window_width"

        if "book" in class_lower:
            # Standard hardcover book height ~9 inches
            ANCHOR_DIMENSIONS["book_height"] = 9.0
            return "book_height"

        if "laptop" in class_lower:
            # Typical laptop screen ~13-15 inches diagonal
            ANCHOR_DIMENSIONS["laptop_width"] = 13.0
            return "laptop_width"

        if "bottle" in class_lower:
            # Standard water bottle height ~8-10 inches
            ANCHOR_DIMENSIONS["bottle_height"] = 9.0
            return "bottle_height"

        if "cup" in class_lower or "mug" in class_lower:
            # Standard coffee mug ~3-4 inches diameter
            ANCHOR_DIMENSIONS["cup_width"] = 3.5
            return "cup_width"

        if "person" in class_lower:
            # Use for scale - average adult height ~66 inches (5'6")
            # But use shoulder width as more reliable anchor ~18 inches
            if aspect_ratio > 1.5:  # Standing person
                ANCHOR_DIMENSIONS["person_height"] = 66.0
                return "person_height"
            else:  # Sitting or partial
                ANCHOR_DIMENSIONS["person_torso"] = 18.0
                return "person_torso"

        if "clock" in class_lower:
            # Wall clock diameter ~10-12 inches
            ANCHOR_DIMENSIONS["clock_diameter"] = 11.0
            return "clock_diameter"

        if "vase" in class_lower:
            # Typical vase height ~10-14 inches
            ANCHOR_DIMENSIONS["vase_height"] = 12.0
            return "vase_height"

        if "potted plant" in class_lower or "plant" in class_lower:
            # Typical potted plant pot diameter ~8-12 inches
            ANCHOR_DIMENSIONS["plant_pot"] = 10.0
            return "plant_pot"

        if "keyboard" in class_lower:
            # Standard keyboard width ~18 inches
            ANCHOR_DIMENSIONS["keyboard_width"] = 18.0
            return "keyboard_width"

        if "mouse" in class_lower:
            # Computer mouse length ~4-5 inches
            ANCHOR_DIMENSIONS["mouse_length"] = 4.5
            return "mouse_length"

        if "remote" in class_lower:
            # TV remote length ~8 inches
            ANCHOR_DIMENSIONS["remote_length"] = 8.0
            return "remote_length"

        if "cell phone" in class_lower or "phone" in class_lower:
            # Smartphone height ~6 inches
            ANCHOR_DIMENSIONS["phone_height"] = 6.0
            return "phone_height"

        if "backpack" in class_lower or "suitcase" in class_lower:
            # Standard backpack height ~18-20 inches
            ANCHOR_DIMENSIONS["backpack_height"] = 19.0
            return "backpack_height"

        if "bowl" in class_lower:
            # Cereal/salad bowl diameter ~6-8 inches
            ANCHOR_DIMENSIONS["bowl_diameter"] = 7.0
            return "bowl_diameter"

        if "scissors" in class_lower:
            # Standard scissors length ~8 inches
            ANCHOR_DIMENSIONS["scissors_length"] = 8.0
            return "scissors_length"

        if "umbrella" in class_lower:
            # Closed umbrella length ~12 inches
            ANCHOR_DIMENSIONS["umbrella_length"] = 12.0
            return "umbrella_length"

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
