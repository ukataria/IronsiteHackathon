"""
Depth Estimation Module

Uses Depth Anything V2 Large for monocular depth estimation.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
import cv2


class DepthEstimator:
    """Depth estimation using Depth Anything V2."""

    def __init__(self, model_size: str = "large", device: str = "cuda"):
        """
        Initialize depth estimator.

        Args:
            model_size: Model size ("small", "base", "large")
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self.model_size = model_size

        try:
            import transformers
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")

        # Load Depth Anything V2 from HuggingFace
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}"

        print(f"Loading {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def estimate_depth(
        self,
        image_path: Union[str, Path],
        output_vis_path: str = None
    ) -> np.ndarray:
        """
        Estimate depth map for an image.

        Args:
            image_path: Path to input image
            output_vis_path: Optional path to save visualization

        Returns:
            Depth map as numpy array (H x W), normalized to [0, 1]
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Infer depth
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],  # (height, width)
            mode="bicubic",
            align_corners=False,
        )

        # Convert to numpy
        depth_map = prediction.squeeze().cpu().numpy()

        # Normalize to [0, 1] for consistency
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        # Save visualization if requested
        if output_vis_path:
            self._save_visualization(depth_map, output_vis_path)

        return depth_map

    def _save_visualization(self, depth_map: np.ndarray, output_path: str):
        """Save depth map as a visualization image."""
        # Apply colormap
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        # Save
        cv2.imwrite(output_path, depth_colored)

    def get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """
        Get depth value at a specific pixel.

        Args:
            depth_map: Depth map array
            x: X coordinate (column)
            y: Y coordinate (row)

        Returns:
            Depth value at (x, y)
        """
        h, w = depth_map.shape
        y = int(np.clip(y, 0, h - 1))
        x = int(np.clip(x, 0, w - 1))
        return float(depth_map[y, x])

    def get_depth_in_bbox(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        method: str = "median"
    ) -> float:
        """
        Get representative depth value within a bounding box.

        Args:
            depth_map: Depth map array
            bbox: (x1, y1, x2, y2) bounding box
            method: Aggregation method ("median", "mean", "min", "max")

        Returns:
            Aggregate depth value
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape

        # Clip to image bounds
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))

        # Extract region
        region = depth_map[y1:y2, x1:x2]

        # Aggregate
        if method == "median":
            return float(np.median(region))
        elif method == "mean":
            return float(np.mean(region))
        elif method == "min":
            return float(np.min(region))
        elif method == "max":
            return float(np.max(region))
        else:
            raise ValueError(f"Unknown method: {method}")
