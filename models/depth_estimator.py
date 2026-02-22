"""
Depth Estimation Module

Supports multiple depth estimation models:
- Depth Anything V2 (Small, Base, Large)
- ZoeDepth (NK, N, K variants)
- MiDaS (Small, DPT-Hybrid, DPT-Large)
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
import cv2


class DepthEstimator:
    """Multi-model depth estimation."""

    def __init__(
        self,
        model_type: str = "depth_anything_v2",
        model_size: str = "large",
        device: str = "cuda"
    ):
        """
        Initialize depth estimator.

        Args:
            model_type: Model type ("depth_anything_v2", "zoe", "midas")
            model_size: Model size variant (model-specific)
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self.model_type = model_type.lower()
        self.model_size = model_size.lower()

        if self.model_type == "depth_anything_v2":
            self._init_depth_anything_v2()
        elif self.model_type == "zoe":
            self._init_zoe()
        elif self.model_type == "midas":
            self._init_midas()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _init_depth_anything_v2(self):
        """Initialize Depth Anything V2."""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            raise ImportError("transformers not installed. Run: uv add transformers")

        # Depth Anything V2 model paths on HuggingFace
        model_paths = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf"
        }

        model_name = model_paths.get(self.model_size, model_paths["large"])

        print(f"Loading Depth Anything V2 ({self.model_size})...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _init_zoe(self):
        """Initialize ZoeDepth."""
        try:
            import torch.hub
        except ImportError:
            raise ImportError("torch not installed")

        # ZoeDepth variants: NK (indoor+outdoor), N (NYU/indoor), K (KITTI/outdoor)
        variant_map = {
            "nk": "ZoeD_NK",  # Hybrid indoor+outdoor (recommended)
            "n": "ZoeD_N",    # NYU-trained (indoor only)
            "k": "ZoeD_K",    # KITTI-trained (outdoor only)
        }

        variant = variant_map.get(self.model_size, "ZoeD_NK")

        print(f"Loading ZoeDepth ({variant})...")
        repo = "isl-org/ZoeDepth"
        self.model = torch.hub.load(repo, variant, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def _init_midas(self):
        """Initialize MiDaS."""
        try:
            import torch.hub
        except ImportError:
            raise ImportError("torch not installed")

        # MiDaS variants
        variant_map = {
            "small": "MiDAS_small",
            "hybrid": "DPT_Hybrid",
            "large": "DPT_Large",
        }

        variant = variant_map.get(self.model_size, "DPT_Large")

        print(f"Loading MiDaS ({variant})...")
        self.model = torch.hub.load("intel-isl/MiDaS", variant)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if variant == "DPT_Large" or variant == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

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
        if self.model_type == "depth_anything_v2":
            return self._estimate_depth_anything_v2(image_path, output_vis_path)
        elif self.model_type == "zoe":
            return self._estimate_zoe(image_path, output_vis_path)
        elif self.model_type == "midas":
            return self._estimate_midas(image_path, output_vis_path)

    def _estimate_depth_anything_v2(
        self,
        image_path: Union[str, Path],
        output_vis_path: str = None
    ) -> np.ndarray:
        """Depth Anything V2 inference."""
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

    def _estimate_zoe(
        self,
        image_path: Union[str, Path],
        output_vis_path: str = None
    ) -> np.ndarray:
        """ZoeDepth inference."""
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (W, H)

        # ZoeDepth expects PIL image directly
        with torch.no_grad():
            depth = self.model.infer_pil(image)

        # depth is already a numpy array (H, W)
        depth_map = depth

        # Normalize to [0, 1]
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        # Save visualization if requested
        if output_vis_path:
            self._save_visualization(depth_map, output_vis_path)

        return depth_map

    def _estimate_midas(
        self,
        image_path: Union[str, Path],
        output_vis_path: str = None
    ) -> np.ndarray:
        """MiDaS inference."""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]  # (H, W)

        # Transform
        input_batch = self.transform(image).to(self.device)

        # Infer
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=original_shape,
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to [0, 1]
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
