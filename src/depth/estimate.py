"""Depth estimation using Depth Anything V2 via HuggingFace transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import numpy as np
import torch

from src.utils import get_image_id, load_image, save_image, save_json, setup_logger

logger = setup_logger("depth")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

DEPTH_MODEL_IDS: dict[str, str] = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

_depth_model: Any | None = None
_depth_processor: Any | None = None
_loaded_model_size: str | None = None


def load_depth_model(model_size: str = "small", device: str = "cpu") -> tuple[Any, Any]:
    """Load Depth Anything V2. Cached after first load per model size."""
    global _depth_model, _depth_processor, _loaded_model_size

    if _depth_model is not None and _loaded_model_size == model_size:
        return _depth_model, _depth_processor

    if model_size not in DEPTH_MODEL_IDS:
        raise ValueError(f"model_size must be one of {list(DEPTH_MODEL_IDS)}. Got: {model_size}")

    model_id = DEPTH_MODEL_IDS[model_size]
    logger.info(f"Loading Depth Anything V2 ({model_size}) from {model_id}...")

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        _depth_processor = AutoImageProcessor.from_pretrained(model_id)
        _depth_model = AutoModelForDepthEstimation.from_pretrained(model_id)
        _depth_model = _depth_model.to(device)
        _depth_model.eval()
        _loaded_model_size = model_size
        logger.info("Depth Anything V2 loaded.")
    except Exception as e:
        logger.error(f"Failed to load depth model: {e}")
        raise

    return _depth_model, _depth_processor


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------


def _normalize_depth(raw: np.ndarray) -> np.ndarray:
    """Normalize raw depth output to [0, 1] float32."""
    dmin, dmax = raw.min(), raw.max()
    if dmax - dmin < 1e-8:
        return np.zeros_like(raw, dtype=np.float32)
    return ((raw - dmin) / (dmax - dmin)).astype(np.float32)


def _colorize_depth(depth_norm: np.ndarray) -> np.ndarray:
    """Convert normalized depth [0,1] to a jet-colorized RGB uint8 image."""
    colored = cm.jet(depth_norm)[:, :, :3]  # drop alpha
    return (colored * 255).astype(np.uint8)


def estimate_depth(
    image_path: str,
    output_dir: str,
    model_size: str = "small",
    device: str = "cpu",
) -> np.ndarray:
    """
    Run Depth Anything V2 on an image.

    Returns normalized depth map as float32 array [0, 1] (H, W).
    Saves:
      {output_dir}/{image_id}_depth.npy   — raw normalized array
      {output_dir}/{image_id}_depth.png   — colorized visualization
    """
    from PIL import Image as PILImage

    image_id = get_image_id(image_path)
    out_npy = str(Path(output_dir) / f"{image_id}_depth.npy")
    out_png = str(Path(output_dir) / f"{image_id}_depth.png")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    img_rgb = load_image(image_path)
    pil_img = PILImage.fromarray(img_rgb)

    try:
        model, processor = load_depth_model(model_size, device)
        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, H, W)

        raw = predicted_depth.squeeze().cpu().numpy()
        # Resize to original image dimensions
        import cv2
        h, w = img_rgb.shape[:2]
        raw_resized = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_norm = _normalize_depth(raw_resized)

    except Exception as e:
        logger.error(f"Depth estimation failed for {image_id}: {e}")
        h, w = load_image(image_path).shape[:2]
        depth_norm = np.zeros((h, w), dtype=np.float32)

    np.save(out_npy, depth_norm)
    colorized = _colorize_depth(depth_norm)
    save_image(colorized, out_png)
    logger.info(f"{image_id}: depth saved → {out_npy}")
    return depth_norm


# ---------------------------------------------------------------------------
# Depth querying helpers
# ---------------------------------------------------------------------------


def get_anchor_depth(
    depth_map: np.ndarray,
    box_pixels: list[float],
    image_width: int,
    image_height: int,
) -> float:
    """
    Get the median depth value within an anchor bounding box.
    box_pixels: [x1, y1, x2, y2] in absolute pixel coordinates.
    Returns a float in [0, 1] (normalized depth).
    """
    x1, y1, x2, y2 = [int(v) for v in box_pixels]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width - 1, x2)
    y2 = min(image_height - 1, y2)
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return 0.5  # fallback mid-depth
    return float(np.median(region))


def group_anchors_by_plane(
    anchors: list[dict],
    depth_map: np.ndarray,
    image_width: int,
    image_height: int,
    depth_tolerance: float = 0.05,
) -> dict[str, list[dict]]:
    """
    Group anchors that share the same depth plane (within tolerance).
    Returns dict mapping plane_id → list of anchor dicts (augmented with 'depth' key).
    """
    if not anchors:
        return {}

    # Augment each anchor with its median depth
    for a in anchors:
        a["depth"] = get_anchor_depth(depth_map, a["box_pixels"], image_width, image_height)

    # Sort by depth
    sorted_anchors = sorted(anchors, key=lambda a: a["depth"])

    planes: dict[str, list[dict]] = {}
    plane_depths: dict[str, float] = {}
    plane_idx = 0

    for anchor in sorted_anchors:
        d = anchor["depth"]
        placed = False
        for plane_id, plane_d in plane_depths.items():
            if abs(d - plane_d) <= depth_tolerance:
                planes[plane_id].append(anchor)
                # Update running median depth for plane
                plane_depths[plane_id] = float(
                    np.median([a["depth"] for a in planes[plane_id]])
                )
                placed = True
                break
        if not placed:
            plane_id = f"plane_{plane_idx}"
            planes[plane_id] = [anchor]
            plane_depths[plane_id] = d
            plane_idx += 1

    logger.info(f"Grouped {len(anchors)} anchors into {len(planes)} depth planes.")
    return planes


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python src/depth/estimate.py <image_path> [output_dir] [model_size]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/depth"
    size = sys.argv[3] if len(sys.argv) > 3 else "small"
    depth = estimate_depth(img_path, out_dir, model_size=size)
    print(f"Depth map shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
