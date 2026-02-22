"""Canny edge detection for construction images. Saves visualization for VLM input."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import get_image_id, load_image, save_image, setup_logger

logger = setup_logger("edges")

# Default Canny thresholds (low, high); tune for construction scenes
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 150


def compute_canny_edges(
    image_path: str,
    output_dir: str,
    low_threshold: int = DEFAULT_CANNY_LOW,
    high_threshold: int = DEFAULT_CANNY_HIGH,
) -> str:
    """
    Run Canny edge detection and save a visualization PNG.

    Reads from image_path, writes {output_dir}/{image_id}_canny.png
    (RGB copy of edge map so it can be used as a standard image input).

    Returns:
        Path to the saved canny PNG.
    """
    image_id = get_image_id(image_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_png = str(Path(output_dir) / f"{image_id}_canny.png")

    img_rgb = load_image(image_path)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    # Save as 3-channel so it works with save_image and VLM image APIs
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    save_image(edges_rgb, out_png)
    logger.info(f"{image_id}: Canny edges saved → {out_png}")
    return out_png
