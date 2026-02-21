"""Depth estimation using Depth Anything V2."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from src.utils import DEPTH_DIR, save_depth, frame_stem

log = logging.getLogger(__name__)

# Model variant: "small" | "base" | "large" — use "large" on Vast.ai GPU
DEPTH_MODEL_SIZE = "large"
DEPTH_MODEL_ID = f"depth-anything/Depth-Anything-V2-{DEPTH_MODEL_SIZE.capitalize()}-hf"

_pipeline = None


def _get_pipeline():
    """Lazy-load Depth Anything V2 pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from transformers import pipeline as hf_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading Depth Anything V2 ({DEPTH_MODEL_SIZE}) on {device}")
    _pipeline = hf_pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL_ID,
        device=device,
    )
    log.info("Depth model loaded.")
    return _pipeline


def estimate_depth(image_path: str | Path) -> np.ndarray:
    """Run Depth Anything V2 on a single image.

    Args:
        image_path: Path to the input frame (PNG/JPG).

    Returns:
        Depth map as float32 numpy array (higher = closer to camera).
    """
    from PIL import Image as PILImage

    image_path = Path(image_path)
    pipe = _get_pipeline()
    img = PILImage.open(str(image_path)).convert("RGB")
    result = pipe(img)
    depth = np.array(result["depth"], dtype=np.float32)
    return depth


def estimate_and_save(image_path: str | Path, stem: str | None = None) -> tuple[Path, Path]:
    """Estimate depth and save .npy + visualization .png.

    Args:
        image_path: Input frame path.
        stem: Output filename stem. Defaults to input stem.

    Returns:
        (npy_path, png_path)
    """
    image_path = Path(image_path)
    if stem is None:
        stem = image_path.stem

    depth = estimate_depth(image_path)
    npy_path, png_path = save_depth(depth, stem)
    log.info(f"Depth saved: {npy_path} | {png_path}")
    return npy_path, png_path


def batch_estimate(frame_paths: list[Path]) -> list[tuple[Path, Path]]:
    """Run depth estimation on a list of frames.

    Args:
        frame_paths: List of frame image paths.

    Returns:
        List of (npy_path, png_path) tuples.
    """
    results = []
    for i, fp in enumerate(frame_paths):
        log.info(f"Depth [{i+1}/{len(frame_paths)}]: {fp.name}")
        npy, png = estimate_and_save(fp)
        results.append((npy, png))
    return results


def get_depth_planes(depth: np.ndarray, n_planes: int = 4) -> list[np.ndarray]:
    """Segment depth map into N planes by quantile thresholds.

    Returns a list of boolean masks, from closest to farthest.
    """
    thresholds = np.quantile(depth, np.linspace(0, 1, n_planes + 1))
    masks = []
    for i in range(n_planes):
        mask = (depth >= thresholds[i]) & (depth < thresholds[i + 1])
        masks.append(mask)
    return masks


def depth_to_surface_normal(depth: np.ndarray) -> np.ndarray:
    """Estimate surface normals from depth map via gradient.

    Returns an HxWx3 float32 array of unit normal vectors.
    """
    gy, gx = np.gradient(depth)
    ones = np.ones_like(depth)
    normals = np.stack([-gx, -gy, ones], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    return (normals / norm).astype(np.float32)
