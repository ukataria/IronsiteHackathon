"""Shared utilities for the temporal reconstruction pipeline."""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
DERIVED = DATA / "derived"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def save_json(data: dict[str, Any], output_path: Path | str) -> None:
    """Save data as JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(input_path: Path | str) -> dict[str, Any]:
    """Load JSON file."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")

    with open(input_path) as f:
        return json.load(f)


def load_image(image_path: Path | str) -> np.ndarray:
    """Load image as numpy array (RGB)."""
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, output_path: Path | str) -> None:
    """Save numpy array as image file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_path), image)


def compute_blur_score(image: np.ndarray) -> float:
    """Compute blur score using Laplacian variance.

    Higher values = sharper image.
    Typical threshold: 100.0 (images below this are considered blurry).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_image_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity (SSIM) between two images.

    Returns value between 0 and 1 (1 = identical).
    """
    from skimage.metrics import structural_similarity as ssim

    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = img2

    # Resize if dimensions don't match
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    return ssim(gray1, gray2)


def create_video_id_dir(video_id: str) -> Path:
    """Create and return directory for a video's derived data."""
    video_dir = DERIVED / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


def get_video_frames(video_path: Path | str) -> list[Path]:
    """Get list of frame paths for a video (sorted by frame number)."""
    video_path = Path(video_path)
    video_id = video_path.stem
    frames_dir = DERIVED / video_id / "frames"

    if not frames_dir.exists():
        return []

    frames = sorted(frames_dir.glob("*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    return frames
