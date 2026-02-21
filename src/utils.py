"""Shared utilities — image I/O, caching, logging, constants."""

import base64
import hashlib
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANCHOR_DIMENSIONS: dict[str, float] = {
    "stud_face_width": 3.5,         # 2x4 face width in inches
    "stud_depth": 1.5,              # 2x4 depth in inches
    "stud_face_width_2x6": 5.5,     # 2x6 face width in inches
    "cmu_length": 15.625,           # CMU block length in inches
    "cmu_height": 7.625,            # CMU block height in inches
    "rebar_4_diameter": 0.5,        # #4 rebar diameter in inches
    "rebar_5_diameter": 0.625,      # #5 rebar diameter in inches
    "electrical_box_single_width": 2.0,
    "electrical_box_single_height": 3.0,
    "electrical_box_double_width": 4.0,
    "electrical_box_double_height": 3.0,
    "door_rough_width": 38.5,       # Standard door rough opening in inches
    "door_rough_height": 82.5,
    "hardhat_width": 12.0,
    "brick_length": 7.625,
    "brick_height": 2.25,
    "plywood_width": 48.0,
    "plywood_height": 96.0,
}

# Map anchor type names to their primary measurement dimension key
ANCHOR_PRIMARY_DIMENSION: dict[str, str] = {
    "stud": "stud_face_width",
    "rebar": "rebar_4_diameter",
    "cmu": "cmu_length",
    "electrical_box": "electrical_box_single_width",
    "door": "door_rough_width",
    "hardhat": "hardhat_width",
    "brick": "brick_length",
}

CACHE_DIR = Path("data/cache")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(name: str) -> logging.Logger:
    """Return a configured logger for a pipeline stage."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def load_image(path: str) -> np.ndarray:
    """Load image from disk, return as RGB uint8 array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: str) -> None:
    """Save RGB numpy array to disk as PNG/JPG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def image_to_base64(image_path: str) -> str:
    """Encode image file as base64 string for API calls."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def save_json(data: dict, path: str) -> None:
    """Write dict to JSON file, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    """Load JSON file and return as dict."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def get_image_id(image_path: str) -> str:
    """Return stem of image filename, e.g. 'frame_001' from 'data/frames/frame_001.jpg'."""
    return Path(image_path).stem


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def cached_api_call(
    prompt: str,
    call_fn,
    image_b64: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """
    Call an API function with disk caching.
    call_fn is called with no args and should return the response string.
    Never recomputes if cache hit exists.
    """
    cache_path_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    cache_path_dir.mkdir(parents=True, exist_ok=True)

    cache_key = hashlib.md5(f"{prompt}{image_b64 or ''}".encode()).hexdigest()
    cache_file = cache_path_dir / f"{cache_key}.json"

    if cache_file.exists():
        return load_json(str(cache_file))["response"]

    response = call_fn()
    save_json({"prompt_hash": cache_key, "response": response}, str(cache_file))
    return response


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def draw_boxes(
    img: np.ndarray,
    boxes: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes on image copy.
    Each box dict must have keys: box [x1,y1,x2,y2] in pixels, and optionally 'label'.
    """
    out = img.copy()
    h, w = out.shape[:2]
    for item in boxes:
        box = item.get("box_pixels", item.get("box", []))
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = item.get("label", item.get("type", ""))
        conf = item.get("confidence", None)
        text = f"{label} {conf:.2f}" if conf is not None else label
        if text:
            cv2.putText(
                out, text, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
    return out


def draw_measurement_lines(
    img: np.ndarray,
    measurements: list[dict],
    color: tuple[int, int, int] = (255, 165, 0),
) -> np.ndarray:
    """
    Draw horizontal measurement lines between element centers.
    Each measurement dict: {cx_a, cy_a, cx_b, cy_b, inches, compliant}.
    """
    out = img.copy()
    for m in measurements:
        cx_a = int(m.get("cx_a", 0))
        cy_a = int(m.get("cy_a", 0))
        cx_b = int(m.get("cx_b", 0))
        cy_b = int(m.get("cy_b", 0))
        compliant = m.get("compliant", True)
        line_color = (0, 200, 0) if compliant else (220, 50, 50)
        cv2.line(out, (cx_a, cy_a), (cx_b, cy_b), line_color, 2)
        mid_x = (cx_a + cx_b) // 2
        mid_y = (cy_a + cy_b) // 2
        label = f"{m.get('inches', 0):.1f}\""
        cv2.putText(
            out, label, (mid_x - 20, mid_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2, cv2.LINE_AA,
        )
    return out
