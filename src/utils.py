"""Shared helpers: file I/O, image ops, logging."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FRAMES_DIR = DATA / "frames"
DEPTH_DIR = DATA / "depth"
SEGMENTS_DIR = DATA / "segments"
SCENES_DIR = DATA / "scenes"
OVERLAYS_DIR = DATA / "overlays"
COMPOSITES_DIR = DATA / "composites"
CACHE_DIR = DATA / "cache"


def ensure_dirs() -> None:
    """Create all data dirs if they don't exist."""
    for d in [
        FRAMES_DIR, DEPTH_DIR, SEGMENTS_DIR, SCENES_DIR,
        OVERLAYS_DIR, COMPOSITES_DIR, CACHE_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def load_image_pil(path: str | Path) -> Image.Image:
    """Load image as PIL RGB."""
    return Image.open(str(path)).convert("RGB")


def save_image(img: np.ndarray, path: str | Path) -> None:
    """Save BGR numpy array to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to BGR numpy array."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def image_to_base64(path: str | Path) -> str:
    """Encode image file as base64 string."""
    import base64
    with open(str(path), "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# Depth I/O
# ---------------------------------------------------------------------------

def save_depth(depth: np.ndarray, stem: str) -> tuple[Path, Path]:
    """Save depth as .npy and a normalized visualization .png."""
    npy_path = DEPTH_DIR / f"{stem}.npy"
    png_path = DEPTH_DIR / f"{stem}.png"
    np.save(str(npy_path), depth)
    depth_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(png_path), depth_colored)
    return npy_path, png_path


def load_depth(stem: str) -> np.ndarray:
    """Load depth .npy file by stem name."""
    return np.load(str(DEPTH_DIR / f"{stem}.npy"))


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(str(path)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def prompt_hash(prompt: str, extra: str = "") -> str:
    """Hash a prompt string for cache keying."""
    return hashlib.md5((prompt + extra).encode()).hexdigest()[:16]


def cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def load_cache(key: str) -> dict | None:
    p = cache_path(key)
    if p.exists():
        return load_json(p)
    return None


def save_cache(key: str, data: dict) -> None:
    save_json(data, cache_path(key))


# ---------------------------------------------------------------------------
# Frame naming
# ---------------------------------------------------------------------------

def frame_stem(clip_id: str, frame_num: int) -> str:
    """Canonical frame stem: {clip_id}_frame_{XXXX}."""
    return f"{clip_id}_frame_{frame_num:04d}"
