"""Depth estimation using Depth Anything V2.

Two modes:
  - Relative (default): HuggingFace transformers, normalized [0, 1].
  - Metric: Native depth_anything_v2 package, actual meters (outdoor/vkitti, max 80 m).

Metric mode setup:
  git clone https://github.com/DepthAnything/Depth-Anything-V2
  # either `pip install -e .` inside that repo, or add its root to PYTHONPATH
  # Download checkpoint to checkpoints/ dir:
  #   depth_anything_v2_metric_vkitti_{vits|vitb|vitl}.pth
  #   https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import numpy as np
import torch

from src.utils import get_image_id, load_image, save_image, setup_logger

logger = setup_logger("depth")

# ---------------------------------------------------------------------------
# Relative depth — HuggingFace transformers
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
    """Load relative Depth Anything V2 via HF transformers. Cached per size."""
    global _depth_model, _depth_processor, _loaded_model_size

    if _depth_model is not None and _loaded_model_size == model_size:
        return _depth_model, _depth_processor

    if model_size not in DEPTH_MODEL_IDS:
        raise ValueError(f"model_size must be one of {list(DEPTH_MODEL_IDS)}. Got: {model_size}")

    model_id = DEPTH_MODEL_IDS[model_size]
    logger.info(f"Loading Depth Anything V2 ({model_size}) from {model_id}...")

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    _depth_processor = AutoImageProcessor.from_pretrained(model_id)
    _depth_model = AutoModelForDepthEstimation.from_pretrained(model_id)
    _depth_model = _depth_model.to(device)
    _depth_model.eval()
    _loaded_model_size = model_size
    logger.info("Depth Anything V2 (relative) loaded.")
    return _depth_model, _depth_processor


# ---------------------------------------------------------------------------
# Metric depth — native depth_anything_v2 repo, outdoor (vkitti)
# ---------------------------------------------------------------------------

METRIC_OUTDOOR_MAX_DEPTH: float = 80.0  # meters
METRIC_OUTDOOR_DATASET: str = "vkitti"

# Encoder configs match the native repo's model_configs dict
_METRIC_ENCODER_CONFIGS: dict[str, dict] = {
    "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "base":  {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

_metric_model: Any | None = None
_loaded_metric_size: str | None = None


def load_metric_depth_model(
    model_size: str = "large",
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
) -> Any:
    """Load metric Depth Anything V2 (outdoor). Cached per size.

    Requires the depth_anything_v2 package to be importable.
    Checkpoint filename: depth_anything_v2_metric_vkitti_{vits|vitb|vitl}.pth
    """
    global _metric_model, _loaded_metric_size

    if _metric_model is not None and _loaded_metric_size == model_size:
        return _metric_model

    if model_size not in _METRIC_ENCODER_CONFIGS:
        raise ValueError(f"model_size must be one of {list(_METRIC_ENCODER_CONFIGS)}. Got: {model_size}")

    # Add the cloned repo's metric_depth dir to sys.path so depth_anything_v2 is importable
    import sys
    _repo_metric = Path(__file__).parent.parent.parent / "_depth_anything_v2_repo" / "metric_depth"
    if _repo_metric.exists() and str(_repo_metric) not in sys.path:
        sys.path.insert(0, str(_repo_metric))

    try:
        from depth_anything_v2.dpt import DepthAnythingV2 as _DAV2
    except ImportError as exc:
        raise ImportError(
            f"depth_anything_v2 package not found. Expected repo at: {_repo_metric}\n"
            "  git clone https://github.com/DepthAnything/Depth-Anything-V2 _depth_anything_v2_repo"
        ) from exc

    cfg = _METRIC_ENCODER_CONFIGS[model_size]
    encoder_tag = cfg["encoder"]  # vits / vitb / vitl
    ckpt_name = f"depth_anything_v2_metric_{METRIC_OUTDOOR_DATASET}_{encoder_tag}.pth"
    ckpt_path = Path(checkpoint_dir) / ckpt_name

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large"
        )

    logger.info(f"Loading metric Depth Anything V2 ({model_size}, outdoor) from {ckpt_path}...")
    model = _DAV2(**{**cfg, "max_depth": METRIC_OUTDOOR_MAX_DEPTH})
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    _metric_model = model
    _loaded_metric_size = model_size
    logger.info("Metric Depth Anything V2 (outdoor) loaded.")
    return _metric_model


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalize_depth(raw: np.ndarray) -> np.ndarray:
    """Normalize raw depth to [0, 1] float32."""
    dmin, dmax = raw.min(), raw.max()
    if dmax - dmin < 1e-8:
        return np.zeros_like(raw, dtype=np.float32)
    return ((raw - dmin) / (dmax - dmin)).astype(np.float32)


def _colorize_depth(depth: np.ndarray, max_depth: float | None = None) -> np.ndarray:
    """Convert depth array to jet-colorized RGB uint8.

    For metric depth pass max_depth (e.g. 80.0) to clamp before colorizing.
    For relative depth leave max_depth=None (array is already [0, 1]).
    """
    if max_depth is not None:
        normed = (np.clip(depth, 0.0, max_depth) / max_depth).astype(np.float32)
    else:
        normed = depth.astype(np.float32)
    colored = cm.jet(normed)[:, :, :3]
    return (colored * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Depth estimation — unified entry point
# ---------------------------------------------------------------------------


def estimate_depth(
    image_path: str,
    output_dir: str,
    model_size: str = "large",
    device: str = "cpu",
    metric: bool = True,
    checkpoint_dir: str = "checkpoints",
) -> np.ndarray:
    """Run Depth Anything V2 on an image.

    Args:
        metric: If True, use outdoor metric model (returns meters, float32).
                If False, use relative model (returns normalized [0, 1]).
        checkpoint_dir: Directory containing metric model .pth files.

    Returns:
        depth map (H, W) float32 — meters if metric=True, [0,1] if metric=False.
    Saves:
        {output_dir}/{image_id}_depth.npy   — depth array
        {output_dir}/{image_id}_depth.png   — jet-colorized visualization
    """
    image_id = get_image_id(image_path)
    out_npy = str(Path(output_dir) / f"{image_id}_depth.npy")
    out_png = str(Path(output_dir) / f"{image_id}_depth.png")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    img_rgb = load_image(image_path)  # H×W×3, uint8, RGB

    try:
        if metric:
            depth = _run_metric(img_rgb, model_size, device, checkpoint_dir)
        else:
            depth = _run_relative(img_rgb, model_size, device)
    except Exception as e:
        logger.error(f"Depth estimation failed for {image_id}: {e}")
        h, w = img_rgb.shape[:2]
        depth = np.zeros((h, w), dtype=np.float32)

    np.save(out_npy, depth)
    max_d = METRIC_OUTDOOR_MAX_DEPTH if metric else None
    colorized = _colorize_depth(depth, max_depth=max_d)
    save_image(colorized, out_png)
    mode_str = "metric (outdoor)" if metric else "relative"
    logger.info(f"{image_id}: {mode_str} depth saved → {out_npy}, range [{depth.min():.2f}, {depth.max():.2f}]")
    return depth


def _run_metric(img_rgb: np.ndarray, model_size: str, device: str, checkpoint_dir: str) -> np.ndarray:
    """Run metric depth inference. Returns HxW float32 array in meters."""
    import cv2

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    model = load_metric_depth_model(model_size, checkpoint_dir, device)
    with torch.no_grad():
        depth = model.infer_image(img_bgr)  # HxW, meters, float32
    return depth.astype(np.float32)


def _run_relative(img_rgb: np.ndarray, model_size: str, device: str) -> np.ndarray:
    """Run relative depth inference. Returns HxW float32 array in [0, 1]."""
    import cv2
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(img_rgb)
    model, processor = load_depth_model(model_size, device)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        raw = model(**inputs).predicted_depth.squeeze().cpu().numpy()

    h, w = img_rgb.shape[:2]
    raw_resized = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
    return _normalize_depth(raw_resized)


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
