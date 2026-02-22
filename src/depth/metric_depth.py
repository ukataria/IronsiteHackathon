"""Metric depth estimation using Depth Anything V2 Metric — outputs real distances in metres."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.cm as cm
import numpy as np
import torch

from src.utils import get_image_id, load_image, load_json, save_image, save_json, setup_logger

logger = setup_logger("metric_depth")

# Metric models output depth in metres (max range ~20 m indoors, ~80 m outdoors).
# Use Indoor for construction interiors; Outdoor if shooting exteriors.
METRIC_MODEL_IDS: dict[str, str] = {
    "small":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base":   "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "outdoor-large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}

_model: Any | None = None
_processor: Any | None = None
_loaded_size: str | None = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_metric_model(model_size: str = "small", device: str = "cpu") -> tuple[Any, Any]:
    """Load Depth Anything V2 Metric model. Cached after first call."""
    global _model, _processor, _loaded_size

    if _model is not None and _loaded_size == model_size:
        return _model, _processor

    if model_size not in METRIC_MODEL_IDS:
        raise ValueError(f"model_size must be one of {list(METRIC_MODEL_IDS)}. Got: {model_size}")

    model_id = METRIC_MODEL_IDS[model_size]
    logger.info(f"Loading metric depth model ({model_size}) from {model_id}...")

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    _processor = AutoImageProcessor.from_pretrained(model_id)
    _model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    _model.eval()
    _loaded_size = model_size
    logger.info("Metric depth model loaded.")

    return _model, _processor


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------


def estimate_metric_depth(
    image_path: str,
    output_dir: str,
    model_size: str = "small",
    device: str = "cpu",
) -> np.ndarray:
    """
    Run Depth Anything V2 Metric on an image.

    Returns depth map in metres as float32 array (H, W).
    Saves:
      {output_dir}/{image_id}_metric_depth.npy  — depth in metres
      {output_dir}/{image_id}_metric_depth.png  — colorized visualization
    """
    from PIL import Image as PILImage

    image_id = get_image_id(image_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_npy = str(Path(output_dir) / f"{image_id}_metric_depth.npy")
    out_png = str(Path(output_dir) / f"{image_id}_metric_depth.png")

    img_rgb = load_image(image_path)
    h, w = img_rgb.shape[:2]
    pil_img = PILImage.fromarray(img_rgb)

    model, processor = load_metric_model(model_size, device)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        depth_raw = outputs.predicted_depth  # (1, H', W') in metres

    depth_m = depth_raw.squeeze().cpu().numpy().astype(np.float32)
    depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_LINEAR)

    np.save(out_npy, depth_m)

    # Colorize: clamp to 95th percentile so outliers don't wash out the palette
    vmax = float(np.percentile(depth_m, 95))
    vmin = float(depth_m.min())
    depth_norm = np.clip((depth_m - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    colorized = (cm.plasma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    save_image(colorized, out_png)

    logger.info(f"{image_id}: metric depth saved → {out_npy}")
    logger.info(f"  Depth range: {depth_m.min():.2f} m – {depth_m.max():.2f} m  (median {np.median(depth_m):.2f} m)")
    return depth_m


# ---------------------------------------------------------------------------
# Per-object distance queries
# ---------------------------------------------------------------------------


def depth_at_box(depth_m: np.ndarray, box: list[float]) -> float:
    """Return median depth in metres within a bounding box [x1, y1, x2, y2]."""
    h, w = depth_m.shape
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    region = depth_m[y1:y2, x1:x2]
    return float(np.median(region)) if region.size > 0 else 0.0


def annotate_anchors_with_depth(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    anchors: list[dict],
) -> np.ndarray:
    """Draw bounding boxes + distance labels (metres and feet) on image copy."""
    out = image_rgb.copy()
    for anchor in anchors:
        x1, y1, x2, y2 = [int(v) for v in anchor["box_pixels"]]
        dist_m = depth_at_box(depth_m, anchor["box_pixels"])
        dist_ft = dist_m * 3.28084

        label = f"{anchor['type']}  {dist_m:.1f}m / {dist_ft:.1f}ft"
        color = (50, 220, 50)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Background pill for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(y1 - th - 8, 0)), (x1 + tw + 4, max(y1, th + 8)), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 2, max(y1 - 4, th + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python src/depth/metric_depth.py <image_path> [--model small|base|large] [--anchors data/detections/{id}_anchors.json] [--device cpu|cuda] [output_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    model_size = "small"
    device_choice = "cpu"
    anchors_json = None
    out_dir = "data/depth"

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--model" and i + 1 < len(sys.argv):
            model_size = sys.argv[i + 1]; i += 2
        elif arg == "--anchors" and i + 1 < len(sys.argv):
            anchors_json = sys.argv[i + 1]; i += 2
        elif arg == "--device" and i + 1 < len(sys.argv):
            device_choice = sys.argv[i + 1]; i += 2
        elif not arg.startswith("--"):
            out_dir = arg; i += 1
        else:
            i += 1

    # Run metric depth
    depth_m = estimate_metric_depth(img_path, out_dir, model_size=model_size, device=device_choice)
    image_id = get_image_id(img_path)

    print(f"\nDepth range: {depth_m.min():.2f} m – {depth_m.max():.2f} m")
    print(f"Median scene depth: {np.median(depth_m):.2f} m / {np.median(depth_m) * 3.28084:.2f} ft")

    # If anchors JSON provided, annotate each object with its distance
    if anchors_json and Path(anchors_json).exists():
        data = load_json(anchors_json)
        anchors = data.get("anchors", [])

        print(f"\nDistances to {len(anchors)} detected anchors:")
        for a in anchors:
            d = depth_at_box(depth_m, a["box_pixels"])
            print(f"  [{a['type']}] conf={a['confidence']:.2f}  →  {d:.2f} m / {d * 3.28084:.2f} ft")

        # Save annotated image
        from src.utils import load_image, save_image
        img_rgb = load_image(img_path)
        annotated = annotate_anchors_with_depth(img_rgb, depth_m, anchors)
        ann_path = str(Path(out_dir) / f"{image_id}_metric_annotated.png")
        save_image(annotated, ann_path)
        print(f"\nAnnotated image: {ann_path}")

        # Save distances JSON
        distances = [
            {**a, "distance_m": round(depth_at_box(depth_m, a["box_pixels"]), 3),
             "distance_ft": round(depth_at_box(depth_m, a["box_pixels"]) * 3.28084, 3)}
            for a in anchors
        ]
        out_json = str(Path(out_dir) / f"{image_id}_distances.json")
        save_json({"image_id": image_id, "anchors_with_distances": distances}, out_json)
        print(f"Distances JSON: {out_json}")
    else:
        print(f"\nDepth map: {out_dir}/{image_id}_metric_depth.npy")
        print(f"Visualization: {out_dir}/{image_id}_metric_depth.png")
        print(f"\nTip: pass --anchors data/detections/{image_id}_anchors.json to get per-object distances.")
