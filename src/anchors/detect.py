"""Anchor detection — YOLOv8 (fine-tuned) primary, GroundingDINO fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils import (
    ANCHOR_DIMENSIONS,
    ANCHOR_PRIMARY_DIMENSION,
    draw_boxes,
    get_image_id,
    load_image,
    save_image,
    save_json,
    setup_logger,
)

logger = setup_logger("anchors")

# ---------------------------------------------------------------------------
# YOLO config
# ---------------------------------------------------------------------------

# Fine-tuned weights path (relative to project root).
# Falls back to a vanilla YOLOv8n if the fine-tuned file is absent.
YOLO_MODEL_PATH = "yolo_weights/finetune_1_weights/best.pt"
YOLO_BASE_MODEL = "yolov8n.pt"

# Map every class name the fine-tuned model can produce → anchor type key.
# Model classes: {0: 'brick', 1: 'outlet', 2: 'block'}
YOLO_CLASS_MAP: dict[str, str] = {
    # brick
    "brick": "brick",
    "bricks-masonry": "brick",
    # outlet / electrical box
    "outlet": "electrical_box",
    "electrical-outlet": "electrical_box",
    "electrical_outlet": "electrical_box",
    # cmu / block
    "block": "cmu",
    "1": "cmu",          # Roboflow auto-label artifact
    "cmu": "cmu",
    "cinder": "cmu",
    "concrete block": "cmu",
}

# ---------------------------------------------------------------------------
# GroundingDINO config (fallback)
# ---------------------------------------------------------------------------

ANCHOR_TEXT_PROMPTS: dict[str, str] = {
    "stud": "wood stud . vertical wood board . wall stud",
    "rebar": "rebar . steel bar",
    "cmu": "cinder block . concrete block",
    "electrical_box": "electrical box . outlet box",
    "door": "door frame . door opening",
    "hardhat": "hard hat . helmet",
    "brick": "brick",
}
ALL_ANCHORS_PROMPT = " . ".join(ANCHOR_TEXT_PROMPTS.values())

# ---------------------------------------------------------------------------
# Model caches
# ---------------------------------------------------------------------------

_yolo_model: Any | None = None
_gdino_model: Any | None = None
_gdino_processor: Any | None = None


def load_yolo(device: str = "cpu") -> Any:
    """Load YOLO model. Uses fine-tuned weights if available, else base model."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    from ultralytics import YOLO

    weights = YOLO_MODEL_PATH if Path(YOLO_MODEL_PATH).exists() else YOLO_BASE_MODEL
    logger.info(f"Loading YOLO from {weights}...")
    _yolo_model = YOLO(weights)
    logger.info(f"YOLO loaded ({weights}).")
    return _yolo_model


def load_grounding_dino(device: str = "cpu") -> tuple[Any, Any]:
    """Load GroundingDINO model + processor. Cached after first load."""
    global _gdino_model, _gdino_processor
    if _gdino_model is not None:
        return _gdino_model, _gdino_processor

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-base"
    logger.info(f"Loading GroundingDINO from {model_id}...")
    _gdino_processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    _gdino_model = _gdino_model.to(device)
    _gdino_model.eval()
    logger.info("GroundingDINO loaded.")
    return _gdino_model, _gdino_processor


# ---------------------------------------------------------------------------
# Label classification helpers
# ---------------------------------------------------------------------------


def _yolo_label_to_anchor(label: str) -> str:
    """Map a YOLO class name string to an anchor type key."""
    return YOLO_CLASS_MAP.get(label.lower().strip(), "unknown")


def _gdino_label_to_anchor(label: str) -> str:
    """Map a GroundingDINO text label to an anchor type key."""
    l = label.lower()
    if any(k in l for k in ["stud", "2x4", "lumber", "framing", "board"]):
        return "stud"
    if any(k in l for k in ["rebar", "steel bar", "reinforcement", "rod"]):
        return "rebar"
    if any(k in l for k in ["cmu", "cinder", "concrete block", "masonry unit"]):
        return "cmu"
    if any(k in l for k in ["electrical", "outlet", "junction", "switch box"]):
        return "electrical_box"
    if any(k in l for k in ["door", "opening", "jamb"]):
        return "door"
    if any(k in l for k in ["hard hat", "hardhat", "helmet"]):
        return "hardhat"
    if "brick" in l:
        return "brick"
    return "unknown"


# ---------------------------------------------------------------------------
# Per-detector raw detection functions
# ---------------------------------------------------------------------------


def _detect_with_yolo(
    image_path: str,
    conf_threshold: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run YOLO inference. Returns (boxes_xyxy, scores, label_strings)."""
    model = load_yolo(device)
    results = model.predict(image_path, conf=conf_threshold, verbose=False, device=device)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), []

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    labels = [model.names[c] for c in class_ids]

    logger.info(
        f"  YOLO raw: {len(scores)} boxes, "
        f"max={scores.max():.3f} mean={scores.mean():.3f} min={scores.min():.3f}"
    )
    return boxes, scores, labels


def _detect_with_gdino(
    image_path: str,
    anchor_types: list[str] | None,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run GroundingDINO inference. Returns (boxes_xyxy, scores, label_strings)."""
    from PIL import Image as PILImage

    img_rgb = load_image(image_path)
    h, w = img_rgb.shape[:2]
    pil_img = PILImage.fromarray(img_rgb)

    if anchor_types:
        parts = [ANCHOR_TEXT_PROMPTS[t] for t in anchor_types if t in ANCHOR_TEXT_PROMPTS]
        text_prompt = " . ".join(parts) + " ."
    else:
        text_prompt = ALL_ANCHORS_PROMPT + " ."

    model, processor = load_grounding_dino(device)
    inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    try:
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, target_sizes=[(h, w)]
        )[0]
    except TypeError:
        res = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )[0]

    boxes = res["boxes"].cpu().numpy()
    scores = res["scores"].cpu().numpy()
    labels = list(res.get("text_labels", res.get("labels", [])))

    keep = scores >= box_threshold
    boxes, scores = boxes[keep], scores[keep]
    labels = [l for l, k in zip(labels, keep) if k]

    if len(scores):
        logger.info(
            f"  GDino raw: {len(scores)} boxes, "
            f"max={scores.max():.3f} mean={scores.mean():.3f} min={scores.min():.3f}"
        )
    else:
        logger.info("  GDino raw: 0 boxes")
    return boxes, scores, labels


# ---------------------------------------------------------------------------
# Shared post-processing filters
# ---------------------------------------------------------------------------

MAX_ASPECT_RATIOS: dict[str, tuple[float, float]] = {
    "stud": (0.02, 0.8),
    "cmu": (1.0, 6.0),
    "brick": (1.5, 5.0),
    "electrical_box": (0.3, 3.0),
    "door": (0.1, 2.0),
}


def _apply_filters(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    image_h: int,
    image_w: int,
    classify_fn: Any,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Area + aspect-ratio filters shared by both detectors."""
    if len(scores) == 0:
        return boxes, scores, labels

    # Area filter: reject full-scene and sub-pixel noise
    image_area = image_h * image_w
    area_mask = np.array([
        0.005 <= ((b[2] - b[0]) * (b[3] - b[1])) / image_area <= 0.40
        for b in boxes
    ])
    boxes, scores = boxes[area_mask], scores[area_mask]
    labels = [l for l, k in zip(labels, area_mask) if k]
    logger.info(f"  After area filter: {len(scores)} boxes remain")

    # Aspect ratio filter
    classified = [classify_fn(str(l)) for l in labels]
    ratio_mask = []
    for box, atype in zip(boxes, classified):
        bw, bh = box[2] - box[0], box[3] - box[1]
        ratio = bw / bh if bh > 0 else 999
        lo, hi = MAX_ASPECT_RATIOS.get(atype, (0.01, 100.0))
        ratio_mask.append(lo <= ratio <= hi)
    ratio_arr = np.array(ratio_mask)
    boxes, scores = boxes[ratio_arr], scores[ratio_arr]
    labels = [l for l, k in zip(labels, ratio_mask) if k]
    logger.info(f"  After aspect ratio filter: {len(scores)} boxes remain")

    return boxes, scores, labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_anchor_pixel_dimensions(
    box_pixels: list[float],
    anchor_type: str,
) -> tuple[float, float]:
    """Return (pixel_width, pixel_height) for a bounding box."""
    x1, y1, x2, y2 = box_pixels
    return abs(x2 - x1), abs(y2 - y1)


def detect_anchors(
    image_path: str,
    output_dir: str,
    anchor_types: list[str] | None = None,
    box_threshold: float = 0.15,
    text_threshold: float = 0.25,
    device: str = "cpu",
    use_yolo: bool = True,
) -> dict:
    """
    Detect known-dimension objects in a construction image.

    Primary detector: fine-tuned YOLOv8 (use_yolo=True, default).
    Fallback: GroundingDINO (use_yolo=False or on YOLO failure).

    Saves:
      {output_dir}/{image_id}_anchors.json
      {output_dir}/{image_id}_annotated.png

    Returns the detections dict (same as the saved JSON).
    """
    image_id = get_image_id(image_path)
    out_json = str(Path(output_dir) / f"{image_id}_anchors.json")
    out_img = str(Path(output_dir) / f"{image_id}_annotated.png")

    img_rgb = load_image(image_path)
    h, w = img_rgb.shape[:2]

    detector_used = "yolo" if use_yolo else "gdino"
    boxes: np.ndarray = np.zeros((0, 4))
    scores: np.ndarray = np.zeros(0)
    labels: list[str] = []
    classify_fn = _yolo_label_to_anchor

    try:
        if use_yolo:
            boxes, scores, labels = _detect_with_yolo(image_path, box_threshold, device)
            classify_fn = _yolo_label_to_anchor
        else:
            boxes, scores, labels = _detect_with_gdino(
                image_path, anchor_types, box_threshold, text_threshold, device
            )
            classify_fn = _gdino_label_to_anchor

        boxes, scores, labels = _apply_filters(boxes, scores, labels, h, w, classify_fn)

    except Exception as e:
        logger.error(f"Detection failed for {image_id} [{detector_used}]: {e}")
        if use_yolo:
            # Try GDino as fallback
            logger.info("Falling back to GroundingDINO...")
            detector_used = "gdino"
            classify_fn = _gdino_label_to_anchor
            try:
                boxes, scores, labels = _detect_with_gdino(
                    image_path, anchor_types, box_threshold, text_threshold, device
                )
                boxes, scores, labels = _apply_filters(boxes, scores, labels, h, w, classify_fn)
            except Exception as e2:
                logger.error(f"GDino fallback also failed: {e2}")
                boxes, scores, labels = np.zeros((0, 4)), np.zeros(0), []

    # Build structured anchor list
    anchors = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        anchor_type = classify_fn(str(label))
        if anchor_types and anchor_type not in anchor_types:
            continue
        if anchor_type == "unknown":
            continue

        x1, y1, x2, y2 = box.tolist()
        px_w, px_h = extract_anchor_pixel_dimensions([x1, y1, x2, y2], anchor_type)
        dim_key = ANCHOR_PRIMARY_DIMENSION.get(anchor_type, "stud_face_width")
        real_width = ANCHOR_DIMENSIONS[dim_key]

        anchors.append({
            "id": i,
            "type": anchor_type,
            "label_raw": str(label),
            "box_pixels": [x1, y1, x2, y2],
            "confidence": float(score),
            "pixel_width": px_w,
            "pixel_height": px_h,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
            "real_width_inches": real_width,
        })

    result = {
        "image_id": image_id,
        "image_width": w,
        "image_height": h,
        "n_anchors": len(anchors),
        "detector": detector_used,
        "anchors": anchors,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_json(result, out_json)

    box_dicts = [
        {"box_pixels": a["box_pixels"], "label": a["type"], "confidence": a["confidence"]}
        for a in anchors
    ]
    annotated = draw_boxes(img_rgb, box_dicts)
    save_image(annotated, out_img)

    logger.info(f"{image_id}: {len(anchors)} anchors [{detector_used}] → {out_json}")
    return result


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python src/anchors/detect.py <image_path> [output_dir] [--gdino]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "data/detections"
    force_gdino = "--gdino" in sys.argv

    result = detect_anchors(img_path, out_dir, use_yolo=not force_gdino)
    print(f"Found {result['n_anchors']} anchors (detector={result['detector']}):")
    for a in result["anchors"]:
        print(f"  [{a['type']}] conf={a['confidence']:.2f} width={a['pixel_width']:.1f}px → {a['real_width_inches']}in")
