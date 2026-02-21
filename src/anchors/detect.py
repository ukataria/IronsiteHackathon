"""Anchor detection using Grounding DINO + optional SAM segmentation."""

from __future__ import annotations

import os
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
# Text prompts — describes each anchor type in GroundingDINO-friendly syntax
# (period-separated phrases, no commas)
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

# Combined prompt for single-pass detection of all anchors
ALL_ANCHORS_PROMPT = " . ".join(ANCHOR_TEXT_PROMPTS.values())

# ---------------------------------------------------------------------------
# Model loading (cached globally to avoid reloading between calls)
# ---------------------------------------------------------------------------

_gdino_model: Any | None = None
_gdino_processor: Any | None = None


def load_grounding_dino(device: str = "cpu") -> tuple[Any, Any]:
    """Load GroundingDINO model + processor. Cached after first load."""
    global _gdino_model, _gdino_processor

    if _gdino_model is not None:
        return _gdino_model, _gdino_processor

    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = "IDEA-Research/grounding-dino-base"
        logger.info(f"Loading GroundingDINO from {model_id}...")
        _gdino_processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        _gdino_model = _gdino_model.to(device)
        _gdino_model.eval()
        logger.info("GroundingDINO loaded.")
    except Exception as e:
        logger.error(f"Failed to load GroundingDINO: {e}")
        raise

    return _gdino_model, _gdino_processor


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def _classify_label(label: str) -> str:
    """Map a GroundingDINO output label string to an anchor type key."""
    label_lower = label.lower()
    if any(k in label_lower for k in ["stud", "2x4", "lumber", "framing", "board"]):
        return "stud"
    if any(k in label_lower for k in ["rebar", "steel bar", "reinforcement", "rod"]):
        return "rebar"
    if any(k in label_lower for k in ["cmu", "cinder", "concrete block", "masonry unit"]):
        return "cmu"
    if any(k in label_lower for k in ["electrical", "outlet", "junction", "switch box"]):
        return "electrical_box"
    if any(k in label_lower for k in ["door", "opening", "jamb"]):
        return "door"
    if any(k in label_lower for k in ["hard hat", "hardhat", "helmet"]):
        return "hardhat"
    if any(k in label_lower for k in ["brick"]):
        return "brick"
    return "unknown"


def extract_anchor_pixel_dimensions(
    box_pixels: list[float],
    anchor_type: str,
) -> tuple[float, float]:
    """
    Return (pixel_width, pixel_height) of a bounding box.
    box_pixels: [x1, y1, x2, y2] in absolute pixel coords.
    """
    x1, y1, x2, y2 = box_pixels
    return abs(x2 - x1), abs(y2 - y1)


def detect_anchors(
    image_path: str,
    output_dir: str,
    anchor_types: list[str] | None = None,
    box_threshold: float = 0.15,
    text_threshold: float = 0.25,
    device: str = "cpu",
) -> dict:
    """
    Detect known-dimension objects in a construction image using GroundingDINO.

    Saves:
      {output_dir}/{image_id}_anchors.json
      {output_dir}/{image_id}_annotated.png

    Returns the detections dict (same as the saved JSON).
    """
    from PIL import Image as PILImage

    image_id = get_image_id(image_path)
    out_json = str(Path(output_dir) / f"{image_id}_anchors.json")
    out_img = str(Path(output_dir) / f"{image_id}_annotated.png")

    # Build text prompt — single-pass for all types or filtered subset
    if anchor_types:
        prompt_parts = [ANCHOR_TEXT_PROMPTS[t] for t in anchor_types if t in ANCHOR_TEXT_PROMPTS]
        text_prompt = " . ".join(prompt_parts) + " ."
    else:
        text_prompt = ALL_ANCHORS_PROMPT + " ."

    img_rgb = load_image(image_path)
    h, w = img_rgb.shape[:2]
    pil_img = PILImage.fromarray(img_rgb)

    try:
        model, processor = load_grounding_dino(device)
        inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Try new API first (no threshold args), fall back to old API
        try:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[(h, w)],
            )[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(h, w)],
            )[0]

        boxes_xyxy = results["boxes"].cpu().numpy()                        # [N, 4] absolute pixels
        scores = results["scores"].cpu().numpy()                         # [N]
        labels = results.get("text_labels", results.get("labels", []))  # str labels, future-proof

        # Log score distribution to help tune threshold
        if len(scores) > 0:
            logger.info(f"  Raw detections: {len(scores)} boxes, scores: max={scores.max():.3f} mean={scores.mean():.3f} min={scores.min():.3f}")
        else:
            logger.info("  Raw detections: 0 boxes returned by model")

        # Filter by score threshold
        keep = scores >= box_threshold
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        labels = [l for l, k in zip(labels, keep) if k]

        # Filter by bounding box area — reject whole-scene and tiny noise detections
        image_area = h * w
        area_keep = []
        for box in boxes_xyxy:
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            frac = box_area / image_area
            area_keep.append(0.005 <= frac <= 0.40)
        area_keep_arr = np.array(area_keep)
        boxes_xyxy = boxes_xyxy[area_keep_arr]
        scores = scores[area_keep_arr]
        labels = [l for l, k in zip(labels, area_keep) if k]
        logger.info(f"  After area filter: {len(scores)} boxes remain")

        # Aspect ratio sanity check per anchor type
        # Studs are tall+narrow (height >> width), CMU blocks are ~2:1 wide, etc.
        MAX_ASPECT_RATIOS: dict[str, tuple[float, float]] = {
            "stud": (0.02, 0.8),    # width/height: full-height stud can be ~0.03
            "cmu": (1.0, 6.0),      # roughly square to 2x wide
            "brick": (1.5, 5.0),
            "electrical_box": (0.3, 3.0),
            "door": (0.1, 2.0),
        }
        ratio_keep = []
        ratio_labels_classified = [_classify_label(str(l)) for l in labels]
        for box, atype in zip(boxes_xyxy, ratio_labels_classified):
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            ratio = bw / bh if bh > 0 else 999
            lo, hi = MAX_ASPECT_RATIOS.get(atype, (0.01, 100.0))
            ratio_keep.append(lo <= ratio <= hi)
        ratio_keep_arr = np.array(ratio_keep)
        boxes_xyxy = boxes_xyxy[ratio_keep_arr]
        scores = scores[ratio_keep_arr]
        labels = [l for l, k in zip(labels, ratio_keep) if k]
        logger.info(f"  After aspect ratio filter: {len(scores)} boxes remain")

    except Exception as e:
        logger.error(f"Detection failed for {image_id}: {e}")
        boxes_xyxy, scores, labels = np.zeros((0, 4)), np.zeros(0), []

    # Build structured anchor list
    anchors = []
    for i, (box, score, label) in enumerate(zip(boxes_xyxy, scores, labels)):
        anchor_type = _classify_label(str(label))
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
        "anchors": anchors,
    }

    # Save JSON
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_json(result, out_json)

    # Save annotated image
    box_dicts = [
        {"box_pixels": a["box_pixels"], "label": a["type"], "confidence": a["confidence"]}
        for a in anchors
    ]
    annotated = draw_boxes(img_rgb, box_dicts)
    save_image(annotated, out_img)

    logger.info(f"{image_id}: detected {len(anchors)} anchors → {out_json}")
    return result


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python src/anchors/detect.py <image_path> [output_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/detections"
    result = detect_anchors(img_path, out_dir)
    print(f"Found {result['n_anchors']} anchors:")
    for a in result["anchors"]:
        print(f"  [{a['type']}] conf={a['confidence']:.2f} width={a['pixel_width']:.1f}px → {a['real_width_inches']}in")
