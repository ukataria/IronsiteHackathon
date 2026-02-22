"""Anchor detection using Microsoft Florence-2 (local, no API key needed)."""

from __future__ import annotations

import sys
from pathlib import Path

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

logger = setup_logger("anchors_florence")

MODEL_ID = "microsoft/Florence-2-base"  # swap to Florence-2-large for better quality

# Period-separated prompt — Florence-2 OPEN_VOCABULARY_DETECTION format
DETECTION_PROMPT = (
    "wood stud . vertical wood board . "
    "rebar . steel reinforcement bar . "
    "cinder block . concrete block . "
    "electrical box . outlet box . "
    "door frame . door opening . "
    "hard hat . construction helmet . "
    "brick"
)

# Map Florence-2 label strings back to our anchor type keys
_LABEL_MAP: list[tuple[list[str], str]] = [
    (["stud", "wood", "board", "framing"], "stud"),
    (["rebar", "steel", "reinforcement"], "rebar"),
    (["cinder", "concrete block", "cmu"], "cmu"),
    (["electrical", "outlet", "junction", "switch"], "electrical_box"),
    (["door"], "door"),
    (["hard hat", "hardhat", "helmet"], "hardhat"),
    (["brick"], "brick"),
]


def _classify_label(label: str) -> str:
    """Map a Florence-2 output label string to an anchor type key."""
    label_lower = label.lower()
    for keywords, anchor_type in _LABEL_MAP:
        if any(kw in label_lower for kw in keywords):
            return anchor_type
    return "unknown"


# ---------------------------------------------------------------------------
# Model loading (cached globally)
# ---------------------------------------------------------------------------

_florence_model = None
_florence_processor = None


def load_florence(device: str = "cpu"):
    """Load Florence-2 model + processor. Cached after first load."""
    global _florence_model, _florence_processor

    if _florence_model is not None:
        return _florence_model, _florence_processor

    from transformers import AutoModelForCausalLM, AutoProcessor

    logger.info(f"Loading Florence-2 from {MODEL_ID} (first run downloads ~1.5GB)...")
    _florence_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    _florence_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    _florence_model.eval()
    logger.info("Florence-2 loaded.")

    return _florence_model, _florence_processor


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def detect_anchors_florence(
    image_path: str,
    output_dir: str,
    device: str = "cpu",
    min_confidence: float = 0.0,  # Florence-2 doesn't produce confidence scores
) -> dict:
    """
    Detect construction anchors using Florence-2 open-vocabulary detection.

    Saves:
      {output_dir}/{image_id}_florence_anchors.json
      {output_dir}/{image_id}_florence_annotated.png

    Returns the detections dict.
    """
    from PIL import Image as PILImage

    image_id = get_image_id(image_path)
    out_json = str(Path(output_dir) / f"{image_id}_florence_anchors.json")
    out_img = str(Path(output_dir) / f"{image_id}_florence_annotated.png")

    img_rgb = load_image(image_path)
    h, w = img_rgb.shape[:2]
    pil_img = PILImage.fromarray(img_rgb)

    model, processor = load_florence(device)

    task = "<OPEN_VOCABULARY_DETECTION>"
    full_prompt = f"{task}{DETECTION_PROMPT}"

    inputs = processor(text=full_prompt, images=pil_img, return_tensors="pt").to(device)

    logger.info(f"Running Florence-2 on {image_id} ({w}x{h})...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    raw_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        raw_text,
        task=task,
        image_size=(w, h),
    )

    detections = parsed.get(task, {})
    boxes = detections.get("bboxes", [])
    labels = detections.get("labels", [])

    logger.info(f"  Raw detections: {len(boxes)} boxes")

    # Build structured anchor list
    anchors = []
    for i, (box, label) in enumerate(zip(boxes, labels)):
        anchor_type = _classify_label(label)
        if anchor_type == "unknown":
            continue

        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        x2, y2 = min(x2, w), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            continue

        dim_key = ANCHOR_PRIMARY_DIMENSION.get(anchor_type, "stud_face_width")
        anchors.append({
            "id": i,
            "type": anchor_type,
            "label_raw": label,
            "box_pixels": [x1, y1, x2, y2],
            "confidence": 1.0,  # Florence-2 doesn't output scores
            "pixel_width": abs(x2 - x1),
            "pixel_height": abs(y2 - y1),
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
            "real_width_inches": ANCHOR_DIMENSIONS[dim_key],
        })

    result = {
        "image_id": image_id,
        "image_width": w,
        "image_height": h,
        "model": MODEL_ID,
        "n_anchors": len(anchors),
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

    logger.info(f"{image_id}: {len(anchors)} anchors → {out_json}")
    logger.info(f"  Annotated image → {out_img}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python src/anchors/detect_florence.py <image_path> [--device cpu|cuda] [output_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    device_choice = "cpu"
    out_dir = "data/detections"

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--device" and i + 1 < len(sys.argv):
            device_choice = sys.argv[i + 1]
        elif not arg.startswith("--"):
            out_dir = arg

    result = detect_anchors_florence(img_path, out_dir, device=device_choice)
    print(f"\nFound {result['n_anchors']} anchors:")
    for a in result["anchors"]:
        print(f"  [{a['type']}] raw='{a['label_raw']}'  {a['pixel_width']:.0f}x{a['pixel_height']:.0f}px  → {a['real_width_inches']}in real")
    print(f"\nAnnotated image: {out_dir}/{result['image_id']}_florence_annotated.png")
