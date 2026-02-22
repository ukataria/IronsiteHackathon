"""VLM-based anchor detection using Claude or GPT-4o instead of GroundingDINO."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

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
from src.vlm.clients import call_claude, call_gpt4o, call_ollama

logger = setup_logger("anchors_vlm")

ANCHOR_TYPES = ["stud", "rebar", "cmu", "electrical_box", "door", "hardhat", "brick"]

DETECTION_PROMPT = """You are a construction site object detector.

Examine this construction image and locate any of these objects:
- stud: wooden 2x4/2x6 wall framing members (vertical boards)
- rebar: steel reinforcement bars / metal rods
- cmu: concrete masonry units / cinder blocks / concrete blocks
- electrical_box: electrical boxes, outlet boxes, junction boxes, switch boxes
- door: door openings, door frames, rough openings
- hardhat: hard hats / construction helmets
- brick: bricks / masonry bricks

For EACH object found, return its bounding box as pixel coordinates.
The image is {width}x{height} pixels.

Return ONLY valid JSON in this exact format, no other text:
{{
  "detections": [
    {{
      "type": "electrical_box",
      "box_pixels": [x1, y1, x2, y2],
      "confidence": 0.9,
      "notes": "single-gang outlet box on left wall"
    }}
  ]
}}

Rules:
- box_pixels must be [x1, y1, x2, y2] in integer pixel coords (top-left to bottom-right)
- confidence is your certainty from 0.0 to 1.0
- If nothing is found, return {{"detections": []}}
- Do not include objects you are not confident about (confidence < 0.4)
"""


def _parse_detections(raw: str, img_w: int, img_h: int) -> list[dict]:
    """Extract detection list from raw VLM JSON response."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(cleaned)
        detections = data.get("detections", [])
    except json.JSONDecodeError:
        # Try to find a JSON block within the text
        match = re.search(r'\{.*"detections".*\}', cleaned, re.DOTALL)
        if not match:
            logger.warning("Could not parse VLM response as JSON.")
            return []
        try:
            data = json.loads(match.group())
            detections = data.get("detections", [])
        except json.JSONDecodeError:
            logger.warning("Fallback JSON parse also failed.")
            return []

    valid = []
    for d in detections:
        anchor_type = d.get("type", "")
        if anchor_type not in ANCHOR_TYPES:
            continue
        box = d.get("box_pixels", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        x2 = min(x2, img_w)
        y2 = min(y2, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        valid.append({
            "type": anchor_type,
            "box_pixels": [x1, y1, x2, y2],
            "confidence": float(d.get("confidence", 0.5)),
            "notes": d.get("notes", ""),
        })

    return valid


def detect_anchors_vlm(
    image_path: str,
    output_dir: str,
    vlm: str = "claude",
    min_confidence: float = 0.4,
) -> dict:
    """
    Detect construction anchors using a VLM (Claude or GPT-4o).

    Saves:
      {output_dir}/{image_id}_vlm_anchors.json
      {output_dir}/{image_id}_vlm_annotated.png

    Returns the detections dict.
    """
    image_id = get_image_id(image_path)
    out_json = str(Path(output_dir) / f"{image_id}_vlm_anchors.json")
    out_img = str(Path(output_dir) / f"{image_id}_vlm_annotated.png")

    img_rgb = load_image(image_path)
    h, w = img_rgb.shape[:2]

    prompt = DETECTION_PROMPT.format(width=w, height=h)
    cache_key = f"{image_id}_vlm_detect_{vlm}"

    if vlm == "claude":
        caller = call_claude
    elif vlm == "ollama":
        caller = call_ollama
    else:
        caller = call_gpt4o
    logger.info(f"Calling {vlm} for detection on {image_id} ({w}x{h})...")

    raw_response = caller(prompt, image_path=image_path, cache_key=cache_key)
    logger.info(f"  Raw response length: {len(raw_response)} chars")

    detections = _parse_detections(raw_response, w, h)
    detections = [d for d in detections if d["confidence"] >= min_confidence]

    # Enrich with real-world dimensions from constants
    anchors = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["box_pixels"]
        dim_key = ANCHOR_PRIMARY_DIMENSION.get(d["type"], "stud_face_width")
        anchors.append({
            "id": i,
            "type": d["type"],
            "box_pixels": d["box_pixels"],
            "confidence": d["confidence"],
            "notes": d["notes"],
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
        "vlm": vlm,
        "n_anchors": len(anchors),
        "anchors": anchors,
        "raw_response": raw_response,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_json(result, out_json)

    box_dicts = [
        {"box_pixels": a["box_pixels"], "label": a["type"], "confidence": a["confidence"]}
        for a in anchors
    ]
    annotated = draw_boxes(img_rgb, box_dicts)
    save_image(annotated, out_img)

    logger.info(f"{image_id}: {len(anchors)} anchors via {vlm} → {out_json}")
    logger.info(f"  Annotated image → {out_img}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python src/anchors/detect_vlm.py <image_path> [--vlm claude|gpt4o] [output_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    vlm_choice = "claude"
    out_dir = "data/detections"

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--vlm" and i + 1 < len(sys.argv):
            vlm_choice = sys.argv[i + 1]
        elif not arg.startswith("--"):
            out_dir = arg

    result = detect_anchors_vlm(img_path, out_dir, vlm=vlm_choice)
    print(f"\nFound {result['n_anchors']} anchors via {vlm_choice}:")
    for a in result["anchors"]:
        print(f"  [{a['type']}] conf={a['confidence']:.2f}  {a['pixel_width']:.0f}x{a['pixel_height']:.0f}px  → {a['real_width_inches']}in real  | {a['notes']}")
    print(f"\nAnnotated image: data/detections/{result['image_id']}_vlm_annotated.png")
