"""Anchor detection using finetuned YOLO model."""

from __future__ import annotations

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

logger = setup_logger("anchors.yolo")

DEFAULT_WEIGHTS = "yolo_weights/finetune_5_weights/best.pt"
DEFAULT_CONF = 0.25

# ---------------------------------------------------------------------------
# Class name → anchor type mapping
# Handles all variants that may appear in the finetuned model's names dict.
# ---------------------------------------------------------------------------

YOLO_CLASS_TO_ANCHOR: dict[str, str] = {
    # brick variants
    "brick": "brick",
    "bricks-masonry": "brick",
    "bricks masonry": "brick",
    # cmu / cinder block variants  (finetune_5 uses "block")
    "block": "cmu",
    "cmu": "cmu",
    "cinder block": "cmu",
    "concrete block": "cmu",
    "1": "cmu",  # labeled "1" in the cmu_blocks roboflow dataset
    # electrical outlet variants  (finetune_5 uses "outlet")
    "outlet": "electrical_box",
    "electrical-outlet": "electrical_box",
    "electrical outlet": "electrical_box",
    "elec": "electrical_box",
}

# Cache loaded model globally
_yolo_model = None
_loaded_weights: str | None = None


def load_yolo_model(weights_path: str = DEFAULT_WEIGHTS):
    """Load YOLO model from weights path. Cached after first load."""
    global _yolo_model, _loaded_weights

    if _yolo_model is not None and _loaded_weights == weights_path:
        return _yolo_model

    from ultralytics import YOLO

    logger.info(f"Loading YOLO model from {weights_path}...")
    _yolo_model = YOLO(weights_path)
    _loaded_weights = weights_path
    logger.info(f"YOLO loaded. Classes: {_yolo_model.names}")
    return _yolo_model


def _classify_yolo_label(raw_label: str) -> str:
    """Map a YOLO class name to a canonical anchor type key."""
    return YOLO_CLASS_TO_ANCHOR.get(raw_label.lower().strip(), "unknown")


def detect_anchors_yolo(
    image_path: str,
    output_dir: str,
    weights_path: str = DEFAULT_WEIGHTS,
    conf: float = DEFAULT_CONF,
) -> dict:
    """
    Detect known-dimension objects in a construction image using the finetuned YOLO model.

    Saves:
      {output_dir}/{image_id}_anchors.json
      {output_dir}/{image_id}_annotated.png

    Returns the detections dict (same schema as detect_anchors in detect.py).
    """
    image_id = get_image_id(image_path)
    out_json = str(Path(output_dir) / f"{image_id}_anchors.json")
    out_img = str(Path(output_dir) / f"{image_id}_annotated.png")

    img_rgb = load_image(image_path)
    h, w = img_rgb.shape[:2]

    anchors = []
    try:
        model = load_yolo_model(weights_path)
        results = model(image_path, conf=conf, verbose=False)
        r = results[0]

        logger.info(f"  YOLO raw detections: {len(r.boxes)} boxes")

        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls)
            raw_label = model.names[cls_id]
            anchor_type = _classify_yolo_label(raw_label)

            if anchor_type == "unknown":
                logger.debug(f"  Skipping unknown class: {raw_label!r}")
                continue

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            px_w = abs(x2 - x1)
            px_h = abs(y2 - y1)
            confidence = float(box.conf)

            dim_key = ANCHOR_PRIMARY_DIMENSION.get(anchor_type, "brick_length")
            real_width = ANCHOR_DIMENSIONS[dim_key]

            anchors.append({
                "id": i,
                "type": anchor_type,
                "label_raw": raw_label,
                "box_pixels": [x1, y1, x2, y2],
                "confidence": confidence,
                "pixel_width": px_w,
                "pixel_height": px_h,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
                "real_width_inches": real_width,
            })

    except Exception as e:
        logger.error(f"YOLO detection failed for {image_id}: {e}")

    result = {
        "image_id": image_id,
        "image_width": w,
        "image_height": h,
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

    logger.info(f"{image_id}: detected {len(anchors)} anchors → {out_json}")
    return result


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python src/anchors/detect_yolo.py <image_path> [output_dir] [weights_path]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/detections"
    weights = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_WEIGHTS
    result = detect_anchors_yolo(img_path, out_dir, weights)
    print(f"Found {result['n_anchors']} anchors:")
    for a in result["anchors"]:
        print(f"  [{a['type']}] conf={a['confidence']:.2f}  raw={a['label_raw']!r}  width={a['pixel_width']:.1f}px → {a['real_width_inches']}in")
