"""Anchor detection using finetuned YOLO model."""

from __future__ import annotations

from pathlib import Path

from src.utils import (
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

# Minimum long-axis pixel size to trust a detection for calibration.
# Smaller boxes are likely far away and produce noisy scale estimates.
MIN_ANCHOR_PX = 20


def _infer_face(
    anchor_type: str,
    px_w: float,
    px_h: float,
) -> tuple[str, float, float]:
    """
    Infer which face of the object is visible and return
    (orientation, pixel_measure, real_measure_inches).

    Brick faces (ASTM C216 standard modular):
      Stretcher (most common): 7.625" × 2.25"  pixel aspect ≈ 3.39
      Header:                  7.625" × 3.625" pixel aspect ≈ 2.10
      End/soldier:             3.625" × 2.25"  pixel aspect ≈ 1.61
      → if pixel aspect < 2.0: roughly square → end/soldier face → long dim = 3.625"
      → if pixel aspect ≥ 2.0: elongated → stretcher/header face → long dim = 7.625"

    CMU faces (ASTM C90 8×8×16):
      Front: 15.625" × 7.625"  pixel aspect ≈ 2.05
      End:    7.625" × 7.625"  pixel aspect ≈ 1.00
      → if pixel aspect < 1.5: roughly square → end face → long dim = 7.625"
      → if pixel aspect ≥ 1.5: elongated → front face → long dim = 15.625"

    Electrical box (single-gang, NEC):
      Always mounted flat on wall → always face-on (2.0" × 3.0")
      → match long pixel dim to 3.0", short pixel dim to 2.0"
    """
    px_long = max(px_w, px_h)
    px_short = min(px_w, px_h)
    aspect = px_long / max(px_short, 1.0)

    if anchor_type == "brick":
        if aspect < 2.0:
            # Near-square → seeing the end/soldier face (3.625" × 2.25")
            return "end_on", px_long, 3.625
        else:
            # Elongated → seeing the stretcher/header face; long axis = 7.625"
            orientation = "horizontal" if px_w >= px_h else "vertical"
            return orientation, px_long, 7.625

    elif anchor_type == "cmu":
        if aspect < 1.5:
            # Near-square → seeing the end face (7.625" × 7.625")
            return "end_on", px_long, 7.625
        else:
            # Elongated → seeing the front face; long axis = 15.625"
            orientation = "horizontal" if px_w >= px_h else "vertical"
            return orientation, px_long, 15.625

    elif anchor_type == "electrical_box":
        # Always face-on to wall: 2.0" wide × 3.0" tall
        # Long pixel dim → 3.0", short pixel dim → 2.0"
        orientation = "vertical" if px_h >= px_w else "horizontal"
        return orientation, px_long, 3.0

    else:
        orientation = "horizontal" if px_w >= px_h else "vertical"
        return orientation, px_long, 7.625

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

            orientation, pixel_measure, real_measure = _infer_face(anchor_type, px_w, px_h)

            # Boxes smaller than MIN_ANCHOR_PX are too far away for reliable calibration.
            # Set pixel_width=0 so calibrate.py skips them (it filters s > 0).
            too_small = max(px_w, px_h) < MIN_ANCHOR_PX
            if too_small:
                logger.debug(
                    f"  Anchor {i} ({anchor_type}) too small for calibration "
                    f"({max(px_w, px_h):.1f}px < {MIN_ANCHOR_PX}px)"
                )

            anchors.append({
                "id": i,
                "type": anchor_type,
                "label_raw": raw_label,
                "box_pixels": [x1, y1, x2, y2],
                "confidence": confidence,
                "pixel_width": 0.0 if too_small else pixel_measure,
                "pixel_width_raw": px_w,
                "pixel_height": px_h,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
                "real_width_inches": real_measure,
                "orientation": orientation,
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
