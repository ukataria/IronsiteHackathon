"""Grounded SAM segmentation: GroundingDINO bounding boxes → SAM masks."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from src.anchors.detect import detect_anchors
from src.utils import get_image_id, load_image, save_image, save_json, setup_logger

logger = setup_logger("segmentation")

SAM_MODEL_ID = "facebook/sam-vit-base"  # ~375MB; swap to sam-vit-large or sam-vit-huge for better quality

# One distinct color per anchor type (RGB)
ANCHOR_COLORS: dict[str, tuple[int, int, int]] = {
    "stud":           (255, 180,  50),   # orange
    "rebar":          (220,  60,  60),   # red
    "cmu":            (100, 180, 255),   # blue
    "electrical_box": (60,  220,  60),   # green
    "door":           (180,  60, 220),   # purple
    "hardhat":        (255, 255,  60),   # yellow
    "brick":          (200, 130,  80),   # brown
    "unknown":        (180, 180, 180),   # grey
}

# ---------------------------------------------------------------------------
# Model loading (cached globally)
# ---------------------------------------------------------------------------

_sam_model = None
_sam_processor = None


def load_sam(device: str = "cpu"):
    """Load SAM model + processor from HuggingFace. Cached after first call."""
    global _sam_model, _sam_processor

    if _sam_model is not None:
        return _sam_model, _sam_processor

    from transformers import SamModel, SamProcessor

    logger.info(f"Loading SAM from {SAM_MODEL_ID} (first run downloads ~375MB)...")
    _sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    _sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    _sam_model.eval()
    logger.info("SAM loaded.")

    return _sam_model, _sam_processor


# ---------------------------------------------------------------------------
# Mask prediction
# ---------------------------------------------------------------------------


def predict_masks_for_boxes(
    image_rgb: np.ndarray,
    boxes: list[list[float]],
    device: str = "cpu",
) -> list[np.ndarray]:
    """
    Run SAM on each bounding box and return binary masks (H x W bool arrays).
    boxes: list of [x1, y1, x2, y2] in pixel coords.
    """
    from PIL import Image as PILImage

    if not boxes:
        return []

    model, processor = load_sam(device)
    pil_img = PILImage.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    masks_out = []
    for box in boxes:
        try:
            # SamProcessor expects input_boxes as [[[x1, y1, x2, y2]]]
            inputs = processor(
                images=pil_img,
                input_boxes=[[[box]]],
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process — returns list of masks per image
            result = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            # result[0] shape: (1, 3, H, W) — take best mask (index 0)
            mask_tensor = result[0][0, 0]   # (H, W) bool
            masks_out.append(mask_tensor.numpy().astype(bool))

        except Exception as e:
            logger.warning(f"SAM failed for box {box}: {e}")
            # Fallback: fill the bounding box as a mask
            fallback = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = [int(v) for v in box]
            fallback[y1:y2, x1:x2] = True
            masks_out.append(fallback)

    return masks_out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def draw_masks_and_boxes(
    image_rgb: np.ndarray,
    anchors: list[dict],
    masks: list[np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay colored masks and labeled bounding boxes on image copy."""
    out = image_rgb.copy().astype(np.float32)

    for anchor, mask in zip(anchors, masks):
        color = np.array(ANCHOR_COLORS.get(anchor["type"], ANCHOR_COLORS["unknown"]), dtype=np.float32)
        # Blend mask region
        out[mask] = out[mask] * (1 - alpha) + color * alpha

    out = np.clip(out, 0, 255).astype(np.uint8)

    # Draw boxes and labels on top
    for anchor in anchors:
        x1, y1, x2, y2 = [int(v) for v in anchor["box_pixels"]]
        color_bgr = tuple(int(c) for c in reversed(ANCHOR_COLORS.get(anchor["type"], ANCHOR_COLORS["unknown"])))
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{anchor['type']} {anchor['confidence']:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def segment_anchors(
    image_path: str,
    output_dir: str,
    detections_dir: str = "data/detections",
    device: str = "cpu",
    box_threshold: float = 0.15,
) -> dict:
    """
    Run GroundingDINO + SAM on an image to produce segmentation masks per anchor.

    Saves:
      {output_dir}/{image_id}_segments.json    — mask bounding rects + anchor info
      {output_dir}/{image_id}_segmented.png    — image with colored overlays

    Returns results dict.
    """
    image_id = get_image_id(image_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: GroundingDINO detection
    logger.info(f"[1/2] Running GroundingDINO on {image_id}...")
    detection = detect_anchors(
        image_path, detections_dir,
        box_threshold=box_threshold, device=device,
    )
    anchors = detection["anchors"]
    logger.info(f"  → {len(anchors)} anchors detected")

    if not anchors:
        logger.warning("No anchors found — skipping SAM.")
        return {"image_id": image_id, "n_segments": 0, "segments": []}

    # Step 2: SAM masking
    logger.info(f"[2/2] Running SAM on {len(anchors)} boxes...")
    image_rgb = load_image(image_path)
    boxes = [a["box_pixels"] for a in anchors]
    masks = predict_masks_for_boxes(image_rgb, boxes, device=device)

    # Build results with mask bounding rects
    segments = []
    for anchor, mask in zip(anchors, masks):
        ys, xs = np.where(mask)
        mask_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else anchor["box_pixels"]
        segments.append({**anchor, "mask_bbox": mask_bbox, "mask_area_px": int(mask.sum())})

    # Save annotated image
    annotated = draw_masks_and_boxes(image_rgb, anchors, masks)
    out_img = str(Path(output_dir) / f"{image_id}_segmented.png")
    save_image(annotated, out_img)

    # Save JSON
    result = {"image_id": image_id, "n_segments": len(segments), "segments": segments}
    out_json = str(Path(output_dir) / f"{image_id}_segments.json")
    save_json(result, out_json)

    logger.info(f"Done → {out_img}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python src/segmentation/segment.py <image_path> [--device cpu|cuda] [output_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    device_choice = "cpu"
    out_dir = "data/segments"

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--device" and i + 1 < len(sys.argv):
            device_choice = sys.argv[i + 1]
        elif not arg.startswith("--"):
            out_dir = arg

    result = segment_anchors(img_path, out_dir, device=device_choice)
    print(f"\nSegmented {result['n_segments']} anchors:")
    for s in result["segments"]:
        print(f"  [{s['type']}] conf={s['confidence']:.2f}  mask_area={s['mask_area_px']}px²")
    print(f"\nOutput: {out_dir}/{result['image_id']}_segmented.png")
