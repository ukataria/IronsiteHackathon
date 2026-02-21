"""GroundedSAM / SAM2 segmentation for construction elements."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.utils import SEGMENTS_DIR, save_image, load_image

log = logging.getLogger(__name__)

# Construction element prompts for GroundingDINO text detection
CONSTRUCTION_LABELS = [
    "wooden stud",
    "concrete floor",
    "ceiling joist",
    "electrical box",
    "pipe",
    "hvac duct",
    "window opening",
    "door opening",
    "plywood subfloor",
    "insulation",
    "temporary bracing",
]

# Map raw labels to clean category names
LABEL_CATEGORY_MAP = {
    "wooden stud": "studs",
    "concrete floor": "floor",
    "ceiling joist": "ceiling",
    "electrical box": "electrical",
    "pipe": "plumbing",
    "hvac duct": "hvac",
    "window opening": "windows",
    "door opening": "doors",
    "plywood subfloor": "floor",
    "insulation": "insulation",
    "temporary bracing": "studs",
}

_grounding_model = None
_sam_predictor = None


def _load_grounded_sam():
    """Lazy-load GroundingDINO + SAM2 models."""
    global _grounding_model, _sam_predictor
    if _grounding_model is not None:
        return _grounding_model, _sam_predictor

    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading GroundingDINO on {device}")

    model_id = "IDEA-Research/grounding-dino-tiny"
    _grounding_model = {
        "processor": AutoProcessor.from_pretrained(model_id),
        "model": AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device),
        "device": device,
    }
    log.info("GroundingDINO loaded.")
    return _grounding_model, _sam_predictor


def detect_elements(
    image_path: str | Path,
    labels: list[str] | None = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> list[dict]:
    """Run GroundingDINO to detect construction elements.

    Returns:
        List of dicts with keys: label, score, box (xyxy).
    """
    import torch
    from PIL import Image as PILImage

    image_path = Path(image_path)
    labels = labels or CONSTRUCTION_LABELS
    grounding, _ = _load_grounded_sam()

    img = PILImage.open(str(image_path)).convert("RGB")
    text_prompt = ". ".join(labels) + "."

    processor = grounding["processor"]
    model = grounding["model"]
    device = grounding["device"]

    inputs = processor(images=img, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        target_sizes=[img.size[::-1]],
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": label,
            "category": LABEL_CATEGORY_MAP.get(label, label),
            "score": float(score),
            "box": [float(x) for x in box.tolist()],  # [x1, y1, x2, y2]
        })

    log.info(f"{image_path.name}: {len(detections)} elements detected")
    return detections


def box_to_mask(box: list[float], image_shape: tuple[int, int]) -> np.ndarray:
    """Convert bounding box [x1,y1,x2,y2] to binary mask."""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    mask[y1:y2, x1:x2] = 255
    return mask


def segment_by_category(
    image_path: str | Path,
    detections: list[dict],
) -> dict[str, np.ndarray]:
    """Merge detection boxes into per-category binary masks.

    Returns:
        Dict mapping category name → binary mask (HxW uint8, 0 or 255).
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    category_masks: dict[str, np.ndarray] = {}
    for det in detections:
        cat = det["category"]
        box_mask = box_to_mask(det["box"], (h, w))
        if cat in category_masks:
            category_masks[cat] = cv2.bitwise_or(category_masks[cat], box_mask)
        else:
            category_masks[cat] = box_mask

    return category_masks


def save_masks(
    masks: dict[str, np.ndarray],
    stem: str,
) -> dict[str, Path]:
    """Save per-category masks to data/segments/.

    Returns:
        Dict mapping category → saved path.
    """
    saved = {}
    for category, mask in masks.items():
        out_path = SEGMENTS_DIR / f"{stem}_{category}.png"
        cv2.imwrite(str(out_path), mask)
        saved[category] = out_path
        log.info(f"Mask saved: {out_path}")
    return saved


def build_wall_mask(
    image_path: str | Path,
    depth: np.ndarray,
    depth_percentile: float = 60.0,
) -> np.ndarray:
    """Heuristic: wall mask = pixels at mid-range depth (not floor/ceiling extremes).

    Useful as a fallback when GroundingDINO doesn't detect studs explicitly.
    """
    h, w = cv2.imread(str(image_path)).shape[:2]
    threshold = np.percentile(depth, depth_percentile)
    wall_mask = (depth >= threshold).astype(np.uint8) * 255
    # Erode to remove edges
    kernel = np.ones((15, 15), np.uint8)
    wall_mask = cv2.erode(wall_mask, kernel, iterations=1)
    # Resize to image size if needed
    if wall_mask.shape != (h, w):
        wall_mask = cv2.resize(wall_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return wall_mask


def run_segmentation(
    image_path: str | Path,
    stem: str | None = None,
) -> dict[str, Path]:
    """Full segmentation pipeline: detect → mask → save.

    Returns:
        Dict mapping category → mask path.
    """
    image_path = Path(image_path)
    if stem is None:
        stem = image_path.stem

    detections = detect_elements(image_path)
    masks = segment_by_category(image_path, detections)

    if not masks:
        log.warning(f"No elements detected in {image_path.name}, using full-image fallback masks")
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        masks["walls"] = np.ones((h, w), dtype=np.uint8) * 255

    return save_masks(masks, stem)
