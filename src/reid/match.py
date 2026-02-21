"""Object re-identification across multiple views using CLIP embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils import setup_logger

logger = setup_logger("reid")

_clip_model: Any | None = None
_clip_processor: Any | None = None


def load_clip(device: str = "cpu") -> tuple[Any, Any]:
    """Load CLIP model for feature extraction. Cached after first load."""
    global _clip_model, _clip_processor

    if _clip_model is not None:
        return _clip_model, _clip_processor

    try:
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-base-patch32"
        logger.info(f"Loading CLIP from {model_id}...")
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id).to(device)
        _clip_model.eval()
        logger.info("CLIP loaded.")
    except Exception as e:
        logger.error(f"Failed to load CLIP: {e}")
        raise

    return _clip_model, _clip_processor


def extract_clip_features(
    image: np.ndarray,
    boxes: list[list[float]],
    device: str = "cpu",
) -> np.ndarray:
    """
    Crop each bounding box from image and embed with CLIP.

    boxes: list of [x1, y1, x2, y2] in absolute pixel coords
    Returns: (N, 512) float32 feature matrix. Rows correspond to boxes.
    """
    import torch
    from PIL import Image as PILImage

    if not boxes:
        return np.zeros((0, 512), dtype=np.float32)

    model, processor = load_clip(device)
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image  # fallback: use full image
        crops.append(PILImage.fromarray(crop))

    try:
        inputs = processor(images=crops, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.error(f"CLIP feature extraction failed: {e}")
        return np.zeros((len(boxes), 512), dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute (N, M) cosine similarity matrix between two feature matrices."""
    # Both should already be L2-normalized
    return a @ b.T


def match_objects_across_views(
    detections_a: list[dict],
    detections_b: list[dict],
    features_a: np.ndarray,
    features_b: np.ndarray,
    similarity_threshold: float = 0.85,
) -> list[tuple[int, int]]:
    """
    Match detected objects between two views by cosine similarity.

    Returns list of (index_in_a, index_in_b) matched pairs.
    Uses greedy matching: each detection can only be matched once.
    """
    if features_a.shape[0] == 0 or features_b.shape[0] == 0:
        return []

    sim_matrix = cosine_similarity_matrix(features_a, features_b)
    matched_b = set()
    matches = []

    # Sort candidates by similarity (highest first) for greedy matching
    candidates = sorted(
        [(i, j, sim_matrix[i, j]) for i in range(len(detections_a)) for j in range(len(detections_b))],
        key=lambda x: -x[2],
    )

    matched_a = set()
    for i, j, sim in candidates:
        if sim < similarity_threshold:
            break
        if i in matched_a or j in matched_b:
            continue
        matches.append((i, j))
        matched_a.add(i)
        matched_b.add(j)

    logger.info(f"ReID: matched {len(matches)} objects across views (threshold={similarity_threshold})")
    return matches


def merge_measurements(
    measurements_a: dict,
    measurements_b: dict,
    matches: list[tuple[int, int]],
) -> dict:
    """
    Merge measurements from two views for matched elements.
    Averages numeric values for matched pairs; appends unmatched elements from both views.
    """
    merged = dict(measurements_a)  # start with view A

    # Average stud spacings for matched elements (simple: average all spacings)
    spacings_a = measurements_a.get("stud_spacings", [])
    spacings_b = measurements_b.get("stud_spacings", [])

    if spacings_a and spacings_b:
        merged_spacings = []
        for s_a, s_b in zip(spacings_a, spacings_b):
            avg_inches = (s_a["inches"] + s_b["inches"]) / 2
            merged_spacings.append({
                **s_a,
                "inches": round(avg_inches, 2),
                "merged": True,
            })
        merged["stud_spacings"] = merged_spacings

    # Scale: use average of both
    ppi_a = measurements_a.get("scale_pixels_per_inch", 0)
    ppi_b = measurements_b.get("scale_pixels_per_inch", 0)
    if ppi_a > 0 and ppi_b > 0:
        merged["scale_pixels_per_inch"] = round((ppi_a + ppi_b) / 2, 4)

    merged["merged_from_views"] = 2
    return merged
