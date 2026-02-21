"""Alpha compositing engine — ghost overlay with per-layer transparency."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from src.utils import COMPOSITES_DIR, save_image

log = logging.getLogger(__name__)

# Ghost tint color in BGR (light blue)
GHOST_TINT_BGR = (255, 200, 150)  # light blue-ish tint

LAYER_ORDER = ["floor", "walls", "ceiling", "electrical", "plumbing", "hvac", "fixtures"]


def apply_ghost_tint(
    overlay: np.ndarray,
    mask: np.ndarray,
    tint_color: tuple[int, int, int] = GHOST_TINT_BGR,
    tint_strength: float = 0.25,
) -> np.ndarray:
    """Apply a subtle color tint to the overlay region to make it visually distinct.

    Args:
        overlay: BGR overlay image.
        mask: Binary mask (255=tint, 0=skip) as HxW uint8.
        tint_color: BGR tint color.
        tint_strength: How much tint to apply (0=none, 1=full color).

    Returns:
        Tinted overlay (BGR).
    """
    tinted = overlay.copy().astype(np.float32)
    tint = np.array(tint_color, dtype=np.float32)
    mask_bool = (mask > 127)

    for c in range(3):
        tinted[:, :, c][mask_bool] = (
            tinted[:, :, c][mask_bool] * (1 - tint_strength) + tint[c] * tint_strength
        )

    return np.clip(tinted, 0, 255).astype(np.uint8)


def alpha_blend(
    base: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Alpha-blend overlay onto base within the masked region.

    Args:
        base: BGR base image (original construction frame).
        overlay: BGR overlay image (inpainted finished element).
        mask: HxW binary mask (255=blend, 0=show base only).
        alpha: Overlay opacity (0=transparent, 1=opaque).

    Returns:
        Composited BGR image.
    """
    result = base.copy().astype(np.float32)
    mask_norm = (mask.astype(np.float32) / 255.0) * alpha

    for c in range(3):
        result[:, :, c] = (
            result[:, :, c] * (1 - mask_norm) +
            overlay[:, :, c].astype(np.float32) * mask_norm
        )

    return np.clip(result, 0, 255).astype(np.uint8)


def composite_layers(
    base_path: str | Path,
    overlay_paths: dict[str, Path],
    mask_paths: dict[str, Path],
    layer_alphas: dict[str, float],
    stem: str,
    apply_tint: bool = True,
) -> Path:
    """Composite multiple overlay layers onto the base frame.

    Layers are applied in LAYER_ORDER (floor first, ceiling last).

    Args:
        base_path: Original construction frame.
        overlay_paths: Dict of layer_name → overlay image path.
        mask_paths: Dict of layer_name → mask path.
        layer_alphas: Dict of layer_name → opacity (0.0–1.0).
        stem: Frame stem for output naming.
        apply_tint: Whether to apply ghost tint to overlays.

    Returns:
        Path to final composited image.
    """
    out_path = COMPOSITES_DIR / f"{stem}_composite.png"

    base = cv2.imread(str(base_path))
    result = base.copy()

    # Process in defined layer order
    ordered_layers = [l for l in LAYER_ORDER if l in overlay_paths]
    # Add any layers not in LAYER_ORDER at the end
    remaining = [l for l in overlay_paths if l not in LAYER_ORDER]
    ordered_layers += remaining

    for layer_name in ordered_layers:
        overlay_path = overlay_paths.get(layer_name)
        mask_path = mask_paths.get(layer_name)
        alpha = layer_alphas.get(layer_name, 0.4)

        if overlay_path is None or mask_path is None:
            continue
        if alpha <= 0.0:
            continue

        overlay = cv2.imread(str(overlay_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if overlay is None or mask is None:
            log.warning(f"Could not load layer {layer_name}, skipping.")
            continue

        # Resize overlay/mask to base size if needed
        h, w = base.shape[:2]
        if overlay.shape[:2] != (h, w):
            overlay = cv2.resize(overlay, (w, h))
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if apply_tint:
            overlay = apply_ghost_tint(overlay, mask)

        result = alpha_blend(result, overlay, mask, alpha=alpha)
        log.info(f"Composited layer: {layer_name} (alpha={alpha:.2f})")

    save_image(result, out_path)
    log.info(f"Composite saved: {out_path}")
    return out_path


def make_side_by_side(
    original_path: str | Path,
    composite_path: str | Path,
    stem: str,
    label_left: str = "CONSTRUCTION",
    label_right: str = "GHOST BLUEPRINT",
) -> Path:
    """Create a side-by-side comparison image.

    Returns:
        Path to side-by-side image.
    """
    out_path = COMPOSITES_DIR / f"{stem}_sidebyside.png"

    left = cv2.imread(str(original_path))
    right = cv2.imread(str(composite_path))

    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    left = cv2.resize(left, (w, h))
    right = cv2.resize(right, (w, h))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(left, label_left, (20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, label_right, (20, 40), font, 1.0, (150, 255, 200), 2, cv2.LINE_AA)

    divider = np.ones((h, 4, 3), dtype=np.uint8) * 200
    combined = np.concatenate([left, divider, right], axis=1)

    save_image(combined, out_path)
    log.info(f"Side-by-side saved: {out_path}")
    return out_path


def make_ghost_only(
    overlay_paths: dict[str, Path],
    mask_paths: dict[str, Path],
    base_shape: tuple[int, int],
    stem: str,
) -> Path:
    """Render ghost overlay on a black background (no base frame).

    Useful for the 'ghost only' view mode in the demo.
    """
    out_path = COMPOSITES_DIR / f"{stem}_ghost_only.png"
    h, w = base_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for layer_name in LAYER_ORDER:
        overlay_path = overlay_paths.get(layer_name)
        mask_path = mask_paths.get(layer_name)
        if overlay_path is None or mask_path is None:
            continue

        overlay = cv2.imread(str(overlay_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if overlay is None or mask is None:
            continue

        if overlay.shape[:2] != (h, w):
            overlay = cv2.resize(overlay, (w, h))
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_bool = mask > 127
        canvas[mask_bool] = overlay[mask_bool]

    save_image(canvas, out_path)
    return out_path
