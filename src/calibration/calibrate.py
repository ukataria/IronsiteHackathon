"""Scale calibration — convert pixel distances to real-world inches."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import get_image_id, load_json, save_json, setup_logger

logger = setup_logger("calibration")

# Coefficient of variation threshold above which we flag low confidence
CV_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Single-anchor math
# ---------------------------------------------------------------------------


def compute_scale_factor(
    anchor_pixel_width: float,
    anchor_real_width_inches: float,
) -> float:
    """
    Compute pixels-per-inch scale from a single anchor.
    Returns 0.0 if anchor_pixel_width is invalid.
    """
    if anchor_pixel_width <= 0:
        return 0.0
    return anchor_pixel_width / anchor_real_width_inches


def depth_adjusted_scale(
    base_scale: float,
    anchor_depth: float,
    target_depth: float,
) -> float:
    """
    Adjust pixels-per-inch scale for an object at a different depth than the anchor.
    Uses the inverse depth ratio: closer objects appear larger per inch.
    Assumes relative (not absolute) depth where higher value = farther away.
    """
    if target_depth <= 0 or anchor_depth <= 0:
        return base_scale
    return base_scale * (anchor_depth / target_depth)


# ---------------------------------------------------------------------------
# Multi-anchor plane calibration
# ---------------------------------------------------------------------------


def calibrate_plane(anchors_on_plane: list[dict]) -> dict:
    """
    Compute calibrated scale for a set of anchors on the same depth plane.

    Each anchor dict must have:
      - pixel_width: float
      - real_width_inches: float
      - (optionally) depth: float

    Returns:
      pixels_per_inch, confidence, n_anchors, scale_estimates, outlier_indices
    """
    if not anchors_on_plane:
        return {
            "pixels_per_inch": 0.0,
            "confidence": 0.0,
            "n_anchors": 0,
            "scale_estimates": [],
            "outlier_indices": [],
        }

    scale_estimates = []
    for a in anchors_on_plane:
        s = compute_scale_factor(a.get("pixel_width", 0), a.get("real_width_inches", 1))
        if s > 0:
            scale_estimates.append(s)

    if not scale_estimates:
        return {
            "pixels_per_inch": 0.0,
            "confidence": 0.0,
            "n_anchors": 0,
            "scale_estimates": scale_estimates,
            "outlier_indices": [],
        }

    arr = np.array(scale_estimates)
    median_scale = float(np.median(arr))
    mean_scale = float(np.mean(arr))

    # Coefficient of variation — lower is better
    cv = float(np.std(arr) / mean_scale) if mean_scale > 0 else 1.0
    confidence = max(0.0, 1.0 - cv)

    # Outlier detection: anchors more than 2 std devs from median
    std = float(np.std(arr))
    outlier_indices = [
        i for i, s in enumerate(scale_estimates)
        if abs(s - median_scale) > 2 * std
    ]

    return {
        "pixels_per_inch": median_scale,
        "confidence": round(confidence, 4),
        "n_anchors": len(scale_estimates),
        "scale_estimates": [round(s, 4) for s in scale_estimates],
        "outlier_indices": outlier_indices,
    }


# ---------------------------------------------------------------------------
# Perspective skew detection
# ---------------------------------------------------------------------------


def detect_perspective_skew(anchors: list[dict]) -> float | None:
    """
    Detect if scale factor varies linearly across image width (camera angle skew).
    If skew is detected, returns the slope (scale gradient per pixel).
    Returns None if fewer than 3 anchors or no significant skew.
    """
    if len(anchors) < 3:
        return None

    cx_list = []
    scale_list = []
    for a in anchors:
        s = compute_scale_factor(a.get("pixel_width", 0), a.get("real_width_inches", 1))
        if s > 0:
            cx_list.append(a.get("center_x", 0))
            scale_list.append(s)

    if len(cx_list) < 3:
        return None

    cx_arr = np.array(cx_list)
    s_arr = np.array(scale_list)

    # Linear regression: scale ~ slope * cx + intercept
    coeffs = np.polyfit(cx_arr, s_arr, 1)
    slope = float(coeffs[0])

    # Significant if slope explains more than 10% variation relative to mean scale
    mean_s = float(np.mean(s_arr))
    img_width_estimate = float(cx_arr.max() - cx_arr.min())
    total_variation = abs(slope * img_width_estimate)

    if mean_s > 0 and (total_variation / mean_s) > 0.10:
        logger.warning(f"Perspective skew detected: slope={slope:.5f} px/inch per pixel")
        return slope

    return None


# ---------------------------------------------------------------------------
# Full image calibration
# ---------------------------------------------------------------------------


def calibrate_image(
    anchors_json_path: str,
    depth_npy_path: str,
    output_dir: str,
    fallback_ppi: float = 0.0,
) -> dict:
    """
    Full calibration pipeline for one image.

    Reads:
      anchors JSON from detect_anchors()
      depth .npy from estimate_depth()

    Saves:
      {output_dir}/{image_id}_calibration.json

    Returns the calibration dict.
    """
    import numpy as np

    anchors_data = load_json(anchors_json_path)
    image_id = anchors_data["image_id"]
    image_width = anchors_data["image_width"]
    image_height = anchors_data["image_height"]
    anchors = anchors_data.get("anchors", [])

    # Load depth map and augment anchors with depth values
    depth_map = np.load(depth_npy_path)

    from src.depth.estimate import get_anchor_depth, group_anchors_by_plane

    planes = group_anchors_by_plane(anchors, depth_map, image_width, image_height)

    calibrated_planes = []
    for plane_id, plane_anchors in planes.items():
        depths = [a.get("depth", 0.5) for a in plane_anchors]
        plane_cal = calibrate_plane(plane_anchors)
        skew = detect_perspective_skew(plane_anchors)
        calibrated_planes.append({
            "plane_id": plane_id,
            "depth_range": [round(min(depths), 4), round(max(depths), 4)],
            "median_depth": round(float(np.median(depths)), 4),
            "pixels_per_inch": round(plane_cal["pixels_per_inch"], 4),
            "confidence": plane_cal["confidence"],
            "n_anchors": plane_cal["n_anchors"],
            "anchor_types": [a["type"] for a in plane_anchors],
            "perspective_correction": round(skew, 6) if skew is not None else None,
        })

    # Primary plane = highest confidence with at least 1 anchor
    primary = None
    if calibrated_planes:
        primary = max(
            (p for p in calibrated_planes if p["n_anchors"] > 0),
            key=lambda p: (p["confidence"], p["n_anchors"]),
            default=None,
        )

    # No anchors detected — carry forward last known good scale if available
    if primary is None and fallback_ppi > 0:
        primary = {
            "plane_id": "carried_forward",
            "pixels_per_inch": fallback_ppi,
            "confidence": 0.3,  # lower confidence to signal it's inferred
        }
        logger.info(f"{image_id}: no anchors — using carried-forward scale {fallback_ppi:.2f} px/in")

    result = {
        "image_id": image_id,
        "planes": calibrated_planes,
        "primary_plane": primary["plane_id"] if primary else None,
        "primary_pixels_per_inch": primary["pixels_per_inch"] if primary else 0.0,
        "primary_confidence": primary["confidence"] if primary else 0.0,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / f"{image_id}_calibration.json")
    save_json(result, out_path)
    logger.info(
        f"{image_id}: calibration complete — "
        f"{result['primary_pixels_per_inch']:.2f} px/in "
        f"(conf={result['primary_confidence']:.2f}) → {out_path}"
    )
    return result


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python src/calibration/calibrate.py <anchors.json> <depth.npy> [output_dir]")
        sys.exit(1)

    anchors_path = sys.argv[1]
    depth_path = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "data/calibrations"
    cal = calibrate_image(anchors_path, depth_path, out_dir)
    print(f"Primary scale: {cal['primary_pixels_per_inch']:.2f} px/in  confidence: {cal['primary_confidence']:.2f}")
