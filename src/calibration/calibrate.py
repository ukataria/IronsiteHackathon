"""Scale calibration — convert pixel distances to real-world inches.

Calibration is derived purely from known physical object dimensions:
  - Brick:        7.625" long  (ASTM C216 standard modular)
  - CMU block:   15.625" long  (ASTM C90 standard 8×8×16)
  - Outlet box:   2.0"   wide  (single-gang device box)

No depth map required. Each detected anchor gives one scale estimate
(pixel_width / real_width_inches). Per-type medians are cross-validated
for confidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import get_image_id, load_json, save_json, setup_logger

logger = setup_logger("calibration")


# ---------------------------------------------------------------------------
# Single-anchor math
# ---------------------------------------------------------------------------


def compute_scale_factor(
    anchor_pixel_width: float,
    anchor_real_width_inches: float,
) -> float:
    """Return pixels-per-inch from one anchor. Returns 0.0 on bad input."""
    if anchor_pixel_width <= 0 or anchor_real_width_inches <= 0:
        return 0.0
    return anchor_pixel_width / anchor_real_width_inches


# ---------------------------------------------------------------------------
# Per-type calibration
# ---------------------------------------------------------------------------


def calibrate_type(anchors: list[dict]) -> dict:
    """
    Compute calibrated scale from anchors of a single type.

    Returns pixels_per_inch, confidence, n_anchors, and individual estimates.
    Confidence = 1 - coefficient_of_variation (lower spread → higher confidence).
    """
    scales = [
        compute_scale_factor(a.get("pixel_width", 0), a.get("real_width_inches", 1))
        for a in anchors
    ]
    scales = [s for s in scales if s > 0]

    if not scales:
        return {"pixels_per_inch": 0.0, "confidence": 0.0, "n_anchors": 0, "scale_estimates": []}

    arr = np.array(scales)
    median_ppi = float(np.median(arr))
    mean_ppi = float(np.mean(arr))
    cv = float(np.std(arr) / mean_ppi) if mean_ppi > 0 else 1.0
    confidence = round(max(0.0, 1.0 - cv), 4)

    return {
        "pixels_per_inch": round(median_ppi, 4),
        "confidence": confidence,
        "n_anchors": len(scales),
        "scale_estimates": [round(s, 4) for s in scales],
    }


# ---------------------------------------------------------------------------
# Perspective skew detection
# ---------------------------------------------------------------------------


def detect_perspective_skew(anchors: list[dict]) -> float | None:
    """
    Detect linear scale gradient across image width (camera angle skew).
    Returns slope (px/in per pixel) if significant, else None.
    """
    if len(anchors) < 3:
        return None

    cx_list, scale_list = [], []
    for a in anchors:
        s = compute_scale_factor(a.get("pixel_width", 0), a.get("real_width_inches", 1))
        if s > 0:
            cx_list.append(a.get("center_x", 0))
            scale_list.append(s)

    if len(cx_list) < 3:
        return None

    cx_arr, s_arr = np.array(cx_list), np.array(scale_list)
    slope = float(np.polyfit(cx_arr, s_arr, 1)[0])
    mean_s = float(np.mean(s_arr))
    span = float(cx_arr.max() - cx_arr.min())

    if mean_s > 0 and (abs(slope) * span / mean_s) > 0.10:
        logger.warning(f"Perspective skew detected: slope={slope:.5f} px/in per pixel")
        return slope
    return None


# ---------------------------------------------------------------------------
# Full image calibration — no depth map needed
# ---------------------------------------------------------------------------


def calibrate_image(
    anchors_json_path: str,
    output_dir: str,
) -> dict:
    """
    Calibrate pixel-to-inch scale using known physical object dimensions.

    For each detected anchor type (brick, cmu, electrical_box) computes a
    per-type scale from pixel_width / real_width_inches. Cross-validates
    across types for overall confidence.

    Saves:  {output_dir}/{image_id}_calibration.json
    Returns the calibration dict.

    Output keys (backward-compatible with downstream consumers):
      primary_pixels_per_inch  — overall median scale across all anchors
      primary_confidence       — confidence score [0, 1]
      per_type                 — per-type breakdown
    """
    anchors_data = load_json(anchors_json_path)
    image_id = anchors_data["image_id"]
    anchors = anchors_data.get("anchors", [])

    # Group anchors by type
    by_type: dict[str, list[dict]] = {}
    for a in anchors:
        by_type.setdefault(a["type"], []).append(a)

    # Per-type calibration
    per_type: dict[str, dict] = {}
    for atype, type_anchors in by_type.items():
        per_type[atype] = calibrate_type(type_anchors)
        logger.info(
            f"  [{atype}] {per_type[atype]['n_anchors']} anchors → "
            f"{per_type[atype]['pixels_per_inch']:.2f} px/in "
            f"(conf={per_type[atype]['confidence']:.2f})"
        )

    # Overall scale = median of all individual estimates across every type
    all_scales = [
        s
        for t in per_type.values()
        for s in t["scale_estimates"]
    ]

    if not all_scales:
        primary_ppi = 0.0
        primary_conf = 0.0
    else:
        arr = np.array(all_scales)
        primary_ppi = float(np.median(arr))

        # Within-anchor consistency
        mean_all = float(np.mean(arr))
        cv_all = float(np.std(arr) / mean_all) if mean_all > 0 else 1.0
        within_conf = max(0.0, 1.0 - cv_all)

        # Cross-type agreement (penalise if types disagree with each other)
        type_ppis = [v["pixels_per_inch"] for v in per_type.values() if v["pixels_per_inch"] > 0]
        if len(type_ppis) > 1:
            cross_cv = float(np.std(type_ppis) / np.mean(type_ppis))
            cross_conf = max(0.0, 1.0 - cross_cv)
        else:
            cross_conf = within_conf  # only one type — use within-type confidence

        primary_conf = round((within_conf + cross_conf) / 2, 4)

    # Perspective skew across all anchors
    skew = detect_perspective_skew(anchors)

    result = {
        "image_id": image_id,
        "calibration_method": "known_dimensions",
        "primary_pixels_per_inch": round(primary_ppi, 4),
        "primary_confidence": primary_conf,
        "per_type": per_type,
        "n_anchors_total": len(anchors),
        "perspective_skew": round(skew, 6) if skew is not None else None,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / f"{image_id}_calibration.json")
    save_json(result, out_path)
    logger.info(
        f"{image_id}: calibration → {primary_ppi:.2f} px/in "
        f"(conf={primary_conf:.2f}, method=known_dimensions) → {out_path}"
    )
    return result


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python src/calibration/calibrate.py <anchors.json> [output_dir]")
        sys.exit(1)

    anchors_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/calibrations"
    cal = calibrate_image(anchors_path, out_dir)
    print(f"Primary scale: {cal['primary_pixels_per_inch']:.2f} px/in  confidence: {cal['primary_confidence']:.2f}")
    for atype, t in cal["per_type"].items():
        print(f"  [{atype}] {t['n_anchors']} anchors -> {t['pixels_per_inch']:.2f} px/in (conf={t['confidence']:.2f})")
