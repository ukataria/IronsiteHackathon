"""Spatial measurement — extract real-world distances using calibrated scale."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import (
    draw_measurement_lines,
    get_image_id,
    load_image,
    load_json,
    save_image,
    save_json,
    setup_logger,
)

logger = setup_logger("measurement")

# ---------------------------------------------------------------------------
# Construction compliance tolerances (inches)
# ---------------------------------------------------------------------------

TOLERANCES: dict[str, float] = {
    "stud_spacing_oc": 0.5,        # 16" OC ± 0.5"
    "rebar_spacing": 0.75,         # 12" OC ± 0.75"
    "electrical_box_height": 1.0,  # 12" to center ± 1"
    "cmu_spacing": 0.5,
}

TARGET_VALUES: dict[str, float] = {
    "stud_spacing_oc": 16.0,
    "rebar_spacing": 12.0,
    "electrical_box_height": 12.0,
}


# ---------------------------------------------------------------------------
# Core measurement functions
# ---------------------------------------------------------------------------


def measure_element_spacing(
    elements: list[dict],
    pixels_per_inch: float,
    axis: str = "horizontal",
) -> list[dict]:
    """
    Compute center-to-center spacing between elements sorted along an axis.

    elements: list of anchor dicts with center_x, center_y
    axis: 'horizontal' (sort by center_x) or 'vertical' (sort by center_y)
    Returns list of spacing dicts with pixel and inch measurements.
    """
    if len(elements) < 2 or pixels_per_inch <= 0:
        return []

    key = "center_x" if axis == "horizontal" else "center_y"
    sorted_els = sorted(elements, key=lambda e: e.get(key, 0))

    spacings = []
    for i in range(len(sorted_els) - 1):
        a = sorted_els[i]
        b = sorted_els[i + 1]
        cx_a = a.get("center_x", 0)
        cy_a = a.get("center_y", 0)
        cx_b = b.get("center_x", 0)
        cy_b = b.get("center_y", 0)

        delta_px = abs(b.get(key, 0) - a.get(key, 0))
        inches = delta_px / pixels_per_inch

        spacings.append({
            "from_id": a.get("id", i),
            "to_id": b.get("id", i + 1),
            "cx_a": cx_a,
            "cy_a": cy_a,
            "cx_b": cx_b,
            "cy_b": cy_b,
            "pixels": round(delta_px, 2),
            "inches": round(inches, 2),
        })

    return spacings


def _check_compliance(
    inches: float,
    element_type: str,
) -> bool:
    """Return True if measurement is within tolerance for its standard value."""
    spacing_key = f"{element_type}_spacing_oc"
    if spacing_key not in TARGET_VALUES:
        return True  # unknown type — assume compliant
    target = TARGET_VALUES[spacing_key]
    tol = TOLERANCES.get(spacing_key, 0.5)
    return abs(inches - target) <= tol


def measure_height_from_reference(
    element: dict,
    reference_y_pixels: float,
    pixels_per_inch: float,
) -> float:
    """
    Measure element center height above a reference horizontal line.
    In image coords, y increases downward, so height = reference_y - center_y.
    Returns height in inches.
    """
    if pixels_per_inch <= 0:
        return 0.0
    center_y = element.get("center_y", reference_y_pixels)
    delta_px = reference_y_pixels - center_y  # positive = above reference
    return round(delta_px / pixels_per_inch, 2)


def measure_gap(
    element_a: dict,
    element_b: dict,
    pixels_per_inch: float,
    axis: str = "horizontal",
) -> float:
    """
    Measure edge-to-edge gap between two elements in inches.
    Uses bounding boxes, not centers.
    """
    if pixels_per_inch <= 0:
        return 0.0

    box_a = element_a.get("box_pixels", [0, 0, 0, 0])
    box_b = element_b.get("box_pixels", [0, 0, 0, 0])

    if axis == "horizontal":
        # gap between right edge of a and left edge of b (sorted left→right)
        if box_a[0] < box_b[0]:
            gap_px = max(0, box_b[0] - box_a[2])
        else:
            gap_px = max(0, box_a[0] - box_b[2])
    else:
        if box_a[1] < box_b[1]:
            gap_px = max(0, box_b[1] - box_a[3])
        else:
            gap_px = max(0, box_a[1] - box_b[3])

    return round(gap_px / pixels_per_inch, 2)


# ---------------------------------------------------------------------------
# Full measurement extraction for one image
# ---------------------------------------------------------------------------


def extract_measurements(
    anchors_json_path: str,
    calibration_json_path: str,
    output_dir: str,
    image_path: str | None = None,
) -> dict:
    """
    Extract all spatial measurements for one image using calibrated scale.

    Reads:
      anchors JSON (from detect_anchors)
      calibration JSON (from calibrate_image)

    Saves:
      {output_dir}/{image_id}_measurements.json
      {output_dir}/{image_id}_measured.png  (if image_path provided)

    Returns the measurements dict.
    """
    anchors_data = load_json(anchors_json_path)
    cal_data = load_json(calibration_json_path)

    image_id = anchors_data["image_id"]
    image_height = anchors_data.get("image_height", 1080)
    anchors = anchors_data.get("anchors", [])
    pixels_per_inch = cal_data.get("primary_pixels_per_inch", 0.0)
    confidence = cal_data.get("primary_confidence", 0.0)

    if pixels_per_inch <= 0:
        logger.warning(f"{image_id}: calibration has zero scale — measurements will be zero.")

    # Group anchors by type
    by_type: dict[str, list[dict]] = {}
    for a in anchors:
        by_type.setdefault(a["type"], []).append(a)

    # --- Stud spacings ---
    studs = by_type.get("stud", [])
    stud_spacings_raw = measure_element_spacing(studs, pixels_per_inch, axis="horizontal")
    stud_spacings = []
    for s in stud_spacings_raw:
        compliant = _check_compliance(s["inches"], "stud")
        stud_spacings.append({**s, "compliant": compliant})

    # --- Rebar spacings ---
    rebars = by_type.get("rebar", [])
    rebar_spacings_raw = measure_element_spacing(rebars, pixels_per_inch, axis="horizontal")
    rebar_spacings = []
    for s in rebar_spacings_raw:
        compliant = _check_compliance(s["inches"], "rebar")
        rebar_spacings.append({**s, "compliant": compliant})

    # --- Electrical box heights ---
    boxes = by_type.get("electrical_box", [])
    reference_y = float(image_height)  # bottom of image as floor proxy
    elec_heights = []
    for i, box in enumerate(boxes):
        height_in = measure_height_from_reference(box, reference_y, pixels_per_inch)
        target = TARGET_VALUES.get("electrical_box_height", 12.0)
        tol = TOLERANCES.get("electrical_box_height", 1.0)
        compliant = abs(height_in - target) <= tol
        elec_heights.append({
            "box_id": i,
            "anchor_id": box.get("id"),
            "height_inches": height_in,
            "compliant": compliant,
        })

    # --- Summary string ---
    summary_parts = []
    if stud_spacings:
        n_fail = sum(1 for s in stud_spacings if not s["compliant"])
        summary_parts.append(
            f"{len(stud_spacings) - n_fail}/{len(stud_spacings)} stud spacings compliant."
        )
        for s in stud_spacings:
            if not s["compliant"]:
                summary_parts.append(
                    f"  Non-compliant bay: {s['inches']:.1f}\" (expected 16.0\" OC ±0.5\")."
                )
    if rebar_spacings:
        n_fail = sum(1 for s in rebar_spacings if not s["compliant"])
        summary_parts.append(
            f"{len(rebar_spacings) - n_fail}/{len(rebar_spacings)} rebar spacings compliant."
        )
    if elec_heights:
        n_fail = sum(1 for h in elec_heights if not h["compliant"])
        summary_parts.append(
            f"{len(elec_heights) - n_fail}/{len(elec_heights)} electrical box heights compliant."
        )

    result = {
        "image_id": image_id,
        "scale_pixels_per_inch": round(pixels_per_inch, 4),
        "calibration_confidence": round(confidence, 4),
        "stud_spacings": stud_spacings,
        "rebar_spacings": rebar_spacings,
        "electrical_box_heights": elec_heights,
        "element_counts": {k: len(v) for k, v in by_type.items()},
        "summary": " ".join(summary_parts) if summary_parts else "No measurements extracted.",
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_json = str(Path(output_dir) / f"{image_id}_measurements.json")
    save_json(result, out_json)

    # Visual overlay if image provided
    if image_path:
        img = load_image(image_path)
        all_spacings = [
            {**s, "cx_a": s["cx_a"], "cy_a": s["cy_a"], "cx_b": s["cx_b"], "cy_b": s["cy_b"]}
            for s in stud_spacings + rebar_spacings
        ]
        img_measured = draw_measurement_lines(img, all_spacings)
        out_img = str(Path(output_dir) / f"{image_id}_measured.png")
        save_image(img_measured, out_img)

    logger.info(f"{image_id}: measurements saved → {out_json}")
    return result


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python src/measurement/measure.py <anchors.json> <calibration.json> [output_dir] [image.jpg]")
        sys.exit(1)

    anchors_p = sys.argv[1]
    cal_p = sys.argv[2]
    out_d = sys.argv[3] if len(sys.argv) > 3 else "data/measurements"
    img_p = sys.argv[4] if len(sys.argv) > 4 else None
    m = extract_measurements(anchors_p, cal_p, out_d, img_p)
    print(m["summary"])
