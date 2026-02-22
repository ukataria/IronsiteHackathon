"""Perspective rectification using YOLO anchor bounding boxes.

Computes a homography that maps the image to a frontal view of the target plane
by matching detected anchor centroids to their ideal grid positions.

Uses bounding-box CENTERS (not corners) as correspondences. AABB corners are
unreliable because they don't correspond to actual object corners when viewed at
an angle — only the centroid is stable across camera angles.

Requires ≥4 anchors of the same type and orientation (end_on excluded).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import get_image_id, load_image, load_json, save_image, save_json, setup_logger

logger = setup_logger("rectification")

# On-center spacing (x_oc_in, y_oc_in) for each type+orientation grid.
_OC_SPACING: dict[tuple[str, str], tuple[float, float]] = {
    ("brick",          "horizontal"): (8.0,   2.625),  # 7.625 + 3/8" mortar
    ("brick",          "vertical"):   (4.0,   8.0),    # soldier course
    ("cmu",            "horizontal"): (16.0,  8.0),    # 15.625 + 3/8" mortar
    ("cmu",            "vertical"):   (8.0,   16.0),
    ("electrical_box", "vertical"):   (2.0,   3.0),
    ("electrical_box", "horizontal"): (3.0,   2.0),
}

MIN_ANCHORS = 4  # homography needs ≥4 point correspondences


# ---------------------------------------------------------------------------
# Anchor selection + grouping
# ---------------------------------------------------------------------------


def _usable(anchors: list[dict], atype: str, orient: str) -> list[dict]:
    """Filter anchors to those matching type and non-end_on orientation."""
    return [a for a in anchors if a.get("type") == atype and a.get("orientation") == orient]


def _select_best_group(anchors: list[dict]) -> tuple[str, str] | None:
    """Return (anchor_type, orientation) with the most usable anchors (≥ MIN_ANCHORS)."""
    best: tuple[str, str] | None = None
    best_count = MIN_ANCHORS - 1
    for key in _OC_SPACING:
        count = len(_usable(anchors, *key))
        if count > best_count:
            best_count = count
            best = key
    return best


def _cluster_rows(anchors: list[dict], tol_px: float) -> list[list[dict]]:
    """Group anchors into rows by center_y proximity (greedy, sorted)."""
    if not anchors:
        return []
    sorted_a = sorted(anchors, key=lambda a: a["center_y"])
    rows: list[list[dict]] = [[sorted_a[0]]]
    for a in sorted_a[1:]:
        if a["center_y"] - rows[-1][-1]["center_y"] <= tol_px:
            rows[-1].append(a)
        else:
            rows.append([a])
    return rows


# ---------------------------------------------------------------------------
# Grid assignment + point correspondence
# ---------------------------------------------------------------------------


def _assign_grid(
    anchors: list[dict],
    atype: str,
    orient: str,
) -> list[tuple[dict, float, float]]:
    """
    Assign real-world center (x_in, y_in) to each anchor.

    Clusters into rows by center_y, sorts each row left→right by center_x,
    then maps (col, row) indices → real-world inches via OC spacing.
    Returns list of (anchor, x_real_in, y_real_in).
    """
    oc_x, oc_y = _OC_SPACING[(atype, orient)]
    med_h = float(np.median([a.get("pixel_height", 20) for a in anchors]))
    rows = _cluster_rows(anchors, max(med_h * 0.6, 8.0))

    result: list[tuple[dict, float, float]] = []
    for row_idx, row in enumerate(rows):
        for col_idx, a in enumerate(sorted(row, key=lambda a: a["center_x"])):
            result.append((a, col_idx * oc_x, row_idx * oc_y))
    return result


def _estimate_px_per_inch(anchors: list[dict]) -> float:
    """Derive px/inch from long-axis pixel size vs known real dimension."""
    estimates = [
        max(a.get("pixel_width_raw", 0), a.get("pixel_height", 0)) / a["real_width_inches"]
        for a in anchors
        if a.get("real_width_inches", 0) > 0
    ]
    return float(np.median(estimates)) if estimates else 20.0


def _build_correspondences(
    assigned: list[tuple[dict, float, float]],
    px_per_inch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (src_pts, dst_pts) using anchor CENTERS only.

    src: actual pixel centroids from YOLO (center_x, center_y)
    dst: ideal frontal-view positions at px_per_inch scale

    Returns float32 arrays of shape (N, 2).
    """
    src = [[a["center_x"], a["center_y"]] for a, _, _ in assigned]
    dst = [[x_in * px_per_inch, y_in * px_per_inch] for _, x_in, y_in in assigned]
    return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)


# ---------------------------------------------------------------------------
# Homography computation
# ---------------------------------------------------------------------------


def compute_homography(anchors_data: dict) -> dict | None:
    """
    Compute homography H mapping image pixels → rectified frontal-view pixels.

    Returns a metadata dict with H (3×3 list), output_w, output_h, px_per_inch,
    anchor_type, orientation — or None if insufficient anchors.
    """
    anchors = anchors_data.get("anchors", [])
    group = _select_best_group(anchors)
    if group is None:
        logger.warning("Rectification: not enough usable anchors (need ≥%d).", MIN_ANCHORS)
        return None

    atype, orient = group
    usable = _usable(anchors, atype, orient)
    px_per_inch = _estimate_px_per_inch(usable)
    assigned = _assign_grid(usable, atype, orient)
    src_pts, dst_pts = _build_correspondences(assigned, px_per_inch)

    H, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC,
        ransacReprojThreshold=max(px_per_inch * 0.25, 3.0),
    )
    if H is None:
        logger.error("Rectification: findHomography returned None.")
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    logger.info(
        f"Homography: {inliers}/{len(src_pts)} inliers  "
        f"{atype}/{orient}  {px_per_inch:.1f} px/in"
    )

    # Project image corners through H to determine output canvas extent
    iw = anchors_data.get("image_width", 1920)
    ih = anchors_data.get("image_height", 1080)
    img_corners = np.array([[0,0],[iw,0],[iw,ih],[0,ih]], dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(img_corners, H).reshape(-1, 2)

    all_pts = np.vstack([dst_pts, proj])
    pad = px_per_inch * 1.5                          # 1.5-inch border
    x_min, y_min = all_pts.min(axis=0) - pad
    x_max, y_max = all_pts.max(axis=0) + pad
    out_w = int(np.clip(x_max - x_min, 100, iw * 3))
    out_h = int(np.clip(y_max - y_min, 100, ih * 3))

    # Translate H so the canvas origin is (0, 0)
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    H_final = T @ H

    return {
        "H": H_final.tolist(),
        "anchor_type": atype,
        "orientation": orient,
        "px_per_inch": round(px_per_inch, 4),
        "output_w": out_w,
        "output_h": out_h,
        "inliers": inliers,
        "total_points": len(src_pts),
    }


# ---------------------------------------------------------------------------
# Image warping
# ---------------------------------------------------------------------------


def rectify_image(
    image_path: str,
    anchors_json_path: str,
    output_dir: str,
) -> dict:
    """
    Warp image to frontal view using YOLO anchor detections.

    Saves:
      {output_dir}/{image_id}_rectified.png
      {output_dir}/{image_id}_rectify_meta.json
    Returns metadata dict with 'status': 'ok' or 'error'.
    """
    image_id = get_image_id(image_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    meta_path = str(Path(output_dir) / f"{image_id}_rectify_meta.json")

    hom = compute_homography(load_json(anchors_json_path))
    if hom is None:
        meta = {"status": "error", "reason": "insufficient anchors", "image_id": image_id}
        save_json(meta, meta_path)
        return meta

    H = np.array(hom["H"], dtype=np.float64)
    img_bgr = cv2.cvtColor(load_image(image_path), cv2.COLOR_RGB2BGR)
    warped = cv2.warpPerspective(img_bgr, H, (hom["output_w"], hom["output_h"]), flags=cv2.INTER_LINEAR)

    out_img = str(Path(output_dir) / f"{image_id}_rectified.png")
    save_image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), out_img)

    meta = {**hom, "status": "ok", "image_id": image_id, "rectified_path": out_img}
    save_json(meta, meta_path)
    logger.info(f"{image_id}: rectified → {out_img} ({hom['output_w']}×{hom['output_h']})")
    return meta


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python src/rectification/rectify.py <image> <anchors.json> [output_dir]")
        sys.exit(1)

    img_p = sys.argv[1]
    anc_p = sys.argv[2]
    out_d = sys.argv[3] if len(sys.argv) > 3 else "data/rectifications"
    result = rectify_image(img_p, anc_p, out_d)
    if result["status"] == "ok":
        print(f"Rectified: {result['rectified_path']}")
        print(f"  type={result['anchor_type']}/{result['orientation']}  px/in={result['px_per_inch']}  size={result['output_w']}×{result['output_h']}")
        print(f"  inliers={result['inliers']}/{result['total_points']}")
    else:
        print(f"Failed: {result['reason']}")
