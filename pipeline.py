"""End-to-end PreCheck pipeline: image → anchors → depth → calibrate → measure → VLM."""

from __future__ import annotations

import sys
from pathlib import Path

from src.utils import get_image_id, setup_logger

logger = setup_logger("pipeline")

DATA_DIRS = {
    "detections": "data/detections",
    "depth": "data/depth",
    "calibrations": "data/calibrations",
    "measurements": "data/measurements",
    "results": "data/results",
}


def run_pipeline(
    image_path: str,
    run_reid: bool = False,
    device: str = "cpu",
    vlm: str = "claude",
    question: str = "What deficiencies exist in this construction work? Provide a full inspection report.",
    model_size: str = "small",
    skip_vlm: bool = False,
) -> dict:
    """
    Full pipeline for a single image.

    Stages:
      1. Anchor detection (GroundingDINO)
      2. Depth estimation (Depth Anything V2)
      3. Scale calibration (pure math)
      4. Spatial measurement
      5. VLM inspection (all 3 conditions)

    All intermediates saved to data/ subdirs.
    Returns final results dict with all condition outputs.
    """
    image_id = get_image_id(image_path)
    logger.info(f"=== Pipeline START: {image_id} ===")

    results = {"image_id": image_id, "image_path": image_path, "stages": {}}

    # -------------------------------------------------------------------------
    # Stage 1: Anchor Detection
    # -------------------------------------------------------------------------
    logger.info(f"[1/5] Anchor detection...")
    try:
        from src.anchors.detect import detect_anchors

        anchors = detect_anchors(image_path, DATA_DIRS["detections"], device=device)
        results["stages"]["anchors"] = {
            "status": "ok",
            "n_anchors": anchors["n_anchors"],
            "json_path": str(Path(DATA_DIRS["detections"]) / f"{image_id}_anchors.json"),
        }
        logger.info(f"  → {anchors['n_anchors']} anchors detected.")
    except Exception as e:
        logger.error(f"Anchor detection failed: {e}")
        results["stages"]["anchors"] = {"status": "error", "error": str(e)}
        anchors = {"n_anchors": 0, "anchors": [], "image_width": 1920, "image_height": 1080}

    # -------------------------------------------------------------------------
    # Stage 2: Depth Estimation
    # -------------------------------------------------------------------------
    logger.info(f"[2/5] Depth estimation...")
    depth_npy_path = str(Path(DATA_DIRS["depth"]) / f"{image_id}_depth.npy")
    depth_png_path = str(Path(DATA_DIRS["depth"]) / f"{image_id}_depth.png")
    try:
        from src.depth.estimate import estimate_depth

        depth_map = estimate_depth(image_path, DATA_DIRS["depth"], model_size=model_size, device=device)
        results["stages"]["depth"] = {
            "status": "ok",
            "shape": list(depth_map.shape),
            "npy_path": depth_npy_path,
            "png_path": depth_png_path,
        }
        logger.info(f"  → Depth map {depth_map.shape} saved.")
    except Exception as e:
        logger.error(f"Depth estimation failed: {e}")
        results["stages"]["depth"] = {"status": "error", "error": str(e)}
        depth_png_path = None

    # -------------------------------------------------------------------------
    # Stage 3: Scale Calibration
    # -------------------------------------------------------------------------
    logger.info(f"[3/5] Scale calibration...")
    anchors_json = str(Path(DATA_DIRS["detections"]) / f"{image_id}_anchors.json")
    cal_json = str(Path(DATA_DIRS["calibrations"]) / f"{image_id}_calibration.json")

    cal_ok = Path(anchors_json).exists() and Path(depth_npy_path).exists()
    if cal_ok:
        try:
            from src.calibration.calibrate import calibrate_image

            cal = calibrate_image(anchors_json, depth_npy_path, DATA_DIRS["calibrations"])
            results["stages"]["calibration"] = {
                "status": "ok",
                "pixels_per_inch": cal["primary_pixels_per_inch"],
                "confidence": cal["primary_confidence"],
                "json_path": cal_json,
            }
            logger.info(
                f"  → {cal['primary_pixels_per_inch']:.2f} px/in "
                f"(conf={cal['primary_confidence']:.2f})"
            )
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            results["stages"]["calibration"] = {"status": "error", "error": str(e)}
    else:
        logger.warning("Skipping calibration — missing anchors or depth file.")
        results["stages"]["calibration"] = {"status": "skipped"}

    # -------------------------------------------------------------------------
    # Stage 4: Spatial Measurement
    # -------------------------------------------------------------------------
    logger.info(f"[4/5] Spatial measurement...")
    measurements_json = str(Path(DATA_DIRS["measurements"]) / f"{image_id}_measurements.json")

    meas_ok = Path(anchors_json).exists() and Path(cal_json).exists()
    if meas_ok:
        try:
            from src.measurement.measure import extract_measurements

            meas = extract_measurements(
                anchors_json, cal_json, DATA_DIRS["measurements"], image_path=image_path
            )
            results["stages"]["measurement"] = {
                "status": "ok",
                "summary": meas["summary"],
                "json_path": measurements_json,
            }
            logger.info(f"  → {meas['summary']}")
        except Exception as e:
            logger.error(f"Measurement failed: {e}")
            results["stages"]["measurement"] = {"status": "error", "error": str(e)}
    else:
        logger.warning("Skipping measurement — missing calibration.")
        results["stages"]["measurement"] = {"status": "skipped"}

    # -------------------------------------------------------------------------
    # Stage 5: VLM Inspection (all 3 conditions)
    # -------------------------------------------------------------------------
    if not skip_vlm:
        logger.info(f"[5/5] VLM inspection (3 conditions)...")
        from src.vlm.clients import run_inspection

        conditions = ["baseline", "depth", "anchor_calibrated"]
        vlm_results = {}

        for condition in conditions:
            logger.info(f"  Running {condition} / {vlm}...")
            try:
                r = run_inspection(
                    image_path=image_path,
                    measurements_json_path=measurements_json,
                    depth_png_path=depth_png_path,
                    condition=condition,
                    vlm=vlm,
                    output_dir=DATA_DIRS["results"],
                    question=question,
                )
                vlm_results[condition] = {"status": "ok", "response": r["response"]}
            except Exception as e:
                logger.error(f"VLM {condition} failed: {e}")
                vlm_results[condition] = {"status": "error", "error": str(e)}

        results["vlm_results"] = vlm_results
    else:
        logger.info("[5/5] VLM skipped (skip_vlm=True).")

    logger.info(f"=== Pipeline DONE: {image_id} ===")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python pipeline.py <image_path> [--vlm claude|gpt4o] [--skip-vlm]")
        sys.exit(1)

    img = sys.argv[1]
    vlm_choice = "claude"
    skip = False

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--vlm" and i + 1 < len(sys.argv):
            vlm_choice = sys.argv[i + 1]
        if arg == "--skip-vlm":
            skip = True

    out = run_pipeline(img, vlm=vlm_choice, skip_vlm=skip)
    print("\n=== Pipeline Summary ===")
    for stage, info in out.get("stages", {}).items():
        print(f"  {stage}: {info.get('status', '?')}")
    if "vlm_results" in out:
        print("\n--- Anchor-Calibrated Inspection ---")
        print(out["vlm_results"].get("anchor_calibrated", {}).get("response", "N/A"))
