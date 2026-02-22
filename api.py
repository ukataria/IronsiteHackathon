"""Flask API — serves pipeline outputs to the React frontend."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, send_file, abort, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATA_DIRS = {
    "frames": Path("data/frames"),
    "detections": Path("data/detections"),
    "depth": Path("data/depth"),
    "calibrations": Path("data/calibrations"),
    "measurements": Path("data/measurements"),
    "results": Path("data/results"),
}

# Backend anchor type → frontend type
ANCHOR_TYPE_MAP: dict[str, str] = {
    "stud": "stud",
    "rebar": "rebar",
    "cmu": "cmu",
    "electrical_box": "elec_box",
    "brick": "brick",
    "door": "stud",
    "hardhat": "stud",
}

# Measurement display metadata per anchor type
MEASUREMENT_META: dict[str, dict] = {
    "stud":  {"expected": 16.0, "tolerance": 0.5,  "prefix": "Stud"},
    "rebar": {"expected": 12.0, "tolerance": 0.75, "prefix": "Rebar"},
    "brick": {"expected": 8.0,  "tolerance": 0.25, "prefix": "Brick H"},
    "cmu":   {"expected": 16.0, "tolerance": 0.5,  "prefix": "CMU"},
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _get_image_ids() -> list[str]:
    """Return sorted list of image IDs that have been processed (measurements or results)."""
    ids: set[str] = set()

    # Prefer measurements dir — populated even without VLM (--skip-vlm)
    if DATA_DIRS["measurements"].exists():
        for p in DATA_DIRS["measurements"].iterdir():
            if p.name.endswith("_measurements.json"):
                ids.add(p.name[: -len("_measurements.json")])

    # Also pull any VLM result files
    if DATA_DIRS["results"].exists():
        for p in DATA_DIRS["results"].iterdir():
            if p.suffix != ".json":
                continue
            stem = p.stem
            for cond in ("_anchor_calibrated_", "_baseline_", "_depth_"):
                if cond in stem:
                    ids.add(stem.split(cond)[0])
                    break

    return sorted(ids)


def _build_frame_data(image_id: str) -> dict:
    anchors_data = _load(DATA_DIRS["detections"] / f"{image_id}_anchors.json")
    cal_data     = _load(DATA_DIRS["calibrations"] / f"{image_id}_calibration.json")
    meas_data    = _load(DATA_DIRS["measurements"] / f"{image_id}_measurements.json")

    image_width  = anchors_data.get("image_width", 1920)
    image_height = anchors_data.get("image_height", 1080)
    raw_anchors  = anchors_data.get("anchors", [])

    # Anchors
    anchors = [
        {
            "id": a["id"],
            "type": ANCHOR_TYPE_MAP.get(a["type"], "stud"),
            "box": a["box_pixels"],
            "confidence": a["confidence"],
            "label": a["type"].replace("_", " "),
        }
        for a in raw_anchors
    ]

    # Calibration
    ppi        = cal_data.get("primary_pixels_per_inch", 0.0)
    confidence = cal_data.get("primary_confidence", 0.0)
    types_seen = {a["type"] for a in raw_anchors}

    if "brick" in types_seen:
        method, method_label = "brick_7.625in", "Brick\n(7.625\" known width)"
    elif "cmu" in types_seen:
        method, method_label = "cmu_15.625in", "CMU Block\n(15.625\" known width)"
    elif "electrical_box" in types_seen:
        method, method_label = "elec_box_2in", "Elec Box\n(2\" known width)"
    else:
        method, method_label = "unknown", "Unknown"

    calibration = {
        "pixels_per_inch": ppi,
        "confidence": confidence,
        "method": method,
        "method_label": method_label,
    }

    # Measurements
    measurements: list[dict] = []

    def _add_spacings(spacings: list[dict], mtype: str) -> None:
        meta = MEASUREMENT_META[mtype]
        for s in spacings:
            measurements.append({
                "id": f"{mtype}_{s['from_id']}_{s['to_id']}",
                "type": "stud_spacing",
                "label": f"{meta['prefix']} {s['from_id'] + 1}→{s['to_id'] + 1}",
                "from_anchor": s.get("from_id", 0),
                "to_anchor":   s.get("to_id", 1),
                "cx_a": s["cx_a"], "cy_a": s["cy_a"],
                "cx_b": s["cx_b"], "cy_b": s["cy_b"],
                "inches":    s["inches"],
                "compliant": s["compliant"],
                "expected":  meta["expected"],
                "tolerance": meta["tolerance"],
                "severity":  "ok" if s["compliant"] else "critical",
            })

    _add_spacings(meas_data.get("stud_spacings", []),    "stud")
    _add_spacings(meas_data.get("rebar_spacings", []),   "rebar")
    _add_spacings(meas_data.get("brick_h_spacings", []), "brick")
    _add_spacings(meas_data.get("cmu_spacings", []),     "cmu")

    # Electrical box heights
    anchor_by_id = {a["id"]: a for a in raw_anchors}
    for h in meas_data.get("electrical_box_heights", []):
        aid = h.get("anchor_id")
        anc = anchor_by_id.get(aid, {})
        cx  = anc.get("center_x", image_width / 2)
        cy  = anc.get("center_y", image_height / 2)
        measurements.append({
            "id":          f"elec_{h['box_id']}",
            "type":        "box_height",
            "label":       f"Box {h['box_id'] + 1} Height",
            "from_anchor": aid or 0,
            "to_anchor":   aid or 0,
            "cx_a": cx, "cy_a": image_height,  # floor reference = bottom of image
            "cx_b": cx, "cy_b": cy,
            "inches":    h["height_inches"],
            "compliant": h["compliant"],
            "expected":  12.0,
            "tolerance": 1.0,
            "severity":  "ok" if h["compliant"] else "warning",
        })

    return {
        "image_id":     image_id,
        "image_width":  image_width,
        "image_height": image_height,
        "calibration":  calibration,
        "anchors":      anchors,
        "measurements": measurements,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/frames")
def list_frames():
    return jsonify(_get_image_ids())


@app.route("/api/frame/<image_id>")
def get_frame(image_id: str):
    return jsonify(_build_frame_data(image_id))


@app.route("/api/image/<image_id>")
def get_image(image_id: str):
    for ext in (".jpg", ".jpeg", ".png"):
        p = DATA_DIRS["frames"] / f"{image_id}{ext}"
        if p.exists():
            return send_file(str(p.resolve()))
    abort(404)


@app.route("/api/image/<image_id>/annotated")
def get_annotated(image_id: str):
    p = DATA_DIRS["detections"] / f"{image_id}_annotated.png"
    if p.exists():
        return send_file(str(p.resolve()))
    abort(404)


@app.route("/api/image/<image_id>/depth")
def get_depth(image_id: str):
    p = DATA_DIRS["depth"] / f"{image_id}_depth.png"
    if p.exists():
        return send_file(str(p.resolve()))
    abort(404)


@app.route("/api/vlm/<image_id>/<condition>")
def get_vlm(image_id: str, condition: str):
    for vlm in ("claude", "gpt4o"):
        p = DATA_DIRS["results"] / f"{image_id}_{condition}_{vlm}.json"
        if p.exists():
            data = json.loads(p.read_text())
            return jsonify({"response": data.get("response", ""), "vlm": vlm})
    return jsonify({"response": "", "vlm": "none"})


@app.route("/api/chat/<image_id>", methods=["POST"])
def chat(image_id: str):
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    history: list[dict] = body.get("history", [])

    if not question:
        return jsonify({"response": ""}), 400

    # Locate frame image and depth map
    image_path: str | None = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = DATA_DIRS["frames"] / f"{image_id}{ext}"
        if p.exists():
            image_path = str(p)
            break
    depth_p = DATA_DIRS["depth"] / f"{image_id}_depth.png"
    depth_image_path = str(depth_p) if depth_p.exists() else None

    # Load all pipeline context for this frame
    meas_data    = _load(DATA_DIRS["measurements"] / f"{image_id}_measurements.json")
    anchors_data = _load(DATA_DIRS["detections"]   / f"{image_id}_anchors.json")

    from src.vlm.prompts import (
        ANCHOR_CALIBRATED_PROMPT_TEMPLATE,
        ELEMENT_DIMENSIONS_BLOCK,
        STANDARDS_BLOCK,
        build_calibration_summary,
        build_reference_objects_block,
        format_measurements_block,
    )

    # Bounding box summary so Claude knows exact pixel locations
    bbox_lines = []
    for a in anchors_data.get("anchors", []):
        b = a["box_pixels"]
        bbox_lines.append(
            f"  [{a['type']}] id={a['id']} conf={a['confidence']:.2f} "
            f"box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] "
            f"center=({a['center_x']:.0f},{a['center_y']:.0f}) "
            f"real_width={a['real_width_inches']}in"
        )
    bbox_block = "━━━ DETECTED ANCHOR BOUNDING BOXES (image pixels) ━━━\n"
    bbox_block += "\n".join(bbox_lines) if bbox_lines else "  None detected."

    # Conversation history text
    history_block = "\n\n".join(
        f"User: {h['question']}\nAssistant: {h['answer']}"
        for h in history[-4:] if h.get("question")
    )

    prompt = ANCHOR_CALIBRATED_PROMPT_TEMPLATE.format(
        question=question,
        dimensions_block=ELEMENT_DIMENSIONS_BLOCK,
        calibration_summary=build_calibration_summary(meas_data),
        measurements_block=format_measurements_block(meas_data),
        standards_block=STANDARDS_BLOCK,
        reference_objects_block=build_reference_objects_block(meas_data),
    )
    if history_block:
        prompt = f"CONVERSATION SO FAR:\n{history_block}\n\n---\n\n{prompt}"
    if bbox_lines:
        prompt = f"{bbox_block}\n\n{prompt}"

    # Append brevity instruction to follow-up questions
    prompt += "\n\nAnswer concisely — 3-5 sentences or a short table. No full re-report."

    try:
        from src.vlm.clients import call_claude
        response = call_claude(
            prompt,
            image_path=image_path,
            secondary_image_path=depth_image_path,
            cache_key=f"chat:{image_id}:{hash(prompt)}",
        )
    except Exception as e:
        response = f"Error calling Claude: {e}"

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
