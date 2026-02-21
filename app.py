"""PreCheck Streamlit demo — side-by-side 3-condition inspection comparison."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PreCheck — Spatial Anchor Calibration",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = Path("data/results")
FRAMES_DIR = Path("data/frames")
DEPTH_DIR = Path("data/depth")
DETECTIONS_DIR = Path("data/detections")
MEASUREMENTS_DIR = Path("data/measurements")

CONDITIONS = {
    "baseline": "Baseline (Raw VLM)",
    "depth": "Depth-Augmented",
    "anchor_calibrated": "Anchor-Calibrated (PreCheck)",
}

CONDITION_DESCRIPTIONS = {
    "baseline": "Original image + spatial question only. No augmentation.",
    "depth": "Image + depth map visualization. Relative depth, no real-world scale.",
    "anchor_calibrated": "Full PreCheck pipeline: image + calibrated real-world measurements injected into prompt.",
}


# ---------------------------------------------------------------------------
# Data loading helpers (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def get_available_images() -> list[str]:
    """Return sorted list of image IDs that have at least one result."""
    if not RESULTS_DIR.exists():
        return []
    jsons = list(RESULTS_DIR.glob("*_baseline_*.json")) + list(RESULTS_DIR.glob("*_anchor_calibrated_*.json"))
    ids = sorted({p.name.split("_baseline_")[0].split("_anchor_calibrated_")[0] for p in jsons})
    return ids


@st.cache_data
def load_result(image_id: str, condition: str, vlm: str) -> dict | None:
    """Load a pre-computed VLM result JSON."""
    path = RESULTS_DIR / f"{image_id}_{condition}_{vlm}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_measurements(image_id: str) -> dict | None:
    path = MEASUREMENTS_DIR / f"{image_id}_measurements.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_image_file(path: str):
    """Load image for st.image display."""
    from PIL import Image
    return Image.open(path)


def find_image_file(image_id: str) -> str | None:
    """Find the source image file for an image_id."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = FRAMES_DIR / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None


def find_annotated_image(image_id: str) -> str | None:
    p = DETECTIONS_DIR / f"{image_id}_annotated.png"
    return str(p) if p.exists() else None


def find_depth_image(image_id: str) -> str | None:
    p = DEPTH_DIR / f"{image_id}_depth.png"
    return str(p) if p.exists() else None


def find_measured_image(image_id: str) -> str | None:
    p = MEASUREMENTS_DIR / f"{image_id}_measured.png"
    return str(p) if p.exists() else None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_pass_fail_badge(compliant: bool) -> str:
    if compliant:
        return "🟢 PASS"
    return "🔴 FAIL"


def render_measurements_panel(measurements: dict) -> None:
    """Render a structured measurements summary."""
    ppi = measurements.get("scale_pixels_per_inch", 0)
    conf = measurements.get("calibration_confidence", 0)

    col1, col2 = st.columns(2)
    col1.metric("Scale (px/inch)", f"{ppi:.2f}")
    col2.metric("Calibration Confidence", f"{conf:.0%}")

    stud_spacings = measurements.get("stud_spacings", [])
    if stud_spacings:
        st.markdown("**Stud Spacings (center-to-center)**")
        for i, s in enumerate(stud_spacings):
            badge = render_pass_fail_badge(s.get("compliant", True))
            st.markdown(f"  Bay {i+1}: **{s['inches']:.1f}\"** {badge}")

    elec_heights = measurements.get("electrical_box_heights", [])
    if elec_heights:
        st.markdown("**Electrical Box Heights**")
        for h in elec_heights:
            badge = render_pass_fail_badge(h.get("compliant", True))
            st.markdown(f"  Box {h['box_id']+1}: **{h['height_inches']:.1f}\"** to center {badge}")

    rebar = measurements.get("rebar_spacings", [])
    if rebar:
        st.markdown("**Rebar Spacings**")
        for i, s in enumerate(rebar):
            badge = render_pass_fail_badge(s.get("compliant", True))
            st.markdown(f"  Bay {i+1}: **{s['inches']:.1f}\"** {badge}")

    summary = measurements.get("summary", "")
    if summary:
        st.info(summary)


def render_response_panel(result: dict | None, condition: str) -> None:
    """Render VLM response for one condition."""
    if result is None:
        st.warning("No result available for this condition. Run the pipeline first.")
        return

    response = result.get("response", "")
    if response.startswith("ERROR"):
        st.error(response)
    else:
        st.markdown(response)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🏗️ PreCheck")
st.sidebar.markdown("*Spatial Anchor Calibration for Construction Inspection*")
st.sidebar.divider()

available_images = get_available_images()

if not available_images:
    st.sidebar.warning("No pre-computed results found in `data/results/`.")
    st.sidebar.markdown("Run the pipeline first:")
    st.sidebar.code("uv run python pipeline.py data/frames/your_image.jpg")
    selected_image_id = None
else:
    selected_image_id = st.sidebar.selectbox(
        "Select image",
        available_images,
        format_func=lambda x: x,
    )

vlm_choice = st.sidebar.selectbox("VLM", ["claude", "gpt4o"], index=0)

st.sidebar.divider()

# Live inference section
with st.sidebar.expander("⚡ Live Inference (slow)", expanded=False):
    uploaded = st.file_uploader("Upload a construction photo", type=["jpg", "jpeg", "png"])
    if uploaded and st.button("Run Pipeline"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        with st.spinner("Running PreCheck pipeline..."):
            from pipeline import run_pipeline
            out = run_pipeline(tmp_path, vlm=vlm_choice)
        st.success("Done! Refresh and select the new image.")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("PreCheck — Spatial Anchor Calibration for Construction Inspection")
st.markdown(
    "**How it works:** We detect known-dimension construction objects (studs, rebar, CMU blocks) "
    "as calibration anchors, compute a pixel-to-inch scale factor, extract real measurements, "
    "and inject those into the VLM prompt — enabling accurate spatial reasoning."
)
st.divider()

if selected_image_id is None:
    st.info("No results found. Add images to `data/frames/` and run the pipeline.")
    st.stop()

# ---------------------------------------------------------------------------
# Image header row
# ---------------------------------------------------------------------------

img_col1, img_col2, img_col3 = st.columns(3)

src_path = find_image_file(selected_image_id)
annotated_path = find_annotated_image(selected_image_id)
depth_path = find_depth_image(selected_image_id)
measured_path = find_measured_image(selected_image_id)

with img_col1:
    st.markdown("**Original Image**")
    if src_path:
        st.image(load_image_file(src_path), use_container_width=True)
    else:
        st.caption("Source image not found in data/frames/")

with img_col2:
    st.markdown("**Anchor Detection + Depth Map**")
    if depth_path:
        st.image(load_image_file(depth_path), use_container_width=True)
    elif annotated_path:
        st.image(load_image_file(annotated_path), use_container_width=True)
    else:
        st.caption("Depth/annotated image not found.")

with img_col3:
    st.markdown("**Measured Overlay**")
    if measured_path:
        st.image(load_image_file(measured_path), use_container_width=True)
    elif annotated_path:
        st.image(load_image_file(annotated_path), use_container_width=True)
    else:
        st.caption("Measurement overlay not found.")

st.divider()

# ---------------------------------------------------------------------------
# Measurements panel
# ---------------------------------------------------------------------------

measurements = load_measurements(selected_image_id)
if measurements:
    with st.expander("📐 Calibrated Spatial Measurements", expanded=True):
        render_measurements_panel(measurements)

st.divider()

# ---------------------------------------------------------------------------
# 3-condition comparison columns
# ---------------------------------------------------------------------------

st.subheader("Inspection Report — 3-Condition Comparison")

cond_cols = st.columns(3)

for col, (condition_key, condition_label) in zip(cond_cols, CONDITIONS.items()):
    with col:
        if condition_key == "anchor_calibrated":
            st.markdown(f"### ✅ {condition_label}")
        else:
            st.markdown(f"### {condition_label}")
        st.caption(CONDITION_DESCRIPTIONS[condition_key])
        st.divider()

        result = load_result(selected_image_id, condition_key, vlm_choice)
        render_response_panel(result, condition_key)

st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85em;'>"
    "PreCheck · Spatial Anchor Calibration · Built at IronsiteHackathon"
    "</div>",
    unsafe_allow_html=True,
)
