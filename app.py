"""PreCheck — Video-centric construction inspection demo."""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PreCheck",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }

:root {
    --bg:       #0B0F19;
    --s1:       #131929;
    --s2:       #1C2333;
    --b1:       #252D40;
    --b2:       #2E3A52;
    --amber:    #F59E0B;
    --adim:     #F59E0B22;
    --green:    #22C55E;
    --gdim:     #22C55E1A;
    --red:      #EF4444;
    --rdim:     #EF44441A;
    --blue:     #3B82F6;
    --bdim:     #3B82F61A;
    --tx:       #E2E8F0;
    --txm:      #7A8AA0;
    --txd:      #3A4A5E;
}

.stApp { background: var(--bg); }

[data-testid="stSidebar"] {
    background: var(--s1) !important;
    border-right: 1px solid var(--b1);
}

[data-testid="stMetric"] {
    background: var(--s2); border: 1px solid var(--b1);
    border-radius: 10px; padding: 1rem 1.2rem !important;
}
[data-testid="stMetric"] label { color: var(--txm) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: var(--amber) !important; font-size: 1.5rem !important; font-weight: 700; }

details { background: var(--s2) !important; border: 1px solid var(--b1) !important; border-radius: 10px !important; }
summary { color: var(--tx) !important; }

[data-testid="stSelectbox"] > div > div {
    background: var(--s2) !important; border-color: var(--b1) !important; color: var(--tx) !important;
}

/* Frame card grid */
.frame-card {
    background: var(--s2); border: 2px solid var(--b1);
    border-radius: 10px; overflow: hidden; cursor: pointer;
    transition: border-color 0.15s;
}
.frame-card.selected { border-color: var(--amber); box-shadow: 0 0 0 1px var(--amber), 0 4px 20px #F59E0B18; }
.frame-card.fail     { border-color: #EF444466; }
.frame-card.pass     { border-color: #22C55E44; }

.frame-label {
    padding: 6px 10px;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase;
    display: flex; align-items: center; justify-content: space-between;
}

.badge {
    display: inline-flex; align-items: center; gap: 3px;
    padding: 2px 7px; border-radius: 100px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.04em;
}
.bp { background: var(--gdim); color: var(--green); border: 1px solid #22C55E44; }
.bf { background: var(--rdim); color: var(--red);   border: 1px solid #EF444444; }
.bn { background: var(--s2);   color: var(--txm);   border: 1px solid var(--b1); }

/* Section headers */
.sh {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--txd); margin-bottom: 12px;
}

/* Condition cards */
.cc { background: var(--s2); border: 1px solid var(--b1); border-radius: 14px; overflow: hidden; }
.cc.win { border-color: var(--amber); box-shadow: 0 0 0 1px var(--amber), 0 8px 32px #F59E0B18; }
.cc-hd { padding: 14px 18px 12px; border-bottom: 1px solid var(--b1); }
.cc-title { font-size: 0.85rem; font-weight: 700; margin-bottom: 3px; }
.cc-desc { font-size: 0.72rem; color: var(--txm); line-height: 1.5; }
.cc-body { padding: 16px 18px; font-size: 0.84rem; color: var(--tx); line-height: 1.75; }

.wchip {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--amber); color: #000;
    font-size: 0.6rem; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 2px 7px; border-radius: 100px;
    margin-left: 7px; vertical-align: middle;
}

/* Measure rows */
.mr {
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 0; border-bottom: 1px solid var(--b1); font-size: 0.86rem;
}
.mr:last-child { border-bottom: none; }
.ml { color: var(--txm); }
.mv { font-weight: 600; color: var(--tx); font-family: 'JetBrains Mono', monospace; }

/* Img stage labels */
.il {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; padding: 8px 12px; margin-bottom: 4px;
    border-radius: 6px; display: inline-block;
}
.il-raw   { color: #94A3B8; background: #94A3B814; }
.il-det   { color: var(--blue); background: var(--bdim); }
.il-meas  { color: var(--green); background: var(--gdim); }

.sidebar-logo { font-size: 1.3rem; font-weight: 800; letter-spacing: -0.02em; color: var(--tx); }
.sidebar-logo span { color: var(--amber); }
.sidebar-sub { font-size: 0.73rem; color: var(--txm); margin-bottom: 14px; }

.step {
    display: flex; align-items: center; gap: 8px;
    background: var(--s2); border: 1px solid var(--b1);
    border-radius: 7px; padding: 7px 12px;
    font-size: 0.8rem; color: var(--txm); margin-bottom: 5px;
}
.sn {
    background: var(--amber); color: #000; font-size: 0.62rem; font-weight: 800;
    width: 17px; height: 17px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
}

.hero-label {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--adim); border: 1px solid var(--amber);
    color: var(--amber); font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 4px 10px; border-radius: 100px; margin-bottom: 10px;
}

.summary-box {
    padding: 12px 16px; margin-top: 14px;
    background: #F59E0B0D; border: 1px solid #F59E0B33;
    border-radius: 8px; font-size: 0.84rem; color: #CBD5E1;
}

.no-data {
    text-align: center; padding: 40px 24px; color: var(--txm);
    background: var(--s2); border: 1px dashed var(--b2);
    border-radius: 12px; font-size: 0.88rem;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--s1); }
::-webkit-scrollbar-thumb { background: var(--b2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR      = Path("data/results")
FRAMES_DIR       = Path("data/frames")
DEPTH_DIR        = Path("data/depth")
DETECTIONS_DIR   = Path("data/detections")
MEASUREMENTS_DIR = Path("data/measurements")

CONDITIONS = {
    "baseline":          "Baseline",
    "depth":             "Depth-Augmented",
    "anchor_calibrated": "PreCheck",
}

CONDITION_DESCRIPTIONS = {
    "baseline":          "Raw VLM — image + question only, no spatial context.",
    "depth":             "Adds depth map. Relative depth only, no real-world scale.",
    "anchor_calibrated": "Full pipeline: calibrated px→inch scale injected into prompt.",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data
def get_available_images() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    jsons = (
        list(RESULTS_DIR.glob("*_baseline_*.json"))
        + list(RESULTS_DIR.glob("*_anchor_calibrated_*.json"))
    )
    return sorted({
        p.name.split("_baseline_")[0].split("_anchor_calibrated_")[0]
        for p in jsons
    })


def group_by_video(image_ids: list[str]) -> dict[str, list[str]]:
    """Group image IDs by source video. Pattern: {video}_f{number}"""
    groups: dict[str, list[str]] = {}
    for img_id in image_ids:
        m = re.match(r"^(.+)_f(\d+)$", img_id)
        key = m.group(1) if m else img_id
        groups.setdefault(key, []).append(img_id)
    return groups


@st.cache_data
def load_result(image_id: str, condition: str, vlm: str) -> dict | None:
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
    from PIL import Image
    return Image.open(path)


def find_image_file(image_id: str) -> str | None:
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


def frame_compliance(image_id: str) -> str:
    """Return 'pass', 'fail', or 'unknown' for a frame."""
    m = load_measurements(image_id)
    if m is None:
        return "unknown"
    items = (
        [s.get("compliant", True) for s in m.get("stud_spacings", [])]
        + [s.get("compliant", True) for s in m.get("rebar_spacings", [])]
        + [h.get("compliant", True) for h in m.get("electrical_box_heights", [])]
    )
    if not items:
        return "unknown"
    return "pass" if all(items) else "fail"


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def badge(compliant: bool) -> str:
    if compliant:
        return '<span class="badge bp">✓ PASS</span>'
    return '<span class="badge bf">✗ FAIL</span>'


def render_measurements(measurements: dict) -> None:
    ppi  = measurements.get("scale_pixels_per_inch", 0)
    conf = measurements.get("calibration_confidence", 0)
    counts = measurements.get("element_counts", {})

    m1, m2, m3 = st.columns(3)
    m1.metric("Scale", f"{ppi:.1f} px/in")
    m2.metric("Confidence", f"{conf:.0%}")
    m3.metric("Anchors", str(sum(counts.values())))

    rows = ""
    for i, s in enumerate(measurements.get("stud_spacings", [])):
        rows += (
            f'<div class="mr"><span class="ml">Stud bay {i+1}</span>'
            f'<span><span class="mv">{s["inches"]:.1f}"</span> &nbsp;{badge(s.get("compliant", True))}</span></div>'
        )
    for i, s in enumerate(measurements.get("rebar_spacings", [])):
        rows += (
            f'<div class="mr"><span class="ml">Rebar bay {i+1}</span>'
            f'<span><span class="mv">{s["inches"]:.1f}"</span> &nbsp;{badge(s.get("compliant", True))}</span></div>'
        )
    for h in measurements.get("electrical_box_heights", []):
        rows += (
            f'<div class="mr"><span class="ml">Elec. box {h["box_id"]+1} height</span>'
            f'<span><span class="mv">{h["height_inches"]:.1f}"</span> &nbsp;{badge(h.get("compliant", True))}</span></div>'
        )
    if rows:
        st.markdown(f'<div style="margin-top:16px">{rows}</div>', unsafe_allow_html=True)

    summary = measurements.get("summary", "")
    if summary:
        st.markdown(f'<div class="summary-box">💡 {summary}</div>', unsafe_allow_html=True)


def render_condition(col, key: str, label: str, image_id: str, vlm: str) -> None:
    is_win = key == "anchor_calibrated"
    result = load_result(image_id, key, vlm)
    if result is None:
        body = "<em style='color:#3A4A5E'>No result — run the pipeline first.</em>"
    else:
        raw = result.get("response", "")
        body = f'<span style="color:#EF4444">{raw}</span>' if raw.startswith("ERROR") else raw

    chip = '<span class="wchip">★ PreCheck</span>' if is_win else ""
    title_color = "var(--amber)" if is_win else "var(--tx)"

    with col:
        st.markdown(
            f'<div class="cc {"win" if is_win else ""}">'
            f'  <div class="cc-hd">'
            f'    <div class="cc-title" style="color:{title_color}">{label}{chip}</div>'
            f'    <div class="cc-desc">{CONDITION_DESCRIPTIONS[key]}</div>'
            f'  </div>'
            f'  <div class="cc-body">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

if "selected_frame" not in st.session_state:
    st.session_state.selected_frame = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">Pre<span>Check</span></div>'
        '<div class="sidebar-sub">Construction Video Inspection · AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:#252D40; margin:10px 0 14px">', unsafe_allow_html=True)

    all_images = get_available_images()
    grouped = group_by_video(all_images)
    video_options = list(grouped.keys())

    if not video_options:
        st.warning("No processed frames found.")
        st.code("uv run python scripts/run_batch.py", language="bash")
        selected_video = None
    else:
        selected_video = st.selectbox(
            "Inspection video",
            video_options,
            format_func=lambda x: f"📹 {x}",
        )

    vlm_choice = st.selectbox(
        "Model",
        ["claude", "gpt4o"],
        format_func=lambda x: "Claude Sonnet 4.6" if x == "claude" else "GPT-4o",
    )

    st.markdown('<hr style="border-color:#252D40; margin:14px 0">', unsafe_allow_html=True)

    with st.expander("⚡ Live Inference", expanded=False):
        st.caption("Upload a video to extract + inspect frames, or upload a single image.")
        uploaded_video = st.file_uploader(
            "Video (.mp4 / .mov)",
            type=["mp4", "mov", "avi"],
            key="vid_upload",
        )
        uploaded_img = st.file_uploader(
            "Single image",
            type=["jpg", "jpeg", "png"],
            key="img_upload",
        )
        if uploaded_video and st.button("Extract & Inspect Video"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name
            with st.spinner("Extracting frames…"):
                from scripts.extract_frames import extract_frames
                saved = extract_frames(tmp_path, "data/frames")
            st.success(f"Extracted {len(saved)} frames.")
            with st.spinner("Running inspection pipeline…"):
                from pipeline import run_pipeline
                for fp in saved:
                    run_pipeline(fp, vlm=vlm_choice)
            st.success("Done! Select the video above.")
            st.cache_data.clear()
        if uploaded_img and st.button("Inspect Image"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(uploaded_img.read())
                tmp_path = tmp.name
            with st.spinner("Running pipeline…"):
                from pipeline import run_pipeline
                run_pipeline(tmp_path, vlm=vlm_choice)
            st.success("Done! Refresh to see result.")
            st.cache_data.clear()

    st.markdown('<hr style="border-color:#252D40; margin:14px 0">', unsafe_allow_html=True)
    st.markdown('<div class="sh">How it works</div>', unsafe_allow_html=True)
    for n, label in [
        ("1", "Extract frames from video"),
        ("2", "Detect calibration anchors"),
        ("3", "Compute px → inch scale"),
        ("4", "Extract real measurements"),
        ("5", "Inject into VLM prompt"),
        ("6", "Flag code violations"),
    ]:
        st.markdown(
            f'<div class="step"><span class="sn">{n}</span>{label}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero-label">⬡ PreCheck · IronsiteHackathon 2025</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<h1 style='font-size:2.2rem;font-weight:800;color:#E2E8F0;margin:0 0 6px'>"
    "VLMs that <span style='color:#F59E0B'>measure</span>, not guess.</h1>"
    "<p style='color:#7A8AA0;font-size:0.95rem;line-height:1.65;max-width:660px;margin:0 0 4px'>"
    "Drop in a construction video. PreCheck extracts frames, detects known-dimension objects "
    "as calibration anchors, converts pixels to inches, and runs an AI inspection on every frame — "
    "flagging code violations before they get buried behind drywall."
    "</p>",
    unsafe_allow_html=True,
)

if not video_options or selected_video is None:
    st.markdown(
        '<div class="no-data" style="margin-top:32px">📂 No processed videos yet.<br>'
        'Upload a video in the sidebar or run <code>uv run python scripts/run_batch.py</code></div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown('<hr style="border:none;border-top:1px solid #252D40;margin:20px 0">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Frame timeline — all frames from selected video
# ---------------------------------------------------------------------------

frames = grouped[selected_video]  # sorted list of image IDs

st.markdown(
    f'<div class="sh">Frame Timeline — {len(frames)} frame{"s" if len(frames) != 1 else ""} inspected</div>',
    unsafe_allow_html=True,
)

# Build columns for the frame grid (up to 6 per row)
PER_ROW = min(6, max(3, len(frames)))
rows_of_frames = [frames[i : i + PER_ROW] for i in range(0, len(frames), PER_ROW)]

for row in rows_of_frames:
    cols = st.columns(len(row), gap="small")
    for col, fid in zip(cols, row):
        status = frame_compliance(fid)
        src = find_image_file(fid)
        is_selected = st.session_state.selected_frame == fid

        badge_html = {
            "pass":    '<span class="badge bp">✓ PASS</span>',
            "fail":    '<span class="badge bf">✗ FAIL</span>',
            "unknown": '<span class="badge bn">• N/A</span>',
        }[status]

        # Short label: just frame number if patterned, else full id
        m = re.match(r"^.+_f(\d+)$", fid)
        short_label = f"f{int(m.group(1)):04d}" if m else fid

        with col:
            border_color = {
                "pass":    "#22C55E44",
                "fail":    "#EF444466",
                "unknown": "#252D40",
            }[status]
            if is_selected:
                border_color = "#F59E0B"
                box_shadow = "box-shadow: 0 0 0 1px #F59E0B, 0 4px 16px #F59E0B18;"
            else:
                box_shadow = ""

            st.markdown(
                f'<div style="background:#1C2333;border:2px solid {border_color};'
                f'border-radius:9px;overflow:hidden;{box_shadow}">',
                unsafe_allow_html=True,
            )
            if src:
                st.image(load_image_file(src), width="stretch")
            st.markdown(
                f'<div style="padding:6px 8px;display:flex;align-items:center;'
                f'justify-content:space-between;font-size:0.68rem;">'
                f'<span style="color:#7A8AA0;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:0.04em">{short_label}</span>{badge_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Inspect", key=f"sel_{fid}", use_container_width=True):
                st.session_state.selected_frame = fid
                st.rerun()

# ---------------------------------------------------------------------------
# Frame detail view
# ---------------------------------------------------------------------------

selected = st.session_state.selected_frame
# Default to first frame if none selected
if selected is None and frames:
    selected = frames[0]

if selected not in frames:
    selected = frames[0]

st.markdown('<hr style="border:none;border-top:1px solid #252D40;margin:24px 0 20px">', unsafe_allow_html=True)
st.markdown(
    f'<div class="sh">Inspection Detail — <span style="color:#E2E8F0;font-weight:700">{selected}</span></div>',
    unsafe_allow_html=True,
)

src_path        = find_image_file(selected)
annotated_path  = find_annotated_image(selected)
depth_path      = find_depth_image(selected)
measured_path   = find_measured_image(selected)

# Pipeline stage images
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown('<span class="il il-raw">① Raw Frame</span>', unsafe_allow_html=True)
    if src_path:
        st.image(load_image_file(src_path), width="stretch")
    else:
        st.caption("Frame not found.")

with c2:
    st.markdown('<span class="il il-det">② Anchor Detection</span>', unsafe_allow_html=True)
    if annotated_path:
        st.image(load_image_file(annotated_path), width="stretch")
    elif depth_path:
        st.image(load_image_file(depth_path), width="stretch")
    else:
        st.caption("Run pipeline to generate.")

with c3:
    st.markdown('<span class="il il-meas">③ Measurements</span>', unsafe_allow_html=True)
    if measured_path:
        st.image(load_image_file(measured_path), width="stretch")
    elif annotated_path:
        st.image(load_image_file(annotated_path), width="stretch")
    else:
        st.caption("Run pipeline to generate.")

# Measurements
measurements = load_measurements(selected)
if measurements:
    st.markdown('<hr style="border:none;border-top:1px solid #252D40;margin:20px 0 16px">', unsafe_allow_html=True)
    st.markdown('<div class="sh">Calibrated Spatial Measurements</div>', unsafe_allow_html=True)
    render_measurements(measurements)

# 3-condition comparison
st.markdown('<hr style="border:none;border-top:1px solid #252D40;margin:24px 0 16px">', unsafe_allow_html=True)
st.markdown('<div class="sh">AI Inspection — 3-Condition Comparison</div>', unsafe_allow_html=True)

cond_cols = st.columns(3, gap="medium")
for col, (ckey, clabel) in zip(cond_cols, CONDITIONS.items()):
    render_condition(col, ckey, clabel, selected, vlm_choice)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div style="text-align:center;color:#2E3A52;font-size:0.75rem;margin-top:48px;padding-bottom:24px">'
    'PreCheck · Spatial Anchor Calibration · IronsiteHackathon 2025'
    '</div>',
    unsafe_allow_html=True,
)
