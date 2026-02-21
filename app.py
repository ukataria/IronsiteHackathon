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
# Custom CSS — warm courtroom palette
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', system-ui, sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }

:root {
    --bg:      #FFE4AD;
    --s1:      #FFF1D6;
    --s2:      #FFFBF2;
    --panel:   #EBDFD8;
    --b1:      #3B1F22;
    --b1a:     rgba(59,31,34,0.15);
    --b1b:     rgba(59,31,34,0.08);
    --gold:    #964804;
    --golddim: rgba(150,72,4,0.10);
    --goldbdr: rgba(150,72,4,0.30);
    --green:   #307351;
    --gdim:    rgba(48,115,81,0.10);
    --gbdr:    rgba(48,115,81,0.30);
    --red:     #C8283C;
    --rdim:    rgba(200,40,60,0.08);
    --rbdr:    rgba(200,40,60,0.28);
    --teal:    #069494;
    --tealdim: rgba(6,148,148,0.10);
    --tx:      #3B1F22;
    --txm:     #7C6058;
    --txd:     #C4A898;
}

.stApp { background: var(--bg); }

[data-testid="stSidebar"] {
    background: var(--s1) !important;
    border-right: 2px solid var(--b1a) !important;
}

[data-testid="stMetric"] {
    background: var(--s1) !important;
    border: 1px solid var(--b1a) !important;
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetric"] label {
    color: var(--txm) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: var(--gold) !important;
    font-size: 1.5rem !important;
    font-weight: 700;
    font-family: 'Space Mono', monospace !important;
}

details {
    background: var(--s1) !important;
    border: 1px solid var(--b1a) !important;
    border-radius: 8px !important;
    margin-top: 8px !important;
}
summary {
    color: var(--txm) !important;
    font-size: 0.76rem !important;
    padding: 9px 14px !important;
    cursor: pointer;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace !important;
}

[data-testid="stSelectbox"] > div > div {
    background: var(--s1) !important;
    border-color: var(--b1a) !important;
    color: var(--tx) !important;
}
[data-testid="stFileUploader"] {
    background: var(--s1) !important;
    border: 1px dashed var(--b1a) !important;
    border-radius: 10px !important;
}

/* Badges */
.badge {
    display: inline-flex; align-items: center; gap: 3px;
    padding: 2px 8px; border-radius: 5px;
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}
.bp { background: var(--gdim); color: var(--green); border: 1px solid var(--gbdr); }
.bf { background: var(--rdim); color: var(--red);   border: 1px solid var(--rbdr); }
.bn { background: var(--b1b);  color: var(--txm);   border: 1px solid var(--b1a); }

/* Section headers */
.sh {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--txd);
    margin-bottom: 14px;
    font-family: 'Space Mono', monospace;
}

/* Condition cards */
.cc {
    background: var(--s1);
    border: 1px solid var(--b1a);
    border-radius: 14px;
    overflow: hidden;
}
.cc.win {
    border-color: var(--gold);
    box-shadow: 0 0 0 1px var(--gold), 0 6px 28px rgba(150,72,4,0.10);
}
.cc-hd {
    padding: 15px 18px 13px;
    border-bottom: 1px solid var(--b1a);
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
}
.cc-title {
    font-size: 0.87rem;
    font-weight: 700;
    margin-bottom: 3px;
    display: flex; align-items: center; gap: 7px;
}
.cc-desc {
    font-size: 0.68rem;
    color: var(--txm);
    line-height: 1.5;
}
.cc-body { padding: 14px 18px 18px; }

/* Verdict badge */
.vb {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 7px;
    font-size: 0.68rem; font-weight: 800;
    letter-spacing: 0.07em; text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    flex-shrink: 0;
}
.vb-pass { background: var(--gdim); color: var(--green); border: 1px solid var(--gbdr); }
.vb-fail { background: var(--rdim); color: var(--red);   border: 1px solid var(--rbdr); }
.vb-unk  { background: var(--b1b);  color: var(--txm);   border: 1px solid var(--b1a); }
.vb-err  { background: var(--rdim); color: var(--red);   border: 1px solid var(--rbdr); }

/* Highlight rows */
.hl-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--b1b);
    font-size: 0.81rem;
    line-height: 1.55;
    color: var(--txm);
}
.hl-row:last-child { border-bottom: none; }
.hl-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--txd); flex-shrink: 0;
    margin-top: 7px;
}
.hl-dot.pass { background: var(--green); }
.hl-dot.fail { background: var(--red); }

.wchip {
    background: var(--gold); color: #fff;
    font-size: 0.57rem; font-weight: 800; letter-spacing: 0.07em;
    text-transform: uppercase; padding: 2px 7px; border-radius: 4px;
    vertical-align: middle; font-family: 'Space Mono', monospace;
}

/* Measure rows */
.mr {
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 0; border-bottom: 1px solid var(--b1b); font-size: 0.83rem;
}
.mr:last-child { border-bottom: none; }
.ml { color: var(--txm); }
.mv {
    font-weight: 700; color: var(--tx);
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}

/* Stage labels */
.il {
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 5px 11px; margin-bottom: 8px;
    border-radius: 5px; display: inline-block;
    font-family: 'Space Mono', monospace;
}
.il-raw  { color: var(--txm);  background: var(--b1b); }
.il-det  { color: var(--teal); background: var(--tealdim); }
.il-meas { color: var(--green); background: var(--gdim); }

.sidebar-logo {
    font-size: 1.35rem; font-weight: 800;
    letter-spacing: -0.02em; color: var(--tx);
}
.sidebar-logo span { color: var(--gold); }
.sidebar-sub { font-size: 0.69rem; color: var(--txm); margin-bottom: 14px; }

.step {
    display: flex; align-items: center; gap: 9px;
    background: var(--panel); border: 1px solid var(--b1a);
    border-radius: 7px; padding: 7px 11px;
    font-size: 0.76rem; color: var(--txm); margin-bottom: 5px;
}
.sn {
    background: var(--gold); color: #fff;
    font-size: 0.58rem; font-weight: 800;
    width: 18px; height: 18px; border-radius: 50%;
    display: inline-flex; align-items: center;
    justify-content: center; flex-shrink: 0;
    font-family: 'Space Mono', monospace;
}

.hero-label {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--golddim); border: 1px solid var(--goldbdr);
    color: var(--gold); font-size: 0.64rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 4px 12px; border-radius: 100px; margin-bottom: 12px;
    font-family: 'Space Mono', monospace;
}

.summary-box {
    padding: 11px 15px; margin-top: 14px;
    background: var(--golddim); border: 1px solid var(--goldbdr);
    border-radius: 7px; font-size: 0.82rem; color: var(--tx);
}

/* Upload hero card */
.upload-hero {
    background: var(--s1);
    border: 2px dashed var(--b1a);
    border-radius: 20px;
    padding: 56px 40px;
    text-align: center;
    max-width: 580px;
    margin: 0 auto;
}
.upload-icon {
    font-size: 2.8rem;
    margin-bottom: 16px;
    display: block;
}
.upload-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--tx);
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}
.upload-sub {
    font-size: 0.85rem;
    color: var(--txm);
    line-height: 1.65;
    margin-bottom: 28px;
}

.stButton > button {
    background: var(--gold) !important;
    border: none !important;
    color: #fff !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.2rem !important;
    transition: opacity 0.15s !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Override for small frame "Inspect" buttons */
.frame-btn > button {
    background: transparent !important;
    border: 1px solid var(--b1a) !important;
    color: var(--txm) !important;
    border-radius: 6px !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    padding: 0.3rem 0.6rem !important;
}
.frame-btn > button:hover {
    border-color: var(--gold) !important;
    color: var(--gold) !important;
    background: var(--golddim) !important;
    opacity: 1 !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--b1a); border-radius: 3px; }
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
    "baseline":          "Raw VLM — image only, no spatial context.",
    "depth":             "Depth map added. No real-world scale.",
    "anchor_calibrated": "Calibrated px→inch scale injected.",
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
# Response parsing
# ---------------------------------------------------------------------------

def _infer_verdict(text: str) -> str:
    t = text.lower()
    if re.search(r"overall[:\s]+pass|recommendation[:\s]+pass|overall.*\bpass\b", t):
        return "PASS"
    if re.search(r"overall[:\s]+fail|recommendation[:\s]+fail|overall.*\bfail\b", t):
        return "FAIL"
    fail_n = len(re.findall(r"\bfail\b|\bdeficien|\bviolation\b|\bnon.compliant\b|\bmissing\b", t))
    pass_n = len(re.findall(r"\bpass\b|\bcompliant\b|\bwithin tolerance\b|\bno deficien", t))
    if fail_n > pass_n:
        return "FAIL"
    if pass_n > fail_n and pass_n > 0:
        return "PASS"
    return "UNKNOWN"


def _extract_highlights(text: str, n: int = 4) -> list[tuple[str, str]]:
    highlights: list[tuple[str, str]] = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if len(highlights) >= n:
            break
        if re.match(r"^#+\s|^={3,}|^-{3,}", line):
            continue
        if line.isupper() and len(line) < 60:
            continue
        if re.match(r"^[-•*►✓✗✘]\s|^\d+[.)]\s", line):
            clean = re.sub(r"^[-•*►✓✗✘]\s*|\d+[.)]\s*", "", line).strip()
            if 12 < len(clean) < 220:
                dot = ("fail" if re.search(r"\bfail\b|\bdeficien|\bnon.compliant\b|\bmissing\b", clean.lower())
                       else "pass" if re.search(r"\bpass\b|\bcompliant\b|\bwithin\b", clean.lower())
                       else "")
                highlights.append((dot, clean))
        elif re.search(r"(spacing|height|stud|rebar|box|nail|plate|clear|gap|compliant|fail|pass)", line.lower()):
            if 20 < len(line) < 200:
                dot = ("fail" if re.search(r"\bfail\b|\bdeficien|\bnon.compliant\b", line.lower())
                       else "pass" if re.search(r"\bpass\b|\bcompliant\b", line.lower())
                       else "")
                highlights.append((dot, line))
    if not highlights:
        for line in lines[:8]:
            if len(line) > 30 and not line.startswith("#"):
                highlights.append(("", line[:200]))
                if len(highlights) >= 3:
                    break
    return highlights[:n]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def badge(compliant: bool) -> str:
    return '<span class="badge bp">✓ Pass</span>' if compliant else '<span class="badge bf">✗ Fail</span>'


def render_measurements(measurements: dict) -> None:
    ppi   = measurements.get("scale_pixels_per_inch", 0)
    conf  = measurements.get("calibration_confidence", 0)
    counts = measurements.get("element_counts", {})

    m1, m2, m3 = st.columns(3)
    m1.metric("Scale", f"{ppi:.1f} px/in")
    m2.metric("Confidence", f"{conf:.0%}")
    m3.metric("Anchors", str(sum(counts.values())))

    rows = ""
    for i, s in enumerate(measurements.get("stud_spacings", [])):
        rows += (
            f'<div class="mr"><span class="ml">Stud bay {i+1}</span>'
            f'<span><span class="mv">{s["inches"]:.1f}"</span>&nbsp;{badge(s.get("compliant", True))}</span></div>'
        )
    for i, s in enumerate(measurements.get("rebar_spacings", [])):
        rows += (
            f'<div class="mr"><span class="ml">Rebar bay {i+1}</span>'
            f'<span><span class="mv">{s["inches"]:.1f}"</span>&nbsp;{badge(s.get("compliant", True))}</span></div>'
        )
    for h in measurements.get("electrical_box_heights", []):
        rows += (
            f'<div class="mr"><span class="ml">Box {h["box_id"]+1} height</span>'
            f'<span><span class="mv">{h["height_inches"]:.1f}"</span>&nbsp;{badge(h.get("compliant", True))}</span></div>'
        )
    if rows:
        st.markdown(f'<div style="margin-top:14px">{rows}</div>', unsafe_allow_html=True)

    summary = measurements.get("summary", "")
    if summary:
        st.markdown(f'<div class="summary-box">⬡ {summary}</div>', unsafe_allow_html=True)


def render_condition(col, key: str, label: str, image_id: str, vlm: str) -> None:
    is_win = key == "anchor_calibrated"
    result = load_result(image_id, key, vlm)

    win_class   = "win" if is_win else ""
    chip        = '<span class="wchip">★ PreCheck</span>' if is_win else ""
    title_color = "var(--gold)" if is_win else "var(--tx)"

    with col:
        if result is None:
            st.markdown(
                f'<div class="cc {win_class}">'
                f'  <div class="cc-hd"><div>'
                f'    <div class="cc-title" style="color:{title_color}">{label} {chip}</div>'
                f'    <div class="cc-desc">{CONDITION_DESCRIPTIONS[key]}</div>'
                f'  </div></div>'
                f'  <div class="cc-body" style="color:var(--txd);font-style:italic;font-size:0.81rem">'
                f'    No result — run the pipeline first.'
                f'  </div></div>',
                unsafe_allow_html=True,
            )
            return

        response = result.get("response", "")

        if response.startswith("ERROR"):
            vb_html    = '<span class="vb vb-err">⚠ Error</span>'
            highlights = [("", response[:200])]
            full_text  = response
        else:
            verdict = _infer_verdict(response)
            vb_html = {
                "PASS":    '<span class="vb vb-pass">✓ Pass</span>',
                "FAIL":    '<span class="vb vb-fail">✗ Fail</span>',
                "UNKNOWN": '<span class="vb vb-unk">— Unknown</span>',
            }[verdict]
            highlights = _extract_highlights(response)
            full_text  = response

        hl_rows = "".join(
            f'<div class="hl-row"><div class="hl-dot {dot}"></div><span>{text}</span></div>'
            for dot, text in highlights
        ) or '<div style="color:var(--txd);font-size:0.79rem;font-style:italic">No structured findings.</div>'

        st.markdown(
            f'<div class="cc {win_class}">'
            f'  <div class="cc-hd">'
            f'    <div>'
            f'      <div class="cc-title" style="color:{title_color}">{label} {chip}</div>'
            f'      <div class="cc-desc">{CONDITION_DESCRIPTIONS[key]}</div>'
            f'    </div>'
            f'    {vb_html}'
            f'  </div>'
            f'  <div class="cc-body">{hl_rows}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if full_text and len(full_text) > 80:
            with st.expander("Full report"):
                st.markdown(full_text)


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
        '<div class="sidebar-sub">Construction Inspection · AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:var(--b1a); margin:10px 0 16px">', unsafe_allow_html=True)

    all_images    = get_available_images()
    grouped       = group_by_video(all_images)
    video_options = list(grouped.keys())

    if video_options:
        selected_video = st.selectbox(
            "Inspection video",
            video_options,
            format_func=lambda x: f"📹 {x}",
        )
    else:
        selected_video = None

    vlm_choice = st.selectbox(
        "Model",
        ["claude", "gpt4o"],
        format_func=lambda x: "Claude Sonnet 4.6" if x == "claude" else "GPT-4o",
    )

    st.markdown('<hr style="border-color:var(--b1a); margin:14px 0">', unsafe_allow_html=True)
    st.markdown('<div class="sh">How it works</div>', unsafe_allow_html=True)
    for n, lbl in [
        ("1", "Extract frames from video"),
        ("2", "Detect calibration anchors"),
        ("3", "Compute px → inch scale"),
        ("4", "Extract real measurements"),
        ("5", "Inject into VLM prompt"),
        ("6", "Flag code violations"),
    ]:
        st.markdown(
            f'<div class="step"><span class="sn">{n}</span>{lbl}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Hero header (always shown)
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-label">⬡ PreCheck · IronsiteHackathon 2025</div>', unsafe_allow_html=True)
st.markdown(
    "<h1 style='font-size:2.1rem;font-weight:800;color:#3B1F22;margin:0 0 6px;letter-spacing:-0.02em'>"
    "VLMs that <span style='color:#964804'>measure</span>, not guess.</h1>"
    "<p style='color:#7C6058;font-size:0.91rem;line-height:1.7;max-width:600px;margin:0'>"
    "Known-dimension objects — studs, rebar, CMU — calibrate pixel-to-inch scale. "
    "PreCheck injects real measurements into the VLM prompt so it reasons over inches, not visual guesses."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown('<hr style="border:none;border-top:1px solid var(--b1a);margin:22px 0">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload-first flow — show upload UI until videos are processed
# ---------------------------------------------------------------------------

if not video_options:
    st.markdown(
        '<div class="upload-hero">'
        '  <span class="upload-icon">🎥</span>'
        '  <div class="upload-title">Upload a construction video</div>'
        '  <div class="upload-sub">'
        '    PreCheck extracts frames, detects calibration anchors, converts pixels to inches, '
        '    and runs a 3-condition AI inspection on every frame.'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _, upload_col, _ = st.columns([1, 2, 1])
    with upload_col:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        uploaded_video = st.file_uploader(
            "Video file",
            type=["mp4", "mov", "avi"],
            label_visibility="collapsed",
        )
        uploaded_img = st.file_uploader(
            "Or upload a single image",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )

        if uploaded_video:
            if st.button("Extract & Inspect Video", use_container_width=True):
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
                st.success("Done!")
                st.cache_data.clear()
                st.rerun()

        if uploaded_img:
            if st.button("Inspect Image", use_container_width=True):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(uploaded_img.read())
                    tmp_path = tmp.name
                with st.spinner("Running pipeline…"):
                    from pipeline import run_pipeline
                    run_pipeline(tmp_path, vlm=vlm_choice)
                st.success("Done!")
                st.cache_data.clear()
                st.rerun()

        st.markdown(
            '<div style="text-align:center;margin-top:16px;color:var(--txd);'
            'font-size:0.74rem;font-family:Space Mono,monospace">'
            'or run: uv run python scripts/run_batch.py'
            '</div>',
            unsafe_allow_html=True,
        )
    st.stop()

# ---------------------------------------------------------------------------
# Frame timeline (only shown when videos exist)
# ---------------------------------------------------------------------------

frames = grouped[selected_video]

st.markdown(
    f'<div class="sh">Frame Timeline — {len(frames)} frame{"s" if len(frames) != 1 else ""} · {selected_video}</div>',
    unsafe_allow_html=True,
)

PER_ROW = min(6, max(3, len(frames)))
rows_of_frames = [frames[i : i + PER_ROW] for i in range(0, len(frames), PER_ROW)]

for row in rows_of_frames:
    cols = st.columns(len(row), gap="small")
    for col, fid in zip(cols, row):
        status     = frame_compliance(fid)
        src        = find_image_file(fid)
        is_selected = st.session_state.selected_frame == fid

        badge_html = {
            "pass":    '<span class="badge bp">✓ Pass</span>',
            "fail":    '<span class="badge bf">✗ Fail</span>',
            "unknown": '<span class="badge bn">· N/A</span>',
        }[status]

        m = re.match(r"^.+_f(\d+)$", fid)
        short_label = f"f{int(m.group(1)):04d}" if m else fid

        border_color = {
            "pass":    "var(--gbdr)",
            "fail":    "var(--rbdr)",
            "unknown": "var(--b1a)",
        }[status]
        extra = "box-shadow:0 0 0 2px var(--gold);" if is_selected else ""
        if is_selected:
            border_color = "var(--gold)"

        with col:
            st.markdown(
                f'<div style="background:var(--s1);border:2px solid {border_color};'
                f'border-radius:10px;overflow:hidden;{extra}">',
                unsafe_allow_html=True,
            )
            if src:
                st.image(load_image_file(src), width="stretch")
            st.markdown(
                f'<div style="padding:6px 8px;display:flex;align-items:center;'
                f'justify-content:space-between;">'
                f'<span style="color:var(--txm);font-size:0.62rem;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.06em;font-family:Space Mono,monospace">'
                f'{short_label}</span>{badge_html}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="frame-btn">', unsafe_allow_html=True)
            if st.button("Inspect", key=f"sel_{fid}", use_container_width=True):
                st.session_state.selected_frame = fid
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Frame detail
# ---------------------------------------------------------------------------

selected = st.session_state.selected_frame
if selected is None and frames:
    selected = frames[0]
if selected not in frames:
    selected = frames[0]

st.markdown('<hr style="border:none;border-top:1px solid var(--b1a);margin:28px 0 22px">', unsafe_allow_html=True)
st.markdown(
    f'<div class="sh">Inspection Detail — <span style="color:var(--tx);font-weight:700">{selected}</span></div>',
    unsafe_allow_html=True,
)

src_path       = find_image_file(selected)
annotated_path = find_annotated_image(selected)
depth_path     = find_depth_image(selected)
measured_path  = find_measured_image(selected)

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

measurements = load_measurements(selected)
if measurements:
    st.markdown('<hr style="border:none;border-top:1px solid var(--b1a);margin:22px 0 18px">', unsafe_allow_html=True)
    st.markdown('<div class="sh">Calibrated Spatial Measurements</div>', unsafe_allow_html=True)
    render_measurements(measurements)

st.markdown('<hr style="border:none;border-top:1px solid var(--b1a);margin:28px 0 18px">', unsafe_allow_html=True)
st.markdown('<div class="sh">AI Inspection — 3-Condition Comparison</div>', unsafe_allow_html=True)

cond_cols = st.columns(3, gap="medium")
for col, (ckey, clabel) in zip(cond_cols, CONDITIONS.items()):
    render_condition(col, ckey, clabel, selected, vlm_choice)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div style="text-align:center;color:var(--txd);font-size:0.7rem;'
    'margin-top:52px;padding-bottom:28px;font-family:Space Mono,monospace">'
    'PreCheck · Spatial Anchor Calibration · IronsiteHackathon 2025'
    '</div>',
    unsafe_allow_html=True,
)
