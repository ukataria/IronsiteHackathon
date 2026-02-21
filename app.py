"""Ghost Blueprint — Streamlit Demo App."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image as PILImage

from src.utils import (
    FRAMES_DIR, DEPTH_DIR, SEGMENTS_DIR, SCENES_DIR,
    OVERLAYS_DIR, COMPOSITES_DIR, ensure_dirs,
)
from src.composite.blend import composite_layers, make_side_by_side, LAYER_ORDER

ensure_dirs()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ghost Blueprint",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #e0f0ff; }
    .subtitle { font-size: 1rem; color: #8ab0cc; margin-top: -10px; }
    .scene-box { background: #0d1b2a; border-radius: 8px; padding: 16px; font-size: 0.85rem; color: #cce; }
    .layer-badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
                   background: #1a3a5c; color: #90d0ff; font-size: 0.78rem; margin: 2px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def list_frames() -> list[str]:
    """List all available frames in data/frames/."""
    paths = sorted(FRAMES_DIR.glob("*.png"))
    return [p.stem for p in paths]


@st.cache_data
def load_pil(path: Path) -> PILImage.Image | None:
    if path.exists():
        return PILImage.open(str(path)).convert("RGB")
    return None


@st.cache_data
def load_scene_json(stem: str) -> dict | None:
    p = SCENES_DIR / f"{stem}_scene.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


@st.cache_data
def load_future_json(stem: str) -> dict | None:
    p = SCENES_DIR / f"{stem}_future.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def get_available_layers(stem: str) -> list[str]:
    """Return layer names that have both an overlay and a mask."""
    layers = []
    for layer_name in LAYER_ORDER:
        overlay = OVERLAYS_DIR / f"{stem}_{layer_name}.png"
        mask = SEGMENTS_DIR / f"{stem}_{layer_name}.png"
        if overlay.exists() and mask.exists():
            layers.append(layer_name)
    return layers


def composite_on_the_fly(
    stem: str,
    active_layers: list[str],
    layer_alphas: dict[str, float],
) -> PILImage.Image | None:
    """Re-composite with current slider values and return as PIL image."""
    frame_path = FRAMES_DIR / f"{stem}.png"
    if not frame_path.exists():
        return None

    overlay_paths = {
        ln: OVERLAYS_DIR / f"{stem}_{ln}.png"
        for ln in active_layers
        if (OVERLAYS_DIR / f"{stem}_{ln}.png").exists()
    }
    mask_paths = {
        ln: SEGMENTS_DIR / f"{stem}_{ln}.png"
        for ln in active_layers
        if (SEGMENTS_DIR / f"{stem}_{ln}.png").exists()
    }

    if not overlay_paths:
        return load_pil(frame_path)

    out_path = composite_layers(
        base_path=frame_path,
        overlay_paths=overlay_paths,
        mask_paths=mask_paths,
        layer_alphas=layer_alphas,
        stem=f"{stem}_live",
    )
    return load_pil(out_path)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 👻 Ghost Blueprint")
    st.markdown("*X-Ray Vision for Construction*")
    st.divider()

    # Frame selector
    available_stems = list_frames()
    if not available_stems:
        st.warning("No frames found in data/frames/. Add videos and run the pipeline.")
        st.stop()

    selected_stem = st.selectbox(
        "Frame",
        options=available_stems,
        format_func=lambda s: s,
    )

    st.divider()

    # Layer toggles
    st.markdown("**Overlay Layers**")
    available_layers = get_available_layers(selected_stem)

    layer_active: dict[str, bool] = {}
    layer_alpha: dict[str, float] = {}

    LAYER_COLORS = {
        "walls": "🟦",
        "floor": "🟫",
        "ceiling": "⬜",
        "electrical": "🟡",
        "plumbing": "🔵",
        "hvac": "🟠",
        "fixtures": "🟣",
    }

    for layer in LAYER_ORDER:
        if layer not in available_layers:
            continue
        col1, col2 = st.columns([1, 2])
        with col1:
            active = st.checkbox(
                f"{LAYER_COLORS.get(layer, '⚪')} {layer.title()}",
                value=True,
                key=f"toggle_{layer}",
            )
        layer_active[layer] = active

    st.divider()

    # Per-layer alpha sliders (only for active layers)
    st.markdown("**Opacity**")
    for layer in LAYER_ORDER:
        if layer not in available_layers or not layer_active.get(layer):
            continue
        alpha = st.slider(
            layer.title(),
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            key=f"alpha_{layer}",
        )
        layer_alpha[layer] = alpha

    st.divider()

    # View mode
    view_mode = st.radio(
        "View Mode",
        options=["Ghost Overlay", "Original Only", "Side by Side", "Ghost Only"],
    )

    st.divider()

    # Run pipeline button
    st.markdown("**Run Pipeline**")
    run_pipeline = st.button("▶ Process This Frame", type="primary")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">Ghost Blueprint</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">X-Ray Vision for Construction — Ironsite Spatial Intelligence Hackathon</div>', unsafe_allow_html=True)
st.markdown("")

frame_path = FRAMES_DIR / f"{selected_stem}.png"
composite_path = COMPOSITES_DIR / f"{selected_stem}_live_composite.png"
sbs_path = COMPOSITES_DIR / f"{selected_stem}_sidebyside.png"
ghost_only_path = COMPOSITES_DIR / f"{selected_stem}_ghost_only.png"

# -- Run pipeline if requested --
if run_pipeline:
    with st.spinner("Running Ghost Blueprint pipeline... (this may take a few minutes)"):
        try:
            from pipeline import run_frame
            run_frame(frame_path)
            st.success("Pipeline complete!")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")

# -- Main image display --
active_layer_names = [ln for ln in LAYER_ORDER if layer_active.get(ln)]

if view_mode == "Original Only":
    img = load_pil(frame_path)
    if img:
        st.image(img, caption="Original Construction Frame", use_container_width=True)
    else:
        st.info("Frame not found.")

elif view_mode == "Ghost Overlay":
    if active_layer_names:
        img = composite_on_the_fly(selected_stem, active_layer_names, layer_alpha)
    else:
        img = load_pil(frame_path)
    if img:
        st.image(img, caption="Ghost Blueprint Overlay", use_container_width=True)
    else:
        st.info("Run the pipeline to generate overlays for this frame.")

elif view_mode == "Side by Side":
    col1, col2 = st.columns(2)
    with col1:
        orig = load_pil(frame_path)
        if orig:
            st.image(orig, caption="Construction (Now)", use_container_width=True)
    with col2:
        if active_layer_names:
            ghost = composite_on_the_fly(selected_stem, active_layer_names, layer_alpha)
        else:
            ghost = None
        if ghost:
            st.image(ghost, caption="Ghost Blueprint (Finished State)", use_container_width=True)
        else:
            st.info("Run the pipeline to generate overlays.")

elif view_mode == "Ghost Only":
    if ghost_only_path.exists():
        st.image(str(ghost_only_path), caption="Ghost Overlay (No Base)", use_container_width=True)
    else:
        st.info("Ghost-only view not yet generated. Run the pipeline first.")

# ---------------------------------------------------------------------------
# Scene Analysis panel
# ---------------------------------------------------------------------------

st.divider()
col_scene, col_future = st.columns(2)

with col_scene:
    st.markdown("#### Scene Analysis")
    scene = load_scene_json(selected_stem)
    if scene:
        phase = scene.get("construction_phase", "unknown")
        dims = scene.get("room_dimensions", {})
        st.markdown(f"**Phase:** `{phase}`")
        if dims:
            w = dims.get("estimated_width_ft", "?")
            d = dims.get("estimated_depth_ft", "?")
            h_ = dims.get("ceiling_height_ft", "?")
            st.markdown(f"**Dimensions:** {w}' × {d}' × {h_}' ceiling")
        surfaces = scene.get("surfaces", {})
        if surfaces:
            st.markdown(f"**Walls:** {surfaces.get('walls', '?')} | **Floor:** {surfaces.get('floor', '?')} | **Ceiling:** {surfaces.get('ceiling', '?')}")
        elements = scene.get("elements_present", [])
        if elements:
            st.markdown(f"**Elements detected:** {len(elements)}")
            for el in elements[:5]:
                st.markdown(f"- `{el.get('type', '?')}` @ {el.get('location', '?')}: {el.get('description', '')}")
        notes = scene.get("notable_observations", "")
        if notes:
            st.markdown(f"**Notes:** {notes}")
    else:
        st.info("Scene analysis will appear here after running the pipeline.")

with col_future:
    st.markdown("#### Finished State Prediction")
    future = load_future_json(selected_stem)
    if future:
        st.markdown(future.get("finished_description", "*No description generated.*"))
        layers = future.get("layers", {})
        if layers:
            walls = layers.get("walls", {})
            floor = layers.get("floor", {})
            ceil_ = layers.get("ceiling", {})
            if walls:
                st.markdown(f"**Walls:** {walls.get('material', '?')}, {walls.get('color', '?')}")
            if floor:
                st.markdown(f"**Floor:** {floor.get('material', '?')}, {floor.get('color', '?')}")
            if ceil_:
                st.markdown(f"**Ceiling:** {ceil_.get('material', '?')}, fixtures: {', '.join(ceil_.get('fixtures', []))}")
        electrical = layers.get("electrical", []) if layers else []
        if electrical:
            st.markdown(f"**Electrical:** {len(electrical)} elements planned")
            for el in electrical[:3]:
                st.markdown(f"- {el.get('type', '?')} @ {el.get('height_inches_from_floor', '?')}\" from floor, {el.get('location', '')}")
    else:
        st.info("Future state prediction will appear here after running the pipeline.")

# ---------------------------------------------------------------------------
# Depth map display
# ---------------------------------------------------------------------------

with st.expander("Depth Map", expanded=False):
    depth_png = DEPTH_DIR / f"{selected_stem}.png"
    if depth_png.exists():
        col1, col2 = st.columns(2)
        with col1:
            st.image(str(frame_path), caption="Original", use_container_width=True)
        with col2:
            st.image(str(depth_png), caption="Depth Map (brighter = closer)", use_container_width=True)
    else:
        st.info("Depth map not yet generated.")

# ---------------------------------------------------------------------------
# Segmentation masks display
# ---------------------------------------------------------------------------

with st.expander("Segmentation Masks", expanded=False):
    mask_files = list(SEGMENTS_DIR.glob(f"{selected_stem}_*.png"))
    if mask_files:
        cols = st.columns(min(len(mask_files), 4))
        for i, mf in enumerate(mask_files):
            layer_name = mf.stem.replace(f"{selected_stem}_", "")
            with cols[i % 4]:
                st.image(str(mf), caption=layer_name, use_container_width=True)
    else:
        st.info("Segmentation masks not yet generated.")
