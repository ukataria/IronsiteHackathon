# CLAUDE.md — Ghost Blueprint | Ironsite Spatial Intelligence Hackathon

## Project: Ghost Blueprint

"X-ray vision for construction." Take raw construction footage and overlay what the finished space will look like — drywall over studs, outlets in place, flooring down, fixtures installed — spatially accurate to the real 3D geometry. Toggle layers on and off. The future state of the building, ghosted onto reality.

## Team

5 people. Parallelize aggressively. Interfaces between components are files (images, JSON, numpy arrays). If you're blocked on someone else's output, mock it and keep moving.

## Core Philosophy

- **Demo-driven development** — every decision should make the demo better. If it doesn't show well, it doesn't matter.
- **Get ugly results fast, then iterate** — a janky overlay on real footage by Saturday morning beats a perfect pipeline that's not integrated until Sunday.
- **Precompute everything for the demo** — real-time inference is a nice-to-have, precomputed hero frames are the priority.
- **Save every intermediate output to disk** — frames, depth maps, segmentation masks, inpainted layers, API responses. Cache everything.
- **Commit after every working milestone** — small checkpoints you can roll back to.

## Tech Stack

- **Python 3.11+** with uv (NEVER pip)
- **GPU Compute**: Vast.ai for heavy inference (depth estimation, segmentation, inpainting)
- **Depth Estimation**: Depth Anything V2
- **Segmentation**: GroundedSAM / SAM2 (segment construction elements by text prompt)
- **Inpainting**: SDXL Inpainting or Stable Diffusion Inpainting + ControlNet (depth-conditioned)
- **LLM APIs**: Anthropic (Claude), Google (Gemini) — scene understanding, construction knowledge
- **Compositing**: OpenCV, Pillow (alpha blending, layer management)
- **Demo UI**: Streamlit
- **Data**: 12 Ironsite construction video clips (~20 min each) in `data/raw/`
- API keys loaded from `.env` via python-dotenv. NEVER hardcode keys.

## Package Management

- Install: `uv add package`
- Run: `uv run python script.py` or `uv run streamlit run app.py`
- NEVER use `pip install`, `uv pip install`, or `@latest`

## Project Structure

```
├── CLAUDE.md
├── pyproject.toml
├── .env                      # API keys (gitignored)
├── data/
│   ├── raw/                  # Ironsite video clips
│   ├── frames/               # Extracted keyframes (frame_XXXX.png)
│   ├── depth/                # Depth maps (depth_XXXX.png + depth_XXXX.npy)
│   ├── segments/             # Segmentation masks per element type
│   ├── scenes/               # LLM scene descriptions (JSON)
│   ├── overlays/             # Inpainted finished-state layers
│   ├── composites/           # Final composited ghost overlays
│   └── cache/                # Cached API responses
├── src/
│   ├── video/                # Frame extraction, keyframe selection
│   │   └── extract.py
│   ├── depth/                # Depth estimation pipeline
│   │   └── estimate.py
│   ├── segmentation/         # GroundedSAM element segmentation
│   │   └── segment.py
│   ├── scene/                # LLM scene understanding
│   │   ├── describe.py       # Generate structured scene descriptions
│   │   └── predict.py        # Generate finished-state descriptions
│   ├── inpaint/              # Inpainting pipeline
│   │   ├── generate.py       # Generate finished-state overlays
│   │   └── controlnet.py     # Depth-conditioned generation
│   ├── composite/            # Layer compositing and ghost overlay
│   │   └── blend.py
│   ├── prompts.py            # ALL prompts as string constants
│   ├── llm.py                # Thin API wrappers with caching
│   └── utils.py              # Shared helpers (file I/O, image ops)
├── app.py                    # Streamlit demo
├── pipeline.py               # End-to-end: frame → depth → segment → describe → inpaint → composite
└── scripts/
    ├── process_videos.py     # Batch process all clips on Vast.ai
    └── select_hero_frames.py # Pick best frames for demo
```

## Pipeline Flow

```
Raw Video Frame
    │
    ├──→ Depth Anything V2 ──→ Depth Map (.npy + visualization .png)
    │
    ├──→ GroundedSAM ──→ Segmentation Masks per element
    │        (studs, floor, ceiling joists, pipes, wiring, openings)
    │
    ├──→ LLM Scene Understanding ──→ Structured Scene JSON
    │        {current_state: [...], dimensions: {...}, phase: "framing"}
    │
    └──→ LLM Future State Prediction ──→ Finished State Description
             {walls: "drywall, painted white", floor: "hardwood",
              electrical: [{type: "outlet", height: 12in, location: ...}]}
                 │
                 ▼
         SDXL Inpainting (per element layer)
         + ControlNet depth conditioning
                 │
                 ▼
         Overlay Layers (drywall, electrical, flooring, ceiling, fixtures)
                 │
                 ▼
         Alpha Compositing ──→ Final Ghost Blueprint Frame
         (adjustable transparency per layer)
```

## Work Split

- **Person 1 (Video + Depth)**: Frame extraction, keyframe selection, Depth Anything V2 on Vast.ai. Delivers: `data/frames/`, `data/depth/`
- **Person 2 (Segmentation + Scene)**: GroundedSAM segmentation, LLM scene descriptions. Delivers: `data/segments/`, `data/scenes/`
- **Person 3 (Inpainting)**: SDXL inpainting pipeline, ControlNet depth conditioning, prompt engineering for realistic results. Delivers: `data/overlays/`
- **Person 4 (Compositing)**: Alpha blending engine, layer management, frame-to-frame consistency. Delivers: `data/composites/`
- **Person 5 (Frontend + Demo)**: Streamlit app, interactive controls, demo video recording, README. Delivers: `app.py`, video, docs

Interface contract: everyone reads/writes files to the shared `data/` directories. Use consistent naming: `{clip_id}_frame_{XXXX}` everywhere.

## Code Rules

- Type hints on all function signatures
- Docstrings on public functions (one-liner is fine)
- snake_case functions/variables, PascalCase classes, UPPER_SNAKE_CASE constants
- Keep functions under 30 lines — split if longer
- Early returns over nested ifs
- f-strings for formatting
- Line length: 88 chars max

## LLM Patterns

```python
# All prompts in src/prompts.py
SCENE_DESCRIBE_PROMPT = """Analyze this construction scene. Given the image and depth map...
Return JSON: {current_elements: [...], dimensions: {...}, construction_phase: "..."}"""

FUTURE_STATE_PROMPT = """Given this construction scene in {phase} phase...
Describe the finished state. Return JSON: {walls: "...", floor: "...", electrical: [...]}"""
```

- Always cache responses: `data/cache/{prompt_hash}.json`
- Check cache before every API call — never waste credits
- Log token usage on every call
- Retry once on failure, return error string, don't crash
- Use `asyncio.gather()` to call multiple APIs in parallel

## Inpainting Guidelines

- Inpaint one element layer at a time (studs→drywall, subfloor→hardwood, etc.)
- Use segmentation mask to constrain inpainting region
- Use depth map as ControlNet conditioning for spatial accuracy
- Iterate on prompts — construction-specific details matter ("smooth drywall with latex paint" not just "wall")
- Save every generated layer separately for compositing
- If a result looks bad, regenerate with a different seed — don't waste time debugging the model

## Compositing Rules

- Each overlay layer has its own alpha channel
- Default ghost opacity: 0.4 (translucent enough to see construction underneath)
- Use a subtle color tint (light blue) on overlay layers to distinguish from real footage
- Layers must respect depth ordering — closer surfaces overlay farther ones
- For video consistency: if doing multiple frames, use same random seeds and similar prompts

## Streamlit Demo Structure

```
┌─────────────────────────────────────────────┐
│  Ghost Blueprint — X-Ray Vision for         │
│  Construction                               │
├──────────────┬──────────────────────────────┤
│ Controls     │  Main View                   │
│              │  ┌──────────────────────────┐ │
│ □ Drywall    │  │                          │ │
│ □ Electrical │  │   Video frame with       │ │
│ □ Flooring   │  │   ghost overlay          │ │
│ □ Ceiling    │  │                          │ │
│ □ Fixtures   │  │                          │ │
│              │  └──────────────────────────┘ │
│ Opacity ───○ │                              │
│              │  [Before] [Ghost] [Side by   │
│ Frame ────○  │   side]                      │
│              │                              │
│ [▶ Play]     │  Scene Analysis:             │
│              │  "Framing phase, north wall, │
│              │   12x14 ft room..."          │
└──────────────┴──────────────────────────────┘
```

- Sidebar: layer toggles, opacity slider, frame selector
- Main area: video frame with overlay
- View modes: original only, ghost only, side-by-side, overlay
- Below: LLM scene description + finished state description
- Cache everything with @st.cache_data — demo must feel snappy

## Hero Frames Strategy

Pick 5-8 "hero frames" from the Ironsite footage that:
- Show clear construction elements (studs, framing, rough openings)
- Have good lighting and camera angle
- Represent different stages or areas
- Are visually interesting

Spend extra time making these overlays perfect. The demo video uses these. Live demo can fall back to these if real-time processing is slow.

## Git Discipline

- Commit after every working milestone
- Commit messages: short, present tense ("add depth pipeline", "wire up inpainting", "build streamlit app")
- Never mention co-authored-by or tooling in commits
- Branch: just use main, it's a hackathon
- `.gitignore`: data/, .env, __pycache__, *.pyc, .venv/

## Risk Mitigation

1. **Inpainting looks bad**: Fall back to simpler overlays — colored transparent shapes showing where elements will go (blue rectangles for drywall, yellow dots for outlets). Less pretty but still communicates the concept.
2. **Depth estimation is inaccurate**: Use it directionally (relative depth) rather than absolute. The overlay just needs to look right, not be survey-grade accurate.
3. **Frame-to-frame consistency is poor**: Don't try video — demo with individual frames and a frame selector slider. Still impressive.
4. **Running out of time**: Prioritize 3 perfect hero frames over 100 mediocre ones. A polished demo of 3 frames beats a broken demo of the full video.
5. **Vast.ai goes down**: Have a lightweight fallback pipeline that runs on CPU with smaller models.

## When Stuck

1. Is the demo still working? If yes, keep building. If no, revert to last commit.
2. Can you precompute it? Pre-baked results are fine for the demo.
3. Is this feature visible in the 3-5 min video? If not, skip it.
4. Would a simpler version still look good? Do that first, upgrade if time allows.
5. Sleep > debugging at 4am.

## Deliverables Checklist

- [ ] GitHub repo (clean, documented)
- [ ] README with approach, process, findings
- [ ] 3-5 min demo video
- [ ] Optional PDF report
- [ ] Streamlit app runs locally from repo
- [ ] At least 5 hero frames with polished ghost overlays
- [ ] Side-by-side comparison: raw model output vs Ghost Blueprint