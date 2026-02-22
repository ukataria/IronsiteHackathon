# DeepAnchored

**Does grounding a vision-language model's outputs in scene-derived spatial measurements improve its ability to reason accurately about physical dimensions in visual environments?**

DeepAnchored answers this by using objects of known physical size already present in a scene as calibration anchors — converting pixel distances into real-world measurements and injecting them into a VLM prompt before inspection. No tape measures. No external sensors. No manual input.

---

## The Problem

Vision-language models can describe what they see but operate entirely in pixel-space. Given a photo, a model cannot tell you whether a gap is 14 inches or 20 inches — nothing in its input establishes a unit of measurement. This makes AI-driven visual inspection unreliable for any task where exact dimensions matter.

## The Approach

Certain objects have standardized physical dimensions that appear consistently across scenes. DeepAnchored detects these **spatial anchors** automatically, uses them to compute a pixel-to-inch conversion, and injects the resulting measurements as structured context into the VLM prompt.

| Anchor Object | Known Dimension |
|---|---|
| 2×4 stud (face) | 3.5 inches |
| CMU block | 15.625 inches |
| #4 Rebar | 0.5 inches |
| Electrical box | 4.0 inches |

This is evaluated across three conditions:

| Condition | VLM Input |
|---|---|
| **Baseline** | Raw image only |
| **Depth-augmented** | Image + monocular depth map |
| **DeepAnchored** | Image + calibrated real-world measurements |

---

## Pipeline

```
Image / Video
     │
     ▼
1. Frame Extraction       — sample frames, filter blurry
     │
     ▼
2. Anchor Detection       — YOLOv8 (fine-tuned) + GroundingDINO fallback
     │
     ▼
3. Depth Estimation       — Depth Anything V2
     │
     ▼
4. Scale Calibration      — pixel-to-inch conversion from anchor bounding boxes
     │
     ▼
5. Spatial Measurement    — distances between structural elements in inches
     │
     ▼
6. VLM Inspection         — Claude / GPT-4o with injected measurements
     │
     ▼
  Inspection Report (pass/fail per frame, violation details)
```

---

## Quickstart

### Requirements
- Python 3.11+
- Node.js 18+
- `uv` package manager
- API key for Anthropic or OpenAI (set in `.env`)

### Setup

```bash
# Clone and install Python deps
git clone https://github.com/ukataria/IronsiteHackathon
cd IronsiteHackathon
uv sync

# Install frontend deps
cd frontend && npm install && cd ..

# Add API keys
# Create a .env file with:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

### Run

**Terminal 1 — Backend:**
```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend && npm run dev
```

Open [http://localhost:8080](http://localhost:8080).

### CLI — Process a single image

```bash
uv run python pipeline.py data/frames/your_image.jpg
```

### CLI — Batch process

```bash
uv run python scripts/run_batch.py --vlm claude
```

---

## Project Structure

```
├── api.py                  # FastAPI backend
├── pipeline.py             # End-to-end pipeline
├── src/
│   ├── anchors/detect.py   # YOLOv8 + GroundingDINO anchor detection
│   ├── depth/estimate.py   # Depth Anything V2
│   ├── calibration/        # Pixel-to-inch scale math
│   ├── measurement/        # Spatial fact extraction
│   ├── vlm/                # Claude / GPT-4o wrappers + prompts
│   └── utils.py            # Shared helpers, caching
├── frontend/               # React + TypeScript + Tailwind UI
├── scripts/
│   ├── extract_frames.py   # Pull frames from video
│   └── run_batch.py        # Batch inference
└── data/
    ├── frames/             # Input images
    ├── detections/         # Anchor detection outputs
    ├── calibrations/       # Scale calibration results
    ├── measurements/       # Spatial measurements per frame
    └── results/            # VLM responses (all 3 conditions)
```

---

## Environment Variables

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

## Built at IronSite Hackathon · 36 hours · 2026
