# CLAUDE.md — PreCheck | Code Style & Project Conventions

## What Is This Project

PreCheck: Spatial Anchor Calibration for Automated Construction Inspection.
We teach VLMs to measure real-world dimensions from construction photos by using known-dimension objects (studs, rebar, CMU blocks) as calibration anchors, then inject those measurements into VLM prompts to enable accurate spatial inspection.

36-hour hackathon. 5 people. Demo Sunday morning.

## Philosophy

- **Demo-driven development.** If it doesn't improve the demo, don't build it.
- **Get ugly results fast, then iterate.** A janky pipeline that produces numbers by Saturday morning beats a clean architecture that's not integrated until Sunday.
- **Cache everything.** API responses, depth maps, detection results, calibration outputs. Never recompute what you already have. Never waste API credits on duplicate calls.
- **Print intermediate results.** Every pipeline stage should print or save its output so you can debug by reading, not guessing.
- **Commit after every working milestone.** Small checkpoints you can roll back to.
- **Pre-compute for demo.** Real-time is a nice-to-have. Pre-baked hero results are the priority.

## Package Management

- **ONLY use uv. NEVER pip.**
- Install: `uv add package`
- Run: `uv run python script.py` or `uv run streamlit run app.py`
- Dev deps: `uv add --dev package`
- FORBIDDEN: `pip install`, `uv pip install`, `@latest`

## Project Structure

```
├── CLAUDE.md                  # This file — code style & conventions
├── TECHNICAL.md               # Detailed technical approach & architecture
├── README.md                  # Submission README
├── pyproject.toml
├── .env                       # API keys (gitignored)
├── .gitignore
│
├── data/
│   ├── raw/                   # Source images & Ironsite footage
│   ├── frames/                # Extracted video frames
│   ├── depth/                 # Depth maps (.png vis + .npy arrays)
│   ├── detections/            # Anchor detection outputs (JSON + annotated images)
│   ├── calibrations/          # Scale calibration results (JSON)
│   ├── measurements/          # Spatial measurements per image (JSON)
│   ├── results/               # VLM responses for all 3 conditions (JSON)
│   └── cache/                 # Cached API responses by prompt hash
│
├── src/
│   ├── __init__.py
│   ├── anchors/               # Anchor detection (studs, rebar, CMU, boxes)
│   │   ├── __init__.py
│   │   └── detect.py
│   ├── depth/                 # Depth estimation pipeline
│   │   ├── __init__.py
│   │   └── estimate.py
│   ├── calibration/           # Pixel-to-real-world scale math
│   │   ├── __init__.py
│   │   └── calibrate.py
│   ├── measurement/           # Spatial fact extraction
│   │   ├── __init__.py
│   │   └── measure.py
│   ├── reid/                  # Object re-identification across views
│   │   ├── __init__.py
│   │   └── match.py
│   ├── vlm/                   # VLM API wrappers + prompt injection
│   │   ├── __init__.py
│   │   ├── clients.py         # Thin wrappers for Claude, Gemini, GPT-5
│   │   └── prompts.py         # ALL prompts as string constants
│   └── utils.py               # Shared helpers (image I/O, caching, logging)
│
├── app.py                     # Streamlit demo
├── pipeline.py                # End-to-end: image → anchors → depth → calibrate → measure → VLM
├── evaluate.py                # Run 3-condition benchmark
└── scripts/
    ├── extract_frames.py      # Pull frames from Ironsite video clips
    └── run_batch.py           # Batch process multiple images through pipeline
```

## Code Style

### Python Basics
- **Python 3.11+**
- Type hints on ALL function signatures, no exceptions
- Docstrings on all public functions (one-liner is fine for hackathon)
- snake_case for functions and variables
- PascalCase for classes
- UPPER_SNAKE_CASE for constants
- f-strings for all string formatting
- Line length: 88 chars max

### Function Design
- Keep functions under 30 lines. If longer, split.
- Early returns over nested ifs. Always.
- One function, one job. No god functions.
- Prefix handler functions with `handle_`
- Prefix helper functions descriptively: `compute_`, `extract_`, `detect_`, `format_`

```python
# GOOD
def compute_scale_factor(anchor_pixels: int, anchor_real_inches: float) -> float:
    """Compute pixels-to-inches conversion from a known anchor."""
    if anchor_pixels <= 0:
        return 0.0
    return anchor_real_inches / anchor_pixels

# BAD
def do_stuff(data):
    # 80 lines of untyped chaos
    ...
```

### Constants & Configuration
- All known construction dimensions go in a constants file or at the top of the relevant module
- All prompts go in `src/vlm/prompts.py` as string constants
- API keys from `.env` via `python-dotenv`, NEVER hardcoded

```python
# src/vlm/prompts.py
INSPECTION_SYSTEM_PROMPT = """You are a construction inspection AI..."""

SPATIAL_CONTEXT_TEMPLATE = """
CALIBRATED SPATIAL MEASUREMENTS:
{measurements}

CONSTRUCTION STANDARDS:
{standards}

Analyze this construction scene for inspection readiness.
"""
```

```python
# Constants for known anchor dimensions
ANCHOR_DIMENSIONS: dict[str, float] = {
    "stud_face_width": 3.5,       # 2x4 face in inches
    "stud_depth": 1.5,            # 2x4 depth in inches
    "cmu_length": 15.625,         # CMU block length in inches
    "cmu_height": 7.625,          # CMU block height in inches
    "rebar_4_diameter": 0.5,      # #4 rebar in inches
    "rebar_5_diameter": 0.625,    # #5 rebar in inches
    "door_rough_width": 38.5,     # Standard door rough opening
    "door_rough_height": 82.5,    # Standard door rough opening
}
```

### Data Flow
- Every pipeline stage reads files from one `data/` subdirectory and writes to another
- Use consistent naming: `{image_id}_suffix.ext` everywhere
- JSON for structured data, .npy for numpy arrays, .png for images
- Every function that calls an API should accept a `cache_key` parameter

```python
# GOOD: clear data flow
def detect_anchors(image_path: str, output_dir: str) -> dict:
    """Detect known-dimension objects. Saves annotated image + JSON."""
    image_id = Path(image_path).stem
    # ... detection logic ...
    save_json(results, f"{output_dir}/{image_id}_anchors.json")
    save_image(annotated, f"{output_dir}/{image_id}_annotated.png")
    return results
```

### Caching Pattern
```python
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("data/cache")

def cached_api_call(prompt: str, image_b64: str | None = None) -> str:
    """Call API with disk caching. Never waste credits on duplicates."""
    cache_key = hashlib.md5(f"{prompt}{image_b64 or ''}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.json"
    
    if cache_path.exists():
        return json.loads(cache_path.read_text())["response"]
    
    response = _call_api(prompt, image_b64)  # actual API call
    cache_path.write_text(json.dumps({"prompt": prompt, "response": response}))
    return response
```

### Error Handling
- Never crash the pipeline. Catch, log, return a fallback.
- API errors: retry once, then return error string
- Detection failures: return empty results, don't block downstream
- Log with context: which image, which stage, what failed

```python
try:
    response = call_claude(prompt, image_b64, cache_key=image_id)
except Exception as e:
    logger.error(f"VLM call failed for {image_id}: {e}")
    response = "ERROR: VLM call failed"
```

### Saving Intermediate Outputs
- EVERY pipeline stage saves its output visually for debugging
- Depth maps: save both the .npy array AND a colorized .png visualization
- Detections: save annotated images with bounding boxes drawn
- Measurements: save images with measurement lines and labels overlaid
- This is non-negotiable — you will need to debug visually at 2am

## Streamlit Conventions

- Keep `app.py` thin — import logic from `src/`
- Use `st.cache_data` for any expensive computation
- Use `st.columns` for side-by-side comparisons
- Use `st.sidebar` for controls (image selection, condition toggle)
- Pre-compute demo results and load from disk — don't run inference live unless you have to

## Git

- Commit messages: short, present tense ("add anchor detection", "wire up calibration", "build streamlit demo")
- Never mention co-authored-by or tooling in commits
- One logical change per commit
- Branch: just use main
- `.gitignore`: data/, .env, __pycache__, *.pyc, .venv/, *.npy (large files)

## Team Interfaces

Each person owns a pipeline stage. Interfaces between stages are files in `data/`.
If you're blocked on someone else's output, create mock data in the expected format and keep building.

| Person | Owns | Reads from | Writes to |
|--------|------|-----------|-----------|
| P1 - Anchor Detection | `src/anchors/` | `data/frames/` | `data/detections/` |
| P2 - Depth + Calibration | `src/depth/`, `src/calibration/` | `data/frames/`, `data/detections/` | `data/depth/`, `data/calibrations/` |
| P3 - VLM + Evaluation | `src/vlm/`, `evaluate.py` | `data/measurements/` | `data/results/` |
| P4 - ReID + Multi-view | `src/reid/`, `src/measurement/` | `data/detections/`, `data/calibrations/` | `data/measurements/` |
| P5 - Demo + Docs | `app.py`, README | `data/results/`, `data/composites/` | Final demo |

## When Stuck

1. Is the demo still working? If yes, keep going. If no, revert.
2. Can you pre-compute it? Pre-baked results are fine for presentation.
3. Is this feature visible in the 3-5 min video? If not, skip it.
4. Would a simpler version still demonstrate the technique? Do that first.
5. Sleep > debugging at 4am with diminishing returns.