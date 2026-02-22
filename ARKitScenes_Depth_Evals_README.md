# ARKitScenes Depth Measurement Evals (VLM + Depth Baselines)

Goal: evaluate **metric depth accuracy (in meters)** on **ARKitScenes**
and compare: 1) VLMs (API-based) answering depth/measurement questions
2) Depth-model baselines (local) producing metric depth maps

This repo is intentionally "eval-first": no training/finetuning until
you have a solid measurement baseline.

------------------------------------------------------------------------

## 1) What you will measure

### Primary: depth error vs ground truth (meters)

For each RGB frame with ground-truth depth: - Predict depth
`D_pred(u,v)` (meters) - Compare to ground-truth depth `D_gt(u,v)`
(meters) on valid pixels

Report: - **AbsRel**: mean(\|D_pred - D_gt\| / D_gt) - **RMSE**:
sqrt(mean((D_pred - D_gt)\^2)) - **MAE (meters)**: mean(\|D_pred -
D_gt\|) - **δ thresholds**: % of pixels with max(D_pred/D_gt,
D_gt/D_pred) \< 1.25, 1.25², 1.25³

### Optional: derived measurement tasks (still depth-only)

From depth + intrinsics, compute geometric measurements: -
point-to-point distance (meters) - plane distance (e.g., distance to
wall plane) - object-to-object distance (if you have segmentation masks)

These are evaluated by comparing predicted measurement to measurement
computed from ground-truth depth.

------------------------------------------------------------------------

## 2) Dataset: ARKitScenes

Official dataset + tooling: - https://github.com/apple/ARKitScenes

You need: - RGB frames - high-quality depth / mesh-derived depth
(metric) - camera intrinsics (fx, fy, cx, cy) - valid-depth mask (if
provided)

**Recommendation:** start with a small subset (e.g., 1--3 scans) to
validate the pipeline.

------------------------------------------------------------------------

## 3) Repo layout

    .
    ├── data/
    │   ├── raw/                  # ARKitScenes downloads (unmodified)
    │   └── processed/
    │       ├── frames.csv        # index of eval frames with filepaths + intrinsics
    │       └── samples/          # optional small curated sample set
    ├── eval/
    │   ├── configs/
    │   │   ├── models.yaml       # model endpoints + settings
    │   │   └── split.yaml        # scan/frame split config
    │   ├── runners/
    │   │   ├── eval_depth_maps.py
    │   │   ├── eval_vlm_points.py
    │   │   └── aggregate.py
    │   ├── metrics.py            # AbsRel/RMSE/MAE/delta
    │   └── viz.py                # qualitative overlays
    ├── models/
    │   ├── baselines/
    │   │   ├── zoedepth/
    │   │   └── midas_dpt/
    │   └── vlm_clients/
    │       ├── openai_client.py
    │       ├── anthropic_client.py
    │       └── google_client.py
    ├── outputs/
    │   ├── predictions/
    │   │   ├── depth_maps/       # .npy or 16-bit png predicted depth (meters)
    │   │   └── vlm_answers/      # jsonl of VLM responses
    │   ├── reports/
    │   │   ├── metrics.json
    │   │   └── summary.md
    │   └── viz/                  # depth comparisons + error heatmaps
    └── README.md

------------------------------------------------------------------------

## 4) Environment setup

### Python

Recommended: - Python 3.10+ - CUDA optional (only for baseline depth
models)

Install:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### API keys (VLM evals)

Export keys as env vars:

    export OPENAI_API_KEY="..."
    export ANTHROPIC_API_KEY="..."
    export GOOGLE_API_KEY="..."

Model selection is configured in `eval/configs/models.yaml`.

------------------------------------------------------------------------

## 5) Frame indexing (build a clean eval table)

Create a `frames.csv` that contains one row per evaluated frame: -
`scan_id` - `frame_id` - `rgb_path` - `depth_gt_path` -
`intrinsics_fx, intrinsics_fy, intrinsics_cx, intrinsics_cy` -
`valid_mask_path` (optional) - `split` (train/val/test; for eval-only,
just "test")

Command:

    python eval/runners/build_index.py \
      --arkit_root data/raw/ARKitScenes \
      --out_csv data/processed/frames.csv \
      --max_frames_per_scan 200

**Tip:** for early debugging, set `--max_frames_per_scan 30` and use 1
scan.

------------------------------------------------------------------------

## 6) Baseline depth models (local) --- required

You need a non-VLM baseline to know what "good" looks like.

### Baseline A: ZoeDepth

Produces metric-ish depth from RGB; can be calibrated to meters if
needed.

### Baseline B: DPT/MiDaS

Strong relative depth; typically needs scale alignment for metric
evaluation.

Run baselines:

    python eval/runners/predict_depth_baselines.py \
      --frames_csv data/processed/frames.csv \
      --out_dir outputs/predictions/depth_maps \
      --models zoedepth midas_dpt \
      --device cuda

Output format: - store `D_pred` as float32 meters in `.npy` aligned to
RGB resolution

------------------------------------------------------------------------

## 7) VLM depth measurement evals (API-based)

VLMs do not output dense depth maps reliably. Evaluate them in two ways:

### (A) Point depth queries (recommended for VLMs)

For each frame: 1) sample K pixel locations (u,v) 2) ask the VLM: "What
is the distance from camera to the point at (u,v) in meters?" 3) compare
to `D_gt(u,v)` at those pixels

This tests: "can the VLM approximate depth in meters at specific
points?"

Command:

    python eval/runners/eval_vlm_points.py \
      --frames_csv data/processed/frames.csv \
      --out_jsonl outputs/predictions/vlm_answers/points.jsonl \
      --model gpt-4o \
      --points_per_image 32 \
      --prompt_template prompts/point_depth.txt

Prompt template (`prompts/point_depth.txt`) should: - instruct "answer
with a single number in meters" - include camera intrinsics (optional
but recommended) - include image coordinate convention (u,v origin
top-left)

------------------------------------------------------------------------

## 8) Metric evaluation (the core report)

### Dense depth-map evaluation (baselines)

    python eval/runners/eval_depth_maps.py \
      --frames_csv data/processed/frames.csv \
      --pred_dir outputs/predictions/depth_maps \
      --out_json outputs/reports/metrics_depth_maps.json

### VLM point-depth evaluation

    python eval/runners/eval_vlm_points_metrics.py \
      --frames_csv data/processed/frames.csv \
      --answers_jsonl outputs/predictions/vlm_answers/points.jsonl \
      --out_json outputs/reports/metrics_vlm_points.json

### Aggregate summary

    python eval/runners/aggregate.py \
      --depth_metrics outputs/reports/metrics_depth_maps.json \
      --vlm_metrics outputs/reports/metrics_vlm_points.json \
      --out_md outputs/reports/summary.md

------------------------------------------------------------------------

## 9) Scale alignment (important for "meters")

Some models output depth up to an unknown scale (common for relative
depth models). Choose one evaluation mode and stick to it:

### Mode 1: No scaling (strict)

-   Evaluate raw `D_pred` vs `D_gt` directly.

### Mode 2: Median scale alignment

-   Compute a single scalar `s` per image (or per scan):
    `s = median(D_gt) / median(D_pred)`
-   Evaluate `s * D_pred` vs `D_gt`

Report both modes if possible.

------------------------------------------------------------------------

## 10) Visual debugging

Generate qualitative outputs: - RGB - GT depth (colorized) - Pred depth
(colorized) - Error heatmap (\|pred-gt\| meters)

    python eval/runners/viz_depth.py \
      --frames_csv data/processed/frames.csv \
      --pred_dir outputs/predictions/depth_maps \
      --out_dir outputs/viz \
      --num_examples 50

------------------------------------------------------------------------

## 11) Recommended initial experiment plan

1)  Download 1 scan from ARKitScenes
2)  Build `frames.csv` with \~30 frames
3)  Run ZoeDepth baseline
4)  Run eval metrics (AbsRel/RMSE/MAE)
5)  Run VLM point-depth queries (K=16) on the same frames
6)  Compare VLM point MAE vs ZoeDepth dense MAE

------------------------------------------------------------------------

## 12) Notes / pitfalls

-   Intrinsics must match the RGB frame resolution. If you resize RGB,
    update intrinsics.
-   Depth GT has invalid pixels. Always mask invalid depth.
-   VLM numeric answers are noisy. Enforce strict output formatting and
    parse failures robustly.
-   Don't mix HR/LR depth without tracking which you used.

------------------------------------------------------------------------

## 13) Output expectations

For indoor scenes, a decent metric-depth system often achieves: - MAE:
\~0.10--0.30 m - AbsRel: \~0.05--0.15

Your report should end with: - strict metric accuracy (meters) - aligned
accuracy (shape-only) - qualitative examples

------------------------------------------------------------------------

## 14) License / usage

ARKitScenes is for research use under Apple's terms. Review and comply
with the dataset license in the official repository before using results
in commercial contexts.
