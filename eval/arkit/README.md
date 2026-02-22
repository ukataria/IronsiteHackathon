# ARKit LiDAR Benchmark

Benchmark VLMs on 3D distance estimation using ARKit LiDAR depth data from iPhone/iPad Pro.

## Quick Start

### 1. Prepare Your Data

If you have ARKit captures from iPhone/iPad Pro:

```bash
# Create directory structure and default intrinsics
python eval/arkit/prepare_arkit_data.py \
  --output_dir data/arkit \
  --width 1920 \
  --height 1440 \
  --create_structure
```

Then place your files:
- RGB images in `data/arkit/rgb/` (0000.jpg, 0001.jpg, ...)
- Depth maps in `data/arkit/depth/` (0000.npy, 0001.npy, ... in meters)

### 2. Run the Benchmark

```bash
python eval/arkit/benchmark_arkit_all_models.py \
  --arkit_data_dir data/arkit \
  --num_images 10 \
  --pairs_per_image 3 \
  --device cuda \
  --models gpt-4o claude-sonnet-4 two-head-claude
```

## Models Tested

- **gpt-4o**: OpenAI's multimodal model
- **claude-sonnet-4**: Anthropic's Claude Sonnet 4
- **gemini-2.0-flash**: Google's Gemini 2.0 Flash
- **two-head-claude**: Your two-head architecture (Depth + VLM)

## Data Format

Expected directory structure:
```
data/arkit/
├── rgb/
│   ├── 0000.jpg
│   ├── 0001.jpg
│   └── ...
├── depth/
│   ├── 0000.npy  # numpy array, float32, in meters
│   ├── 0001.npy
│   └── ...
└── intrinsics.npy  # dict with fx, fy, cx, cy, width, height
```

## Output

The benchmark produces:
- `arkit_benchmark_results.json` - Full results for all models
- `arkit_benchmark_comparison.png` - Visualization comparing models
- `marked_images/` - Images with marked test points

## Metrics

For each model:
- **MAE** (Mean Absolute Error in meters)
- **Median Error** (meters)
- **Success Rate** (% of valid responses)
- **Std Error** (standard deviation)

## Requirements

Set environment variables for API-based models:
```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

For two-head model, you need GPU for depth estimation.

## Notes

- ARKit LiDAR depth is high-quality metric depth (in meters)
- Typical indoor accuracy: 1-5mm at close range
- Works best for scenes 0.5m - 5m from camera
- The benchmark reuses NYU utils for camera projections
