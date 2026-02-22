# NYU Depth V2 Distance Evaluation Setup

Branch: `nyu-distance-evals`

## Overview

Comprehensive evaluation framework for testing VLM 3D distance estimation capabilities using the NYU Depth V2 dataset.

## What's Built

### 1. Dataset Infrastructure (`data/nyu_depth_v2/`)
- **download_nyu.py**: Downloads and extracts NYU Depth V2 labeled dataset
  - Downloads 2.8GB .mat file
  - Extracts RGB images, depth maps, and semantic labels
  - Saves camera intrinsics (Kinect parameters)
  - Default: 100 samples extracted

### 2. Evaluation Utilities (`eval/nyu_distance/`)

**nyu_utils.py**:
- `NYUDepthLoader`: Load RGB + depth + intrinsics
- `pixel_to_3d()`: Convert pixel coords to 3D point using pinhole model
- `compute_distance_3d()`: Calculate Euclidean distance between 3D points
- `generate_object_pairs()`: Smart sampling of point pairs for queries

**eval_nyu_distance.py**:
- Main evaluation script
- Creates marked images with red/blue crosses
- Queries Claude Sonnet 4.6 and GPT-4o
- Computes MAE, median error, std dev
- Saves full results JSON

### 3. Setup Scripts
- `quick_setup.sh`: Interactive setup helper
- `README.md`: Complete documentation

## Technical Details

### Ground Truth 3D Distance

Using NYU camera intrinsics and depth maps:

```python
# Pixel to 3D (pinhole camera model)
X = (u - cx) * depth / fx
Y = (v - cy) * depth / fy
Z = depth

# 3D Euclidean distance
d = sqrt((X1-X2)² + (Y1-Y2)² + (Z1-Z2)²)
```

### Camera Intrinsics (NYU Kinect)
```python
fx = 518.857901  # Focal length x
fy = 519.469611  # Focal length y
cx = 325.582244  # Principal point x
cy = 253.736347  # Principal point y
width = 640
height = 480
```

### Object Pair Selection Strategy

Intelligent sampling to find good test cases:
- Grid-based sampling across image
- Filter invalid depths (nan, inf, <=0)
- Require minimum pixel separation (>50px)
- Require depth variation (>0.2m) to ensure different objects
- Generate multiple pairs per image for robust eval

### VLM Query Format

**Input**: RGB image with marked points
- Point A: Red cross at pixel (u1, v1)
- Point B: Blue cross at pixel (u2, v2)

**Prompt**:
```
Estimate the 3D Euclidean distance in meters between
the two marked points (A and B) in this indoor scene.

Output ONLY a number in meters.
```

**Expected**: Single numeric value (e.g., "1.5")

## How to Run

### Quick Start

```bash
# Setup (downloads dataset if needed)
./eval/nyu_distance/quick_setup.sh

# Run evaluation (10 images × 3 pairs = 30 queries)
python eval/nyu_distance/eval_nyu_distance.py \
  --num_images 10 \
  --pairs_per_image 3
```

### Custom Options

```bash
python eval/nyu_distance/eval_nyu_distance.py \
  --nyu_data_dir data/nyu_depth_v2/extracted \
  --num_images 20 \
  --pairs_per_image 5 \
  --out_dir outputs/nyu_distance
```

### Manual Dataset Setup

```bash
# Download full dataset (2.8GB)
python data/nyu_depth_v2/download_nyu.py

# Or download manually from:
# http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
# Then run download_nyu.py to extract
```

## Expected Results

### Metrics
- **MAE**: Mean absolute error in meters
- **Median Error**: Median error in meters
- **Std Error**: Standard deviation of errors
- **Response Rate**: % of successful numeric predictions

### Hypothesis
Based on prior synthetic depth eval results:
- Claude Sonnet 4.6: ~2.0m MAE on point depth
- GPT-4o: ~2.25m MAE on point depth

For 3D distance (harder task):
- Expected: 0.5-1.5m MAE on typical indoor distances (1-4m)
- Claude may outperform GPT-4o by 10-15%

### Why This is Harder Than Point Depth
1. **Real vs Synthetic**: Noisy real-world scenes vs clean synthetic
2. **Two-point reasoning**: Must reason about both points simultaneously
3. **3D geometry**: Must understand depth + perspective
4. **Indoor complexity**: Cluttered scenes with occlusions

## Output Files

```
outputs/nyu_distance/
├── nyu_distance_results.json     # Full results with all predictions
└── marked_images/                 # Visualizations
    ├── img0000_pair0.jpg         # Image with marked points A/B
    ├── img0000_pair1.jpg
    └── ...
```

### Results JSON Structure

```json
{
  "claude": [
    {
      "image_idx": 0,
      "pair_idx": 0,
      "point1": [320, 240],
      "point2": [450, 300],
      "gt_distance": 1.234,
      "predicted_distance": 1.5,
      "raw_response": "1.5",
      "model": "claude-sonnet-4-20250514"
    },
    ...
  ],
  "gpt4o": [...]
}
```

## Comparison to Previous Evals

| Eval | Task | Dataset | GT Source | Metric |
|------|------|---------|-----------|--------|
| **Synthetic Depth** | Point depth | 10 synthetic images | Procedural depth | MAE: 1.98m (Claude), 2.25m (GPT) |
| **Two-Head Depth** | Point depth | Same synthetic | Procedural depth | MAE: 1.70m (Perception head) |
| **NYU Distance** | 3D distance | Real RGB-D indoor | Kinect + intrinsics | TBD |

## Next Steps

1. **Run initial eval**: 10 images × 3 pairs = 30 test cases
2. **Analyze failures**: Where do VLMs struggle?
3. **Compare to depth model**: How would Depth Anything V2 perform?
4. **Scale up**: Run on full 100-image set if promising

## Implementation Notes

### Why NYU Depth V2?
- ✅ Real indoor scenes (matches construction use case)
- ✅ Ground truth metric depth (Kinect sensor)
- ✅ Camera intrinsics provided
- ✅ Large dataset (1,449 labeled images)
- ✅ Standard benchmark in depth estimation

### Design Decisions
1. **Point marking**: Visual crosses instead of text coords (easier for VLM)
2. **Grid sampling**: Ensures coverage across entire image
3. **Depth filtering**: Avoids invalid/edge-case points
4. **Forced compliance**: Same prompt strategy as successful prior evals
5. **Parallel queries**: Claude and GPT-4o on identical test cases

### Potential Issues
- **Download time**: 2.8GB dataset takes ~10 min
- **API costs**: ~30 queries × 2 models = 60 VLM calls
- **Depth map quality**: Kinect depth has noise, especially at edges
- **VLM reasoning**: May struggle with absolute scale without anchors

## Files Created

```
data/nyu_depth_v2/
  download_nyu.py          # Dataset downloader + extractor

eval/nyu_distance/
  nyu_utils.py             # Core utilities (3D conversion, pair sampling)
  eval_nyu_distance.py     # Main evaluation script
  quick_setup.sh           # Setup helper
  README.md                # Documentation

NYU_DISTANCE_EVAL.md       # This file
```

## Branch Status

✅ Infrastructure complete
✅ Evaluation script ready
⏳ Dataset download pending (2.8GB)
⏳ Evaluation run pending

Ready to run once dataset is downloaded!
