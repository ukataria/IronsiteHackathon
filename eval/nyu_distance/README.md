# NYU Depth V2 Distance Evaluation

Evaluation of VLMs (Claude Sonnet 4.6 vs GPT-4o) on 3D distance estimation using NYU Depth V2 dataset.

## Task

Given two marked points in an indoor RGB image, estimate the real-world 3D Euclidean distance between them.

## Methodology

### Ground Truth Computation

Using NYU Depth V2's calibrated depth maps and camera intrinsics:

```python
# Convert pixel (u, v) to 3D point
X = (u - cx) * depth(u,v) / fx
Y = (v - cy) * depth(u,v) / fy
Z = depth(u,v)

# Euclidean distance between two points
distance = sqrt((X1-X2)² + (Y1-Y2)² + (Z1-Z2)²)
```

### Camera Intrinsics (NYU Kinect)
- fx = 518.857901
- fy = 519.469611
- cx = 325.582244
- cy = 253.736347
- Resolution: 640x480

### VLM Query Format

- **Input**: RGB image with two points marked (red cross "A" and blue cross "B")
- **Prompt**: "Estimate the 3D distance in meters between point A and point B"
- **Expected output**: Single numeric value in meters

## Setup

### 1. Download NYU Depth V2 Dataset

**Option A: Automatic download**
```bash
python data/nyu_depth_v2/download_nyu.py
```

**Option B: Manual download**
1. Download `nyu_depth_v2_labeled.mat` (~2.8GB) from:
   http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

2. Place in `data/nyu_depth_v2/`

3. Extract samples:
```bash
python data/nyu_depth_v2/download_nyu.py
```

This extracts 100 samples to `data/nyu_depth_v2/extracted/`

### 2. Run Evaluation

```bash
python eval/nyu_distance/eval_nyu_distance.py \
  --nyu_data_dir data/nyu_depth_v2/extracted \
  --num_images 10 \
  --pairs_per_image 3
```

## Expected Results

### Metrics
- **MAE (Mean Absolute Error)**: Average error in meters
- **Median Error**: Median error in meters
- **Response Rate**: % of successful numeric predictions

### Hypothesis
- VLMs will struggle with absolute 3D distance estimation
- Expected MAE: 0.5-1.5m for typical indoor distances of 1-4m
- Claude Sonnet 4.6 may outperform GPT-4o based on prior depth evals

## Output

Results saved to:
- `outputs/nyu_distance/nyu_distance_results.json` - Full results
- `outputs/nyu_distance/marked_images/` - Visualizations with marked points

## Comparison to Depth Eval

| Evaluation | Task | Ground Truth | Metric |
|------------|------|--------------|--------|
| **Previous (Synthetic)** | Point depth estimation | Synthetic depth map | MAE in meters |
| **This (NYU)** | 3D distance between points | Real RGB-D data + intrinsics | MAE in meters |

NYU eval is harder because:
- Real noisy indoor scenes (vs clean synthetic)
- 3D distance requires reasoning about both points
- More realistic test of spatial understanding
