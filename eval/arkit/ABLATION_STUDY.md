# Ablation Study: Spatial Anchor Calibration Components

This ablation study isolates the contribution of each component in the Spatial Anchor Calibration approach.

## Conditions Tested

### 1. VLM Only (Baseline)
- Pure Claude Sonnet 4 vision-language model
- No additional information
- Tests baseline VLM spatial reasoning capability

### 2. VLM + Depth
- Claude Sonnet 4 + monocular depth estimates
- Provides depth values at query points from Depth Anything V2
- Tests whether depth information alone improves accuracy

### 3. VLM + Anchors (No Depth)
- Claude Sonnet 4 + detected objects with known dimensions
- YOLO detects objects, provides their known real-world sizes
- Tests whether anchor information without depth helps calibration

### 4. Full Spatial Anchor Calibration
- Claude Sonnet 4 + depth + anchors + geometric calibration
- Complete pipeline: anchor detection → depth estimation → calibration → measurement
- Tests full system performance

## Running the Ablation Study

```bash
# Basic usage
uv run python eval/arkit/ablation_study.py \
  --data_dir data/arkit \
  --num_images 10 \
  --pairs_per_image 3 \
  --device cuda

# Larger test set
uv run python eval/arkit/ablation_study.py \
  --data_dir data/arkit \
  --num_images 50 \
  --pairs_per_image 5 \
  --output_dir outputs/ablation_large \
  --device cuda
```

## Output

The script generates:

1. **ablation_results.json** - Detailed results for all conditions
2. **ablation_comparison.png** - Visualization with 4 plots:
   - Mean Absolute Error comparison
   - Success rate (% predictions parsed successfully)
   - Median error comparison
   - Error distribution box plots
3. **marked_images/** - Annotated images for each condition

## Expected Results

Based on our hypothesis:

- **VLM Only**: Baseline performance (~2.5-3m MAE)
- **VLM + Depth**: Minor improvement with depth cues (~2.3-2.7m MAE)
- **VLM + Anchors**: Better calibration context (~2.0-2.5m MAE)
- **Full Spatial Anchor**: Best performance with geometric calibration (~1.2-1.5m MAE)

The ablation demonstrates:
- How much each component contributes
- Whether depth alone helps
- Whether anchors alone help
- The synergy of combining depth + anchors + calibration

## Analysis

Key questions answered:

1. **Does depth help?** Compare Condition 1 vs 2
2. **Do anchors help?** Compare Condition 1 vs 3
3. **Is calibration necessary?** Compare Condition 3 vs 4
4. **What's the total gain?** Compare Condition 1 vs 4

## Requirements

- ANTHROPIC_API_KEY environment variable
- CUDA-capable GPU (for depth estimation)
- ARKit LiDAR depth data
