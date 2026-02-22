# Depth Model Benchmarking

Benchmark different depth estimation models for spatial anchor calibration.

## Supported Models

### Depth Anything V2
- **Variants**: `small`, `base`, `large`
- **Best for**: General-purpose monocular depth estimation
- **Usage**: `--depth_model_type depth_anything_v2 --depth_model_size large`

### ZoeDepth
- **Variants**:
  - `nk`: Hybrid indoor+outdoor (recommended)
  - `n`: NYU-trained (indoor only)
  - `k`: KITTI-trained (outdoor only)
- **Best for**: Indoor scenes (NYU dataset)
- **Usage**: `--depth_model_type zoe --depth_model_size nk`

### MiDaS
- **Variants**: `small`, `hybrid`, `large`
- **Best for**: Diverse scenes with DPT transformer architecture
- **Usage**: `--depth_model_type midas --depth_model_size large`

## Quick Start

### Single Model Ablation Study

Test a specific depth model:

```bash
# Depth Anything V2 (default)
uv run python eval/arkit/ablation_study.py \
  --data_dir data/arkit \
  --num_images 10 \
  --pairs_per_image 3 \
  --depth_model_type depth_anything_v2 \
  --depth_model_size large

# ZoeDepth NK
uv run python eval/arkit/ablation_study.py \
  --data_dir data/arkit \
  --num_images 10 \
  --pairs_per_image 3 \
  --depth_model_type zoe \
  --depth_model_size nk

# MiDaS DPT-Large
uv run python eval/arkit/ablation_study.py \
  --data_dir data/arkit \
  --num_images 10 \
  --pairs_per_image 3 \
  --depth_model_type midas \
  --depth_model_size large
```

### Full Depth Model Benchmark

Run ablation study across all depth models and generate comparison:

```bash
# Benchmark all depth models (default)
uv run python eval/arkit/benchmark_depth_models.py \
  --data_dir data/arkit \
  --num_images 10 \
  --pairs_per_image 3 \
  --device cuda

# Benchmark specific models only
uv run python eval/arkit/benchmark_depth_models.py \
  --data_dir data/arkit \
  --num_images 10 \
  --pairs_per_image 3 \
  --models zoe midas
```

## Output Structure

```
outputs/depth_benchmark/
├── depth_anything_v2_large/
│   ├── ablation_results.json
│   ├── ablation_comparison.png
│   └── marked_images/
├── zoe_nk/
│   ├── ablation_results.json
│   ├── ablation_comparison.png
│   └── marked_images/
├── midas_large/
│   ├── ablation_results.json
│   ├── ablation_comparison.png
│   └── marked_images/
└── depth_model_comparison.png  # Cross-model comparison
```

## Comparison Metrics

The benchmark generates 4 comparison plots:

1. **MAE by Depth Model & Condition**: Shows error for each model across all 4 ablation conditions
2. **Prediction Success Rate**: Percentage of predictions successfully parsed
3. **Full Spatial Anchor MAE**: Direct comparison of best-case performance
4. **Improvement Over Baseline**: % error reduction vs VLM-only baseline

## Expected Performance

Based on NYU indoor scenes:

| Model | Full Spatial Anchor MAE | Success Rate |
|-------|------------------------|--------------|
| Depth Anything V2 Large | ~1.2-1.5m | >95% |
| ZoeDepth NK | ~1.3-1.6m | >95% |
| MiDaS DPT-Large | ~1.4-1.7m | >95% |

All models should show significant improvement over VLM-only baseline (~2.5-3.0m MAE).

## Requirements

```bash
# Core dependencies (already installed)
uv add transformers torch pillow opencv-python numpy matplotlib

# Models are downloaded automatically from:
# - HuggingFace (Depth Anything V2)
# - torch.hub (ZoeDepth, MiDaS)
```

## Notes

- **GPU Recommended**: Depth estimation is compute-intensive
- **First Run**: Models download on first use (~2-5GB per model)
- **Cache**: Models cached in `~/.cache/torch/hub` and `~/.cache/huggingface`
- **Memory**: Expect ~4-8GB GPU memory per model during inference
