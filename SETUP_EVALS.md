# Setup Guide for VLM Depth Evals

This guide will help you run depth evaluations using OpenAI and Anthropic VLMs.

## Prerequisites

1. **Python Environment**: Python 3.10+
2. **API Keys**: Already configured in `.env` file
3. **ARKitScenes Data**: You need RGB images and ground truth depth maps

## Installation Steps

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Your Data

You need to create a CSV file that indexes your evaluation frames. The CSV should have these columns:

- `frame_id`: Unique identifier for the frame
- `rgb_path`: Path to RGB image
- `depth_gt_path`: Path to ground truth depth map (`.npy` or 16-bit `.png` in mm)

**Example CSV** (`data/processed/frames.csv`):

```csv
frame_id,rgb_path,depth_gt_path
frame_001,data/raw/scene_001/rgb/0001.jpg,data/raw/scene_001/depth/0001.npy
frame_002,data/raw/scene_001/rgb/0002.jpg,data/raw/scene_001/depth/0002.npy
```

### 3. Run VLM Evaluation

#### Evaluate with OpenAI GPT-4o:

```bash
python eval/runners/eval_vlm_points.py \
  --frames_csv data/processed/frames.csv \
  --model_type openai \
  --model_name gpt-4o \
  --points_per_image 16 \
  --out_jsonl outputs/predictions/vlm_answers/gpt4o_points.jsonl
```

#### Evaluate with Anthropic Claude:

```bash
python eval/runners/eval_vlm_points.py \
  --frames_csv data/processed/frames.csv \
  --model_type anthropic \
  --model_name claude-3-5-sonnet-20241022 \
  --points_per_image 16 \
  --out_jsonl outputs/predictions/vlm_answers/claude_points.jsonl
```

#### Quick Test (limit to 5 images):

```bash
python eval/runners/eval_vlm_points.py \
  --frames_csv data/processed/frames.csv \
  --model_type openai \
  --points_per_image 8 \
  --max_images 5 \
  --out_jsonl outputs/predictions/vlm_answers/test_gpt4o.jsonl
```

## Output

The evaluation will produce:

1. **Predictions JSONL**: Contains all VLM responses and comparisons
   - Located at: `outputs/predictions/vlm_answers/`

2. **Metrics JSON**: Summary statistics
   - MAE (Mean Absolute Error in meters)
   - RMSE (Root Mean Squared Error)
   - AbsRel (Absolute Relative Error)

## Expected Performance

For indoor scenes, good metric depth models typically achieve:
- MAE: 0.10-0.30 meters
- AbsRel: 0.05-0.15

VLMs are expected to have higher error rates initially, as they're not specifically trained for metric depth estimation.

## Troubleshooting

### Missing frames.csv
If you don't have a frames.csv file yet, you need to create one pointing to your ARKitScenes data.

### API Key Errors
Make sure the `.env` file exists and contains valid API keys:
```bash
source .env
```

### Out of Memory
Reduce `--points_per_image` or use `--max_images` to limit evaluation scope.

## Next Steps

1. Compare OpenAI vs Anthropic performance
2. Experiment with different prompt templates
3. Try different sampling strategies (uniform vs stratified)
4. Analyze failure cases to improve prompts
