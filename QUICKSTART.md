# Quick Start Guide - VLM Depth Evals

This is a quick reference to get started with evaluating OpenAI and Anthropic VLMs on depth estimation.

## ✅ What's Already Set Up

- ✓ API keys configured in `.env` file
- ✓ Python dependencies installed
- ✓ VLM clients for OpenAI (GPT-4o) and Anthropic (Claude 3.5 Sonnet)
- ✓ Evaluation framework with metrics
- ✓ Directory structure created

## 📋 What You Need to Do

### 1. Prepare Your Data

Create `data/processed/frames.csv` pointing to your ARKitScenes images and depth maps:

```csv
frame_id,rgb_path,depth_gt_path
frame_001,path/to/rgb/001.jpg,path/to/depth/001.npy
frame_002,path/to/rgb/002.jpg,path/to/depth/002.npy
```

See `data/processed/frames_template.csv` for the format.

### 2. Run Evaluations

**Test with OpenAI GPT-4o** (5 images, 8 points each):
```bash
python eval/runners/eval_vlm_points.py \
  --frames_csv data/processed/frames.csv \
  --model_type openai \
  --model_name gpt-4o \
  --points_per_image 8 \
  --max_images 5 \
  --out_jsonl outputs/predictions/vlm_answers/test_gpt4o.jsonl
```

**Test with Anthropic Claude**:
```bash
python eval/runners/eval_vlm_points.py \
  --frames_csv data/processed/frames.csv \
  --model_type anthropic \
  --model_name claude-3-5-sonnet-20241022 \
  --points_per_image 8 \
  --max_images 5 \
  --out_jsonl outputs/predictions/vlm_answers/test_claude.jsonl
```

**Full evaluation** (more images, more points):
```bash
python eval/runners/eval_vlm_points.py \
  --frames_csv data/processed/frames.csv \
  --model_type openai \
  --points_per_image 16 \
  --out_jsonl outputs/predictions/vlm_answers/gpt4o_full.jsonl
```

### 3. View Results

After running, you'll see metrics printed:
- **MAE**: Mean Absolute Error in meters
- **RMSE**: Root Mean Squared Error
- **AbsRel**: Absolute Relative Error

Results are saved to:
- Predictions: `outputs/predictions/vlm_answers/*.jsonl`
- Metrics: `outputs/predictions/vlm_answers/*_metrics.json`

## 📊 Expected Performance

For indoor scenes, good depth models achieve:
- MAE: 0.10-0.30 meters
- AbsRel: 0.05-0.15

VLMs will likely have higher errors initially since they're not trained specifically for metric depth.

## 🔍 Troubleshooting

**"No such file: data/processed/frames.csv"**
→ You need to create this file first (see step 1 above)

**"API key not found"**
→ Make sure `.env` file exists and is loaded

**"Out of memory"**
→ Reduce `--points_per_image` or use `--max_images`

## 📚 More Details

- See [SETUP_EVALS.md](SETUP_EVALS.md) for detailed setup instructions
- See [ARKitScenes_Depth_Evals_README.md](ARKitScenes_Depth_Evals_README.md) for full evaluation methodology

## 🧪 Verify Setup

Run the test script to check everything is working:
```bash
python test_vlm_setup.py
```

You should see:
```
✓ OpenAI API key found
✓ Anthropic API key found
✓ OpenAI client initialized successfully
✓ Anthropic client initialized successfully
```
