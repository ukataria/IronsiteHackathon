# GPU Server Setup Instructions

**Quick Start Guide for Running Full VLM Benchmark**

---

## ✅ What's Ready

- ✅ **13 VLM models** configured (API + open-source + two-head)
- ✅ **Two-head model** fully implemented (anchor detection + depth + calibration + VLM)
- ✅ **NYU Depth V2 dataset** downloaded (100 test images)
- ✅ **Benchmark framework** with progress tracking and visualization
- ✅ **GPU dependencies** documented in `requirements-gpu.txt`
- ✅ **Automated runner** script: `run_full_benchmark_gpu.sh`

---

## 🚀 Quick Start (Copy-Paste)

```bash
# 1. Navigate to project
cd /path/to/IronsiteHackathon
git checkout nyu-distance-evals
git pull

# 2. Install GPU dependencies
pip install -r requirements-gpu.txt

# 3. Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 4. Set API keys (if not already set)
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
EOF

# 5. Run benchmark (interactive)
./run_full_benchmark_gpu.sh

# Or run specific models directly:
python eval/nyu_distance/benchmark_all_models.py \
  --nyu_data_dir data/nyu_depth_v2/extracted \
  --num_images 10 \
  --pairs_per_image 3 \
  --models two-head-claude gpt-4o claude-sonnet-4 \
  --device cuda \
  --out_dir outputs/gpu_benchmark
```

---

## 📊 Models Available

### Closed Source (API)
- ✅ **gpt-4o** - OpenAI GPT-4o (baseline: ~0.69m MAE)
- ✅ **claude-sonnet-4** - Anthropic Claude Sonnet 4 (baseline: ~0.73m MAE)
- ⚠️ **gemini-2.5-pro** - Google Gemini (VERY SLOW: ~40s/query, may skip)

### Open Research (Requires GPU)
- 🔧 **internvl3** - OpenGVLab InternVL3-8B
- 🔧 **qwen2.5-vl** - Qwen 2.5 VL 7B Instruct
- 🔧 **llava-onevision** - LLaVA OneVision
- 🔧 **pixtral** - Mistral Pixtral
- 🔧 **kimi-vl** - Kimi VL (optional)

### Edge Deployable (CPU OK)
- 🔧 **qwen-vl-7b** - Qwen VL 7B
- 🔧 **minicpm-v** - MiniCPM-V
- 🔧 **phi-multimodal** - Phi Multimodal

### Two-Head (Ours) 🔴
- ✅ **two-head-claude** - Spatial Anchor Calibration + Claude Sonnet 4
  - YOLOv8 anchor detection
  - Depth Anything V2 Large depth estimation
  - Multi-anchor spatial calibration
  - Claude Sonnet 4 VLM reasoning

---

## 🎯 Recommended Test Plan

### Test 1: Two-Head vs Baselines (Priority)
```bash
python eval/nyu_distance/benchmark_all_models.py \
  --nyu_data_dir data/nyu_depth_v2/extracted \
  --num_images 20 \
  --pairs_per_image 3 \
  --models two-head-claude gpt-4o claude-sonnet-4 \
  --device cuda \
  --out_dir outputs/two_head_vs_baselines
```

**Runtime**: ~10-15 minutes (60 queries × 3 models)

### Test 2: Full Open-Source Models
```bash
python eval/nyu_distance/benchmark_all_models.py \
  --nyu_data_dir data/nyu_depth_v2/extracted \
  --num_images 10 \
  --pairs_per_image 3 \
  --models internvl3 qwen2.5-vl llava-onevision \
  --device cuda \
  --out_dir outputs/open_source_models
```

**Runtime**: ~20-30 minutes (depends on model loading time)

### Test 3: Everything (Except Gemini)
```bash
./run_full_benchmark_gpu.sh
# Select option 1 (all models)
```

**Runtime**: ~60-90 minutes

---

## 📁 Output Files

After running, check:

```bash
outputs/gpu_benchmark/
├── benchmark_results.json       # Full results with all predictions
├── benchmark_comparison.png     # 4-plot visualization
└── marked_images/               # Test images with point annotations
    ├── two-head-claude_img0000_pair0.jpg
    ├── gpt-4o_img0000_pair0.jpg
    └── ...
```

**View results**:
```bash
# JSON results
cat outputs/gpu_benchmark/benchmark_results.json | jq '.summary'

# Visualization
open outputs/gpu_benchmark/benchmark_comparison.png

# Or on headless server, copy to local:
scp user@gpu-server:~/IronsiteHackathon/outputs/gpu_benchmark/*.png .
```

---

## 🔍 Expected Results

### Baseline Performance (Already Known)
- **GPT-4o**: 0.689m MAE, 100% success
- **Claude Sonnet 4**: 0.729m MAE, 100% success
- **Gemini 2.5 Pro**: Unknown (too slow to test fully)

### Two-Head Model Hypothesis

**Best Case** (many anchors detected):
- MAE: **0.3-0.5m** (30-40% improvement over baseline)
- Calibration success: 60-80% of queries
- Fallback to VLM: 20-40% of queries

**Likely Case** (NYU dataset has few construction anchors):
- MAE: **0.5-0.6m** (15-25% improvement)
- Calibration success: 30-50% of queries
- Fallback to VLM: 50-70% of queries

**Worst Case** (no anchors detected):
- MAE: **~0.73m** (same as Claude baseline)
- Calibration success: 0%
- Fallback to VLM: 100%

**Why NYU may be challenging**:
- Indoor residential scenes, not construction sites
- Fewer known-dimension objects (no studs, rebar, CMU blocks)
- YOLO trained on generic objects (COCO dataset), not construction-specific

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller models or reduce batch size
# Two-head model needs ~8GB VRAM for Depth Anything V2 Large
# Open-source VLMs need 12-16GB each

# If OOM, try smaller depth model:
# Edit models/depth_estimator.py, change "large" to "base"
```

### Missing Dependencies
```bash
pip install ultralytics transformers accelerate
pip install einops timm  # For InternVL3
```

### Model Download Failures
```bash
# HuggingFace models auto-download on first run
# May take 10-20 minutes for large models
# Ensure internet connection and HF_HOME has space

export HF_HOME=/path/with/space
```

### YOLO/Anchor Detection Fails
```bash
# This is expected on NYU dataset (no construction objects)
# Two-head will automatically fallback to VLM-only mode
# Check results JSON for "calibrated": true/false
```

---

## 📊 Interpreting Results

### Benchmark Summary Table
```
          model category      mae  median_error  success_rate
         gpt-4o   closed 0.689187      0.274120         100.0
claude-sonnet-4   closed 0.728773      0.659645         100.0
two-head-claude two-head 0.XXXXXX      0.XXXXXX         100.0  ← Look here!
```

### Key Questions to Answer

1. **Is two-head better?**
   - Compare `two-head-claude` MAE vs `claude-sonnet-4` MAE
   - If lower by >15%, calibration is working

2. **How often does calibration succeed?**
   ```bash
   cat outputs/gpu_benchmark/benchmark_results.json | jq \
     '[.results."two-head-claude"[] | select(.calibrated == true)] | length'
   ```

3. **Accuracy when calibrated vs uncalibrated?**
   ```bash
   # Filter calibrated queries
   jq '[.results."two-head-claude"[] | select(.calibrated == true) | .error] | add/length' results.json

   # Filter uncalibrated queries
   jq '[.results."two-head-claude"[] | select(.calibrated == false) | .error] | add/length' results.json
   ```

4. **Visualization interpretation**:
   - **Red bar** = Two-head performance
   - **Blue bars** = Closed-source baselines
   - Shorter bar = better (lower MAE)

---

## 🎬 Next Steps After Results

### If Two-Head Wins (MAE < 0.6m)
1. ✅ **Success!** Spatial calibration works
2. Test on actual construction photos (with visible studs/rebar)
3. Fine-tune YOLO on construction-specific anchors
4. Expected improvement: 2-3x better on construction data

### If Two-Head Ties (~0.7m MAE)
1. Check calibration success rate (likely low on NYU)
2. Validate on construction photos with known anchors
3. Two-head architecture is sound, needs better anchor detection

### If Open-Source Models Compete
1. Compare InternVL3 and Qwen2.5-VL to GPT-4o
2. If competitive, huge cost savings (free vs API)
3. Consider two-head + open-source VLM combination

---

## 📞 Support

**Implementation**: Complete ✅
**Documentation**: [TWO_HEAD_MODEL.md](TWO_HEAD_MODEL.md)
**Troubleshooting**: Check this file

**Ready to run on GPU server!** 🚀

---

## ⚡ One-Liner Test Commands

```bash
# Quick test (3 images = 9 queries, ~2 min)
python eval/nyu_distance/benchmark_all_models.py --nyu_data_dir data/nyu_depth_v2/extracted --num_images 3 --pairs_per_image 3 --models two-head-claude gpt-4o --device cuda --out_dir outputs/quick_test

# Full test (20 images = 60 queries, ~15 min)
python eval/nyu_distance/benchmark_all_models.py --nyu_data_dir data/nyu_depth_v2/extracted --num_images 20 --pairs_per_image 3 --models two-head-claude gpt-4o claude-sonnet-4 --device cuda --out_dir outputs/full_test

# Two-head only (for debugging)
python eval/nyu_distance/benchmark_all_models.py --nyu_data_dir data/nyu_depth_v2/extracted --num_images 5 --pairs_per_image 2 --models two-head-claude --device cuda --out_dir outputs/two_head_only
```

Good luck! 🎯
