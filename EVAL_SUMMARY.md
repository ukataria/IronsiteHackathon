# VLM Depth Evaluation - Executive Summary

## Test Overview
- **Dataset:** 10 synthetic frames with ground truth depth (1.5-5.0m range)
- **Evaluation:** 20 point queries across 5 images
- **Models:** OpenAI GPT-4o vs Anthropic Claude 3 Haiku

## Results at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPTH ESTIMATION ACCURACY                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Traditional Depth Models                                       │
│  ████ MAE: 0.10-0.30m  ⭐⭐⭐⭐⭐  BASELINE                      │
│                                                                 │
│  OpenAI GPT-4o                                                  │
│  ██████████████ MAE: 1.41m  ⭐⭐  ~5-14x worse than baseline    │
│                                                                 │
│  Claude 3 Haiku                                                 │
│  ████████████████████████████████████████ MAE: 32.48m          │
│  ⚠️  ~108-325x worse than baseline, 23x worse than GPT-4o      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Metrics

| Model | MAE | RMSE | AbsRel | Valid % |
|-------|-----|------|--------|---------|
| **Depth Models (baseline)** | 0.10-0.30m | - | 0.05-0.15 | - |
| **GPT-4o** | 1.41m | 1.67m | 0.30 | 100% ✓ |
| **Claude 3 Haiku** | 32.48m | 33.34m | 6.47 | 100% ✗ |

## Key Insights

### ✅ What Works
- **Both models respond** with numeric values (100% response rate)
- **GPT-4o shows genuine depth understanding** (errors ~1-2m range)
- **Task comprehension** is good for both models

### ❌ What Doesn't Work
- **Neither model accurate enough** for metric measurements
- **Claude Haiku essentially guesses** (~40m constant answer)
- **Far below traditional depth models** (5-325x worse)

### 🎯 Practical Implications

**For Construction Inspection:**
```
✓ Use depth models for:        ❌ Don't use VLMs for:
  - Stud spacing measurements     - Precise distances
  - Wall planarity checks         - Sub-meter accuracy needs
  - Distance calculations         - Safety-critical measurements
  - Code compliance metrics       - Structural dimensions

✓ Use VLMs (GPT-4o) for:       ⚠️ Use with caution:
  - Defect detection              - Claude Haiku for vision tasks
  - Code violation reasoning      - Any VLM for precise depth
  - Spatial relationships         - Cost-sensitive applications
  - Inspection reports
```

## Recommendations

### Immediate Actions
1. **Use traditional depth models** (Depth Anything, ZoeDepth) for all metric measurements
2. **Use GPT-4o** for spatial reasoning and inspection narrative
3. **Avoid Claude 3 Haiku** for vision-intensive tasks
4. **Combine approaches:** Depth model → measurements → VLM → interpretation

### To Improve VLM Performance
1. ⬆️ **Upgrade Claude tier** - Get Claude 3.5 Sonnet (should match GPT-4o)
2. 📝 **Better prompts** - Add camera intrinsics, reference objects
3. 🎯 **Few-shot learning** - Provide example depth annotations
4. 🖼️ **Real images** - Test with actual construction photos

### For Demo/Production
```python
# Recommended pipeline:
depth_map = depth_model.predict(image)        # ZoeDepth/Depth Anything
measurements = extract_measurements(depth_map) # Accurate metrics
analysis = vlm.analyze(image, measurements)   # GPT-4o interpretation
```

## Cost Analysis

**Per 1000 images (8 points each):**

| Model | Cost | Accuracy | Cost/Performance Ratio |
|-------|------|----------|------------------------|
| GPT-4o | ~$40 | Moderate | Fair ⭐⭐⭐ |
| Claude Haiku | ~$4 | Very Poor | Poor ⭐ (not worth it) |
| Claude 3.5 Sonnet | ~$24 | Expected: Good | Expected: Best ⭐⭐⭐⭐ |

**Recommendation:** Skip Haiku, use GPT-4o or upgrade to Claude 3.5 Sonnet

## Files & Documentation

- 📊 [EVAL_RESULTS.md](EVAL_RESULTS.md) - Detailed technical results
- 🚀 [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- 📖 [SETUP_EVALS.md](SETUP_EVALS.md) - Setup instructions
- 📁 `outputs/predictions/vlm_answers/` - Raw prediction files

## Bottom Line

**Can VLMs replace depth models?** No.

**Can VLMs complement depth models?** Yes, with GPT-4o or Claude 3.5 Sonnet.

**Best approach?** Use depth models for measurements, VLMs for reasoning.

---

*Evaluation completed: 2026-02-21*
*Framework: evals branch*
*Total queries: 40 (20 completed successfully per model)*
