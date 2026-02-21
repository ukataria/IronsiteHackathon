# VLM Depth Eval - Quick Reference Card

## 🎯 Bottom Line

**Can VLMs measure depth accurately?** No. (~10-29x worse than depth models)

**Should you use VLMs for depth?** Not for measurements. Yes for interpretation.

## 📊 Final Scores

| Model | MAE | Response Rate | Use Case |
|-------|-----|---------------|----------|
| Depth Models | 0.10-0.30m | 100% | ✅ All measurements |
| GPT-4o | 2.92m | 100% | ✅ Spatial reasoning |
| Claude Sonnet 4.6 | 2.92m | 5%* | ✅ Conservative analysis |

*Claude mostly refused unrealistic synthetic images (smart!)

## ✅ Use VLMs For

- Defect detection
- Code compliance reasoning
- Spatial relationship descriptions
- Inspection report generation
- Interpreting measurements from depth models

## ❌ Don't Use VLMs For

- Precise distance measurements
- Stud spacing calculations
- Wall planarity checks
- Any task requiring <1m accuracy
- Safety-critical depth sensing

## 💡 Recommended Architecture

```
Image → Depth Model → Measurements → VLM → Analysis
        (accurate)   (0.1-0.3m error)   (reasoning)
```

## 🔧 Model Selection

**GPT-4o:** High compliance, always responds, ~$10/1k images
**Claude Sonnet 4.6:** More cautious, better reasoning, ~$11/1k images
**Pick GPT-4o** if you want consistent API behavior
**Pick Claude** if you value conservative/honest answers

## 📁 Key Files

- [FINAL_EVAL_SUMMARY.md](FINAL_EVAL_SUMMARY.md) - Complete analysis
- [EVAL_RESULTS.md](EVAL_RESULTS.md) - Detailed results
- [QUICKSTART.md](QUICKSTART.md) - How to run evals
- `outputs/predictions/vlm_answers/` - Raw data

## 🚀 For Your Demo

**DO:**
- Use Depth Anything V2 for measurements
- Use GPT-4o for explaining what's wrong
- Show VLM interpreting depth data

**DON'T:**
- Claim VLMs can measure distances
- Use VLMs alone for metric measurements
- Skip the depth model step

---
*Eval Date: 2026-02-21 | Models: GPT-4o, Claude Sonnet 4.6*
