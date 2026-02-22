# DeepAnchor: Spatial Anchor Calibration for Vision-Language Construction Inspection

> **Bridging the gap between visual recognition and real-world spatial reasoning in construction inspection.**

---

## Overview

Modern vision-language models (VLMs) like GPT-4o, Claude, and Gemini are remarkably capable at understanding scenes — but they consistently fail when asked to *measure* them. They can recognize that wall studs should be spaced 16 inches on center. They cannot tell you whether the studs in a photo actually are.

**DeepAnchor** solves this by turning ordinary construction site images into measurable, metrically-grounded environments — no LiDAR, no specialized hardware required. The system exploits a simple but powerful insight: construction sites are full of objects with known, standardized dimensions. By detecting these objects and using them as calibration anchors, DeepAnchor can estimate a pixel-to-meter scale factor and inject physically grounded measurements directly into a VLM's reasoning context.

The result: automated spatial inspection capable of catching defects — wrong stud spacing, improper rebar placement, missing clearances — *before* they get buried behind drywall.

---

## Process 

We were testing vision-language models on construction images — asking them things like "how far apart are these studs?" or "is this outlet at the right height?" — and the models would give these confident, articulate non-answers. They'd describe the scene beautifully and then either refuse to give a number or just make one up. We knew VLMs understood construction. That wasn't the problem. The problem was that they had no way to connect what they were seeing in pixels to what anything actually measured in the real world. That gap felt like a solvable problem, and that's where the project started. We had initial prototypes of the idea at first. Talking to Daniele early on gave us great direction. He encouraged us to build something with real research value and utility, including giving us the idea for ablation testing. We designed a four-condition ablation before writing most of the code — image only, depth cues only, anchors only, and the full pipeline — so that whatever we found, we'd be able to say something specific about why. We used the ARKitScenes dataset for benchmarking because it came with LiDAR-verified metric depth. Building the pipeline was super fun. The core idea — use standardized construction objects like bricks and CMU blocks as calibration anchors, compute a pixel-to-meter scale, inject the measurements into the prompt — came together pretty naturally. YOLO for detection, Depth Anything V2 for plane separation, consistency based on anchors in the same depth planes, median aggregation for robustness, structured JSON into the VLM. Clean stages, clear interfaces.

We worked super well together as a team. We divided tasks up and when tasks finished, there was always more to do. We also helped each other out if needed while having fun along the way. 

## Why This Matters

Construction rework costs the United States more than **$60 billion annually**. A spatial defect caught before enclosure costs hundreds of dollars to fix. The same defect discovered after drywall installation can cost **$50,000 or more** in demolition and rework.

Current AI vision systems can describe a construction scene but cannot reliably answer the one question that matters most to an inspector: *is this measurement within tolerance?*

DeepAnchor addresses this directly, enabling pass/fail inspection decisions grounded in physical units rather than visual guesswork.

---

## Approach

### Core Insight

Construction environments contain **standardized manufactured components** with fixed, known dimensions. These objects serve as natural calibration references — the computational equivalent of a trained inspector mentally calibrating distance against the known 3.5-inch width of a 2×4.


### Pipeline (5 Stages)

```
Raw Image
    │
    ▼
1. Anchor Detection      → YOLOv8 detects known-dimension objects (bricks, CMU blocks, boxes)
    │
    ▼
2. Depth Estimation      → Depth Anything V2 produces a dense relative depth map
    │
    ▼
3. Scale Calibration     → Pixel-to-meter scale computed from anchor bounding boxes;
                           median aggregation across multiple anchors for robustness
    │
    ▼
4. Spatial Measurement   → Distances, heights, and clearances extracted in real-world units;
                           depth-plane consistency enforced to prevent cross-plane errors
    │
    ▼
5. Context Injection     → Structured measurement JSON injected into VLM prompt;
                           model reasons over facts, not pixels
```

---

## Technical Details

### Anchor Detection

A YOLO-family detector fine-tuned on annotated construction imagery identifies anchor objects in each frame. Each detection returns a bounding box, class label, and confidence score. Only detections above a configurable confidence threshold (τ ∈ [0.25, 0.50]) are retained.

### Depth Estimation

Depth Anything V2 produces a dense depth map used not for metric depth recovery (which is scale-ambiguous from monocular images), but for **plane separation** — determining whether objects share a surface or lie at different depths. This prevents a critical failure mode: applying a scale factor derived from an anchor on one plane to measure an object on a different plane.

Per-object depth is estimated as the median depth within each bounding box:

```
z_k = median{ D(u,v) | (u,v) ∈ bounding_box_k }
```

### Scale Calibration

For each detected anchor, an independent scale estimate is computed:

```
s_k = pixel_extent_k / real_world_dimension_k
```

Multiple anchors are aggregated using the **median** to reject outliers caused by occlusion, detection noise, or perspective distortion:

```
s_calibrated = median(s_1, s_2, ..., s_n)
```

### Depth Adjustment

When an anchor and a measurement target lie on different depth planes, a perspective-based scale adjustment is applied:

```
s_adjusted = s_anchor × (depth_anchor / depth_target)
```

This compensates for the fact that farther objects appear smaller under perspective projection.

### Spatial Measurement

Once scale is calibrated, physical distance is recovered as:

```
d_meters = d_pixels / s_calibrated
```

All measurements — distances, heights, clearances — are packaged into a structured JSON object and injected into the VLM prompt alongside the original image. The model is instructed to reason over these quantitative facts rather than attempting visual estimation.

---

## Evaluation

### Benchmark: ARKitScenes LiDAR

We evaluate on the ARKitScenes dataset, which provides RGB frames paired with metrically-accurate LiDAR depth and camera intrinsics. Ground-truth distances are computed by back-projecting pixel coordinates into 3D space and measuring Euclidean distance. This benchmark isolates metric reasoning ability from image quality confounds.

### Ablation Results

| Condition                  | MAE (meters) | MAE (inches) |
|----------------------------|:------------:|:------------:|
| VLM Only                   | 1.27         | 50.0         |
| VLM + Depth Cues           | 2.66         | 104.7        |
| VLM + Anchors (no depth)   | 2.95         | 116.1        |
| **Full DeepAnchor (Ours)** | **1.06**     | **41.7**     |

### Comparison Against Closed-Source VLMs

| System                     | MAE (meters) | MAE (inches) |
|----------------------------|:------------:|:------------:|
| **DeepAnchor (Ours)**      | **1.28**     | **50.4**     |
| GPT-4o-mini                | 2.39         | 94.1         |
| GPT-4o                     | 2.55         | 100.4        |
| Claude Opus 4              | 2.77         | 109.1        |
| Claude Sonnet 4            | 2.83         | 111.4        |

DeepAnchor reduces MAE by **46–55% relative to all evaluated closed-source VLMs**.

---

## Key Findings

**1. Explicit scale grounding is the critical ingredient.**
VLMs already *know* construction standards. The failure is perceptual, not knowledge-based. Providing explicit metric measurements — rather than asking the model to infer them — is what unlocks reliable spatial reasoning.

**2. Depth cues alone can make things worse.**
Injecting relative depth without a calibrated scale causes models to over-trust an incorrect metric interpretation. Depth-only augmentation underperformed even the image-only baseline (2.66m vs. 1.27m MAE). Depth is useful for plane separation, not scale estimation.

**3. Anchors without depth calibration are fragile.**
Naive anchor-based scaling (without depth-plane consistency) performed worst overall (2.95m MAE) because anchors on different planes introduce incorrect scale factors. The full pipeline — anchors + depth-plane alignment — is what achieves the performance gain.

**4. The approach generalizes beyond construction.**
Any environment containing standardized manufactured components (hospitals, warehouses, manufacturing floors) can serve as a calibration field. No specialized hardware required.

---

## Applications

- **Construction QA**: Automated pass/fail verification of stud spacing, rebar placement, electrical box heights, and system clearances before enclosure steps.
- **Insurance & Claims**: Time-stamped, structured spatial evidence of site conditions for audit trails.
- **Compliance Reporting**: Jurisdiction-specific rule encoding with numeric deficiency justification.
- **Progress Monitoring**: Repeated captures compared against dimensional tolerances, reducing in-person walkthrough frequency.

---

## Limitations

- Requires at least one clearly visible, detectable anchor in the frame. Performance degrades when anchors are absent, occluded, or only partially visible.
- Assumes standardized components follow nominal dimensions — mixed sizes of similar-looking objects can introduce ambiguity.
- Monocular depth remains scale-ambiguous and can be noisy; depth errors propagate to depth-adjusted scale estimates.
- Performance is sensitive to detector quality under domain shift (challenging lighting, clutter, motion blur).
- Does not currently handle rotated anchors without orientation-aware detection.

---

## Future Work

- **Real-time video**: Optimize the pipeline for live construction video streams, enabling continuous monitoring.
- **BIM integration**: Automatically compare as-built measurements against Building Information Model specifications.
- **Additional trades**: Extend anchor library and detection models to HVAC, plumbing, and fire protection systems.
- **Foundation model integration**: Incorporate Grounded-SAM for more robust anchor segmentation and orientation handling.

---

## Repository Structure

```
deepanchor/
├── detection/          # YOLOv8 anchor detector (training + inference)
├── depth/              # Depth Anything V2 integration
├── calibration/        # Scale calibration and depth-plane logic
├── measurement/        # Spatial measurement extraction
├── injection/          # VLM context injection and prompt templates
├── evaluation/         # ARKitScenes benchmark scripts
├── data/               # Anchor dimension lookup tables
└── README.md
```

---

## Citation

```bibtex
@article{nakhawa2025deepanchor,
  title={DeepAnchor: Spatial Anchor Calibration for Vision-Language Construction Inspection},
  author={Nakhawa, Ankit and Mandal, Souptik and Kataria, Utsav and Kotha, Vishal and Singh, Eshan},
  year={2025},
  institution={University of Maryland}
}
```

---

*DeepAnchor demonstrates that physically grounded AI perception is achievable today — using commodity images, standardized objects, and a principled calibration pipeline — without waiting for affordable metric depth sensors or next-generation VLMs.*
