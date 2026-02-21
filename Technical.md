# TECHNICAL.md — PreCheck: Spatial Anchor Calibration for Construction Inspection

## 1. Problem Statement

### The Spatial Intelligence Gap
Current vision-language models (GPT-5, Claude, Gemini) can identify objects in construction scenes but fundamentally cannot reason about their spatial relationships in real-world units. Ask any frontier model "what is the spacing between these studs?" and it will either refuse to answer, hallucinate a number, or give a vague qualitative response like "they appear evenly spaced."

This is not a knowledge gap — these models know that studs should be 16 inches on center. It's a **perception gap**. They cannot extract real-world measurements from pixels because they have no mechanism for converting pixel distances to physical distances.

### Why This Matters in Construction
Before any irreversible construction step — pouring concrete, closing walls with drywall, backfilling foundations — the work must be inspected for correctness. The critical inspection questions are inherently spatial:
- Is the rebar spaced at 12 inches on center as specified?
- Are the studs at 16 inches on center?
- Is the electrical box at the correct height?
- Are nail plates installed on all penetrations?
- Is there adequate clearance between systems?

A spatial error caught before closeup costs $500 to fix. The same error caught after costs $50,000-$100,000+ in demolition and rework. Construction rework is estimated at $60+ billion annually in the US, and a significant portion originates from spatial errors that were visually present but not caught at the inspection stage.

### The Human Baseline
A human inspector unconsciously calibrates spatial perception using reference objects. They know a 2x4 is 3.5 inches wide. They know what 16 inches looks like relative to a stud. They mentally convert pixel-level visual information to physical measurements using these anchors. This is the capability we replicate.

---

## 2. Technique: Spatial Anchor Calibration

### Core Insight
Construction environments are uniquely rich in **known-dimension objects** — materials manufactured to precise, standardized sizes. These objects serve as natural calibration anchors that bridge the gap between pixel space and physical space.

### Known Anchor Dimensions
```
MATERIAL                    DIMENSION              USE CASE
─────────────────────────────────────────────────────────────
2x4 lumber (face)           3.5 inches wide        Wall framing calibration
2x4 lumber (edge)           1.5 inches deep         
2x6 lumber (face)           5.5 inches wide        Floor joist calibration
CMU block                   15.625 x 7.625 inches  Foundation wall calibration
Standard rebar #4           0.500 inch diameter     Rebar grid calibration
Standard rebar #5           0.625 inch diameter     
Electrical box (single)     2 x 3 inches           MEP calibration
Electrical box (double)     4 x 3 inches           
Door rough opening          38.5 x 82.5 inches     Large-scale calibration
Hard hat                    ~12 inches across       Personnel area calibration
Standard brick              7.625 x 2.25 inches    Masonry calibration
Plywood sheet               48 x 96 inches         Sheathing calibration
```

### Pipeline Architecture

```
INPUT: Construction photo (stable, roughly perpendicular to work surface)
                    │
                    ▼
    ┌───────────────────────────────┐
    │   STAGE 1: ANCHOR DETECTION  │
    │                               │
    │   Model: YOLO / GroundedSAM  │
    │   Input: Raw image            │
    │   Output: Bounding boxes +    │
    │     classifications for all   │
    │     known-dimension objects   │
    │                               │
    │   Detects: studs, rebar,     │
    │     CMU blocks, elec boxes,  │
    │     door openings, etc.      │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │   STAGE 2: DEPTH ESTIMATION  │
    │                               │
    │   Model: Depth Anything V2   │
    │   Input: Raw image            │
    │   Output: Relative depth map │
    │     (.npy array + .png vis)  │
    │                               │
    │   Purpose: Identify depth    │
    │     planes, group objects on │
    │     same surface, determine  │
    │     which anchors share a    │
    │     plane with target objects│
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │   STAGE 3: SCALE CALIBRATION │
    │                               │
    │   Pure math — no ML needed   │
    │                               │
    │   For each detected anchor:  │
    │   1. Get pixel width (px)    │
    │   2. Get known width (in)    │
    │   3. Get depth value (d)     │
    │   4. Compute: scale = in/px  │
    │      at depth plane d        │
    │                               │
    │   Cross-validate: multiple   │
    │     anchors on same plane    │
    │     should yield consistent  │
    │     scale factors            │
    │                               │
    │   Output: pixels_per_inch    │
    │     for each depth plane     │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STAGE 4: SPATIAL MEASUREMENT│
    │                               │
    │   Using calibrated scale:    │
    │   - Element-to-element       │
    │     spacing (stud-to-stud,   │
    │     rebar-to-rebar)          │
    │   - Element heights from     │
    │     reference plane (floor)  │
    │   - Gap/clearance between    │
    │     elements                 │
    │   - Presence/absence of      │
    │     required elements        │
    │                               │
    │   Output: Structured JSON    │
    │     with all measurements    │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STAGE 5: CONTEXT INJECTION  │
    │                               │
    │   Construct VLM prompt:      │
    │   - Original image           │
    │   - Calibrated measurements  │
    │     as structured text       │
    │   - Relevant construction    │
    │     standards/tolerances     │
    │   - Inspection template      │
    │                               │
    │   The VLM reasons over REAL  │
    │   measurements, not pixels   │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STAGE 6: INSPECTION OUTPUT  │
    │                               │
    │   VLM generates:             │
    │   - Per-element pass/fail    │
    │   - Specific measurements    │
    │   - Deficiency descriptions  │
    │     with locations           │
    │   - Overall recommendation   │
    └───────────────────────────────┘
```

### Stage 3 Deep Dive: The Calibration Math

This is the core technical contribution. The key insight is that perspective projection creates a linear relationship between real-world size and pixel size at a given depth.

**Single-anchor calibration:**
```
Given:
  anchor_pixel_width  = width of detected anchor in pixels
  anchor_real_width   = known real-world width in inches
  anchor_depth        = relative depth value from depth map

Scale factor at this depth:
  pixels_per_inch = anchor_pixel_width / anchor_real_width

To measure distance between two objects at the same depth:
  distance_pixels = |object_A_center - object_B_center| in pixels
  distance_inches = distance_pixels / pixels_per_inch
```

**Multi-anchor cross-validation:**
When multiple anchors exist on the same depth plane (e.g., many studs on one wall), each produces an independent scale estimate. We compute the median scale factor and flag outliers — an inconsistent scale estimate may indicate a detection error or an object that's not on the expected plane.

```
anchors_on_plane = [anchor1, anchor2, anchor3, ...]
scale_estimates = [a.pixel_width / a.real_width for a in anchors_on_plane]
calibrated_scale = median(scale_estimates)
confidence = 1.0 - (std(scale_estimates) / mean(scale_estimates))
```

**Depth-adjusted measurement:**
For objects at different depths (e.g., a pipe in front of a wall), we adjust the scale factor proportionally to the depth ratio:

```
If anchor is at depth d1 with scale s1,
and target object is at depth d2:
  adjusted_scale = s1 * (d1 / d2)
```

Note: Monocular depth estimation gives relative, not absolute depth. This adjustment is approximate but sufficient for "is this within tolerance" decisions.

**Perspective correction:**
For photos taken at a slight angle to the work surface (not perfectly perpendicular), detected anchor dimensions will be foreshortened. We can detect this by comparing scale factors across the image — if the left side consistently yields a different scale than the right side, the camera is angled. We apply a linear correction gradient across the measurement axis.

### Stage 6 Deep Dive: Spatial Context Injection Prompt

The prompt structure is critical. We provide the VLM with:

```
SYSTEM: You are a construction inspection AI. You have been provided
with an image of construction work and calibrated spatial measurements
extracted from that image using known reference dimensions. Your job
is to evaluate whether the work meets the specified construction
standards. Base your analysis ONLY on the provided measurements,
not on visual estimation.

USER:
[IMAGE ATTACHED]

CALIBRATED SPATIAL MEASUREMENTS (computed via anchor calibration):
Calibration anchor: 2x4 stud face (3.5" known width)
Calibration confidence: 0.94 (12 anchors, std dev 0.08)
Scale: 18.7 pixels per inch at primary wall plane

Element measurements:
  Stud 1 → Stud 2: 16.1 inches center-to-center
  Stud 2 → Stud 3: 16.0 inches center-to-center
  Stud 3 → Stud 4: 19.3 inches center-to-center
  Stud 4 → Stud 5: 15.9 inches center-to-center
  Stud 5 → Stud 6: 16.2 inches center-to-center

  Electrical box 1 height from bottom plate: 11.8 inches (center)
  Electrical box 2 height from bottom plate: 12.1 inches (center)

  Nail plates detected: 4 of 6 penetrations
  Missing nail plate locations: 34" from left corner at 52" height

APPLICABLE STANDARDS:
  Stud spacing: 16" on center (tolerance: ±0.5")
  Outlet box height: 12" to center (tolerance: ±1")
  Nail plates: Required on ALL stud penetrations (IRC R602.8)

Generate an inspection report with:
1. Per-element pass/fail with measurements
2. List of deficiencies with precise locations
3. Overall pass/fail recommendation
4. Severity rating for each deficiency (critical/major/minor)
```

---

## 3. Object Re-Identification (ReID) Across Views

### Problem
A single photo may not capture all of a wall or inspection area. Workers typically take multiple photos from different positions and angles. Without ReID, the system would analyze each photo independently, potentially double-counting elements or missing the connection between views.

### Approach
Use visual feature matching to identify the same physical object across multiple photos:

1. **Feature extraction**: For each detected element, extract a feature vector using CLIP embeddings or ORB/SIFT descriptors on the cropped region.

2. **Spatial consistency check**: Two detections are the same object only if:
   - Their visual features are similar (cosine similarity > threshold)
   - Their position relative to nearby anchors is consistent
   - They appear at a spatially plausible location given the estimated camera movement

3. **Merged spatial model**: After matching, merge measurements from multiple views into a single consistent spatial model of the inspection area. This can improve measurement accuracy (average across views) and increase coverage (see around partial occlusions).

### Implementation
```python
def match_objects_across_views(
    detections_a: list[Detection],
    detections_b: list[Detection],
    features_a: list[np.ndarray],
    features_b: list[np.ndarray],
    similarity_threshold: float = 0.85,
) -> list[tuple[int, int]]:
    """Match detected objects between two views using feature similarity."""
    matches = []
    similarity_matrix = cosine_similarity(features_a, features_b)
    for i, row in enumerate(similarity_matrix):
        best_match = row.argmax()
        if row[best_match] >= similarity_threshold:
            matches.append((i, best_match))
    return matches
```

---

## 4. Evaluation Methodology

### Benchmark Design
We construct a benchmark of spatial questions on construction images where ground truth measurements are known (either from the images themselves using visible tape measures / known dimensions, or from supplementary measurement data).

### Three Experimental Conditions

**Condition 1 — Baseline (Raw VLM)**
- Input: Original image + spatial question
- No augmentation
- Tests the VLM's native spatial reasoning

**Condition 2 — Depth-Augmented**
- Input: Original image + depth map visualization + spatial question
- Provides relative depth information but no real-world scale
- Tests whether depth alone improves spatial reasoning

**Condition 3 — Anchor-Calibrated (Ours)**
- Input: Original image + calibrated spatial measurements + construction standards + spatial question
- Full PreCheck pipeline
- Tests whether real-world measurements enable accurate spatial reasoning

### Metrics

**Measurement accuracy**: For questions with numeric answers (spacing, height, distance), compute:
- Mean absolute error (MAE) in inches
- Percentage within tolerance (±0.5" for framing, ±1" for MEP)
- Correlation with ground truth

**Deficiency detection**:
- True positive rate: correctly identified actual deficiencies
- False positive rate: incorrectly flagged correct work
- F1 score

**Qualitative assessment**:
- Specificity: Does the response include specific measurements or just qualitative language?
- Actionability: Could a worker act on this response to fix the issue?
- Spatial grounding: Does the response reference specific locations in the image?

### Test Question Categories
```
CATEGORY                  EXAMPLE QUESTION                                DIFFICULTY
──────────────────────────────────────────────────────────────────────────────────────
Element spacing           "What is the center-to-center stud spacing?"   Medium
Height measurement        "How high is the electrical box from floor?"   Medium
Presence/absence          "Are nail plates on all penetrations?"         Easy
Compliance check          "Does stud spacing meet 16" OC requirement?"   Hard
Comparative               "Which bay has the widest stud spacing?"       Medium
Deficiency identification "What deficiencies exist in this framing?"     Hard
Count                     "How many studs are visible?"                  Easy
Clearance                 "Is there adequate clearance at the header?"   Hard
```

### Expected Results Hypothesis
- Condition 1 (baseline) will produce qualitative, non-specific responses with low measurement accuracy
- Condition 2 (depth) will show marginal improvement in relative spatial reasoning but still fail at absolute measurements
- Condition 3 (anchor-calibrated) will show significant improvement in measurement accuracy and deficiency detection, demonstrating that spatial anchor calibration bridges the gap between pixel perception and real-world spatial reasoning

---

## 5. Data Strategy

### Primary Data: Ironsite Construction Footage
- 12 video clips, ~20 minutes each
- Extract stable, well-lit frames showing clear construction elements
- Priority: frames showing framing, rebar, CMU walls, MEP rough-in

### Supplementary Data
- Self-captured photos of construction elements with tape measure visible (provides ground truth measurements)
- Public construction inspection photos from building department databases
- Stock construction photos with identifiable standard-dimension elements

### Ground Truth Collection
For benchmark evaluation, we need known measurements. Approaches:
1. **Visible tape measure**: Photos where a tape measure is visible provide direct ground truth
2. **Known standards**: Count elements and multiply (e.g., 8 studs at 16" OC = 128" total span)
3. **Reference object cross-check**: Measure using one anchor type, validate with a different anchor type in the same image

---

## 6. Risk Analysis & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Anchor detection fails on Ironsite footage | Medium | High | Fall back to manual anchor annotation for demo frames; use GroundedSAM with text prompts as backup |
| Depth estimation too noisy for calibration | Medium | Medium | Use depth only for plane grouping, not for cross-plane adjustment; stick to single-plane measurements |
| Calibration accuracy insufficient | Medium | High | Report confidence intervals with measurements; focus demo on cases where calibration is strong (many anchors, consistent scale) |
| Ironsite footage doesn't show clear inspection scenarios | Medium | Medium | Supplement with our own construction photos; the technique works on any construction image |
| VLM ignores injected measurements and guesses anyway | Low | High | Explicit prompt instruction: "Base your analysis ONLY on the provided measurements, not visual estimation" + few-shot examples |
| Time runs out before integration | Medium | Critical | Each pipeline stage saves files independently; even if integration is partial, we can show individual stages working + manual pipeline for demo |

---

## 7. Future Work (For Demo Video Conclusion)

- **Fine-tuning**: Train VLMs on construction-specific spatial reasoning tasks to internalize measurement knowledge
- **Video mode**: Continuous inspection as camera pans across work, accumulating measurements in real-time
- **BIM integration**: Compare measurements against Building Information Model specifications automatically
- **Historical tracking**: Compare today's measurements against last week's to detect settlement, shifting, or other spatial changes over time
- **Multi-trade support**: Extend anchor library to cover plumbing, HVAC, fire protection, and other specialty trades

---

## 8. Demo Script (3-5 Minute Video)

```
[0:00-0:30] THE HOOK
"This wall has a defect. In 24 hours it'll be covered with drywall
and this defect will cost $50,000 to fix. We asked GPT-5 if anything
was wrong. It said the framing looks standard and properly installed.
It's wrong. Here's how we caught it."

[0:30-1:30] THE TECHNIQUE
"VLMs can't measure. They see pixels, not inches. But construction
sites are full of objects with known dimensions. A 2x4 is always
3.5 inches wide. We use these as calibration anchors."
[Show anchor detection on a frame]
[Show calibration math — pixel width → known width → scale factor]
[Show measurements being extracted]

[1:30-2:30] THE PIPELINE
"Our system: detect anchors, estimate depth, calibrate scale,
extract measurements, inject into VLM prompt. The model now
reasons over real measurements, not pixel guesses."
[Show the full pipeline running on a construction image]
[Show the structured spatial facts being injected]

[2:30-3:30] THE RESULTS
"We tested three conditions across N spatial questions."
[Show baseline vs depth-augmented vs anchor-calibrated accuracy]
[Show specific examples of baseline failing and ours succeeding]
"Anchor calibration improved measurement accuracy by X% and
deficiency detection by Y%."

[3:30-4:00] THE PRODUCT
[Show Streamlit PreCheck demo]
"Upload a photo, get an inspection checklist with real measurements.
Every deficiency caught here saves thousands in rework."
[Show the interactive demo with judges' own selection]

[4:00-4:30] THE IMPACT
"Construction rework costs $60 billion a year. Most of it starts
with a spatial error that was visible but not caught. PreCheck
catches those errors at the only moment they're still cheap to fix.
One measurement. One callout. $50,000 saved."

[4:30-5:00] FUTURE WORK + CLOSE
"With more time: fine-tuning, real-time video mode, BIM integration.
The technique generalizes to any domain with known-dimension objects —
manufacturing, warehousing, infrastructure inspection.
This is PreCheck. Spatial anchor calibration for construction."
```