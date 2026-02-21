# Two-Head Architecture for PreCheck

## Overview

This implements the "Two-Head" model architecture from your Technical.md:
- **Perception Head** (frozen): Depth Anything V2 + Anchor Detector
- **Measurement Head** (learned): Calibrated spatial measurements
- **Reasoning Head** (VLM): Construction inspection analysis

## Architecture

```
Construction Image
       ↓
┌──────────────────────────┐
│   PERCEPTION HEAD        │
│   (Frozen after          │
│    fine-tuning)          │
├──────────────────────────┤
│  • Depth Anything V2     │
│    → Depth map (meters)  │
│                          │
│  • Anchor Detector       │
│    (YOLO/GroundedSAM)    │
│    → Known-dim objects   │
│      (studs, CMU, etc)   │
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   MEASUREMENT HEAD       │
│   (Learned calibration)  │
├──────────────────────────┤
│  • Group anchors by      │
│    depth plane           │
│                          │
│  • Compute scale factor  │
│    pixels/inch per plane │
│                          │
│  • Extract measurements  │
│    (spacing, height,     │
│     clearance)           │
│                          │
│  • Cross-validate with   │
│    multiple anchors      │
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   REASONING HEAD         │
│   (VLM - Claude/GPT-4o)  │
├──────────────────────────┤
│  Input:                  │
│  • Original image        │
│  • Calibrated            │
│    measurements          │
│  • Construction          │
│    standards             │
│                          │
│  Output:                 │
│  • Pass/fail per element │
│  • Deficiency list       │
│  • Overall assessment    │
└──────────────────────────┘
```

## Components

### 1. Perception Head

**Location:** `models/two_head/perception_head.py`

**Components:**
- **Depth Estimation**: Depth Anything V2 (frozen after fine-tuning)
- **Anchor Detection**: Simple heuristic (upgradeable to GroundedSAM/YOLO)

**Known Anchor Dimensions:**
```python
"2x4_stud_face": 3.5 inches
"2x4_stud_edge": 1.5 inches
"2x6_joist_face": 5.5 inches
"cmu_block_width": 15.625 inches
"rebar_4": 0.5 inches
"electrical_box_single": 2.0 x 3.0 inches
```

**Output:**
- Depth map (H, W) in meters
- List of `AnchorDetection` objects with:
  - Bounding box
  - Class name
  - Known real-world width
  - Pixel width
  - Depth value

### 2. Measurement Head

**Location:** `models/two_head/measurement_head.py`

**Algorithm** (from Technical.md Stage 3):

1. **Group Anchors by Depth Plane:**
   ```python
   # Anchors within depth_tolerance are on same plane
   if abs(anchor.depth - plane.depth) <= 0.5m:
       plane.add_anchor(anchor)
   ```

2. **Calibrate Scale per Plane:**
   ```python
   # Each anchor provides scale estimate
   scale_estimates = [
       anchor.pixel_width / anchor.known_width_inches
       for anchor in plane.anchors
   ]

   # Use median for robustness
   calibrated_scale = median(scale_estimates)

   # Confidence from consistency
   confidence = 1.0 - (std(scale_estimates) / mean(scale_estimates))
   ```

3. **Extract Measurements:**
   ```python
   # Stud-to-stud spacing
   pixel_distance = distance(stud1.center, stud2.center)
   inches = pixel_distance / calibrated_scale
   ```

**Output:**
- List of `DepthPlane` objects (depth, anchors, scale, confidence)
- List of `Measurement` objects (type, value, confidence, location)

### 3. Reasoning Head

**Location:** `models/two_head/two_head_model.py`

**Prompt Structure** (from Technical.md Stage 6):

```
SYSTEM: You are a construction inspection AI with calibrated measurements.

CALIBRATED SPATIAL MEASUREMENTS:
Calibration anchor: 2x4 stud face (3.5" known width)
Calibration confidence: 0.94 (12 anchors, std dev 0.08)
Scale: 18.7 pixels per inch at primary wall plane

Element measurements:
  Stud 1 → Stud 2: 16.1 inches center-to-center
  Stud 2 → Stud 3: 16.0 inches center-to-center
  Stud 3 → Stud 4: 19.3 inches center-to-center ← VIOLATION

APPLICABLE STANDARDS:
  Stud spacing: 16" on center (tolerance: ±0.5")

TASK: Analyze for code violations.

OUTPUT: JSON with elements, deficiencies, overall pass/fail
```

**VLM Options:**
- Claude Sonnet 4.6: Best accuracy (1.98m MAE on depth evals)
- GPT-4o: Faster, slightly less accurate (2.25m MAE)

## Training (Fine-tuning)

**Location:** `train/finetune_perception.py`

### Phase 1: Fine-tune Depth Model

```bash
python train/finetune_perception.py \
  --data_dir data/train \
  --depth_epochs 10 \
  --device cuda
```

**Data Format:**
```
data/train/
  rgb/
    0001.jpg
    0002.jpg
  depth/
    0001.npy  # Ground truth depth in meters
    0002.npy
```

**Training:**
- Pre-trained: Depth Anything V2
- Fine-tune on construction scenes with depth GT
- Loss: L1 (MAE) between predicted and GT depth
- Freeze after training

### Phase 2: Fine-tune Anchor Detector

```bash
python train/finetune_perception.py \
  --data_dir data/train \
  --annotations data/train/annotations.json \
  --anchor_epochs 20 \
  --device cuda
```

**Data Format:**
```
data/train/
  rgb/
    0001.jpg
  annotations.json  # COCO format with anchor labels
```

**Categories:**
- 2x4 stud (face/edge)
- CMU block
- Rebar
- Electrical boxes

**Training:**
- Pre-trained: YOLO or GroundedSAM
- Fine-tune on labeled construction anchors
- Loss: Detection loss (bbox + classification)
- Freeze after training

**After Training:** Both components are FROZEN and used as-is in two-head model.

## Evaluation

**Location:** `eval/runners/eval_two_head.py`

### Run Comparative Eval

```bash
python eval/runners/eval_two_head.py \
  --images_dir data/test/construction \
  --vlm_provider anthropic \
  --question "Are there any code violations in this framing?" \
  --out_json outputs/reports/two_head_eval.json
```

### Three Conditions (from Technical.md)

**Condition 1: Baseline VLM**
- Input: Image + question
- No measurements
- Tests native VLM spatial reasoning

**Condition 2: Depth-Augmented** *(not yet implemented)*
- Input: Image + depth visualization + question
- Relative depth info, no scale
- Tests if depth helps

**Condition 3: Two-Head (Anchor-Calibrated)**
- Input: Image + calibrated measurements + standards + question
- Full pipeline
- Tests if real measurements enable accurate inspection

### Metrics

**From Technical.md:**

1. **Measurement Accuracy:**
   - MAE in inches for numeric answers
   - % within tolerance (±0.5" framing, ±1" MEP)

2. **Deficiency Detection:**
   - True positive rate
   - False positive rate
   - F1 score

3. **Qualitative:**
   - Specificity: Uses actual measurements vs qualitative language
   - Actionability: Worker can fix based on response
   - Spatial grounding: References specific locations

### Expected Results (Hypothesis)

From Technical.md:
- **Baseline:** Qualitative, non-specific, low measurement accuracy
- **Depth-Augmented:** Marginal improvement, still no absolute measurements
- **Two-Head:** Significant improvement in accuracy and deficiency detection

## Usage

### Quick Start

```python
from models.two_head.two_head_model import TwoHeadModel
import numpy as np
from PIL import Image

# Initialize
model = TwoHeadModel(
    vlm_provider="anthropic",
    vlm_model="claude-sonnet-4-20250514"
)

# Load image
image = np.array(Image.open("construction.jpg"))

# Run inspection
result = model.run(
    image=image,
    image_path="construction.jpg",
    question="Check stud spacing compliance"
)

# Access results
print(f"Found {result['measurement']['num_measurements']} measurements")
print(f"Calibration confidence: {result['reasoning']['calibration_confidence']:.2f}")
print(f"VLM response: {result['reasoning']['response']}")
```

### Detailed Pipeline

```python
from models.two_head.perception_head import PerceptionHead
from models.two_head.measurement_head import MeasurementHead

# Step 1: Perception
perception = PerceptionHead()
depth_map, anchors = perception.run(image)

print(f"Detected {len(anchors)} anchors")

# Step 2: Measurement
measurement = MeasurementHead()
planes, measurements = measurement.run(anchors)

print(f"Grouped into {len(planes)} depth planes")
print(f"Extracted {len(measurements)} measurements")

for m in measurements:
    print(f"  {m.from_object} → {m.to_object}: {m.value_inches:.1f}\"")

# Step 3: Reasoning (done in TwoHeadModel.query_with_measurements)
```

## File Structure

```
models/
  two_head/
    __init__.py
    perception_head.py       # Depth + anchor detection
    measurement_head.py      # Calibration logic
    two_head_model.py        # Full integrated model

train/
  finetune_perception.py     # Fine-tune depth + anchor

eval/
  runners/
    eval_two_head.py         # Comparative evaluation

outputs/
  reports/
    two_head_eval.json       # Evaluation results
```

## Key Implementation Details

### From Technical.md

1. **Multi-Anchor Cross-Validation:**
   - Use median of scale estimates for robustness
   - Confidence = 1 - (std/mean) of scale estimates
   - Flag inconsistent anchors as potential detection errors

2. **Depth-Adjusted Measurement:**
   - For objects at different depths:
   - `adjusted_scale = anchor_scale * (anchor_depth / object_depth)`
   - Approximate but sufficient for tolerance checks

3. **Spatial Context Injection:**
   - Always include: calibration info, measurements, standards
   - Instruct VLM: "Base analysis ONLY on provided measurements"
   - Request structured JSON output for parsing

## Future Enhancements

1. **Upgrade Anchor Detector:**
   - Replace heuristic with GroundedSAM or fine-tuned YOLO
   - Support more anchor types (doors, bricks, plywood)

2. **Actual Depth Anything V2:**
   - Replace placeholder with real Depth Anything V2
   - Fine-tune on ARKitScenes or construction-specific depth data

3. **Learnable Measurement Head:**
   - Current: Pure math calibration
   - Future: Small neural net to refine scale factors
   - Learn corrections for perspective, lens distortion

4. **Multi-View Fusion:**
   - Object ReID across multiple photos
   - Merge measurements from different viewpoints
   - Increase coverage and accuracy

5. **Video Mode:**
   - Real-time inspection as camera pans
   - Accumulate measurements continuously
   - SLAM for camera tracking

## Citation

Based on the technical approach described in `Technical.md`:
- Spatial Anchor Calibration
- Construction-specific perception
- Measurement-augmented VLM reasoning

## License

Part of PreCheck construction inspection system.
