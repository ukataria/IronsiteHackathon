"""All VLM prompt templates as string constants. Never put prompts anywhere else."""

# ---------------------------------------------------------------------------
# System prompt (shared across all conditions)
# ---------------------------------------------------------------------------

INSPECTION_SYSTEM_PROMPT = """You are a professional construction inspection AI assistant. \
Your role is to evaluate construction work for code compliance and quality.

When provided with calibrated spatial measurements, base your analysis ONLY on those \
measurements — do not visually estimate distances. When measurements are not provided, \
use your best visual judgment and clearly state that you are estimating.

Always structure your output as a clear inspection report with:
1. Per-element pass/fail assessments with measurements
2. Specific deficiency descriptions with locations
3. Severity ratings (critical / major / minor)
4. Overall pass/fail recommendation

Be precise, actionable, and concise. A worker should be able to read your report \
and know exactly what to fix and where."""

# ---------------------------------------------------------------------------
# Construction standards reference
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Standard element dimensions — injected as grounding context for the VLM
# ---------------------------------------------------------------------------

ELEMENT_DIMENSIONS_BLOCK = """\
━━━ STANDARD ELEMENT DIMENSIONS (used for scale calibration) ━━━

Brick (Standard Modular, ASTM C216):
  Length:  7.625 in  │  Nominal with 3/8" mortar joint: 8.0 in
  Height:  2.25  in  │  Nominal course height:          2.625 in
  Width:   3.625 in
  3 courses + joints = 8.0 in nominal vertical

CMU / Concrete Block (Standard 8×8×16, ASTM C90):
  Length:  15.625 in │  Nominal with 3/8" mortar joint: 16.0 in
  Height:   7.625 in │  Nominal course height:           8.0 in
  Width:    7.625 in
  1 course + joint  = 8.0 in nominal vertical

Electrical Outlet Box (Single-gang, NEC Article 314):
  Width:   2.0 in (device box inner)
  Height:  3.0 in (device box inner)
  Depth:   2.5 in (standard)
  Required centerline height from finished floor: 12.0 in (NEC, ±1 in)

Mortar Joints:  3/8 in nominal  (tolerance: ±1/8 in)"""

# ---------------------------------------------------------------------------
# Construction standards reference
# ---------------------------------------------------------------------------

CONSTRUCTION_STANDARDS: dict[str, str] = {
    "stud_spacing": "16 inches on center (tolerance: ±0.5 inches) per IRC R602",
    "stud_spacing_24oc": "24 inches on center (tolerance: ±0.5 inches) for 2x6 walls",
    "rebar_spacing": "12 inches on center (tolerance: ±0.75 inches) — verify per structural drawings",
    "electrical_box_height": "12 inches to center from finished floor (tolerance: ±1 inch) per NEC",
    "nail_plates": "Required on ALL stud penetrations within 1.5 inches of face per IRC R602.8",
    "header_bearing": "Minimum 1.5 inches bearing on each side per IRC R602.7",
    "cmu_joint_thickness": "3/8 inch mortar joints (tolerance: ±1/8 inch)",
    "brick_dimensions": "Standard brick 7.625\" L × 3.625\" W × 2.25\" H; mortar joint 3/8\" nominal",
    "brick_h_spacing": "Horizontal center-to-center: 8.0\" (brick + mortar joint, tolerance: ±0.25\")",
    "brick_course_height": "Vertical course height: 2.625\" center-to-center (tolerance: ±0.25\")",
    "cmu_dimensions": "Standard CMU 15.625\" L × 7.625\" H; mortar joint 3/8\" nominal",
    "cmu_spacing": "Horizontal center-to-center: 16.0\" (CMU + mortar, tolerance: ±0.5\")",
}

STANDARDS_BLOCK = "\n".join(
    f"  {k.replace('_', ' ').title()}: {v}" for k, v in CONSTRUCTION_STANDARDS.items()
)

# ---------------------------------------------------------------------------
# Condition 1 — Baseline (raw VLM, no augmentation)
# ---------------------------------------------------------------------------

BASELINE_PROMPT_TEMPLATE = """Inspect this construction image and answer the following question:

{question}

Provide a structured inspection report including:
1. Your assessment for each visible element
2. Any deficiencies you observe (with approximate locations)
3. Overall pass/fail recommendation
4. Severity of any deficiencies (critical / major / minor)

Note: You are working from visual estimation only — no calibrated measurements are available."""

# ---------------------------------------------------------------------------
# Condition 2 — Depth-augmented (depth map added, no real-world scale)
# ---------------------------------------------------------------------------

DEPTH_AUGMENTED_PROMPT_TEMPLATE = """Inspect this construction image. A depth map visualization \
has been provided as a second image to help you understand spatial relationships.

Question: {question}

The depth map uses a jet colormap where:
  - Blue/cool colors = farther from camera
  - Red/warm colors = closer to camera

Use the depth information to reason about which elements are on the same surface plane \
and their relative spatial positions.

Provide a structured inspection report including:
1. Your assessment for each visible element
2. Any deficiencies you observe (with approximate locations)
3. Overall pass/fail recommendation
4. Severity of any deficiencies (critical / major / minor)

Note: You have relative depth information but no calibrated real-world measurements."""

# ---------------------------------------------------------------------------
# Condition 3 — Anchor-calibrated (full PreCheck pipeline output)
# ---------------------------------------------------------------------------

ANCHOR_CALIBRATED_PROMPT_TEMPLATE = """You are inspecting a construction site. \
Two images have been provided:
  Image 1: Original construction photo
  Image 2: Depth map (jet colormap — red/warm = closer to camera, blue/cool = farther away)

Use the depth map to understand spatial layout and which surfaces share the same plane. \
Use the calibrated measurements below to answer questions in real-world inches.

Question: {question}

{dimensions_block}

━━━ CALIBRATED SPATIAL MEASUREMENTS ━━━
Calibration method: known physical dimensions of detected objects (bricks, CMU, outlet boxes)
{calibration_summary}

{measurements_block}

━━━ APPLICABLE CONSTRUCTION STANDARDS ━━━
{standards_block}

When answering questions about distances or sizes: use the provided scale \
(pixels per inch) combined with the depth map to reason spatially. \
For pass/fail, use ONLY the measured values above — do not visually re-estimate distances.

Provide a structured inspection report:
1. Per-element pass/fail with exact measurements
2. Specific deficiencies with locations
3. Severity ratings (critical / major / minor)
4. Overall pass/fail recommendation"""


def build_chat_opening_prompt(
    measurements: dict,
    question: str = "Provide a full inspection report.",
) -> str:
    """
    Build the first-turn prompt for a chat session.
    Includes dimensions, calibrated measurements, and standards as grounding context.
    """
    mblock = format_measurements_block(measurements)
    cal_summary = build_calibration_summary(measurements)
    return ANCHOR_CALIBRATED_PROMPT_TEMPLATE.format(
        question=question,
        dimensions_block=ELEMENT_DIMENSIONS_BLOCK,
        calibration_summary=cal_summary,
        measurements_block=mblock,
        standards_block=STANDARDS_BLOCK,
    )


def format_measurements_block(measurements: dict) -> str:
    """Format a measurements dict into a readable block for prompt injection."""
    lines = []

    ppi = measurements.get("scale_pixels_per_inch", 0)
    conf = measurements.get("calibration_confidence", 0)
    lines.append(f"Scale: {ppi:.2f} pixels/inch  |  Calibration confidence: {conf:.0%}")
    lines.append("")

    def _spacing_block(label: str, spacings: list[dict]) -> None:
        if not spacings:
            return
        lines.append(f"{label}:")
        for i, s in enumerate(spacings):
            status = "✓ PASS" if s.get("compliant") else "✗ FAIL"
            lines.append(f"  Bay {i+1}: {s['inches']:.2f}\"  [{status}]")

    stud_spacings = measurements.get("stud_spacings", [])
    _spacing_block("Stud Spacing (center-to-center, target 16.0\" OC)", stud_spacings)

    rebar_spacings = measurements.get("rebar_spacings", [])
    if rebar_spacings:
        lines.append("")
    _spacing_block("Rebar Spacing (center-to-center, target 12.0\" OC)", rebar_spacings)

    brick_h = measurements.get("brick_h_spacings", [])
    if brick_h:
        lines.append("")
    _spacing_block("Brick Horizontal Spacing (center-to-center, target 8.0\")", brick_h)

    brick_v = measurements.get("brick_v_spacings", [])
    if brick_v:
        lines.append("")
    _spacing_block("Brick Course Height (center-to-center, target 2.625\")", brick_v)

    cmu_spacings = measurements.get("cmu_spacings", [])
    if cmu_spacings:
        lines.append("")
    _spacing_block("CMU Block Spacing (center-to-center, target 16.0\")", cmu_spacings)

    elec_heights = measurements.get("electrical_box_heights", [])
    if elec_heights:
        lines.append("")
        lines.append("Electrical Box Heights (from floor reference, target 12.0\"):")
        for h in elec_heights:
            status = "✓ PASS" if h.get("compliant") else "✗ FAIL"
            lines.append(f"  Box {h['box_id']+1}: {h['height_inches']:.1f}\" to center  [{status}]")

    counts = measurements.get("element_counts", {})
    if counts:
        lines.append("")
        lines.append("Element counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

    summary = measurements.get("summary", "")
    if summary:
        lines.append("")
        lines.append(f"Pre-check summary: {summary}")

    return "\n".join(lines)


def build_calibration_summary(measurements: dict) -> str:
    """Build short calibration metadata line for the prompt header."""
    ppi = measurements.get("scale_pixels_per_inch", 0)
    conf = measurements.get("calibration_confidence", 0)
    counts = measurements.get("element_counts", {})
    n_anchors = sum(counts.values())
    return (
        f"Anchor count: {n_anchors}  |  "
        f"Scale: {ppi:.2f} px/inch  |  "
        f"Confidence: {conf:.0%}"
    )
