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

Mortar Joints:  3/8 in nominal  (tolerance: ±1/8 in)

Orientation note:
  Bricks appear in multiple orientations — stretcher (long face, 7.625" wide),
  header (end face, 3.625" wide), or soldier (upright, 2.25" wide). CMU blocks
  similarly. When estimating distances, identify which face is visible before
  applying a unit conversion. The scale bar was calibrated using each object's
  longest detected dimension to account for rotation."""

# ---------------------------------------------------------------------------
# Construction standards reference
# ---------------------------------------------------------------------------

# Standards grouped by which anchor type triggers them.
# Only standards for detected element types are shown to the VLM.
_STANDARDS_BY_TYPE: dict[str, dict[str, str]] = {
    "brick": {
        "Brick Dimensions": "Standard brick 7.625\" L × 3.625\" W × 2.25\" H; mortar joint 3/8\" nominal",
        "Brick Horizontal Spacing": "Center-to-center: 8.0\" (brick + mortar, tolerance: ±0.25\") per ASTM C216",
        "Brick Course Height": "Vertical course: 2.625\" center-to-center (tolerance: ±0.25\")",
        "Mortar Joint Thickness": "3/8 inch nominal (tolerance: ±1/8 inch)",
        "Brick Alignment": "Courses must be level; vertical joints should align with every other course (running bond)",
    },
    "cmu": {
        "CMU Dimensions": "Standard CMU 15.625\" L × 7.625\" H; mortar joint 3/8\" nominal per ASTM C90",
        "CMU Horizontal Spacing": "Center-to-center: 16.0\" (CMU + mortar, tolerance: ±0.5\")",
        "CMU Course Height": "Vertical course: 8.0\" center-to-center (tolerance: ±0.5\")",
        "Mortar Joint Thickness": "3/8 inch nominal (tolerance: ±1/8 inch)",
    },
    "stud": {
        "Stud Spacing": "16 inches on center (tolerance: ±0.5 inches) per IRC R602",
        "Stud Spacing 2x6": "24 inches on center (tolerance: ±0.5 inches) for 2x6 walls",
        "Nail Plates": "Required on ALL stud penetrations within 1.5 inches of face per IRC R602.8",
        "Header Bearing": "Minimum 1.5 inches bearing on each side per IRC R602.7",
    },
    "rebar": {
        "Rebar Spacing": "12 inches on center (tolerance: ±0.75 inches) — verify per structural drawings",
    },
    "electrical_box": {
        "Electrical Box Height": "12 inches to center from finished floor (tolerance: ±1 inch) per NEC Article 314",
    },
}


def build_standards_block(measurements: dict) -> str:
    """Build a standards block containing only standards relevant to detected element types."""
    detected = set(measurements.get("element_counts", {}).keys())
    lines = []
    for element_type, standards in _STANDARDS_BY_TYPE.items():
        if element_type not in detected:
            continue
        for label, spec in standards.items():
            lines.append(f"  {label}: {spec}")
    return "\n".join(lines) if lines else "  No element-specific standards applicable."

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
Use the calibrated measurements to answer questions in real-world inches.

Question: {question}

{dimensions_block}

{reference_objects_block}

━━━ CALIBRATED SPATIAL MEASUREMENTS ━━━
Calibration method: known physical dimensions of detected objects (bricks, CMU, outlet boxes)
{calibration_summary}

{measurements_block}

━━━ APPLICABLE CONSTRUCTION STANDARDS ━━━
{standards_block}

For pass/fail assessments: use ONLY the pre-computed measured values above. \
For any other distance question: use reference objects as described above \
and show your reasoning. Do not guess distances without referencing a known object.

Provide a structured inspection report:
1. Per-element pass/fail with exact measurements
2. Specific deficiencies with locations
3. Severity ratings (critical / major / minor)
4. Overall pass/fail recommendation"""


# Human-readable descriptions of each detectable object type's real-world dimensions.
# Used to tell the VLM what it can use as a visual ruler.
_REFERENCE_DIMS: dict[str, str] = {
    "brick": "7.625\" long × 2.25\" tall (8.0\" × 2.625\" nominal with 3/8\" mortar joints)",
    "cmu": "15.625\" long × 7.625\" tall (16.0\" × 8.0\" nominal with 3/8\" mortar joints)",
    "electrical_box": "2.0\" wide × 3.0\" tall (single-gang NEC box)",
}



def build_reference_objects_block(measurements: dict) -> str:
    """
    Build the visual reference section injected into the VLM prompt.
    Describes detected objects with known dimensions so Claude can use them as rulers
    for arbitrary distance questions about anything in the image.
    """
    counts = measurements.get("element_counts", {})
    lines = ["━━━ VISUAL SCALE REFERENCE ━━━"]

    known = {k: v for k, v in counts.items() if k in _REFERENCE_DIMS}
    if known:
        lines.append("Detected objects with known real-world dimensions (use as visual rulers):")
        for obj_type, count in known.items():
            lines.append(
                f"  - {obj_type.replace('_', ' ').title()} "
                f"({count} visible): {_REFERENCE_DIMS[obj_type]}"
            )
        lines.append("  Important: identify the orientation of each object before using it as")
        lines.append("  a ruler — a soldier brick (upright) is 2.25\" wide, not 7.625\".")
    else:
        lines.append("No objects with known dimensions detected.")

    lines.append("")
    lines.append("To estimate any distance in the image:")
    lines.append("  Count how many reference objects span the distance and multiply "
                 "by the known dimension.")
    lines.append("  Always show your reasoning step "
                 "(e.g. \"spans ~2.3 brick-widths = 2.3 × 8.0\\\" = 18.4\\\"\")")
    return "\n".join(lines)



def build_chat_opening_prompt(
    measurements: dict,
    question: str = "Provide a full inspection report.",
) -> str:
    """
    Build the first-turn prompt for a chat session.
    Includes scale reference, dimensions, calibrated measurements, and standards.
    """
    mblock = format_measurements_block(measurements)
    cal_summary = build_calibration_summary(measurements)
    ref_block = build_reference_objects_block(measurements)
    standards = build_standards_block(measurements)
    return ANCHOR_CALIBRATED_PROMPT_TEMPLATE.format(
        question=question,
        dimensions_block=ELEMENT_DIMENSIONS_BLOCK,
        calibration_summary=cal_summary,
        measurements_block=mblock,
        standards_block=standards,
        reference_objects_block=ref_block,
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
