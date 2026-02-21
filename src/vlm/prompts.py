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

CONSTRUCTION_STANDARDS: dict[str, str] = {
    "stud_spacing": "16 inches on center (tolerance: ±0.5 inches) per IRC R602",
    "stud_spacing_24oc": "24 inches on center (tolerance: ±0.5 inches) for 2x6 walls",
    "rebar_spacing": "12 inches on center (tolerance: ±0.75 inches) — verify per structural drawings",
    "electrical_box_height": "12 inches to center from finished floor (tolerance: ±1 inch) per NEC",
    "nail_plates": "Required on ALL stud penetrations within 1.5 inches of face per IRC R602.8",
    "header_bearing": "Minimum 1.5 inches bearing on each side per IRC R602.7",
    "cmu_joint_thickness": "3/8 inch mortar joints (tolerance: ±1/8 inch)",
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

ANCHOR_CALIBRATED_PROMPT_TEMPLATE = """Inspect this construction image. Calibrated spatial \
measurements have been computed from this image using known-dimension reference objects \
(spatial anchor calibration).

Question: {question}

━━━ CALIBRATED SPATIAL MEASUREMENTS ━━━
Calibration Method: Spatial anchor calibration using known-dimension construction materials
{calibration_summary}

{measurements_block}

━━━ APPLICABLE CONSTRUCTION STANDARDS ━━━
{standards_block}

IMPORTANT: Base your pass/fail determinations ONLY on the provided measurements above. \
Do not visually re-estimate distances. The measurements were computed from calibrated \
pixel-to-inch scale factors derived from known-dimension anchor objects in the scene.

Provide a structured inspection report including:
1. Per-element pass/fail with exact measurements from the data above
2. Specific deficiencies with precise locations
3. Severity ratings (critical / major / minor)
4. Overall pass/fail recommendation"""


def format_measurements_block(measurements: dict) -> str:
    """Format a measurements dict into a readable block for prompt injection."""
    lines = []

    ppi = measurements.get("scale_pixels_per_inch", 0)
    conf = measurements.get("calibration_confidence", 0)
    lines.append(f"Scale: {ppi:.2f} pixels/inch  |  Calibration confidence: {conf:.0%}")
    lines.append("")

    stud_spacings = measurements.get("stud_spacings", [])
    if stud_spacings:
        lines.append("Stud Spacing (center-to-center):")
        for i, s in enumerate(stud_spacings):
            status = "✓ PASS" if s.get("compliant") else "✗ FAIL"
            lines.append(f"  Bay {i+1}: {s['inches']:.1f}\"  [{status}]")

    rebar_spacings = measurements.get("rebar_spacings", [])
    if rebar_spacings:
        lines.append("")
        lines.append("Rebar Spacing (center-to-center):")
        for i, s in enumerate(rebar_spacings):
            status = "✓ PASS" if s.get("compliant") else "✗ FAIL"
            lines.append(f"  Bay {i+1}: {s['inches']:.1f}\"  [{status}]")

    elec_heights = measurements.get("electrical_box_heights", [])
    if elec_heights:
        lines.append("")
        lines.append("Electrical Box Heights (from floor reference):")
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
