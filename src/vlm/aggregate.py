"""Aggregate all frame measurements into a project-level summary for the VLM report."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def aggregate_all_frames(measurements_dir: Path) -> dict:
    """Read every *_measurements.json and produce a project-level summary."""
    frames: list[dict] = []
    for p in sorted(measurements_dir.glob("*_measurements.json")):
        try:
            data = json.loads(p.read_text())
            data["_frame_file"] = p.stem.replace("_measurements", "")
            frames.append(data)
        except Exception:
            continue

    if not frames:
        return {"total_frames": 0}

    def _agg_spacings(key: str, target: float, tol: float) -> dict:
        all_vals = []
        compliant = 0
        total = 0
        skipped = 0
        worst_frame = None
        worst_val = None

        # Exclude carry-forward frames (conf=0.3) — only use real anchor detections (conf=1.0).
        # Also reject obvious outliers > 4x target (detection artifact, not real measurement).
        max_plausible = target * 4

        for frame in frames:
            conf = frame.get("calibration_confidence", 0)
            if conf < 0.5:
                continue
            for s in frame.get(key, []):
                v = s.get("inches", 0)
                if v < target * 0.25 or v > max_plausible:
                    skipped += 1
                    continue
                c = s.get("compliant", False)
                all_vals.append(v)
                total += 1
                if c:
                    compliant += 1
                if worst_val is None or abs(v - target) > abs(worst_val - target):
                    worst_val = v
                    worst_frame = frame.get("_frame_file")

        if not all_vals:
            return {}

        return {
            "count": total,
            "compliant": compliant,
            "compliance_pct": round(compliant / total * 100, 1),
            "mean_inches": round(statistics.mean(all_vals), 2),
            "min_inches": round(min(all_vals), 2),
            "max_inches": round(max(all_vals), 2),
            "target_inches": target,
            "tolerance_inches": tol,
            "worst_frame": worst_frame,
            "worst_val": round(worst_val, 2) if worst_val else None,
            "skipped_outliers": skipped,
        }

    def _agg_elec_heights(frames: list[dict]) -> dict:
        all_vals = []
        compliant = 0
        total = 0
        skipped = 0
        worst_frame = None
        worst_val = None
        target = 12.0

        for frame in frames:
            if frame.get("calibration_confidence", 0) < 0.5:
                continue
            for h in frame.get("electrical_box_heights", []):
                v = h.get("height_inches", 0)
                if v <= 0 or v > target * 4:
                    skipped += 1
                    continue
                c = h.get("compliant", False)
                all_vals.append(v)
                total += 1
                if c:
                    compliant += 1
                if worst_val is None or abs(v - target) > abs(worst_val - target):
                    worst_val = v
                    worst_frame = frame.get("_frame_file")

        if not all_vals:
            return {}

        return {
            "count": total,
            "compliant": compliant,
            "compliance_pct": round(compliant / total * 100, 1),
            "mean_inches": round(statistics.mean(all_vals), 2),
            "min_inches": round(min(all_vals), 2),
            "max_inches": round(max(all_vals), 2),
            "target_inches": target,
            "tolerance_inches": 1.0,
            "worst_frame": worst_frame,
            "worst_val": round(worst_val, 2) if worst_val else None,
        }

    # Frames that actually had detections (non-carry-forward calibration)
    frames_with_detections = [
        f for f in frames if f.get("calibration_confidence", 0) >= 0.9
    ]

    result = {
        "total_frames": len(frames),
        "frames_with_detections": len(frames_with_detections),
        "stud_spacings":     _agg_spacings("stud_spacings",    16.0, 0.5),
        "rebar_spacings":    _agg_spacings("rebar_spacings",   12.0, 0.75),
        "brick_h_spacings":  _agg_spacings("brick_h_spacings",  8.0, 0.25),
        "brick_v_spacings":  _agg_spacings("brick_v_spacings",  2.625, 0.25),
        "cmu_spacings":      _agg_spacings("cmu_spacings",      16.0, 0.5),
        "electrical_boxes":  _agg_elec_heights(frames),
    }

    # Strip empty dicts
    result = {k: v for k, v in result.items() if v != {}}
    return result


def pick_representative_frames(measurements_dir: Path, n: int = 3) -> list[str]:
    """
    Pick n representative frame image IDs for the report:
    - worst compliance frame
    - best compliance frame (with detections)
    - middle frame
    """
    scored: list[tuple[float, str]] = []  # (compliance_rate, image_id)

    for p in sorted(measurements_dir.glob("*_measurements.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue

        image_id = p.stem.replace("_measurements", "")
        all_spacings = (
            data.get("stud_spacings", [])
            + data.get("rebar_spacings", [])
            + data.get("brick_h_spacings", [])
            + data.get("brick_v_spacings", [])
            + data.get("cmu_spacings", [])
        )
        # Ignore frames with no measurements at all
        if not all_spacings:
            continue

        compliant = sum(1 for s in all_spacings if s.get("compliant"))
        rate = compliant / len(all_spacings)
        scored.append((rate, image_id))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0])
    worst = scored[0][1]
    best = scored[-1][1]
    mid = scored[len(scored) // 2][1]

    # Deduplicate while preserving order
    seen: set[str] = set()
    picks: list[str] = []
    for fid in [worst, mid, best]:
        if fid not in seen:
            picks.append(fid)
            seen.add(fid)
    return picks[:n]


def format_aggregate_block(agg: dict) -> str:
    """Format the aggregated stats as a readable block for the VLM prompt."""
    lines = [
        f"Total frames analyzed: {agg['total_frames']}",
        f"Frames with anchor detections: {agg.get('frames_with_detections', 'N/A')}",
        "",
    ]

    def _spacing_section(label: str, data: dict) -> None:
        if not data:
            return
        lines.append(f"{label}:")
        lines.append(f"  Measurements: {data['count']} across all frames")
        lines.append(
            f"  Compliance: {data['compliant']}/{data['count']} ({data['compliance_pct']}%)"
        )
        lines.append(f"  Mean spacing: {data['mean_inches']}\"  "
                     f"(target {data['target_inches']}\" ± {data['tolerance_inches']}\")")
        lines.append(f"  Range: {data['min_inches']}\" – {data['max_inches']}\"")
        if data.get("worst_frame"):
            lines.append(
                f"  Worst frame: {data['worst_frame']} "
                f"@ {data['worst_val']}\" (delta {abs(data['worst_val'] - data['target_inches']):.2f}\")"
            )
        lines.append("")

    _spacing_section("Brick Horizontal Spacing (target 8.0\")", agg.get("brick_h_spacings", {}))
    _spacing_section("Brick Course Height (target 2.625\")", agg.get("brick_v_spacings", {}))
    _spacing_section("Stud Spacing (target 16.0\" OC)", agg.get("stud_spacings", {}))
    _spacing_section("Rebar Spacing (target 12.0\" OC)", agg.get("rebar_spacings", {}))
    _spacing_section("CMU Block Spacing (target 16.0\")", agg.get("cmu_spacings", {}))
    _spacing_section("Electrical Box Height (target 12.0\")", agg.get("electrical_boxes", {}))

    return "\n".join(lines)
