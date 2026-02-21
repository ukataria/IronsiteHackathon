"""3-condition benchmark evaluation across a set of construction images."""

from __future__ import annotations

import sys
from pathlib import Path

from src.utils import load_json, save_json, setup_logger

logger = setup_logger("evaluate")

CONDITIONS = ["baseline", "depth", "anchor_calibrated"]


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    image_dir: str,
    vlm: str = "claude",
    conditions: list[str] | None = None,
    question: str = "What deficiencies exist in this construction work? Provide a full inspection report.",
    skip_pipeline: bool = False,
) -> dict:
    """
    Run all conditions on all images in image_dir.

    If skip_pipeline=False, runs the full pipeline first for each image.
    If skip_pipeline=True, assumes pipeline has already been run and results exist.

    Saves aggregate results to data/results/eval_summary.json.
    """
    conditions = conditions or CONDITIONS
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))

    if not image_paths:
        logger.warning(f"No images found in {image_dir}")
        return {}

    logger.info(f"Evaluating {len(image_paths)} images × {len(conditions)} conditions × 1 VLM ({vlm})")

    all_results = {}

    for img_path in sorted(image_paths):
        image_id = img_path.stem
        logger.info(f"\n--- Processing {image_id} ---")

        if not skip_pipeline:
            from pipeline import run_pipeline
            try:
                run_pipeline(str(img_path), vlm=vlm, skip_vlm=False)
            except Exception as e:
                logger.error(f"Pipeline failed for {image_id}: {e}")
                continue

        # Load per-condition results
        image_results = {}
        for condition in conditions:
            result_path = Path("data/results") / f"{image_id}_{condition}_{vlm}.json"
            if result_path.exists():
                r = load_json(str(result_path))
                image_results[condition] = {
                    "response": r.get("response", ""),
                    "measurements": r.get("measurements_used", {}),
                }
            else:
                image_results[condition] = {"response": "NOT_RUN", "measurements": {}}

        all_results[image_id] = image_results

    summary = {
        "n_images": len(all_results),
        "conditions": conditions,
        "vlm": vlm,
        "results": all_results,
    }

    out_path = "data/results/eval_summary.json"
    Path("data/results").mkdir(parents=True, exist_ok=True)
    save_json(summary, out_path)
    logger.info(f"\nEval summary saved → {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _count_specificity(response: str) -> float:
    """
    Heuristic specificity score [0, 1].
    Checks for numeric measurements, pass/fail language, location references.
    """
    if not response or response.startswith("ERROR") or response == "NOT_RUN":
        return 0.0

    score = 0.0
    # Has numeric measurements (inches)
    import re
    if re.search(r'\d+\.?\d*\s*["\']|inch|inches', response, re.IGNORECASE):
        score += 0.4
    # Has pass/fail language
    if re.search(r'pass|fail|compliant|non-compliant|deficien', response, re.IGNORECASE):
        score += 0.3
    # Has severity language
    if re.search(r'critical|major|minor|severity', response, re.IGNORECASE):
        score += 0.15
    # Has location references
    if re.search(r'left|right|center|corner|bay|bottom|top|between', response, re.IGNORECASE):
        score += 0.15
    return min(score, 1.0)


def compute_metrics(results_dir: str = "data/results") -> dict:
    """
    Compute aggregate metrics across all evaluated images.

    Currently computes:
      - specificity_score: heuristic measure of response quality
      - response_length: proxy for detail level
      - error_rate: fraction of failed VLM calls

    Ground truth numeric accuracy requires manually labeled data
    (stud_spacing_gt, etc.) in the measurements JSON.
    """
    summary_path = Path(results_dir) / "eval_summary.json"
    if not summary_path.exists():
        logger.error(f"No eval summary found at {summary_path}. Run run_evaluation() first.")
        return {}

    summary = load_json(str(summary_path))
    results = summary.get("results", {})
    conditions = summary.get("conditions", CONDITIONS)

    metrics: dict[str, dict] = {c: {"specificity": [], "length": [], "errors": 0} for c in conditions}

    for image_id, image_results in results.items():
        for condition in conditions:
            resp = image_results.get(condition, {}).get("response", "")
            if resp.startswith("ERROR") or resp == "NOT_RUN":
                metrics[condition]["errors"] += 1
            else:
                metrics[condition]["specificity"].append(_count_specificity(resp))
                metrics[condition]["length"].append(len(resp))

    aggregated = {}
    for condition in conditions:
        m = metrics[condition]
        n = len(m["specificity"]) or 1
        aggregated[condition] = {
            "mean_specificity": round(sum(m["specificity"]) / n, 3),
            "mean_response_length": round(sum(m["length"]) / n),
            "error_rate": round(m["errors"] / len(results), 3) if results else 0.0,
            "n_evaluated": n,
        }

    return aggregated


def print_results_table(metrics: dict) -> None:
    """Print a formatted comparison table for the 3 conditions."""
    if not metrics:
        print("No metrics to display.")
        return

    col_w = 22
    header = f"{'Metric':<28}" + "".join(f"{c:>{col_w}}" for c in metrics)
    print("\n" + "=" * (28 + col_w * len(metrics)))
    print("PreCheck — 3-Condition Evaluation Results")
    print("=" * (28 + col_w * len(metrics)))
    print(header)
    print("-" * (28 + col_w * len(metrics)))

    metric_names = {
        "mean_specificity": "Specificity Score (0-1)",
        "mean_response_length": "Avg Response Length (chars)",
        "error_rate": "Error Rate",
        "n_evaluated": "Images Evaluated",
    }

    for key, label in metric_names.items():
        row = f"{label:<28}"
        for condition in metrics:
            val = metrics[condition].get(key, "—")
            row += f"{str(val):>{col_w}}"
        print(row)

    print("=" * (28 + col_w * len(metrics)))
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PreCheck 3-condition evaluation")
    parser.add_argument("image_dir", nargs="?", default="data/frames", help="Directory of input images")
    parser.add_argument("--vlm", default="claude", choices=["claude", "gpt4o"])
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip pipeline, load existing results")
    parser.add_argument("--metrics-only", action="store_true", help="Only compute and print metrics")
    args = parser.parse_args()

    if not args.metrics_only:
        run_evaluation(args.image_dir, vlm=args.vlm, skip_pipeline=args.skip_pipeline)

    metrics = compute_metrics()
    print_results_table(metrics)
