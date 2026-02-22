#!/usr/bin/env python3
"""
Benchmark Multiple Depth Models

Runs ablation study across different depth estimation models:
- Depth Anything V2 (Large)
- ZoeDepth (NK - indoor/outdoor hybrid)
- MiDaS (DPT-Large)

Compares performance and generates comparative visualization.
"""

import argparse
import subprocess
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


DEPTH_MODELS = [
    ("depth_anything_v2", "large", "Depth Anything V2 Large"),
    ("zoe", "nk", "ZoeDepth NK (Indoor/Outdoor)"),
    ("midas", "large", "MiDaS DPT-Large"),
]


def run_single_benchmark(
    data_dir: str,
    num_images: int,
    pairs_per_image: int,
    device: str,
    depth_model_type: str,
    depth_model_size: str,
    output_base_dir: str
):
    """Run ablation study for a single depth model."""
    output_dir = Path(output_base_dir) / f"{depth_model_type}_{depth_model_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print(f"BENCHMARKING: {depth_model_type}/{depth_model_size}")
    print("="*80 + "\n")

    cmd = [
        "uv", "run", "python", "eval/arkit/ablation_study.py",
        "--data_dir", data_dir,
        "--num_images", str(num_images),
        "--pairs_per_image", str(pairs_per_image),
        "--output_dir", str(output_dir),
        "--device", device,
        "--depth_model_type", depth_model_type,
        "--depth_model_size", depth_model_size,
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"⚠️  Benchmark failed for {depth_model_type}/{depth_model_size}")
        return None

    # Load results
    results_path = output_dir / "ablation_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    else:
        print(f"⚠️  No results found at {results_path}")
        return None


def compute_mae(results: list) -> float:
    """Compute Mean Absolute Error from results."""
    errors = []
    for r in results:
        if r.get("predicted_distance") is not None:
            error = abs(r["predicted_distance"] - r["gt_distance"])
            errors.append(error)

    return np.mean(errors) if errors else float('nan')


def compute_success_rate(results: list) -> float:
    """Compute success rate (% predictions parsed)."""
    total = len(results)
    successful = sum(1 for r in results if r.get("predicted_distance") is not None)
    return (successful / total * 100) if total > 0 else 0.0


def generate_comparison_plot(all_benchmarks: dict, output_path: str):
    """Generate comparison plot across depth models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Depth Model Comparison: Spatial Anchor Calibration", fontsize=16, fontweight='bold')

    conditions = ["vlm_only", "vlm_plus_depth", "vlm_plus_anchors", "full_spatial_anchor"]
    condition_labels = ["VLM Only", "VLM + Depth", "VLM + Anchors", "Full Spatial Anchor"]

    # Collect data for each depth model
    model_names = []
    condition_maes = {cond: [] for cond in conditions}
    condition_success_rates = {cond: [] for cond in conditions}

    for (model_type, model_size, display_name), results in all_benchmarks.items():
        if results is None:
            continue

        model_names.append(display_name)

        for cond in conditions:
            cond_results = results.get(cond, [])
            mae = compute_mae(cond_results)
            success = compute_success_rate(cond_results)

            condition_maes[cond].append(mae)
            condition_success_rates[cond].append(success)

    if not model_names:
        print("⚠️  No valid results to plot")
        return

    # Plot 1: MAE comparison (grouped bar chart)
    ax = axes[0, 0]
    x = np.arange(len(model_names))
    width = 0.2

    for i, (cond, label) in enumerate(zip(conditions, condition_labels)):
        maes = condition_maes[cond]
        offset = (i - 1.5) * width
        ax.bar(x + offset, maes, width, label=label)

    ax.set_xlabel("Depth Model", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (meters)", fontsize=12)
    ax.set_title("MAE by Depth Model & Condition", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Success rate comparison
    ax = axes[0, 1]

    for i, (cond, label) in enumerate(zip(conditions, condition_labels)):
        success_rates = condition_success_rates[cond]
        offset = (i - 1.5) * width
        ax.bar(x + offset, success_rates, width, label=label)

    ax.set_xlabel("Depth Model", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Prediction Success Rate", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    # Plot 3: Full Spatial Anchor performance comparison
    ax = axes[1, 0]
    full_maes = condition_maes["full_spatial_anchor"]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(model_names, full_maes, color=colors)

    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}m',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel("Mean Absolute Error (meters)", fontsize=12)
    ax.set_title("Full Spatial Anchor Calibration - MAE Comparison", fontsize=14, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Improvement over baseline (VLM Only)
    ax = axes[1, 1]
    improvements = []

    for i in range(len(model_names)):
        vlm_only_mae = condition_maes["vlm_only"][i]
        full_mae = condition_maes["full_spatial_anchor"][i]

        if not (np.isnan(vlm_only_mae) or np.isnan(full_mae)):
            improvement = ((vlm_only_mae - full_mae) / vlm_only_mae) * 100
            improvements.append(improvement)
        else:
            improvements.append(0)

    bars = ax.bar(model_names, improvements, color=colors)

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=10, fontweight='bold')

    ax.set_ylabel("Error Reduction (%)", fontsize=12)
    ax.set_title("Improvement Over VLM-Only Baseline", fontsize=14, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple depth models for spatial anchor calibration"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to ARKit data directory")
    parser.add_argument("--num_images", type=int, default=10,
                       help="Number of images to test per model")
    parser.add_argument("--pairs_per_image", type=int, default=3,
                       help="Number of point pairs per image")
    parser.add_argument("--output_dir", type=str, default="outputs/depth_benchmark",
                       help="Output directory for benchmark results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device for depth estimation")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       help="Specific models to benchmark (default: all)")

    args = parser.parse_args()
    load_dotenv()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter models if specified
    models_to_test = DEPTH_MODELS
    if args.models:
        models_to_test = [
            m for m in DEPTH_MODELS
            if f"{m[0]}_{m[1]}" in args.models or m[0] in args.models
        ]

    print("="*80)
    print("DEPTH MODEL BENCHMARK")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Test size: {args.num_images} images × {args.pairs_per_image} pairs")
    print(f"Device: {args.device}")
    print(f"Models to test: {len(models_to_test)}")
    for model_type, model_size, display_name in models_to_test:
        print(f"  - {display_name} ({model_type}/{model_size})")
    print("="*80)

    # Run benchmarks
    all_results = {}

    for model_type, model_size, display_name in models_to_test:
        results = run_single_benchmark(
            args.data_dir,
            args.num_images,
            args.pairs_per_image,
            args.device,
            model_type,
            model_size,
            args.output_dir
        )

        all_results[(model_type, model_size, display_name)] = results

    # Generate comparison plot
    comparison_plot_path = output_dir / "depth_model_comparison.png"
    generate_comparison_plot(all_results, str(comparison_plot_path))

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    for (model_type, model_size, display_name), results in all_results.items():
        if results is None:
            print(f"\n{display_name}: FAILED")
            continue

        print(f"\n{display_name}:")
        print("-" * 40)

        for cond in ["vlm_only", "vlm_plus_depth", "vlm_plus_anchors", "full_spatial_anchor"]:
            cond_results = results.get(cond, [])
            mae = compute_mae(cond_results)
            success = compute_success_rate(cond_results)

            cond_label = cond.replace("_", " ").title()
            print(f"  {cond_label:25s}: MAE={mae:.3f}m, Success={success:.1f}%")

    print("\n" + "="*80)
    print(f"✓ Benchmark complete! Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
