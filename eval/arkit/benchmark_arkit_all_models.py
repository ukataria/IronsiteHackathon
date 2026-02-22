#!/usr/bin/env python3
"""
Benchmark all VLMs on ARKit LiDAR 3D distance estimation.

Evaluates models across closed, open research, and edge-deployable categories
on ARKit depth data captured from iPhone/iPad Pro LiDAR.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.nyu_distance.nyu_utils import NYUDepthLoader, generate_object_pairs
from models.vlm_clients.anthropic_client import AnthropicVLMClient
from models.vlm_clients.openai_client import OpenAIVLMClient
from models.vlm_clients.gemini_client import GeminiVLMClient
from models.vlm_clients.huggingface_client import HuggingFaceVLMClient
from models.vlm_clients.twohead_client import TwoHeadVLMClient


# Model configurations
MODELS = {
    # Closed source (API-based)
    "gpt-4o": {
        "category": "closed",
        "client_class": OpenAIVLMClient,
        "model_name": "gpt-4o",
        "requires_api": True,
        "env_var": "OPENAI_API_KEY"
    },
    "claude-sonnet-4": {
        "category": "closed",
        "client_class": AnthropicVLMClient,
        "model_name": "claude-sonnet-4-20250514",
        "requires_api": True,
        "env_var": "ANTHROPIC_API_KEY"
    },
    "gemini-2.0-flash": {
        "category": "closed",
        "client_class": GeminiVLMClient,
        "model_name": "gemini-2.0-flash-exp",
        "requires_api": True,
        "env_var": "GEMINI_API_KEY"
    },

    # Two-Head Architecture (Ours)
    "two-head-claude": {
        "category": "two-head",
        "client_class": TwoHeadVLMClient,
        "model_name": "claude-sonnet-4-20250514",
        "requires_api": True,
        "requires_gpu": True,
        "env_var": "ANTHROPIC_API_KEY",
        "description": "Spatial Anchor Calibration + Claude Sonnet 4"
    }
}


def create_marked_image(rgb_path, point1, point2, output_path):
    """Create image with marked points."""
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw point 1 (red cross)
    u1, v1 = point1
    size = 10
    draw.line([(u1-size, v1), (u1+size, v1)], fill="red", width=3)
    draw.line([(u1, v1-size), (u1, v1+size)], fill="red", width=3)
    draw.text((u1+15, v1-15), "A", fill="red")

    # Draw point 2 (blue cross)
    u2, v2 = point2
    draw.line([(u2-size, v2), (u2+size, v2)], fill="blue", width=3)
    draw.line([(u2, v2-size), (u2, v2+size)], fill="blue", width=3)
    draw.text((u2+15, v2-15), "B", fill="blue")

    img.save(output_path)
    return output_path


def query_vlm_distance(vlm_client, marked_image_path, point1, point2):
    """Query VLM for distance between marked points."""
    prompt = f"""You are analyzing an indoor scene with two points marked:
- Point A (red cross) at pixel ({point1[0]}, {point1[1]})
- Point B (blue cross) at pixel ({point2[0]}, {point2[1]})

TASK: Estimate the 3D Euclidean distance in meters between the two marked points.

CRITICAL REQUIREMENTS:
- You MUST provide a numeric estimate - refusal is not acceptable
- Consider depth, perspective, and typical indoor object sizes
- Base your estimate on visual cues in the image

OUTPUT FORMAT:
- Respond with ONLY a single number in meters
- NO explanations, NO refusals, NO disclaimers
- Just the number
- Examples: "1.5" or "0.8" or "2.3"

Your distance estimate in meters (number only):"""

    # Handle different client types
    if hasattr(vlm_client, 'query_distance'):
        return vlm_client.query_distance(marked_image_path, point1, point2, prompt)

    # Fallback for API clients
    import base64
    with open(marked_image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    try:
        if isinstance(vlm_client, AnthropicVLMClient):
            from anthropic import Anthropic
            client = Anthropic()
            response = client.messages.create(
                model=vlm_client.model,
                max_tokens=10,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            raw_response = response.content[0].text

        elif isinstance(vlm_client, OpenAIVLMClient):
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=vlm_client.model,
                max_tokens=10,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            raw_response = response.choices[0].message.content

        else:
            return {"predicted_distance": None, "raw_response": "Unknown client", "model": str(vlm_client)}

        raw_response = raw_response.strip()

        # Parse numeric
        import re
        numbers = re.findall(r'\d+\.?\d*', raw_response)
        predicted_distance = float(numbers[0]) if numbers else None

        return {
            "predicted_distance": predicted_distance,
            "raw_response": raw_response,
            "model": vlm_client.model
        }

    except Exception as e:
        return {
            "predicted_distance": None,
            "raw_response": f"Error: {str(e)}",
            "model": getattr(vlm_client, 'model', 'unknown')
        }


def plot_results(results_by_model, output_path):
    """Create comparison visualization."""
    model_stats = []

    for model_name, results in results_by_model.items():
        successful = [r for r in results if r["predicted_distance"] is not None]

        if successful:
            errors = [abs(r["predicted_distance"] - r["gt_distance"]) for r in successful]
            mae = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)
            success_rate = len(successful) / len(results) * 100
        else:
            mae = np.nan
            median_error = np.nan
            std_error = np.nan
            success_rate = 0

        model_stats.append({
            "model": model_name,
            "category": MODELS[model_name]["category"],
            "mae": mae,
            "median_error": median_error,
            "std_error": std_error,
            "success_rate": success_rate
        })

    df = pd.DataFrame(model_stats).sort_values("mae")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. MAE comparison
    ax = axes[0, 0]
    colors = {"closed": "blue", "open-research": "green", "edge": "orange", "two-head": "red"}
    bar_colors = [colors.get(row["category"], "gray") for _, row in df.iterrows()]

    ax.barh(df["model"], df["mae"], color=bar_colors)
    ax.set_xlabel("MAE (meters)", fontsize=12)
    ax.set_title("Mean Absolute Error by Model (ARKit LiDAR)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Closed Source'),
        Patch(facecolor='red', label='Two-Head (Ours)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # 2. Success rate
    ax = axes[0, 1]
    ax.barh(df["model"], df["success_rate"], color=bar_colors)
    ax.set_xlabel("Success Rate (%)", fontsize=12)
    ax.set_title("Response Success Rate", fontsize=14, fontweight="bold")
    ax.set_xlim([0, 100])
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 3. Median error
    ax = axes[1, 0]
    ax.barh(df["model"], df["median_error"], color=bar_colors)
    ax.set_xlabel("Median Error (meters)", fontsize=12)
    ax.set_title("Median Absolute Error by Model", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 4. Error distribution
    ax = axes[1, 1]
    category_stats = df.groupby("category").agg({"mae": "mean", "success_rate": "mean"})
    category_stats.plot(kind="bar", ax=ax, color=["blue", "green"])
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Average Performance by Category", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(["MAE (m)", "Success Rate (%)"], loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Benchmark all VLMs on ARKit LiDAR distance estimation")
    parser.add_argument("--arkit_data_dir", type=str, required=True, help="Path to ARKit data (with rgb/, depth/, intrinsics.npy)")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--pairs_per_image", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="outputs/arkit_benchmark")
    parser.add_argument("--models", nargs="+", help="Specific models to test (default: all)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()
    load_dotenv()

    print("="*60)
    print("ARKit LiDAR DISTANCE BENCHMARK - ALL MODELS")
    print("="*60)
    print(f"Dataset: {args.arkit_data_dir}")
    print(f"Test size: {args.num_images} images × {args.pairs_per_image} pairs")
    print(f"Device: {args.device}")
    print("="*60)
    print()

    # Determine which models to test
    models_to_test = args.models if args.models else list(MODELS.keys())

    # Filter out models without API keys
    import os
    final_models = []
    for model_name in models_to_test:
        model_config = MODELS[model_name]

        if model_config.get("requires_api"):
            env_var = model_config.get("env_var")
            if env_var and not os.getenv(env_var):
                print(f"⚠️  Skipping {model_name}: {env_var} not set")
                continue

        final_models.append(model_name)

    print(f"Testing {len(final_models)} models:")
    for m in final_models:
        print(f"  • {m} ({MODELS[m]['category']})")
    print()

    # Initialize loader (reuse NYU loader - same interface)
    loader = NYUDepthLoader(args.arkit_data_dir)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "marked_images").mkdir(exist_ok=True)

    # Store results
    all_results = {}

    # Run evaluation for each model
    for model_idx, model_name in enumerate(final_models, 1):
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name} ({model_idx}/{len(final_models)})")
        print(f"{'='*60}")

        model_config = MODELS[model_name]
        client_class = model_config["client_class"]

        # Initialize client
        try:
            if client_class == TwoHeadVLMClient:
                client = client_class(
                    vlm_model=model_config["model_name"],
                    device=args.device
                )
            elif model_config.get("requires_api"):
                client = client_class(model=model_config["model_name"])
            else:
                client = client_class(model_name=model_config["model_name"], device=args.device)
        except Exception as e:
            print(f"✗ Failed to initialize {model_name}: {e}")
            all_results[model_name] = []
            continue

        model_results = []
        total_queries = args.num_images * args.pairs_per_image

        # Run on test images
        for img_idx in range(args.num_images):
            sample = loader.get_sample(img_idx)
            rgb_path = sample["rgb_path"]
            depth_map = sample["depth_map"]

            # Generate pairs
            pairs = generate_object_pairs(depth_map, num_pairs=args.pairs_per_image)

            for pair_idx, pair in enumerate(pairs):
                point1 = pair["point1"]
                point2 = pair["point2"]

                # Compute GT distance
                gt_distance = loader.compute_distance_3d(point1, point2, depth_map)

                # Create marked image
                marked_path = out_dir / "marked_images" / f"{model_name}_img{img_idx:04d}_pair{pair_idx}.jpg"
                create_marked_image(rgb_path, point1, point2, str(marked_path))

                # Print what we're asking
                current_query = img_idx * args.pairs_per_image + pair_idx + 1
                print(f"\n  [{current_query}/{total_queries}] Image {img_idx}, Pair {pair_idx}")
                print(f"  Question: \"What is the 3D Euclidean distance in meters between")
                print(f"            point A (red cross at pixel {point1}) and")
                print(f"            point B (blue cross at pixel {point2})?\"")
                print(f"  Ground Truth: {gt_distance:.2f}m")

                # Query model
                result = query_vlm_distance(client, str(marked_path), point1, point2)

                if result["predicted_distance"] is not None:
                    print(f"  Predicted: {result['predicted_distance']:.2f}m | Error: {abs(result['predicted_distance'] - gt_distance):.2f}m")
                else:
                    print(f"  Predicted: FAILED - {result.get('raw_response', 'Unknown error')}")

                result.update({
                    "image_idx": img_idx,
                    "pair_idx": pair_idx,
                    "point1": point1,
                    "point2": point2,
                    "gt_distance": float(gt_distance)
                })
                model_results.append(result)

        all_results[model_name] = model_results

        # Print interim results
        successful = [r for r in model_results if r["predicted_distance"] is not None]
        if successful:
            errors = [abs(r["predicted_distance"] - r["gt_distance"]) for r in successful]
            mae = np.mean(errors)
            median_error = np.median(errors)
            print(f"\n✓ {model_name} COMPLETE:")
            print(f"  MAE: {mae:.3f}m | Median: {median_error:.3f}m | Success: {len(successful)}/{len(model_results)}")
        else:
            print(f"\n✗ {model_name}: No successful predictions")

    # Save results
    results_path = out_dir / "arkit_benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "dataset": "ARKit LiDAR",
            "config": {
                "num_images": args.num_images,
                "pairs_per_image": args.pairs_per_image,
                "device": args.device
            },
            "results": all_results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # Generate visualization
    print("\nGenerating visualization...")
    plot_path = out_dir / "arkit_benchmark_comparison.png"
    df = plot_results(all_results, plot_path)

    # Print summary table
    print("\n" + "="*60)
    print("ARKIT LIDAR BENCHMARK SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)


if __name__ == "__main__":
    import os
    main()
