#!/usr/bin/env python3
"""
Evaluate VLM depth predictions on point queries.

This script:
1. Loads frames from a CSV index
2. Samples random pixel coordinates per frame
3. Queries VLM for depth at those coordinates
4. Compares predictions to ground truth
5. Saves results and metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vlm_clients.openai_client import OpenAIVLMClient
from models.vlm_clients.anthropic_client import AnthropicVLMClient
from eval.metrics import compute_point_depth_metrics


def load_depth_gt(depth_path: str) -> np.ndarray:
    """Load ground truth depth map."""
    if depth_path.endswith('.npy'):
        return np.load(depth_path)
    elif depth_path.endswith('.png'):
        # Assume 16-bit PNG encoding depth in millimeters
        depth_img = Image.open(depth_path)
        depth_mm = np.array(depth_img, dtype=np.float32)
        return depth_mm / 1000.0  # Convert to meters
    else:
        raise ValueError(f"Unsupported depth format: {depth_path}")


def sample_points_stratified(image_shape: tuple, num_points: int, depth_gt: np.ndarray) -> List[tuple]:
    """
    Sample points using stratified grid sampling.

    Args:
        image_shape: (H, W) of the image
        num_points: Number of points to sample
        depth_gt: Ground truth depth map to avoid invalid regions

    Returns:
        List of (u, v) pixel coordinates
    """
    H, W = image_shape
    points = []

    # Create grid
    grid_size = int(np.sqrt(num_points))
    cell_h = H // grid_size
    cell_w = W // grid_size

    # Sample one point per cell
    for i in range(grid_size):
        for j in range(grid_size):
            # Try to find valid point in this cell
            for _ in range(10):  # Max 10 attempts per cell
                u = np.random.randint(j * cell_w, min((j + 1) * cell_w, W))
                v = np.random.randint(i * cell_h, min((i + 1) * cell_h, H))

                if depth_gt[v, u] > 0 and np.isfinite(depth_gt[v, u]):
                    points.append((u, v))
                    break

            if len(points) >= num_points:
                break
        if len(points) >= num_points:
            break

    return points[:num_points]


def sample_points_uniform(image_shape: tuple, num_points: int, depth_gt: np.ndarray) -> List[tuple]:
    """
    Sample points uniformly at random from valid depth regions.

    Args:
        image_shape: (H, W) of the image
        num_points: Number of points to sample
        depth_gt: Ground truth depth map to filter valid regions

    Returns:
        List of (u, v) pixel coordinates
    """
    H, W = image_shape
    valid_mask = (depth_gt > 0) & np.isfinite(depth_gt)
    valid_coords = np.argwhere(valid_mask)

    if len(valid_coords) < num_points:
        num_points = len(valid_coords)

    sampled_indices = np.random.choice(len(valid_coords), size=num_points, replace=False)
    points = [(int(coord[1]), int(coord[0])) for coord in valid_coords[sampled_indices]]

    return points


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM point depth predictions")
    parser.add_argument("--frames_csv", type=str, required=True, help="Path to frames CSV index")
    parser.add_argument("--config", type=str, default="eval/configs/models.yaml", help="Model config YAML")
    parser.add_argument("--out_jsonl", type=str, default="outputs/predictions/vlm_answers/points.jsonl", help="Output JSONL file")
    parser.add_argument("--model_type", type=str, choices=["openai", "anthropic"], default="openai", help="VLM provider")
    parser.add_argument("--model_name", type=str, help="Model name (overrides config)")
    parser.add_argument("--points_per_image", type=int, default=16, help="Number of points to sample per image")
    parser.add_argument("--max_images", type=int, help="Limit number of images to evaluate")
    parser.add_argument("--sampling_strategy", type=str, choices=["uniform", "stratified"], default="stratified")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Initialize VLM client
    if args.model_type == "openai":
        model_name = args.model_name or config["vlm_models"]["openai"][0]["name"]
        client = OpenAIVLMClient(model=model_name)
        print(f"Using OpenAI model: {model_name}")
    elif args.model_type == "anthropic":
        model_name = args.model_name or config["vlm_models"]["anthropic"][0]["name"]
        client = AnthropicVLMClient(model=model_name)
        print(f"Using Anthropic model: {model_name}")

    # Load frames index
    if not os.path.exists(args.frames_csv):
        print(f"Error: frames CSV not found at {args.frames_csv}")
        print("Please create the frames index first. You can use a simple CSV with columns:")
        print("  frame_id, rgb_path, depth_gt_path")
        return

    frames_df = pd.read_csv(args.frames_csv)

    if args.max_images:
        frames_df = frames_df.head(args.max_images)

    print(f"Loaded {len(frames_df)} frames")

    # Create output directory
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    # Evaluate each frame
    all_results = []

    for idx, row in tqdm(frames_df.iterrows(), total=len(frames_df), desc="Evaluating frames"):
        frame_id = row.get("frame_id", idx)
        rgb_path = row["rgb_path"]
        depth_gt_path = row["depth_gt_path"]

        if not os.path.exists(rgb_path):
            print(f"Warning: RGB image not found: {rgb_path}")
            continue

        if not os.path.exists(depth_gt_path):
            print(f"Warning: Depth GT not found: {depth_gt_path}")
            continue

        # Load ground truth depth
        depth_gt = load_depth_gt(depth_gt_path)
        image = Image.open(rgb_path)
        image_shape = (image.height, image.width)

        # Sample points
        if args.sampling_strategy == "stratified":
            points = sample_points_stratified(image_shape, args.points_per_image, depth_gt)
        else:
            points = sample_points_uniform(image_shape, args.points_per_image, depth_gt)

        # Query VLM for each point
        for point_idx, (u, v) in enumerate(points):
            gt_depth = float(depth_gt[v, u])

            if gt_depth <= 0 or not np.isfinite(gt_depth):
                continue

            # Query VLM
            result = client.query_point_depth(rgb_path, (u, v))

            # Store result
            all_results.append({
                "frame_id": frame_id,
                "point_idx": point_idx,
                "u": u,
                "v": v,
                "gt_depth": gt_depth,
                "predicted_depth": result["predicted_depth"],
                "raw_response": result["raw_response"],
                "model": result["model"],
                "rgb_path": rgb_path
            })

    # Save results to JSONL
    with open(args.out_jsonl, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')

    print(f"\nSaved {len(all_results)} predictions to {args.out_jsonl}")

    # Compute and display metrics
    valid_results = [r for r in all_results if r["predicted_depth"] is not None]

    if len(valid_results) > 0:
        predictions = np.array([r["predicted_depth"] for r in valid_results])
        ground_truths = np.array([r["gt_depth"] for r in valid_results])

        metrics = compute_point_depth_metrics(predictions, ground_truths)

        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        print(f"Valid predictions: {metrics['num_valid_points']} / {metrics['num_total_points']}")
        print(f"MAE (meters):      {metrics['mae']:.3f}")
        print(f"RMSE (meters):     {metrics['rmse']:.3f}")
        print(f"AbsRel:            {metrics['abs_rel']:.3f}")
        print("="*60)

        # Save metrics
        metrics_path = args.out_jsonl.replace('.jsonl', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {metrics_path}")
    else:
        print("\nWarning: No valid predictions to evaluate!")


if __name__ == "__main__":
    main()
