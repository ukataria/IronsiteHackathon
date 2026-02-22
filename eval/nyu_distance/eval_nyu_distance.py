#!/usr/bin/env python3
"""
Evaluate VLMs on 3D distance estimation using NYU Depth V2.

Compares:
1. Claude Sonnet 4.6
2. GPT-4o

Task: Given two marked points in an image, estimate the 3D distance between them.

Ground truth: Computed from depth map + camera intrinsics.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from dotenv import load_dotenv
import base64

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.nyu_distance.nyu_utils import NYUDepthLoader, generate_object_pairs
from models.vlm_clients.anthropic_client import AnthropicVLMClient
from models.vlm_clients.openai_client import OpenAIVLMClient


def create_marked_image(
    rgb_path: str,
    point1: tuple,
    point2: tuple,
    output_path: str
) -> str:
    """
    Create image with two marked points.

    Args:
        rgb_path: Path to RGB image
        point1: (u, v) pixel coords of first point
        point2: (u, v) pixel coords of second point
        output_path: Where to save marked image

    Returns:
        Path to marked image
    """
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


def query_vlm_distance(
    vlm_client,
    marked_image_path: str,
    point1: tuple,
    point2: tuple
) -> dict:
    """
    Query VLM for distance between two marked points.

    Args:
        vlm_client: VLM client (Anthropic or OpenAI)
        marked_image_path: Path to image with marked points
        point1: (u, v) coordinates of point A
        point2: (u, v) coordinates of point B

    Returns:
        Dict with predicted_distance, raw_response, model
    """
    prompt = f"""You are analyzing an indoor scene image with two points marked:
- Point A (red cross) at pixel position ({point1[0]}, {point1[1]})
- Point B (blue cross) at pixel position ({point2[0]}, {point2[1]})

TASK: Estimate the 3D Euclidean distance in meters between the two marked points.

CRITICAL REQUIREMENTS:
- You MUST provide a numeric estimate - refusal is not acceptable
- Consider depth, perspective, and typical indoor object sizes
- Base your estimate on visual cues in the image
- This is a real indoor scene captured with RGB-D camera

OUTPUT FORMAT:
- Respond with ONLY a single number in meters
- NO explanations, NO refusals, NO disclaimers
- Just the number
- Examples: "1.5" or "0.8" or "2.3"

Your distance estimate in meters (number only):"""

    # Encode image
    with open(marked_image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Query based on client type
    try:
        if hasattr(vlm_client, 'model') and 'claude' in vlm_client.model:
            # Anthropic
            from anthropic import Anthropic
            client = Anthropic()
            response = client.messages.create(
                model=vlm_client.model,
                max_tokens=10,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            raw_response = response.content[0].text
        else:
            # OpenAI
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=vlm_client.model,
                max_tokens=10,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            raw_response = response.choices[0].message.content

        raw_response = raw_response.strip()

        # Parse numeric response
        try:
            # Extract first number
            import re
            numbers = re.findall(r'\d+\.?\d*', raw_response)
            if numbers:
                predicted_distance = float(numbers[0])
            else:
                predicted_distance = None
        except:
            predicted_distance = None

        return {
            "predicted_distance": predicted_distance,
            "raw_response": raw_response,
            "model": vlm_client.model
        }

    except Exception as e:
        return {
            "predicted_distance": None,
            "raw_response": f"Error: {str(e)}",
            "model": vlm_client.model
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on NYU Depth V2 distance estimation")
    parser.add_argument("--nyu_data_dir", type=str, default="data/nyu_depth_v2/extracted")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to test")
    parser.add_argument("--pairs_per_image", type=int, default=3, help="Object pairs per image")
    parser.add_argument("--out_dir", type=str, default="outputs/nyu_distance")
    parser.add_argument("--skip_download", action="store_true", help="Skip dataset download check")

    args = parser.parse_args()

    load_dotenv()

    print("="*60)
    print("NYU DEPTH V2 - 3D DISTANCE EVALUATION")
    print("="*60)
    print(f"Dataset: {args.nyu_data_dir}")
    print(f"Testing: {args.num_images} images x {args.pairs_per_image} pairs")
    print("="*60)
    print()

    # Check if dataset exists
    nyu_data_path = Path(args.nyu_data_dir)
    if not nyu_data_path.exists() and not args.skip_download:
        print("Dataset not found. Run download script first:")
        print("  python data/nyu_depth_v2/download_nyu.py")
        return

    # Initialize NYU loader
    print("Loading NYU Depth V2 dataset...")
    loader = NYUDepthLoader(args.nyu_data_dir)

    # Initialize VLMs
    print("Initializing VLMs...")
    claude_client = AnthropicVLMClient(model="claude-sonnet-4-20250514")
    gpt_client = OpenAIVLMClient(model="gpt-4o")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "marked_images").mkdir(exist_ok=True)

    # Run evaluation
    all_results = {
        "claude": [],
        "gpt4o": []
    }

    for img_idx in tqdm(range(args.num_images), desc="Images"):
        # Load sample
        sample = loader.get_sample(img_idx)
        rgb_path = sample["rgb_path"]
        depth_map = sample["depth_map"]

        # Generate object pairs
        pairs = generate_object_pairs(
            depth_map,
            num_pairs=args.pairs_per_image
        )

        for pair_idx, pair in enumerate(pairs):
            point1 = pair["point1"]
            point2 = pair["point2"]

            # Compute ground truth distance
            gt_distance = loader.compute_distance_3d(point1, point2, depth_map)

            # Create marked image
            marked_path = out_dir / "marked_images" / f"img{img_idx:04d}_pair{pair_idx}.jpg"
            create_marked_image(rgb_path, point1, point2, str(marked_path))

            # Query Claude
            claude_result = query_vlm_distance(claude_client, str(marked_path), point1, point2)
            claude_result.update({
                "image_idx": img_idx,
                "pair_idx": pair_idx,
                "point1": point1,
                "point2": point2,
                "gt_distance": float(gt_distance)
            })
            all_results["claude"].append(claude_result)

            # Query GPT-4o
            gpt_result = query_vlm_distance(gpt_client, str(marked_path), point1, point2)
            gpt_result.update({
                "image_idx": img_idx,
                "pair_idx": pair_idx,
                "point1": point1,
                "point2": point2,
                "gt_distance": float(gt_distance)
            })
            all_results["gpt4o"].append(gpt_result)

    # Compute metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    for model_name in ["claude", "gpt4o"]:
        results = all_results[model_name]

        # Filter successful predictions
        successful = [r for r in results if r["predicted_distance"] is not None]

        if successful:
            errors = [abs(r["predicted_distance"] - r["gt_distance"]) for r in successful]
            mae = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)

            print(f"\n{model_name.upper()}:")
            print(f"  Successful predictions: {len(successful)}/{len(results)}")
            print(f"  MAE: {mae:.3f}m")
            print(f"  Median Error: {median_error:.3f}m")
            print(f"  Std Error: {std_error:.3f}m")
        else:
            print(f"\n{model_name.upper()}: No successful predictions")

    # Save results
    results_path = out_dir / "nyu_distance_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("="*60)


if __name__ == "__main__":
    main()
