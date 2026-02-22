#!/usr/bin/env python3
"""
Evaluate Two-Head Model vs Baseline VLM

Compares three conditions from Technical.md:
1. Baseline: Raw VLM only
2. Depth-Augmented: VLM with depth visualization
3. Anchor-Calibrated (Two-Head): VLM with calibrated measurements
"""

import argparse
import json
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.two_head.two_head_model import TwoHeadModel
from models.vlm_clients.openai_client import OpenAIVLMClient
from models.vlm_clients.anthropic_client import AnthropicVLMClient


def run_baseline_vlm(
    image_path: str,
    question: str,
    vlm_client
) -> dict:
    """
    Condition 1: Baseline VLM with no augmentation.

    Args:
        image_path: Path to image
        question: Inspection question
        vlm_client: VLM client (OpenAI or Anthropic)

    Returns:
        Response dict
    """
    # For baseline, we just query the VLM directly with the question
    # No measurements, no depth info

    import base64
    from anthropic import Anthropic
    from openai import OpenAI

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""You are a construction inspection AI.

Look at this construction image and answer: {question}

Provide your analysis based on visual inspection."""

    if isinstance(vlm_client, AnthropicVLMClient):
        client = Anthropic()
        response = client.messages.create(
            model=vlm_client.model,
            max_tokens=500,
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
    else:  # OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=vlm_client.model,
            max_tokens=500,
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

    return {
        "condition": "baseline",
        "response": raw_response,
        "has_measurements": False
    }


def run_two_head_model(
    image_path: str,
    question: str,
    two_head: TwoHeadModel
) -> dict:
    """
    Condition 3: Anchor-calibrated two-head model.

    Args:
        image_path: Path to image
        question: Inspection question
        two_head: TwoHeadModel instance

    Returns:
        Complete pipeline result
    """
    # Load image
    image = np.array(Image.open(image_path))

    # Run two-head pipeline
    result = two_head.run(image, image_path, question)

    return {
        "condition": "two_head",
        "response": result["reasoning"]["response"],
        "raw_response": result["reasoning"]["raw_response"],
        "has_measurements": True,
        "num_measurements": result["measurement"]["num_measurements"],
        "calibration_confidence": result["reasoning"]["calibration_confidence"],
        "full_result": result
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate two-head model vs baseline")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory with test images")
    parser.add_argument("--vlm_provider", type=str, default="anthropic", choices=["openai", "anthropic"])
    parser.add_argument("--vlm_model", type=str, help="VLM model name")
    parser.add_argument("--question", type=str, default="Are there any code violations or deficiencies in this framing? Specifically check stud spacing.")
    parser.add_argument("--out_json", type=str, default="outputs/reports/two_head_eval.json")
    parser.add_argument("--max_images", type=int, help="Limit number of images")

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Set default model names
    if not args.vlm_model:
        if args.vlm_provider == "anthropic":
            args.vlm_model = "claude-sonnet-4-20250514"
        else:
            args.vlm_model = "gpt-4o"

    print("="*60)
    print("TWO-HEAD MODEL EVALUATION")
    print("="*60)
    print(f"VLM: {args.vlm_provider} / {args.vlm_model}")
    print(f"Images: {args.images_dir}")
    print(f"Question: {args.question}")
    print("="*60)
    print()

    # Initialize models
    print("Initializing models...")

    # Baseline VLM
    if args.vlm_provider == "anthropic":
        baseline_vlm = AnthropicVLMClient(model=args.vlm_model)
    else:
        baseline_vlm = OpenAIVLMClient(model=args.vlm_model)

    # Two-head model
    two_head = TwoHeadModel(
        vlm_provider=args.vlm_provider,
        vlm_model=args.vlm_model
    )

    # Get image files
    images_path = Path(args.images_dir)
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))

    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f"Found {len(image_files)} images")
    print()

    # Run evaluations
    results = []

    for img_path in tqdm(image_files, desc="Evaluating"):
        img_name = img_path.name

        print(f"\nProcessing: {img_name}")

        # Condition 1: Baseline
        print("  Running baseline VLM...")
        baseline_result = run_baseline_vlm(str(img_path), args.question, baseline_vlm)

        # Condition 3: Two-head
        print("  Running two-head model...")
        two_head_result = run_two_head_model(str(img_path), args.question, two_head)

        # Store results
        results.append({
            "image": img_name,
            "question": args.question,
            "baseline": baseline_result,
            "two_head": two_head_result
        })

        # Print preview
        print(f"  Baseline response preview: {baseline_result['response'][:100]}...")
        if two_head_result.get('num_measurements', 0) > 0:
            print(f"  Two-head found {two_head_result['num_measurements']} measurements")
        print(f"  Two-head response preview: {str(two_head_result['response'])[:100]}...")

    # Save results
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    with open(args.out_json, 'w') as f:
        json.dump({
            "config": {
                "vlm_provider": args.vlm_provider,
                "vlm_model": args.vlm_model,
                "question": args.question,
                "num_images": len(image_files)
            },
            "results": results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {args.out_json}")
    print(f"{'='*60}")

    # Summary statistics
    baseline_has_measurements = sum(1 for r in results if "inch" in r["baseline"]["response"].lower() or "spacing" in r["baseline"]["response"].lower())
    two_head_has_measurements = sum(1 for r in results if r["two_head"]["has_measurements"])

    print("\nSUMMARY:")
    print(f"  Baseline mentioned measurements: {baseline_has_measurements}/{len(results)}")
    print(f"  Two-head provided measurements: {two_head_has_measurements}/{len(results)}")
    print(f"  Avg measurements per image (two-head): {np.mean([r['two_head'].get('num_measurements', 0) for r in results]):.1f}")


if __name__ == "__main__":
    main()
