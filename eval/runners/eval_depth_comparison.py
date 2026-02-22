#!/usr/bin/env python3
"""
Compare depth estimation MAE:
1. Baseline VLM (Claude direct depth queries) - from existing results
2. Two-Head Perception (depth model from perception head)
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.two_head.perception_head import PerceptionHead

def load_baseline_results(jsonl_path):
    """Load baseline VLM results from JSONL."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results

def evaluate_perception_head(baseline_results):
    """Evaluate two-head perception depth model on same queries."""
    perception = PerceptionHead()

    comparisons = []
    current_image_path = None
    depth_map = None

    for query in tqdm(baseline_results, desc="Perception Head"):
        rgb_path = query['rgb_path']
        u, v = query['u'], query['v']
        depth_gt = query['gt_depth']
        depth_baseline = query.get('predicted_depth')

        # Load depth map (cache for same image)
        if rgb_path != current_image_path:
            image = np.array(Image.open(rgb_path))
            depth_map, _ = perception.run(image)
            current_image_path = rgb_path

        # Sample at query pixel
        h, w = depth_map.shape
        v_clip = min(v, h - 1)
        u_clip = min(u, w - 1)
        depth_perception = depth_map[v_clip, u_clip]

        # Compute errors
        error_baseline = abs(depth_baseline - depth_gt) if depth_baseline is not None else None
        error_perception = abs(depth_perception - depth_gt)

        comparisons.append({
            "frame_id": query['frame_id'],
            "point_idx": query['point_idx'],
            "u": u,
            "v": v,
            "depth_gt": depth_gt,
            "depth_baseline": depth_baseline,
            "depth_perception": float(depth_perception),
            "error_baseline": error_baseline,
            "error_perception": float(error_perception)
        })

    return comparisons

def main():
    parser = argparse.ArgumentParser(description="Compare depth MAE: Baseline vs Two-Head")
    parser.add_argument("--baseline_results", type=str, default="outputs/predictions/vlm_answers/claude_sonnet46_forced.jsonl")
    parser.add_argument("--out_json", type=str, default="outputs/reports/depth_mae_comparison.json")

    args = parser.parse_args()

    print("="*60)
    print("DEPTH ESTIMATION MAE COMPARISON")
    print("="*60)
    print("Comparing:")
    print("  1. Baseline VLM (Claude direct queries)")
    print("  2. Two-Head Perception (depth model)")
    print("="*60)
    print()

    # Load baseline results
    print(f"Loading baseline results from: {args.baseline_results}")
    baseline_results = load_baseline_results(args.baseline_results)
    print(f"Loaded {len(baseline_results)} queries")

    # Run perception head on same queries
    print("\nEvaluating Two-Head Perception...")
    comparisons = evaluate_perception_head(baseline_results)

    # Compute MAE for each method
    errors_baseline = [c['error_baseline'] for c in comparisons if c['error_baseline'] is not None]
    errors_perception = [c['error_perception'] for c in comparisons]

    mae_baseline = np.mean(errors_baseline) if errors_baseline else None
    mae_perception = np.mean(errors_perception)

    improvement = (mae_baseline - mae_perception) if mae_baseline else None
    improvement_pct = (improvement / mae_baseline * 100) if mae_baseline else None

    results = {
        "baseline_vlm": {
            "method": "claude-sonnet-4-20250514",
            "mae": mae_baseline,
            "num_successful": len(errors_baseline),
            "total_queries": len(comparisons)
        },
        "perception_head": {
            "method": "depth_anything_v2_placeholder",
            "mae": mae_perception,
            "num_successful": len(errors_perception),
            "total_queries": len(comparisons)
        },
        "comparison": {
            "baseline_mae": mae_baseline,
            "perception_mae": mae_perception,
            "improvement": improvement,
            "improvement_pct": improvement_pct
        },
        "per_query_results": comparisons
    }

    # Save
    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline VLM (Claude) MAE: {mae_baseline:.2f}m")
    print(f"Perception Head MAE: {mae_perception:.2f}m")
    if improvement:
        improvement_sign = "better" if improvement > 0 else "worse"
        print(f"Perception is {abs(improvement):.2f}m ({abs(improvement_pct):.1f}%) {improvement_sign}")
    print(f"\nResults saved to: {args.out_json}")
    print("="*60)

if __name__ == "__main__":
    main()
