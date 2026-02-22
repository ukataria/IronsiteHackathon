#!/usr/bin/env python3
"""
Ablation Study: Spatial Anchor Calibration Components

Evaluates contribution of each component:
1. VLM only (baseline)
2. VLM + depth information
3. VLM + anchor information (no depth)
4. Full Spatial Anchor Calibration (depth + anchors)

Uses Claude Sonnet 4 as the VLM backbone for all conditions.
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
from models.anchor_detection import AnchorDetector
from models.depth_estimator import DepthEstimator


def create_marked_image(rgb_path, point1, point2, output_path):
    """Create image with marked points."""
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw point 1 (red cross)
    u1, v1 = point1
    size = 10
    draw.line([(u1-size, v1), (u1+size, v1)], fill="red", width=3)
    draw.line([(u1, v1-size), (u1, v1+size)], fill="red", width=3)
    draw.text((u1+15, v1-15), "A", fill="red", font_size=20)

    # Draw point 2 (blue cross)
    u2, v2 = point2
    draw.line([(u2-size, v2), (u2+size, v2)], fill="blue", width=3)
    draw.line([(u2, v2-size), (u2, v2+size)], fill="blue", width=3)
    draw.text((u2+15, v2-15), "B", fill="blue", font_size=20)

    img.save(output_path)
    return output_path


class AblationCondition:
    """Base class for ablation conditions."""

    def __init__(self, vlm_model="claude-sonnet-4-20250514", device="cuda"):
        self.vlm = AnthropicVLMClient(model=vlm_model)
        self.device = device

    def query_distance(self, image_path, point1, point2, prompt, **kwargs):
        """Query distance - to be overridden by each condition."""
        raise NotImplementedError


class VLMOnlyCondition(AblationCondition):
    """Condition 1: VLM only (baseline)"""

    def __init__(self, vlm_model="claude-sonnet-4-20250514", device="cuda"):
        super().__init__(vlm_model, device)
        print("  Condition 1: VLM Only (baseline)")

    def query_distance(self, image_path, point1, point2, prompt, **kwargs):
        """Query VLM directly without any augmentation."""
        # Use the VLM's native query method
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model=self.vlm.model,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": prompt}
            ]}]
        )
        raw_response = response.content[0].text

        return {"response": raw_response, "augmented_prompt": prompt}


class VLMPlusDepthCondition(AblationCondition):
    """Condition 2: VLM + Depth information"""

    def __init__(self, vlm_model="claude-sonnet-4-20250514", device="cuda"):
        super().__init__(vlm_model, device)
        self.depth_estimator = DepthEstimator(model_size="large", device=device)
        print("  Condition 2: VLM + Depth")

    def query_distance(self, image_path, point1, point2, prompt, **kwargs):
        """Augment prompt with depth values at the two points."""
        # Get depth map
        depth_map = self.depth_estimator.estimate_depth(image_path)

        u1, v1 = point1
        u2, v2 = point2

        # Get depth at points (with small region averaging for robustness)
        def get_depth_region(depth, u, v, size=5):
            h, w = depth.shape
            u = int(np.clip(u, size, w-size-1))
            v = int(np.clip(v, size, h-size-1))
            region = depth[v-size:v+size, u-size:u+size]
            return float(np.median(region))

        d1 = get_depth_region(depth_map, u1, v1)
        d2 = get_depth_region(depth_map, u2, v2)

        # Augment prompt with depth information
        depth_info = f"\n\nADDITIONAL DEPTH INFORMATION:\n"
        depth_info += f"- Estimated depth at point A: {d1:.2f} meters\n"
        depth_info += f"- Estimated depth at point B: {d2:.2f} meters\n"
        depth_info += f"This depth information is from a monocular depth estimation model.\n"

        augmented_prompt = prompt + depth_info

        # Query VLM with augmented prompt
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model=self.vlm.model,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": augmented_prompt}
            ]}]
        )
        raw_response = response.content[0].text

        return {"response": raw_response, "augmented_prompt": augmented_prompt, "depth_a": d1, "depth_b": d2}


class VLMPlusAnchorsCondition(AblationCondition):
    """Condition 3: VLM + Anchor information (no depth)"""

    def __init__(self, vlm_model="claude-sonnet-4-20250514", device="cuda"):
        super().__init__(vlm_model, device)
        self.anchor_detector = AnchorDetector(device=device)
        print("  Condition 3: VLM + Anchors (no depth)")

    def query_distance(self, image_path, point1, point2, prompt, **kwargs):
        """Augment prompt with detected anchor information."""
        # Detect anchors
        anchors = self.anchor_detector.detect(image_path, confidence_threshold=0.01)

        if not anchors:
            # No anchors - same as VLM only
            import base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            from anthropic import Anthropic
            client = Anthropic()
            response = client.messages.create(
                model=self.vlm.model,
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            raw_response = response.content[0].text

            return {"response": raw_response, "augmented_prompt": prompt, "anchors_detected": 0}

        # Build anchor information
        anchor_info = f"\n\nKNOWN-DIMENSION OBJECTS DETECTED:\n"
        anchor_info += f"The following objects with known real-world dimensions were detected:\n\n"

        for i, anchor in enumerate(anchors, 1):
            anchor_info += f"{i}. {anchor.class_name.replace('_', ' ').title()}\n"
            anchor_info += f"   - Known dimension: {anchor.known_width:.1f} inches\n"
            anchor_info += f"   - Bounding box: ({anchor.bbox[0]}, {anchor.bbox[1]}) to ({anchor.bbox[2]}, {anchor.bbox[3]})\n"
            anchor_info += f"   - Center: ({anchor.center[0]:.0f}, {anchor.center[1]:.0f})\n"
            anchor_info += f"   - Pixel width: {anchor.pixel_width:.0f} pixels\n"
            anchor_info += f"   - Confidence: {anchor.confidence:.2f}\n\n"

        anchor_info += "You can use these known dimensions to help calibrate your distance estimate.\n"

        augmented_prompt = prompt + anchor_info

        # Query VLM
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model=self.vlm.model,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": augmented_prompt}
            ]}]
        )
        raw_response = response.content[0].text

        return {"response": raw_response, "augmented_prompt": augmented_prompt,
               "anchors_detected": len(anchors)}


class FullSpatialAnchorCondition(AblationCondition):
    """Condition 4: Full Spatial Anchor Calibration (depth + anchors)"""

    def __init__(self, vlm_model="claude-sonnet-4-20250514", device="cuda"):
        super().__init__(vlm_model, device)
        self.anchor_detector = AnchorDetector(device=device)
        self.depth_estimator = DepthEstimator(model_size="large", device=device)
        print("  Condition 4: Full Spatial Anchor Calibration")

    def query_distance(self, image_path, point1, point2, prompt, **kwargs):
        """Full spatial anchor calibration with depth and anchors."""
        # Import spatial calibration
        from models.spatial_calibration import SpatialCalibrator

        # Detect anchors
        anchors = self.anchor_detector.detect(image_path, confidence_threshold=0.01)

        # Get depth map
        depth_map = self.depth_estimator.estimate_depth(image_path)

        if not anchors:
            # Fall back to VLM + depth
            u1, v1 = point1
            u2, v2 = point2

            def get_depth_region(depth, u, v, size=5):
                h, w = depth.shape
                u = int(np.clip(u, size, w-size-1))
                v = int(np.clip(v, size, h-size-1))
                region = depth[v-size:v+size, u-size:u+size]
                return float(np.median(region))

            d1 = get_depth_region(depth_map, u1, v1)
            d2 = get_depth_region(depth_map, u2, v2)

            depth_info = f"\n\nESTIMATED DEPTH INFORMATION:\n"
            depth_info += f"- Depth at point A: {d1:.2f} meters\n"
            depth_info += f"- Depth at point B: {d2:.2f} meters\n"

            augmented_prompt = prompt + depth_info

            # Query VLM
            import base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            from anthropic import Anthropic
            client = Anthropic()
            response = client.messages.create(
                model=self.vlm.model,
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                    {"type": "text", "text": augmented_prompt}
                ]}]
            )
            raw_response = response.content[0].text

            return {"response": raw_response, "augmented_prompt": augmented_prompt,
                   "anchors_detected": 0, "calibrated": False}

        # Perform spatial calibration
        calibrator = SpatialCalibrator()
        depth_planes = calibrator.calibrate(anchors, depth_map)

        if not depth_planes:
            # Calibration failed - fall back to VLM only
            import base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            from anthropic import Anthropic
            client = Anthropic()
            response = client.messages.create(
                model=self.vlm.model,
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            raw_response = response.content[0].text

            return {"response": raw_response, "augmented_prompt": prompt, "anchors_detected": len(anchors),
                   "calibrated": False}

        # Build calibration information
        calib_info = f"\n\nCALIBRATED SPATIAL MEASUREMENTS:\n"
        calib_info += f"Using {len(anchors)} detected object(s) with known dimensions for calibration:\n\n"

        for i, anchor in enumerate(anchors, 1):
            calib_info += f"{i}. {anchor.class_name.replace('_', ' ').title()}: "
            calib_info += f"{anchor.known_width:.1f} inches\n"

        calib_info += f"\nCalibration results:\n"
        for i, plane in enumerate(depth_planes, 1):
            calib_info += f"- Depth plane {i}: {plane.scale:.2f} pixels/inch "
            calib_info += f"(confidence: {plane.confidence:.2f})\n"

        # Compute calibrated measurement
        measurement = calibrator.measure_distance_2d(point1, point2, depth_map)

        if measurement:
            distance_m = measurement['distance_inches'] * 0.0254  # inches to meters
            calib_info += f"\nCALIBRATED MEASUREMENT:\n"
            calib_info += f"- Estimated 3D distance: {distance_m:.2f} meters "
            calib_info += f"({measurement['distance_inches']:.1f} inches)\n"
            calib_info += f"- Scale used: {measurement['scale']:.2f} pixels/inch\n"
            calib_info += f"- Calibration confidence: {measurement['confidence']:.2f}\n"

        augmented_prompt = prompt + calib_info

        # Query VLM
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model=self.vlm.model,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": augmented_prompt}
            ]}]
        )
        raw_response = response.content[0].text

        return {"response": raw_response, "augmented_prompt": augmented_prompt,
               "anchors_detected": len(anchors), "calibrated": True,
               "measurement": measurement}


def extract_distance_from_response(response_text):
    """Extract numerical distance from VLM response."""
    import re

    # Look for patterns like "X.XX meters" or "X.XX m"
    patterns = [
        r'(\d+\.?\d*)\s*meters?',
        r'(\d+\.?\d*)\s*m\b',
        r'approximately\s+(\d+\.?\d*)',
        r'around\s+(\d+\.?\d*)',
        r'about\s+(\d+\.?\d*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text.lower())
        if match:
            try:
                return float(match.group(1))
            except:
                continue

    return None


def run_ablation_study(data_dir, num_images, pairs_per_image, output_dir, device):
    """Run ablation study across all conditions."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ABLATION STUDY: Spatial Anchor Calibration")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Test size: {num_images} images × {pairs_per_image} pairs")
    print(f"Device: {device}")
    print("="*60)
    print()

    # Initialize conditions
    conditions = {
        "vlm_only": VLMOnlyCondition(device=device),
        "vlm_plus_depth": VLMPlusDepthCondition(device=device),
        "vlm_plus_anchors": VLMPlusAnchorsCondition(device=device),
        "full_spatial_anchor": FullSpatialAnchorCondition(device=device)
    }

    # Load data
    print("\nLoading ARKit data...")
    loader = NYUDepthLoader(data_dir)

    # Store results
    all_results = {cond_name: [] for cond_name in conditions.keys()}

    # Run evaluation
    total_pairs = num_images * pairs_per_image

    for cond_name, condition in conditions.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING: {cond_name.upper().replace('_', ' ')}")
        print(f"{'='*60}")

        query_idx = 0

        for img_idx in range(num_images):
            # Load sample
            sample = loader.get_sample(img_idx)
            rgb_path = sample["rgb_path"]
            depth_map = sample["depth_map"]

            # Generate point pairs
            pairs = generate_object_pairs(depth_map, num_pairs=pairs_per_image)

            for pair in pairs:
                point1 = pair["point1"]
                point2 = pair["point2"]
                query_idx += 1

                # Compute ground truth
                gt_distance = loader.compute_distance_3d(point1, point2, depth_map)

                # Create marked image
                marked_dir = output_dir / "marked_images"
                marked_dir.mkdir(exist_ok=True)
                marked_path = marked_dir / f"{cond_name}_img{img_idx:04d}_pair{query_idx}.jpg"
                create_marked_image(rgb_path, point1, point2, marked_path)

                # Create prompt
                prompt = f"""What is the 3D Euclidean distance in meters between
            point A (red cross at pixel {point1}) and
            point B (blue cross at pixel {point2})?

Please provide your answer as a single numerical value in meters."""

                print(f"\n  [{query_idx}/{total_pairs}] Image {img_idx}, Pair {query_idx % pairs_per_image}")
                print(f"  Ground Truth: {gt_distance:.2f}m")

                # Query condition
                result = condition.query_distance(
                    str(marked_path), point1, point2, prompt
                )

                # Extract predicted distance
                predicted_distance = extract_distance_from_response(result["response"])

                if predicted_distance is not None:
                    error = abs(predicted_distance - gt_distance)
                    print(f"  Predicted: {predicted_distance:.2f}m | Error: {error:.2f}m")
                else:
                    print(f"  Predicted: FAILED - Could not parse response")

                # Store result
                all_results[cond_name].append({
                    "image_idx": img_idx,
                    "query_idx": query_idx,
                    "point1": point1,
                    "point2": point2,
                    "gt_distance": float(gt_distance),
                    "predicted_distance": predicted_distance,
                    "response": result["response"],
                    "augmented_prompt": result["augmented_prompt"],
                    **{k: v for k, v in result.items() if k not in ["response", "augmented_prompt"]}
                })

        print(f"\n✓ {cond_name.upper().replace('_', ' ')} COMPLETE")

    # Save results
    results_path = output_dir / "ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # Generate visualization
    plot_ablation_results(all_results, output_dir / "ablation_comparison.png")

    # Print summary
    print_ablation_summary(all_results)


def plot_ablation_results(results, output_path):
    """Create visualization comparing ablation conditions."""

    # Compute statistics for each condition
    stats = []
    for cond_name, cond_results in results.items():
        successful = [r for r in cond_results if r["predicted_distance"] is not None]

        if successful:
            errors = [abs(r["predicted_distance"] - r["gt_distance"]) for r in successful]
            mae = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)
            success_rate = len(successful) / len(cond_results) * 100
        else:
            mae = np.nan
            median_error = np.nan
            std_error = np.nan
            success_rate = 0

        stats.append({
            "condition": cond_name.replace("_", " ").title(),
            "mae": mae,
            "median_error": median_error,
            "std_error": std_error,
            "success_rate": success_rate
        })

    df = pd.DataFrame(stats)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Color scheme
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]  # Red, Blue, Green, Purple

    # 1. MAE comparison
    ax = axes[0, 0]
    bars = ax.barh(df["condition"], df["mae"], color=colors)
    ax.set_xlabel("MAE (meters)", fontsize=11)
    ax.set_title("Mean Absolute Error by Condition", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (_, row) in enumerate(df.iterrows()):
        if not np.isnan(row["mae"]):
            ax.text(row["mae"] + 0.05, i, f'{row["mae"]:.2f}m',
                   va='center', fontsize=9, fontweight='bold')

    # 2. Success rate
    ax = axes[0, 1]
    bars = ax.barh(df["condition"], df["success_rate"], color=colors)
    ax.set_xlabel("Success Rate (%)", fontsize=11)
    ax.set_title("Prediction Success Rate", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 105)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["success_rate"] + 2, i, f'{row["success_rate"]:.0f}%',
               va='center', fontsize=9, fontweight='bold')

    # 3. Median error
    ax = axes[1, 0]
    bars = ax.barh(df["condition"], df["median_error"], color=colors)
    ax.set_xlabel("Median Error (meters)", fontsize=11)
    ax.set_title("Median Absolute Error", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for i, (_, row) in enumerate(df.iterrows()):
        if not np.isnan(row["median_error"]):
            ax.text(row["median_error"] + 0.05, i, f'{row["median_error"]:.2f}m',
                   va='center', fontsize=9, fontweight='bold')

    # 4. Error distribution boxplot
    ax = axes[1, 1]

    error_data = []
    labels = []
    for cond_name, cond_results in results.items():
        successful = [r for r in cond_results if r["predicted_distance"] is not None]
        if successful:
            errors = [abs(r["predicted_distance"] - r["gt_distance"]) for r in successful]
            error_data.append(errors)
            labels.append(cond_name.replace("_", " ").title())

    if error_data:
        bp = ax.boxplot(error_data, tick_labels=labels, vert=False, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors[:len(error_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xlabel("Absolute Error (meters)", fontsize=11)
        ax.set_title("Error Distribution", fontsize=13, fontweight="bold")
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to: {output_path}")


def print_ablation_summary(results):
    """Print summary table of ablation results."""

    stats = []
    for cond_name, cond_results in results.items():
        successful = [r for r in cond_results if r["predicted_distance"] is not None]

        if successful:
            errors = [abs(r["predicted_distance"] - r["gt_distance"]) for r in successful]
            mae = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)
            success_rate = len(successful) / len(cond_results) * 100
        else:
            mae = np.nan
            median_error = np.nan
            std_error = np.nan
            success_rate = 0

        stats.append({
            "condition": cond_name.replace("_", " ").title(),
            "mae": mae,
            "median_error": median_error,
            "std_error": std_error,
            "success_rate": success_rate
        })

    df = pd.DataFrame(stats).sort_values("mae")

    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for Spatial Anchor Calibration components"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to ARKit data directory")
    parser.add_argument("--num_images", type=int, default=10,
                       help="Number of images to test")
    parser.add_argument("--pairs_per_image", type=int, default=3,
                       help="Number of point pairs per image")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation_study",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device for depth estimation")

    args = parser.parse_args()
    load_dotenv()

    run_ablation_study(
        args.data_dir,
        args.num_images,
        args.pairs_per_image,
        args.output_dir,
        args.device
    )


if __name__ == "__main__":
    main()
