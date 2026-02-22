#!/usr/bin/env python3
"""
Create synthetic test data for VLM depth evaluation.

This generates simple test images with known depth information
so we can verify the evaluation pipeline works.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path


def create_synthetic_scene(scene_id: int, num_frames: int = 5):
    """
    Create synthetic RGB images with ground truth depth maps.

    Args:
        scene_id: Scene identifier
        num_frames: Number of frames to generate
    """
    output_dir = Path(f"data/raw/synthetic/scene_{scene_id:03d}")
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    width, height = 640, 480

    for frame_id in range(num_frames):
        # Create RGB image with simple geometric shapes
        img = Image.new('RGB', (width, height), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)

        # Draw some shapes at different "depths"
        # Closer objects (smaller depth values) are larger and darker
        shapes = [
            # (x, y, size, depth_meters, color)
            (160, 120, 80, 1.5, (100, 100, 200)),  # Close blue square
            (480, 120, 60, 2.5, (200, 100, 100)),  # Mid-distance red square
            (320, 300, 40, 4.0, (100, 200, 100)),  # Far green square
        ]

        # Create depth map (H, W) in meters
        # Start with far wall at 5 meters
        depth_map = np.ones((height, width), dtype=np.float32) * 5.0

        for x, y, size, depth, color in shapes:
            # Draw shape on RGB
            draw.rectangle(
                [x - size//2, y - size//2, x + size//2, y + size//2],
                fill=color,
                outline=(0, 0, 0),
                width=2
            )

            # Set depth for this object
            y1, y2 = max(0, y - size//2), min(height, y + size//2)
            x1, x2 = max(0, x - size//2), min(width, x + size//2)
            depth_map[y1:y2, x1:x2] = depth

            # Skip adding text labels - they confuse VLMs that can read text
            # (VLMs will read the label instead of estimating depth)

        # Add some noise to depth
        noise = np.random.normal(0, 0.05, depth_map.shape).astype(np.float32)
        depth_map = np.maximum(0.1, depth_map + noise)

        # Save RGB image
        rgb_path = rgb_dir / f"{frame_id:04d}.jpg"
        img.save(rgb_path, quality=95)

        # Save depth as .npy
        depth_npy_path = depth_dir / f"{frame_id:04d}.npy"
        np.save(depth_npy_path, depth_map)

        # Also save depth visualization (for debugging)
        depth_vis = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        depth_vis_img = Image.fromarray(depth_vis, mode='L')
        depth_vis_path = depth_dir / f"{frame_id:04d}_vis.png"
        depth_vis_img.save(depth_vis_path)

        print(f"Created frame {frame_id}: {rgb_path}, {depth_npy_path}")

    return output_dir


def create_frames_csv(num_scenes: int = 2, frames_per_scene: int = 5):
    """Create frames.csv index for synthetic data."""
    import pandas as pd

    frames = []
    for scene_id in range(num_scenes):
        scene_dir = Path(f"data/raw/synthetic/scene_{scene_id:03d}")

        for frame_id in range(frames_per_scene):
            frames.append({
                'frame_id': f"scene{scene_id:03d}_frame{frame_id:04d}",
                'rgb_path': str(scene_dir / "rgb" / f"{frame_id:04d}.jpg"),
                'depth_gt_path': str(scene_dir / "depth" / f"{frame_id:04d}.npy"),
            })

    df = pd.DataFrame(frames)
    csv_path = "data/processed/frames.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCreated frames index: {csv_path}")
    print(f"Total frames: {len(df)}")

    return csv_path


def main():
    print("Creating synthetic test data...")
    print("=" * 60)

    # Create 2 scenes with 5 frames each
    num_scenes = 2
    frames_per_scene = 5

    for scene_id in range(num_scenes):
        print(f"\nCreating scene {scene_id}...")
        create_synthetic_scene(scene_id, frames_per_scene)

    # Create frames CSV
    print("\n" + "=" * 60)
    csv_path = create_frames_csv(num_scenes, frames_per_scene)

    print("\n" + "=" * 60)
    print("Synthetic data creation complete!")
    print(f"Created {num_scenes} scenes with {frames_per_scene} frames each")
    print(f"Frames index: {csv_path}")
    print("\nYou can now run the evaluation:")
    print("  python eval/runners/eval_vlm_points.py \\")
    print("    --frames_csv data/processed/frames.csv \\")
    print("    --model_type openai \\")
    print("    --points_per_image 8 \\")
    print("    --max_images 5")


if __name__ == "__main__":
    main()
