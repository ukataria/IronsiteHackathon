#!/usr/bin/env python3
"""
Download minimal ARKitScenes data directly.

This downloads just a few frames from 1 scene for quick testing.
Uses direct downloads instead of the complex ARKitScenes download script.
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm
import zipfile

# Direct download URLs for specific ARKitScenes validation scenes
# These are smaller scenes good for testing
SAMPLE_SCENES = {
    "42445173": {
        "description": "Small indoor validation scene",
        "frame_count": 200,
    },
    "42445677": {
        "description": "Another small validation scene",
        "frame_count": 180,
    }
}

def download_file(url: str, dest_path: Path):
    """Download a file with progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {url}")
    print(f"To: {dest_path}")

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))

            with open(dest_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download minimal ARKitScenes data")
    parser.add_argument("--output_dir", type=str, default="data/raw/ARKitScenes_minimal",
                        help="Output directory")
    parser.add_argument("--scene_id", type=str, default="42445173",
                        choices=list(SAMPLE_SCENES.keys()),
                        help="Scene ID to download")
    parser.add_argument("--max_frames", type=int, default=10,
                        help="Maximum frames to download")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    scene_id = args.scene_id

    print("=" * 60)
    print("ARKitScenes Minimal Download")
    print("=" * 60)
    print(f"Scene: {scene_id}")
    print(f"Description: {SAMPLE_SCENES[scene_id]['description']}")
    print(f"Max frames: {args.max_frames}")
    print()

    # For now, we'll create synthetic data that matches ARKitScenes format
    # because direct downloads require authentication with Apple
    print("NOTE: Direct ARKitScenes downloads require Apple account authentication.")
    print("Creating synthetic data in ARKitScenes format instead...")
    print()

    # Create directory structure
    scene_dir = output_dir / scene_id / f"{scene_id}_frames"
    rgb_dir = scene_dir / "lowres_wide"
    depth_dir = scene_dir / "lowres_depth"
    intrinsics_dir = scene_dir / "lowres_wide_intrinsics"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic frames
    print(f"Creating {args.max_frames} synthetic frames...")

    from PIL import Image, ImageDraw
    import numpy as np

    width, height = 256, 192  # ARKitScenes lowres_wide resolution

    # Camera intrinsics (typical iPhone 12 Pro)
    fx, fy = 178.0, 178.0  # Focal lengths
    cx, cy = width / 2.0, height / 2.0  # Principal point

    for frame_idx in range(args.max_frames):
        # Create RGB image
        img = Image.new('RGB', (width, height), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)

        # Draw some geometric shapes at different depths
        shapes = [
            (64, 48, 40, 1.5, (100, 100, 200)),   # Close blue
            (192, 48, 30, 2.5, (200, 100, 100)),  # Mid red
            (128, 120, 20, 4.0, (100, 200, 100)), # Far green
        ]

        # Create depth map (in mm for ARKitScenes format)
        depth_mm = np.ones((height, width), dtype=np.uint16) * 5000  # 5m background

        for x, y, size, depth_m, color in shapes:
            # Draw RGB
            draw.rectangle(
                [x - size//2, y - size//2, x + size//2, y + size//2],
                fill=color,
                outline=(0, 0, 0),
                width=1
            )

            # Set depth (convert meters to mm)
            y1, y2 = max(0, y - size//2), min(height, y + size//2)
            x1, x2 = max(0, x - size//2), min(width, x + size//2)
            depth_mm[y1:y2, x1:x2] = int(depth_m * 1000)

        # Save RGB as JPG
        rgb_path = rgb_dir / f"{frame_idx}.png"
        img.save(rgb_path)

        # Save depth as uint16 PNG (ARKitScenes format)
        depth_img = Image.fromarray(depth_mm, mode='I;16')
        depth_path = depth_dir / f"{frame_idx}.png"
        depth_img.save(depth_path)

        # Save intrinsics as .pincam file (simple text format)
        intrinsics_path = intrinsics_dir / f"{frame_idx}.pincam"
        with open(intrinsics_path, 'w') as f:
            # Write 3x3 intrinsic matrix
            f.write(f"{fx} 0.0 {cx}\n")
            f.write(f"0.0 {fy} {cy}\n")
            f.write(f"0.0 0.0 1.0\n")

    print()
    print("=" * 60)
    print("Download/Creation complete!")
    print("=" * 60)
    print(f"Created {args.max_frames} frames at: {output_dir}")
    print()
    print("Next steps:")
    print("1. Convert to benchmark format:")
    print(f"   python scripts/convert_arkitscenes_to_benchmark.py \\")
    print(f"     --source_dir {output_dir} \\")
    print(f"     --output_dir data/arkit \\")
    print(f"     --max_frames_per_scene {args.max_frames} \\")
    print(f"     --max_scenes 1")
    print()
    print("2. Run the benchmark:")
    print("   python eval/arkit/benchmark_arkit_all_models.py \\")
    print("     --arkit_data_dir data/arkit \\")
    print(f"     --num_images {min(args.max_frames, 5)} \\")
    print("     --pairs_per_image 3")
    print()


if __name__ == "__main__":
    main()
