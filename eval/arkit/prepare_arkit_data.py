#!/usr/bin/env python3
"""
Prepare ARKit LiDAR data for benchmarking.

This script helps you organize your ARKit captures into the expected format:
- data/arkit/rgb/        # RGB images
- data/arkit/depth/      # Depth maps (.npy in meters)
- data/arkit/intrinsics.npy  # Camera intrinsics

For iPhone/iPad captures, you'll need to extract the data from your ARKit recording.
"""

import argparse
import json
import numpy as np
from pathlib import Path


def create_default_intrinsics(width=1920, height=1440):
    """
    Create default iPhone/iPad Pro intrinsics.

    These are approximate values for iPhone 12 Pro and later.
    Adjust based on your device.
    """
    # Typical values for iPhone 12 Pro / 13 Pro / 14 Pro
    fx = 1420.0  # Focal length in pixels
    fy = 1420.0
    cx = width / 2.0   # Principal point (image center)
    cy = height / 2.0

    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height
    }

    return intrinsics


def main():
    parser = argparse.ArgumentParser(description="Prepare ARKit data for benchmarking")
    parser.add_argument("--source_dir", type=str, help="Directory with your ARKit captures")
    parser.add_argument("--output_dir", type=str, default="data/arkit", help="Output directory")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1440, help="Image height")
    parser.add_argument("--create_structure", action="store_true", help="Just create directory structure")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Create directory structure
    (output_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)

    print(f"Created directory structure at: {output_dir}")
    print(f"  - {output_dir / 'rgb'}")
    print(f"  - {output_dir / 'depth'}")

    # Create default intrinsics
    intrinsics = create_default_intrinsics(args.width, args.height)
    intrinsics_path = output_dir / "intrinsics.npy"
    np.save(intrinsics_path, intrinsics)

    print(f"\nCreated intrinsics file: {intrinsics_path}")
    print("Default camera parameters:")
    for key, value in intrinsics.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. Place your RGB images in: {output_dir / 'rgb'}")
    print(f"   Format: 0000.jpg, 0001.jpg, ...")
    print()
    print(f"2. Place your depth maps in: {output_dir / 'depth'}")
    print(f"   Format: 0000.npy, 0001.npy, ... (numpy arrays in meters)")
    print()
    print("3. If using different camera intrinsics, update:")
    print(f"   {intrinsics_path}")
    print()
    print("4. Run the benchmark:")
    print(f"   python eval/arkit/benchmark_arkit_all_models.py \\")
    print(f"     --arkit_data_dir {output_dir} \\")
    print(f"     --num_images 10 \\")
    print(f"     --pairs_per_image 3")
    print("="*60)


if __name__ == "__main__":
    main()
