#!/usr/bin/env python3
"""
Convert ARKitScenes downloaded data to benchmark format.

ARKitScenes format:
- Validation/<video_id>/<video_id>_frames/lowres_wide/*.png (RGB)
- Validation/<video_id>/<video_id>_frames/lowres_depth/*.png (uint16 depth in mm)
- Validation/<video_id>/<video_id>_frames/lowres_wide_intrinsics/*.pincam

Benchmark format:
- data/arkit/rgb/*.jpg
- data/arkit/depth/*.npy (float32 depth in meters)
- data/arkit/intrinsics.npy
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def read_pincam(pincam_path):
    """Read ARKit .pincam file and extract intrinsics."""
    with open(pincam_path, 'r') as f:
        lines = f.readlines()

    # Parse intrinsic matrix (first 3x3 matrix)
    # Format is usually:
    # fx 0 cx
    # 0 fy cy
    # 0 0 1
    intrinsic_lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')]

    # Extract the 3x3 matrix
    matrix = []
    for line in intrinsic_lines[:3]:
        row = [float(x) for x in line.split()]
        matrix.append(row)

    fx = matrix[0][0]
    fy = matrix[1][1]
    cx = matrix[0][2]
    cy = matrix[1][2]

    return fx, fy, cx, cy


def convert_scene(scene_dir, output_dir, max_frames=None):
    """Convert a single ARKitScenes scene to benchmark format."""
    scene_id = scene_dir.name
    frames_dir = scene_dir / f"{scene_id}_frames"

    rgb_dir = frames_dir / "lowres_wide"
    depth_dir = frames_dir / "lowres_depth"
    intrinsics_dir = frames_dir / "lowres_wide_intrinsics"

    if not rgb_dir.exists() or not depth_dir.exists():
        print(f"Skipping {scene_id}: missing RGB or depth data")
        return 0

    # Create output directories
    out_rgb = output_dir / "rgb"
    out_depth = output_dir / "depth"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)

    # Get RGB frames
    rgb_frames = sorted(rgb_dir.glob("*.png"))
    if max_frames:
        rgb_frames = rgb_frames[:max_frames]

    print(f"Converting {len(rgb_frames)} frames from {scene_id}...")

    # Read first intrinsics file as reference
    intrinsics_files = sorted(intrinsics_dir.glob("*.pincam"))
    if intrinsics_files:
        fx, fy, cx, cy = read_pincam(intrinsics_files[0])

        # Get image dimensions from first RGB image
        first_img = Image.open(rgb_frames[0])
        width, height = first_img.size

        intrinsics = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": width,
            "height": height
        }

        # Save intrinsics
        np.save(output_dir / "intrinsics.npy", intrinsics)
        print(f"Saved intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Convert frames
    for idx, rgb_path in enumerate(tqdm(rgb_frames, desc=f"Scene {scene_id}")):
        frame_name = rgb_path.stem

        # Copy/convert RGB
        rgb_img = Image.open(rgb_path)
        rgb_out = out_rgb / f"{idx:04d}.jpg"
        rgb_img.save(rgb_out, "JPEG", quality=95)

        # Convert depth from uint16 mm to float32 meters
        depth_path = depth_dir / f"{frame_name}.png"
        if depth_path.exists():
            depth_img = Image.open(depth_path)
            depth_mm = np.array(depth_img, dtype=np.uint16)
            depth_m = depth_mm.astype(np.float32) / 1000.0  # Convert mm to meters

            depth_out = out_depth / f"{idx:04d}.npy"
            np.save(depth_out, depth_m)

    return len(rgb_frames)


def main():
    parser = argparse.ArgumentParser(description="Convert ARKitScenes to benchmark format")
    parser.add_argument("--source_dir", type=str, default="data/raw/ARKitScenes/Validation",
                        help="ARKitScenes validation directory")
    parser.add_argument("--output_dir", type=str, default="data/arkit",
                        help="Output directory for benchmark format")
    parser.add_argument("--max_frames_per_scene", type=int, default=50,
                        help="Maximum frames to extract per scene")
    parser.add_argument("--max_scenes", type=int, default=1,
                        help="Maximum number of scenes to convert")

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        print("Please run download_arkitscenes_data.sh first")
        return

    # Find all scene directories
    scene_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])

    if not scene_dirs:
        print(f"No scenes found in {source_dir}")
        return

    print(f"Found {len(scene_dirs)} scenes")
    print(f"Converting up to {args.max_scenes} scenes...")
    print()

    total_frames = 0
    for scene_dir in scene_dirs[:args.max_scenes]:
        frames = convert_scene(scene_dir, output_dir, args.max_frames_per_scene)
        total_frames += frames

    print()
    print("="*60)
    print("Conversion complete!")
    print("="*60)
    print(f"Total frames converted: {total_frames}")
    print(f"Output directory: {output_dir}")
    print()
    print("Ready to run benchmark:")
    print(f"  python eval/arkit/benchmark_arkit_all_models.py \\")
    print(f"    --arkit_data_dir {output_dir} \\")
    print(f"    --num_images {min(total_frames, 10)} \\")
    print(f"    --pairs_per_image 3")
    print()


if __name__ == "__main__":
    main()
