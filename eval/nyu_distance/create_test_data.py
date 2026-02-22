#!/usr/bin/env python3
"""
Create small test dataset for NYU distance eval (without downloading 2.8GB).

Uses synthetic data to test the pipeline.
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

def create_test_samples(output_dir: str, num_samples: int = 5):
    """Create synthetic test samples mimicking NYU format."""
    output_dir = Path(output_dir)
    (output_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)

    # NYU Kinect intrinsics
    intrinsics = {
        "fx": 518.857901,
        "fy": 519.469611,
        "cx": 325.582244,
        "cy": 253.736347,
        "width": 640,
        "height": 480
    }

    np.save(output_dir / "intrinsics.npy", intrinsics)

    for i in range(num_samples):
        # Create synthetic RGB image (simple indoor scene simulation)
        img = Image.new('RGB', (640, 480), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Draw some objects at different depths
        # Close object (blue)
        draw.rectangle([100, 150, 200, 350], fill=(100, 150, 200))

        # Medium object (red)
        draw.rectangle([350, 100, 450, 250], fill=(200, 100, 100))

        # Far object (green)
        draw.rectangle([250, 300, 350, 400], fill=(100, 200, 100))

        # Add some variation
        np.random.seed(i)
        noise = np.random.randint(-20, 20, (480, 640, 3))
        img_array = np.array(img).astype(np.int16) + noise
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        # Save RGB
        img.save(output_dir / "rgb" / f"{i:04d}.jpg")

        # Create synthetic depth map (meters)
        depth = np.ones((480, 640), dtype=np.float32) * 3.0  # Background at 3m

        # Close object at 1.5m
        depth[150:350, 100:200] = 1.5

        # Medium object at 2.5m
        depth[100:250, 350:450] = 2.5

        # Far object at 4.0m
        depth[300:400, 250:350] = 4.0

        # Add realistic depth noise
        noise = np.random.normal(0, 0.05, depth.shape)
        depth = depth + noise
        depth = np.clip(depth, 0.5, 10.0)

        # Save depth
        np.save(output_dir / "depth" / f"{i:04d}.npy", depth)

        # Save depth visualization
        depth_vis = (depth / depth.max() * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(output_dir / "depth" / f"{i:04d}_vis.png")

    print(f"Created {num_samples} test samples in {output_dir}")

if __name__ == "__main__":
    create_test_samples("data/nyu_depth_v2/test_samples", num_samples=5)
