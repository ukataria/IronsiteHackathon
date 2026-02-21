#!/usr/bin/env python3
"""
Download NYU Depth V2 dataset.

Official dataset: http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

We'll use the preprocessed version from:
https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""

import os
import urllib.request
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def download_file(url, output_path):
    """Download file with progress bar."""
    print(f"Downloading from {url}")
    print(f"Saving to {output_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rProgress: {percent:.1f}%", end="")

    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    print("\nDownload complete!")

def extract_nyu_samples(h5_file, output_dir, max_samples=100):
    """
    Extract RGB images, depth maps, and labels from NYU .mat file.

    Args:
        h5_file: Path to nyu_depth_v2_labeled.mat
        output_dir: Where to save extracted images
        max_samples: Number of samples to extract
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "rgb").mkdir(exist_ok=True)
    (output_dir / "depth").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)

    print(f"Extracting samples from {h5_file}...")

    with h5py.File(h5_file, 'r') as f:
        # Dataset structure:
        # - images: (3, 640, 480, N) - RGB images
        # - depths: (640, 480, N) - depth in meters
        # - labels: (640, 480, N) - semantic labels

        images = f['images']
        depths = f['depths']
        labels = f['labels'] if 'labels' in f else None

        num_samples = min(images.shape[3], max_samples)

        print(f"Dataset contains {images.shape[3]} images")
        print(f"Extracting {num_samples} samples...")

        for i in tqdm(range(num_samples), desc="Extracting"):
            # Extract RGB (convert from MATLAB format: C,H,W -> H,W,C)
            rgb = images[:, :, :, i].transpose(1, 2, 0)
            rgb = (rgb * 255).astype(np.uint8)

            # Extract depth
            depth = depths[:, :, i]

            # Save RGB as JPEG
            rgb_path = output_dir / "rgb" / f"{i:04d}.jpg"
            Image.fromarray(rgb).save(rgb_path)

            # Save depth as NPY (preserves metric values)
            depth_path = output_dir / "depth" / f"{i:04d}.npy"
            np.save(depth_path, depth)

            # Save depth visualization
            depth_vis = (depth / depth.max() * 255).astype(np.uint8)
            depth_vis_path = output_dir / "depth" / f"{i:04d}_vis.png"
            Image.fromarray(depth_vis).save(depth_vis_path)

            # Save labels if available
            if labels is not None:
                label = labels[:, :, i]
                label_path = output_dir / "labels" / f"{i:04d}.npy"
                np.save(label_path, label)

        print(f"Extracted {num_samples} samples to {output_dir}")

        # Save camera intrinsics (NYU Kinect parameters)
        intrinsics = {
            "fx": 518.857901,  # Focal length x
            "fy": 519.469611,  # Focal length y
            "cx": 325.582244,  # Principal point x
            "cy": 253.736347,  # Principal point y
            "width": 640,
            "height": 480
        }

        intrinsics_path = output_dir / "intrinsics.npy"
        np.save(intrinsics_path, intrinsics)
        print(f"Saved camera intrinsics to {intrinsics_path}")

def main():
    """Download and extract NYU Depth V2 dataset."""

    # Dataset URL (labeled subset)
    # Note: This is a large file (~2.8 GB)
    dataset_url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

    data_dir = Path(__file__).parent
    h5_file = data_dir / "nyu_depth_v2_labeled.mat"

    # Download if not exists
    if not h5_file.exists():
        print("="*60)
        print("DOWNLOADING NYU DEPTH V2 DATASET")
        print("="*60)
        print(f"Size: ~2.8 GB")
        print(f"This may take several minutes...")
        print()

        try:
            download_file(dataset_url, h5_file)
        except Exception as e:
            print(f"\nError downloading: {e}")
            print("\nAlternative: Download manually from:")
            print("http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html")
            print(f"Save to: {h5_file}")
            return
    else:
        print(f"Dataset already downloaded: {h5_file}")

    # Extract samples
    print("\n" + "="*60)
    print("EXTRACTING SAMPLES")
    print("="*60)

    output_dir = data_dir / "extracted"
    extract_nyu_samples(h5_file, output_dir, max_samples=100)

    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print(f"RGB images: {output_dir / 'rgb'}")
    print(f"Depth maps: {output_dir / 'depth'}")
    print(f"Intrinsics: {output_dir / 'intrinsics.npy'}")

if __name__ == "__main__":
    main()
