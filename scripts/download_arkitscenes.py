#!/usr/bin/env python3
"""
Download ARKitScenes sample data for depth evaluation.

ARKitScenes provides high-quality RGB-D data from indoor scenes.
This script downloads a small subset for testing.
"""

import os
import json
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

# ARKitScenes dataset URLs
# Using the validation set which is publicly available
SAMPLE_SCENES = [
    "42445173",  # Small indoor scene
    "42445677",  # Another validation scene
]

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"

def download_file(url: str, dest_path: str):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

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


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def main():
    data_dir = Path("data/raw/ARKitScenes")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ARKitScenes Sample Download")
    print("="*60)
    print(f"\nNote: ARKitScenes requires accepting Apple's terms of use.")
    print(f"Visit: https://github.com/apple/ARKitScenes")
    print()

    # Try to download validation scenes directly
    # Note: The actual download URLs may require authentication
    # For this demo, we'll provide instructions for manual download

    print("Manual Download Instructions:")
    print("-" * 60)
    print("1. Visit: https://github.com/apple/ARKitScenes")
    print("2. Follow the download instructions in their README")
    print("3. Download the 'Validation' split (smaller, publicly available)")
    print("4. Extract to: data/raw/ARKitScenes/")
    print()
    print("Expected structure:")
    print("  data/raw/ARKitScenes/")
    print("    Training/")
    print("      <scene_id>/")
    print("        <scene_id>_frames/")
    print("          lowres_wide/")
    print("          lowres_depth/")
    print("          lowres_wide_intrinsics/")
    print()
    print("Alternatively, for a quick test, you can use any RGB + depth images.")
    print("Just create the frames.csv manually pointing to your images.")
    print("-" * 60)


if __name__ == "__main__":
    main()
