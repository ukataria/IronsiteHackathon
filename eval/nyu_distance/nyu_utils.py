"""
NYU Depth V2 utilities for 3D distance evaluation.

Implements:
- Pixel to 3D point conversion using camera intrinsics
- Ground truth distance computation between objects
- Object pair selection for distance queries
"""

import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
import json


class NYUDepthLoader:
    """Load and process NYU Depth V2 data."""

    def __init__(self, data_dir: str):
        """
        Initialize NYU dataset loader.

        Args:
            data_dir: Path to extracted NYU data (with rgb/, depth/, intrinsics.npy)
        """
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"

        # Load camera intrinsics
        intrinsics_path = self.data_dir / "intrinsics.npy"
        intrinsics = np.load(intrinsics_path, allow_pickle=True).item()

        self.fx = intrinsics["fx"]
        self.fy = intrinsics["fy"]
        self.cx = intrinsics["cx"]
        self.cy = intrinsics["cy"]
        self.width = intrinsics["width"]
        self.height = intrinsics["height"]

    def pixel_to_3d(self, u: int, v: int, depth: float) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to 3D point in camera space.

        Uses pinhole camera model with intrinsics.

        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            depth: Depth value at (u, v) in meters

        Returns:
            (X, Y, Z) coordinates in meters
        """
        # Pinhole camera projection equations
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth

        return (X, Y, Z)

    def compute_distance_3d(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        depth_map: np.ndarray
    ) -> float:
        """
        Compute Euclidean distance between two points in 3D space.

        Args:
            point1: (u1, v1) pixel coordinates of first point
            point2: (u2, v2) pixel coordinates of second point
            depth_map: Depth map in meters (H, W)

        Returns:
            Distance in meters
        """
        u1, v1 = point1
        u2, v2 = point2

        # Get depth values
        depth1 = depth_map[v1, u1]
        depth2 = depth_map[v2, u2]

        # Convert to 3D
        X1, Y1, Z1 = self.pixel_to_3d(u1, v1, depth1)
        X2, Y2, Z2 = self.pixel_to_3d(u2, v2, depth2)

        # Euclidean distance
        distance = np.sqrt(
            (X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2
        )

        return distance

    def get_sample(self, idx: int) -> Dict:
        """
        Load a single NYU sample with RGB, depth, and intrinsics.

        Args:
            idx: Sample index

        Returns:
            Dict with rgb_path, depth_map, intrinsics
        """
        rgb_path = self.rgb_dir / f"{idx:04d}.jpg"
        depth_path = self.depth_dir / f"{idx:04d}.npy"

        if not rgb_path.exists() or not depth_path.exists():
            raise ValueError(f"Sample {idx} not found")

        depth_map = np.load(depth_path)

        return {
            "idx": idx,
            "rgb_path": str(rgb_path),
            "depth_map": depth_map,
            "intrinsics": {
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
                "width": self.width,
                "height": self.height
            }
        }


def generate_object_pairs(
    depth_map: np.ndarray,
    num_pairs: int = 5,
    min_distance: float = 0.5,
    max_distance: float = 3.0,
    grid_size: int = 8
) -> List[Dict]:
    """
    Generate pairs of points for distance queries.

    Strategy: Sample pairs of points that are likely to be on different objects
    (based on depth discontinuities).

    Args:
        depth_map: Depth map (H, W) in meters
        num_pairs: Number of pairs to generate
        min_distance: Minimum 3D distance (meters) for valid pairs
        max_distance: Maximum 3D distance (meters) for valid pairs
        grid_size: Divide image into grid for sampling

    Returns:
        List of dicts with point1, point2, description
    """
    H, W = depth_map.shape

    # Create grid cells
    cell_h = H // grid_size
    cell_w = W // grid_size

    # Sample points from grid cells
    sample_points = []

    for i in range(grid_size):
        for j in range(grid_size):
            # Center of grid cell
            cy = i * cell_h + cell_h // 2
            cx = j * cell_w + cell_w // 2

            # Ensure within bounds
            cy = min(cy, H - 1)
            cx = min(cx, W - 1)

            depth = depth_map[cy, cx]

            # Skip invalid depths
            if depth <= 0 or np.isnan(depth) or np.isinf(depth):
                continue

            sample_points.append({
                "pixel": (cx, cy),
                "depth": depth,
                "grid_pos": (i, j)
            })

    # Generate pairs
    pairs = []
    np.random.seed(42)

    for _ in range(num_pairs * 10):  # Oversample and filter
        if len(pairs) >= num_pairs:
            break

        # Randomly select two points
        if len(sample_points) < 2:
            break

        idx1, idx2 = np.random.choice(len(sample_points), size=2, replace=False)
        p1 = sample_points[idx1]
        p2 = sample_points[idx2]

        # Compute 3D distance (simplified - using depth difference as heuristic)
        depth_diff = abs(p1["depth"] - p2["depth"])
        pixel_dist = np.sqrt(
            (p1["pixel"][0] - p2["pixel"][0])**2 +
            (p1["pixel"][1] - p2["pixel"][1])**2
        )

        # Heuristic: good pairs have some pixel separation and depth variation
        if pixel_dist < 50:  # Too close in image
            continue

        if depth_diff < 0.2:  # Too similar depth (likely same surface)
            continue

        # Create pair
        pair = {
            "point1": p1["pixel"],
            "point2": p2["pixel"],
            "depth1": p1["depth"],
            "depth2": p2["depth"],
            "pixel_distance": pixel_dist,
            "description": f"Distance between two objects in the scene"
        }

        pairs.append(pair)

    return pairs[:num_pairs]
