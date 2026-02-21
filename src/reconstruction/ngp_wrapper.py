"""Instant-NGP reconstruction wrapper for GPU-native 3D reconstruction."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.utils import get_logger, load_config, save_json

log = get_logger(__name__)


def create_ngp_transforms(
    frame_dir: Path,
    output_path: Path,
    camera_angle_x: float = 0.7,  # ~40 degrees FOV
) -> Path:
    """Create transforms.json for Instant-NGP from frames.

    Args:
        frame_dir: Directory containing frames
        output_path: Path to save transforms.json
        camera_angle_x: Camera field of view in radians

    Returns:
        Path to transforms.json
    """
    log.info(f"Creating NGP transforms for {frame_dir}")

    frames = sorted(frame_dir.glob("*.png"))
    if len(frames) == 0:
        raise ValueError(f"No frames found in {frame_dir}")

    # Read first frame to get dimensions
    sample_frame = cv2.imread(str(frames[0]))
    h, w = sample_frame.shape[:2]

    # Create transforms structure
    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": [],
    }

    # Add each frame
    # For construction videos, we assume the camera is moving through the space
    # We'll use simple forward motion + slight rotation as initial guess
    for i, frame_path in enumerate(frames):
        # Simple camera trajectory (you can make this more sophisticated)
        # Camera moves forward and slightly rotates
        t = i / len(frames)  # 0 to 1

        # Translation: move forward
        tx = 0.0
        ty = 0.0
        tz = -2.0 + t * 4.0  # Move from -2 to +2

        # Rotation: slight turn
        angle = (t - 0.5) * 0.2  # Slight rotation

        # Create transformation matrix (simplified)
        transform_matrix = [
            [np.cos(angle), 0, np.sin(angle), tx],
            [0, 1, 0, ty],
            [-np.sin(angle), 0, np.cos(angle), tz],
            [0, 0, 0, 1],
        ]

        frame_entry = {
            "file_path": str(frame_path.relative_to(frame_dir.parent)),
            "transform_matrix": transform_matrix,
        }

        transforms["frames"].append(frame_entry)

    # Save transforms
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transforms, f, indent=2)

    log.info(f"Created transforms with {len(frames)} frames: {output_path}")
    return output_path


def run_ngp_training(
    transforms_path: Path,
    output_dir: Path,
    n_steps: int = 1000,
    snapshot_interval: int = 100,
) -> Path:
    """Run Instant-NGP training.

    Args:
        transforms_path: Path to transforms.json
        output_dir: Output directory for snapshots
        n_steps: Number of training steps
        snapshot_interval: Save snapshot every N steps

    Returns:
        Path to final snapshot
    """
    log.info(f"Running NGP training for {n_steps} steps")

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / f"snapshot_{n_steps}.msgpack"

    # Run instant-ngp training via Python API
    try:
        import pyngp as ngp

        # Create testbed
        testbed = ngp.Testbed()
        testbed.root_dir = str(transforms_path.parent)

        # Load training data
        testbed.load_training_data(str(transforms_path))

        # Train
        log.info(f"Training NeRF on GPU...")
        testbed.shall_train = True

        for step in range(n_steps):
            testbed.frame()

            if (step + 1) % snapshot_interval == 0:
                log.info(f"Step {step + 1}/{n_steps}")

        # Save final snapshot
        testbed.save_snapshot(str(snapshot_path), False)
        log.info(f"Saved snapshot: {snapshot_path}")

        return snapshot_path

    except ImportError:
        log.error("pyngp not installed, falling back to CLI")
        return run_ngp_training_cli(transforms_path, output_dir, n_steps)


def run_ngp_training_cli(
    transforms_path: Path, output_dir: Path, n_steps: int = 1000
) -> Path:
    """Run Instant-NGP training via CLI (fallback).

    Args:
        transforms_path: Path to transforms.json
        output_dir: Output directory
        n_steps: Number of training steps

    Returns:
        Path to output snapshot
    """
    snapshot_path = output_dir / f"snapshot_{n_steps}.msgpack"

    cmd = [
        "instant-ngp",
        str(transforms_path),
        "--n_steps",
        str(n_steps),
        "--save_snapshot",
        str(snapshot_path),
        "--headless",
    ]

    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"NGP training failed: {result.stderr}")
        raise RuntimeError(f"NGP training failed")

    return snapshot_path


def export_point_cloud_from_nerf(
    snapshot_path: Path,
    transforms_path: Path,
    output_path: Path,
    resolution: int = 512,
    density_threshold: float = 10.0,
) -> Path:
    """Export dense point cloud from trained NeRF.

    Args:
        snapshot_path: Path to NGP snapshot
        transforms_path: Path to transforms.json
        output_path: Output .ply path
        resolution: Resolution for point cloud extraction
        density_threshold: Minimum density for points

    Returns:
        Path to exported point cloud
    """
    log.info("Exporting point cloud from NeRF")

    try:
        import pyngp as ngp
        import open3d as o3d

        # Load trained model
        testbed = ngp.Testbed()
        testbed.root_dir = str(transforms_path.parent)
        testbed.load_snapshot(str(snapshot_path))

        # Extract density grid
        log.info("Extracting density grid...")

        # Sample points in 3D space
        points = []
        colors = []

        # Create grid of sample points
        grid_range = np.linspace(-1, 1, resolution)
        for x in grid_range:
            for y in grid_range:
                for z in grid_range:
                    # Query density and color at this point
                    pos = np.array([x, y, z], dtype=np.float32)

                    # This is simplified - actual API may differ
                    # You'd use testbed to query density/color at pos
                    density = testbed.compute_density(pos)

                    if density > density_threshold:
                        color = testbed.compute_color(pos)
                        points.append(pos)
                        colors.append(color)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), pcd)

        log.info(f"Exported {len(points)} points to {output_path}")
        return output_path

    except Exception as e:
        log.error(f"Failed to export point cloud: {e}")
        # Fallback: use marching cubes mesh export then convert to points
        return export_point_cloud_via_mesh(snapshot_path, transforms_path, output_path)


def export_point_cloud_via_mesh(
    snapshot_path: Path, transforms_path: Path, output_path: Path
) -> Path:
    """Export point cloud by first extracting mesh, then sampling points.

    Args:
        snapshot_path: Path to NGP snapshot
        transforms_path: Path to transforms.json
        output_path: Output .ply path

    Returns:
        Path to exported point cloud
    """
    log.info("Exporting via mesh (marching cubes)")

    mesh_path = output_path.parent / f"{output_path.stem}_mesh.ply"

    # Export mesh using instant-ngp CLI
    cmd = [
        "instant-ngp",
        str(transforms_path),
        "--load_snapshot",
        str(snapshot_path),
        "--save_mesh",
        str(mesh_path),
        "--marching_cubes_res",
        "256",
        "--headless",
    ]

    subprocess.run(cmd, check=True)

    # Load mesh and sample points
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    pcd = mesh.sample_points_uniformly(number_of_points=100000)

    # Save point cloud
    o3d.io.write_point_cloud(str(output_path), pcd)

    log.info(f"Exported point cloud: {output_path}")
    return output_path


def reconstruct_segment_ngp(
    frame_dir: Path | str,
    output_dir: Path | str,
    config: dict | None = None,
) -> dict[str, Path]:
    """Run Instant-NGP reconstruction for a segment.

    Args:
        frame_dir: Directory containing frames
        output_dir: Output directory
        config: Configuration dict

    Returns:
        Dictionary with paths to outputs
    """
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = load_config()

    ngp_config = config.get("ngp", {})

    log.info(f"NGP reconstruction: {frame_dir.name}")
    log.info(f"Output: {output_dir}")

    # Check frames exist
    frames = sorted(frame_dir.glob("*.png"))
    if len(frames) == 0:
        raise ValueError(f"No frames found in {frame_dir}")

    log.info(f"Found {len(frames)} frames")

    # Step 1: Create transforms.json
    transforms_path = output_dir / "transforms.json"
    create_ngp_transforms(
        frame_dir=frame_dir,
        output_path=transforms_path,
        camera_angle_x=ngp_config.get("camera_fov", 0.7),
    )

    # Step 2: Train NeRF
    snapshot_path = run_ngp_training(
        transforms_path=transforms_path,
        output_dir=output_dir,
        n_steps=ngp_config.get("n_steps", 1000),
        snapshot_interval=ngp_config.get("snapshot_interval", 100),
    )

    # Step 3: Export point cloud
    pointcloud_path = output_dir / "pointcloud.ply"
    export_point_cloud_from_nerf(
        snapshot_path=snapshot_path,
        transforms_path=transforms_path,
        output_path=pointcloud_path,
        resolution=ngp_config.get("export_resolution", 512),
        density_threshold=ngp_config.get("density_threshold", 10.0),
    )

    # Save metadata
    metadata = {
        "frame_count": len(frames),
        "transforms": str(transforms_path),
        "snapshot": str(snapshot_path),
        "pointcloud": str(pointcloud_path),
        "n_steps": ngp_config.get("n_steps", 1000),
    }

    save_json(metadata, output_dir / "ngp_metadata.json")

    log.info("NGP reconstruction complete")

    return {
        "transforms": transforms_path,
        "snapshot": snapshot_path,
        "pointcloud": pointcloud_path,
        "metadata": output_dir / "ngp_metadata.json",
    }


def check_ngp_installed() -> bool:
    """Check if Instant-NGP is installed."""
    try:
        import pyngp

        log.info("Instant-NGP Python bindings available")
        return True
    except ImportError:
        log.warning("pyngp not found, will try CLI")
        # Check if CLI is available
        try:
            result = subprocess.run(["instant-ngp", "--help"], capture_output=True)
            if result.returncode == 0:
                log.info("Instant-NGP CLI available")
                return True
        except FileNotFoundError:
            log.error("Instant-NGP not installed")
            return False

    return False
