"""COLMAP reconstruction wrapper for 3D point cloud generation."""

import shutil
import subprocess
from pathlib import Path

import numpy as np

from src.utils import get_logger, load_config, save_json

log = get_logger(__name__)


def run_colmap_feature_extraction(
    image_dir: Path,
    database_path: Path,
    camera_model: str = "SIMPLE_RADIAL",
    max_image_size: int = 1600,
) -> None:
    """Extract features using COLMAP.

    Args:
        image_dir: Directory containing input images
        database_path: Path to COLMAP database file
        camera_model: Camera model type
        max_image_size: Maximum image dimension
    """
    log.info("Running COLMAP feature extraction")

    cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.max_image_size",
        str(max_image_size),
    ]

    subprocess.run(cmd, check=True)
    log.info("Feature extraction complete")


def run_colmap_matcher(
    database_path: Path,
    matcher_type: str = "exhaustive",
) -> None:
    """Match features using COLMAP.

    Args:
        database_path: Path to COLMAP database file
        matcher_type: Matcher type ("exhaustive", "sequential", "spatial")
    """
    log.info(f"Running COLMAP {matcher_type} matcher")

    cmd = [
        "colmap",
        f"{matcher_type}_matcher",
        "--database_path",
        str(database_path),
    ]

    subprocess.run(cmd, check=True)
    log.info("Feature matching complete")


def run_colmap_mapper(
    database_path: Path,
    image_dir: Path,
    output_dir: Path,
    min_num_matches: int = 15,
) -> None:
    """Run COLMAP sparse reconstruction.

    Args:
        database_path: Path to COLMAP database file
        image_dir: Directory containing input images
        output_dir: Output directory for sparse reconstruction
        min_num_matches: Minimum number of matches for image pair
    """
    log.info("Running COLMAP mapper")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(output_dir),
        "--Mapper.min_num_matches",
        str(min_num_matches),
    ]

    subprocess.run(cmd, check=True)
    log.info("Sparse reconstruction complete")


def export_point_cloud(
    sparse_dir: Path,
    output_path: Path,
    output_format: str = "PLY",
) -> Path:
    """Export COLMAP reconstruction to point cloud.

    Args:
        sparse_dir: Directory containing sparse reconstruction (model)
        output_path: Output point cloud file path
        output_format: Output format ("PLY", "BIN")

    Returns:
        Path to exported point cloud
    """
    log.info(f"Exporting point cloud to {output_path}")

    # Find the model directory (usually sparse/0/)
    model_dirs = list(sparse_dir.glob("*/"))
    if not model_dirs:
        raise RuntimeError(f"No model found in {sparse_dir}")

    model_dir = model_dirs[0]  # Use first model
    log.info(f"Using model: {model_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap",
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(output_path),
        "--output_type",
        output_format,
    ]

    subprocess.run(cmd, check=True)
    log.info(f"Point cloud exported: {output_path}")

    return output_path


def reconstruct_segment(
    frame_dir: Path | str,
    output_dir: Path | str,
    config: dict | None = None,
) -> dict[str, Path]:
    """Run full COLMAP reconstruction pipeline for a segment.

    Args:
        frame_dir: Directory containing frames
        output_dir: Output directory for reconstruction
        config: Configuration dict

    Returns:
        Dictionary with paths to reconstruction outputs
    """
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = load_config()

    colmap_config = config.get("colmap", {})

    # Setup paths
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    pointcloud_path = output_dir / "pointcloud.ply"

    log.info(f"Reconstructing segment: {frame_dir.name}")
    log.info(f"Output directory: {output_dir}")

    # Check if frames exist
    frames = sorted(frame_dir.glob("*.png"))
    if len(frames) == 0:
        raise ValueError(f"No frames found in {frame_dir}")

    log.info(f"Found {len(frames)} frames")

    # Step 1: Feature extraction
    run_colmap_feature_extraction(
        image_dir=frame_dir,
        database_path=database_path,
        camera_model=colmap_config.get("camera_model", "SIMPLE_RADIAL"),
        max_image_size=colmap_config.get("max_image_size", 1600),
    )

    # Step 2: Feature matching
    run_colmap_matcher(
        database_path=database_path,
        matcher_type=colmap_config.get("matcher", "exhaustive"),
    )

    # Step 3: Sparse reconstruction
    run_colmap_mapper(
        database_path=database_path,
        image_dir=frame_dir,
        output_dir=sparse_dir,
        min_num_matches=colmap_config.get("min_num_matches", 15),
    )

    # Step 4: Export point cloud
    export_point_cloud(
        sparse_dir=sparse_dir,
        output_path=pointcloud_path,
    )

    # Save reconstruction metadata
    metadata = {
        "frame_count": len(frames),
        "database": str(database_path),
        "sparse_model": str(sparse_dir),
        "pointcloud": str(pointcloud_path),
    }

    save_json(metadata, output_dir / "reconstruction_metadata.json")

    log.info("Reconstruction complete")

    return {
        "database": database_path,
        "sparse": sparse_dir,
        "pointcloud": pointcloud_path,
        "metadata": output_dir / "reconstruction_metadata.json",
    }


def check_colmap_installed() -> bool:
    """Check if COLMAP is installed and available."""
    try:
        result = subprocess.run(["colmap", "--version"], capture_output=True, text=True)
        log.info(f"COLMAP version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        log.error("COLMAP not found. Please install COLMAP: https://colmap.github.io/install.html")
        return False
