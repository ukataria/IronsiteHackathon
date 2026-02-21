"""Point cloud alignment using ICP (Iterative Closest Point)."""

from pathlib import Path

import numpy as np
import open3d as o3d

from src.utils import get_logger, load_config, save_json

log = get_logger(__name__)


def load_point_cloud(path: Path | str) -> o3d.geometry.PointCloud:
    """Load point cloud from file.

    Args:
        path: Path to point cloud file (.ply, .pcd, .xyz)

    Returns:
        Open3D point cloud object
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    pcd = o3d.io.read_point_cloud(str(path))
    log.info(f"Loaded point cloud: {len(pcd.points)} points from {path.name}")
    return pcd


def save_point_cloud(pcd: o3d.geometry.PointCloud, path: Path | str) -> None:
    """Save point cloud to file.

    Args:
        pcd: Open3D point cloud object
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    log.info(f"Saved point cloud: {len(pcd.points)} points to {path.name}")


def downsample_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
    """Downsample point cloud using voxel grid.

    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling

    Returns:
        Downsampled point cloud
    """
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    log.info(f"Downsampled: {len(pcd.points)} → {len(downsampled.points)} points (voxel_size={voxel_size})")
    return downsampled


def align_point_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_iterations: int = 100,
    threshold: float = 0.05,
    voxel_size: float = 0.05,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, dict]:
    """Align source point cloud to target using ICP.

    Args:
        source: Source point cloud to be aligned
        target: Target (reference) point cloud
        max_iterations: Maximum ICP iterations
        threshold: Distance threshold for correspondence
        voxel_size: Voxel size for downsampling

    Returns:
        Tuple of (aligned_source, transformation_matrix, registration_result)
    """
    log.info("Running ICP alignment")

    # Downsample for faster ICP
    source_down = downsample_point_cloud(source, voxel_size=voxel_size)
    target_down = downsample_point_cloud(target, voxel_size=voxel_size)

    # Estimate normals for better alignment
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Run point-to-plane ICP
    log.info(f"ICP parameters: max_iter={max_iterations}, threshold={threshold}")

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )

    log.info(f"ICP fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")

    # Apply transformation to full-resolution source
    transformation = result.transformation
    aligned_source = source.transform(transformation)

    registration_info = {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "correspondence_set_size": len(result.correspondence_set),
        "transformation": transformation.tolist(),
    }

    return aligned_source, transformation, registration_info


def align_segment_to_reference(
    segment_cloud_path: Path | str,
    reference_cloud_path: Path | str,
    output_path: Path | str,
    config: dict | None = None,
) -> dict[str, Path | np.ndarray]:
    """Align a segment point cloud to a reference point cloud.

    Args:
        segment_cloud_path: Path to segment point cloud
        reference_cloud_path: Path to reference point cloud
        output_path: Output path for aligned point cloud
        config: Configuration dict

    Returns:
        Dictionary with aligned cloud path and transformation matrix
    """
    if config is None:
        config = load_config()

    align_config = config.get("alignment", {})

    # Load point clouds
    segment_pcd = load_point_cloud(segment_cloud_path)
    reference_pcd = load_point_cloud(reference_cloud_path)

    # Align
    aligned_pcd, transformation, reg_info = align_point_clouds(
        source=segment_pcd,
        target=reference_pcd,
        max_iterations=align_config.get("max_iterations", 100),
        threshold=align_config.get("threshold", 0.05),
        voxel_size=align_config.get("voxel_size", 0.05),
    )

    # Save aligned point cloud
    output_path = Path(output_path)
    save_point_cloud(aligned_pcd, output_path)

    # Save transformation matrix
    transform_path = output_path.parent / f"{output_path.stem}_transform.npy"
    np.save(transform_path, transformation)

    # Save registration metadata
    metadata_path = output_path.parent / f"{output_path.stem}_registration.json"
    save_json(reg_info, metadata_path)

    log.info(f"Alignment complete: {output_path.name}")

    return {
        "aligned_cloud": output_path,
        "transformation": transformation,
        "transform_file": transform_path,
        "metadata": metadata_path,
        "registration_info": reg_info,
    }


def align_all_segments(
    segment_clouds: list[Path],
    reference_index: int,
    output_dir: Path | str,
    config: dict | None = None,
) -> dict[str, list[Path] | Path]:
    """Align all segment point clouds to a reference segment.

    Args:
        segment_clouds: List of segment point cloud paths
        reference_index: Index of reference segment in the list
        output_dir: Output directory for aligned clouds
        config: Configuration dict

    Returns:
        Dictionary with aligned cloud paths and reference
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if reference_index < 0 or reference_index >= len(segment_clouds):
        raise ValueError(f"Invalid reference index: {reference_index}")

    reference_cloud = segment_clouds[reference_index]
    log.info(f"Using reference: {reference_cloud.name}")

    # Copy reference cloud to output
    reference_output = output_dir / "reference.ply"
    reference_pcd = load_point_cloud(reference_cloud)
    save_point_cloud(reference_pcd, reference_output)

    aligned_clouds = []

    for i, segment_cloud in enumerate(segment_clouds):
        if i == reference_index:
            # Reference doesn't need alignment
            aligned_clouds.append(reference_output)
            continue

        output_path = output_dir / f"segment_{i}_aligned.ply"

        result = align_segment_to_reference(
            segment_cloud_path=segment_cloud,
            reference_cloud_path=reference_cloud,
            output_path=output_path,
            config=config,
        )

        aligned_clouds.append(result["aligned_cloud"])

    log.info(f"Aligned {len(aligned_clouds)} segments")

    return {
        "aligned_clouds": aligned_clouds,
        "reference": reference_output,
        "reference_index": reference_index,
    }
