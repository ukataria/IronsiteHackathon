"""Change detection: identify added and removed geometry between temporal segments."""

from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from src.utils import get_logger, load_config, save_json

log = get_logger(__name__)


def detect_changes(
    cloud_a: o3d.geometry.PointCloud,
    cloud_b: o3d.geometry.PointCloud,
    nn_distance: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect added and removed points between two aligned point clouds.

    Args:
        cloud_a: Earlier time point cloud
        cloud_b: Later time point cloud
        nn_distance: Nearest neighbor distance threshold

    Returns:
        Tuple of (added_indices, removed_indices)
    """
    log.info("Detecting changes between point clouds")

    points_a = np.asarray(cloud_a.points)
    points_b = np.asarray(cloud_b.points)

    # Build KD-tree for cloud A
    tree_a = o3d.geometry.KDTreeFlann(cloud_a)

    # Find points in B that are far from A (added geometry)
    added_indices = []
    for i, point in enumerate(points_b):
        [k, idx, _] = tree_a.search_radius_vector_3d(point, nn_distance)
        if k == 0:  # No neighbors in cloud A
            added_indices.append(i)

    # Build KD-tree for cloud B
    tree_b = o3d.geometry.KDTreeFlann(cloud_b)

    # Find points in A that are far from B (removed geometry)
    removed_indices = []
    for i, point in enumerate(points_a):
        [k, idx, _] = tree_b.search_radius_vector_3d(point, nn_distance)
        if k == 0:  # No neighbors in cloud B
            removed_indices.append(i)

    log.info(f"Detected changes: {len(added_indices)} added, {len(removed_indices)} removed")

    return np.array(added_indices), np.array(removed_indices)


def cluster_changes(
    points: np.ndarray,
    eps: float = 0.1,
    min_samples: int = 50,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Cluster change points into coherent geometric regions using DBSCAN.

    Args:
        points: Nx3 array of points
        eps: DBSCAN epsilon (maximum distance between points in cluster)
        min_samples: Minimum points per cluster

    Returns:
        Tuple of (cluster_labels, cluster_point_lists)
    """
    if len(points) == 0:
        return np.array([]), []

    log.info(f"Clustering {len(points)} change points (eps={eps}, min_samples={min_samples})")

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # Get unique clusters (excluding noise label -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)

    clusters = []
    for label in sorted(unique_labels):
        cluster_points = points[labels == label]
        clusters.append(cluster_points)
        log.info(f"Cluster {label}: {len(cluster_points)} points")

    log.info(f"Found {len(clusters)} clusters ({np.sum(labels == -1)} noise points)")

    return labels, clusters


def filter_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Remove statistical outliers from point cloud.

    Args:
        pcd: Input point cloud
        nb_neighbors: Number of neighbors for statistical analysis
        std_ratio: Standard deviation ratio threshold

    Returns:
        Filtered point cloud
    """
    filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    log.info(f"Outlier filtering: {len(pcd.points)} → {len(filtered_pcd.points)} points")
    return filtered_pcd


def colorize_changes(
    added_pcd: o3d.geometry.PointCloud | None,
    removed_pcd: o3d.geometry.PointCloud | None,
    added_color: tuple[float, float, float] = (0.0, 1.0, 0.0),  # Green
    removed_color: tuple[float, float, float] = (1.0, 0.0, 0.0),  # Red
) -> tuple[o3d.geometry.PointCloud | None, o3d.geometry.PointCloud | None]:
    """Apply colors to change point clouds.

    Args:
        added_pcd: Added geometry point cloud
        removed_pcd: Removed geometry point cloud
        added_color: RGB color for added points
        removed_color: RGB color for removed points

    Returns:
        Tuple of (colored_added, colored_removed)
    """
    if added_pcd is not None:
        added_pcd.paint_uniform_color(added_color)

    if removed_pcd is not None:
        removed_pcd.paint_uniform_color(removed_color)

    return added_pcd, removed_pcd


def detect_and_cluster_changes(
    cloud_a_path: Path | str,
    cloud_b_path: Path | str,
    output_dir: Path | str,
    config: dict | None = None,
) -> dict[str, Path | list]:
    """Detect and cluster changes between two aligned point clouds.

    Args:
        cloud_a_path: Path to earlier point cloud
        cloud_b_path: Path to later point cloud
        output_dir: Output directory for change clouds
        config: Configuration dict

    Returns:
        Dictionary with change detection results
    """
    if config is None:
        config = load_config()

    change_config = config.get("changes", {})

    # Load point clouds
    cloud_a = o3d.io.read_point_cloud(str(cloud_a_path))
    cloud_b = o3d.io.read_point_cloud(str(cloud_b_path))

    log.info(f"Comparing clouds: {Path(cloud_a_path).name} vs {Path(cloud_b_path).name}")

    # Detect changes
    added_indices, removed_indices = detect_changes(
        cloud_a=cloud_a,
        cloud_b=cloud_b,
        nn_distance=change_config.get("nn_distance", 0.05),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "added_points": len(added_indices),
        "removed_points": len(removed_indices),
    }

    # Process added geometry
    if len(added_indices) > 0:
        points_b = np.asarray(cloud_b.points)
        added_points = points_b[added_indices]

        added_pcd = o3d.geometry.PointCloud()
        added_pcd.points = o3d.utility.Vector3dVector(added_points)

        # Filter outliers
        added_pcd = filter_outliers(
            added_pcd,
            nb_neighbors=change_config.get("statistical_outlier_nb_neighbors", 20),
            std_ratio=change_config.get("statistical_outlier_std_ratio", 2.0),
        )

        # Cluster
        labels, clusters = cluster_changes(
            np.asarray(added_pcd.points),
            eps=change_config.get("cluster_eps", 0.1),
            min_samples=change_config.get("min_cluster_size", 50),
        )

        # Colorize and save
        added_pcd, _ = colorize_changes(added_pcd, None)
        added_path = output_dir / "added.ply"
        o3d.io.write_point_cloud(str(added_path), added_pcd)

        results["added_cloud"] = added_path
        results["added_clusters"] = len(clusters)

    # Process removed geometry
    if len(removed_indices) > 0:
        points_a = np.asarray(cloud_a.points)
        removed_points = points_a[removed_indices]

        removed_pcd = o3d.geometry.PointCloud()
        removed_pcd.points = o3d.utility.Vector3dVector(removed_points)

        # Filter outliers
        removed_pcd = filter_outliers(
            removed_pcd,
            nb_neighbors=change_config.get("statistical_outlier_nb_neighbors", 20),
            std_ratio=change_config.get("statistical_outlier_std_ratio", 2.0),
        )

        # Cluster
        labels, clusters = cluster_changes(
            np.asarray(removed_pcd.points),
            eps=change_config.get("cluster_eps", 0.1),
            min_samples=change_config.get("min_cluster_size", 50),
        )

        # Colorize and save
        _, removed_pcd = colorize_changes(None, removed_pcd)
        removed_path = output_dir / "removed.ply"
        o3d.io.write_point_cloud(str(removed_path), removed_pcd)

        results["removed_cloud"] = removed_path
        results["removed_clusters"] = len(clusters)

    # Save metadata
    metadata_path = output_dir / "changes_metadata.json"
    save_json(results, metadata_path)

    log.info(f"Change detection complete: {output_dir}")

    return results
