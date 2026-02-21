"""Timeline export and asset packaging for the 3D viewer."""

import shutil
from pathlib import Path

from src.utils import get_logger, save_json

log = get_logger(__name__)


def create_timeline(
    segments: list[dict],
    aligned_clouds: list[Path],
    change_results: list[dict] | None = None,
) -> dict:
    """Create timeline JSON for 3D viewer.

    Args:
        segments: List of segment metadata dictionaries
        aligned_clouds: List of aligned point cloud paths
        change_results: Optional list of change detection results

    Returns:
        Timeline dictionary
    """
    timeline = {
        "total_segments": len(segments),
        "segments": [],
    }

    for i, (segment, cloud_path) in enumerate(zip(segments, aligned_clouds)):
        segment_entry = {
            "id": i,
            "segment_id": segment.get("segment_id", i),
            "start_frame": segment.get("start_frame", 0),
            "end_frame": segment.get("end_frame", 0),
            "start_time": segment.get("start_time", 0.0),
            "end_time": segment.get("end_time", 0.0),
            "duration_sec": segment.get("duration_sec", 0.0),
            "pointcloud": cloud_path.name,
        }

        # Add change information if available
        if change_results and i < len(change_results):
            changes = change_results[i]
            segment_entry["changes"] = {
                "added_points": changes.get("added_points", 0),
                "removed_points": changes.get("removed_points", 0),
                "added_cloud": changes.get("added_cloud", {}).name if "added_cloud" in changes else None,
                "removed_cloud": changes.get("removed_cloud", {}).name if "removed_cloud" in changes else None,
            }

        timeline["segments"].append(segment_entry)

    return timeline


def package_viewer_assets(
    video_id: str,
    aligned_clouds: list[Path],
    reference_cloud: Path,
    output_dir: Path | str,
    segments: list[dict],
    change_results: list[dict] | None = None,
    transforms_dir: Path | None = None,
) -> Path:
    """Package all assets for the 3D viewer application.

    Args:
        video_id: Video identifier
        aligned_clouds: List of aligned point cloud paths
        reference_cloud: Reference point cloud path
        output_dir: Output directory for viewer assets
        segments: Segment metadata
        change_results: Optional change detection results
        transforms_dir: Optional directory containing transformation matrices

    Returns:
        Path to the packaged viewer directory
    """
    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Packaging viewer assets for {video_id}")

    # Copy point clouds to assets
    for cloud_path in aligned_clouds:
        dest = assets_dir / cloud_path.name
        shutil.copy(cloud_path, dest)
        log.info(f"Copied: {cloud_path.name}")

    # Copy reference
    ref_dest = assets_dir / "reference.ply"
    shutil.copy(reference_cloud, ref_dest)
    log.info(f"Copied reference: {reference_cloud.name}")

    # Copy change clouds if available
    if change_results:
        for i, changes in enumerate(change_results):
            if "added_cloud" in changes and changes["added_cloud"]:
                added_path = changes["added_cloud"]
                if isinstance(added_path, Path) and added_path.exists():
                    dest = assets_dir / f"added_{i}.ply"
                    shutil.copy(added_path, dest)
                    log.info(f"Copied added changes: segment {i}")

            if "removed_cloud" in changes and changes["removed_cloud"]:
                removed_path = changes["removed_cloud"]
                if isinstance(removed_path, Path) and removed_path.exists():
                    dest = assets_dir / f"removed_{i}.ply"
                    shutil.copy(removed_path, dest)
                    log.info(f"Copied removed changes: segment {i}")

    # Copy transforms if available
    if transforms_dir and transforms_dir.exists():
        transforms_dest = output_dir / "transforms"
        transforms_dest.mkdir(parents=True, exist_ok=True)

        for transform_file in transforms_dir.glob("*.npy"):
            shutil.copy(transform_file, transforms_dest / transform_file.name)
            log.info(f"Copied transform: {transform_file.name}")

        for transform_json in transforms_dir.glob("*.json"):
            shutil.copy(transform_json, transforms_dest / transform_json.name)

    # Create timeline.json
    timeline = create_timeline(segments, aligned_clouds, change_results)
    timeline_path = output_dir / "timeline.json"
    save_json(timeline, timeline_path)
    log.info(f"Created timeline: {timeline_path}")

    # Create metadata.json
    metadata = {
        "video_id": video_id,
        "total_segments": len(segments),
        "total_clouds": len(aligned_clouds),
        "reference": "reference.ply",
        "timeline": "timeline.json",
    }

    metadata_path = output_dir / "metadata.json"
    save_json(metadata, metadata_path)
    log.info(f"Created metadata: {metadata_path}")

    log.info(f"Viewer assets packaged: {output_dir}")

    return output_dir


def export_for_viewer(
    video_id: str,
    derived_dir: Path | str,
    viewer_output_dir: Path | str,
) -> Path:
    """Export all pipeline outputs for the viewer application.

    Args:
        video_id: Video identifier
        derived_dir: Path to derived data directory
        viewer_output_dir: Output directory for viewer

    Returns:
        Path to viewer directory
    """
    derived_dir = Path(derived_dir)
    viewer_output_dir = Path(viewer_output_dir)

    log.info(f"Exporting {video_id} for viewer")

    # Load segments metadata
    segments_file = derived_dir / "segments" / "segments.json"
    if segments_file.exists():
        import json

        with open(segments_file) as f:
            segments_data = json.load(f)
            segments = segments_data.get("segments", [])
    else:
        log.warning("No segments metadata found, using default")
        segments = []

    # Get aligned clouds
    aligned_dir = derived_dir / "aligned"
    aligned_clouds = sorted(aligned_dir.glob("segment_*_aligned.ply"))
    reference_cloud = aligned_dir / "reference.ply"

    if not reference_cloud.exists():
        raise FileNotFoundError(f"Reference cloud not found: {reference_cloud}")

    # Get change results if available
    changes_dir = derived_dir / "changes"
    change_results = []

    if changes_dir.exists():
        for i in range(len(aligned_clouds)):
            change_metadata = changes_dir / f"segment_{i}_changes" / "changes_metadata.json"
            if change_metadata.exists():
                import json

                with open(change_metadata) as f:
                    change_results.append(json.load(f))
            else:
                change_results.append({})

    # Get transforms
    transforms_dir = derived_dir / "transforms"

    # Package assets
    return package_viewer_assets(
        video_id=video_id,
        aligned_clouds=aligned_clouds,
        reference_cloud=reference_cloud,
        output_dir=viewer_output_dir,
        segments=segments,
        change_results=change_results if change_results else None,
        transforms_dir=transforms_dir if transforms_dir.exists() else None,
    )
