"""End-to-end temporal reconstruction pipeline."""

import argparse
from pathlib import Path

from src.alignment.align import align_all_segments
from src.changes.detect import detect_and_cluster_changes
from src.export.timeline import export_for_viewer
from src.extraction.extract import extract_and_filter
from src.reconstruction.colmap_wrapper import check_colmap_installed, reconstruct_segment
from src.segmentation.segment_time import segment_video
from src.utils import create_video_id_dir, get_logger, load_config, save_json

log = get_logger(__name__)


def run_pipeline(
    video_path: Path | str,
    video_id: str,
    config: dict | None = None,
    skip_segmentation: bool = False,
    skip_extraction: bool = False,
    skip_reconstruction: bool = False,
    skip_alignment: bool = False,
    skip_changes: bool = False,
) -> dict:
    """Run the complete temporal reconstruction pipeline.

    Args:
        video_path: Path to input video
        video_id: Unique identifier for this video
        config: Configuration dictionary
        skip_segmentation: Skip temporal segmentation
        skip_extraction: Skip frame extraction
        skip_reconstruction: Skip COLMAP reconstruction
        skip_alignment: Skip point cloud alignment
        skip_changes: Skip change detection

    Returns:
        Dictionary with pipeline results and paths
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if config is None:
        config = load_config()

    # Create output directory structure
    video_dir = create_video_id_dir(video_id)
    segments_dir = video_dir / "segments"
    aligned_dir = video_dir / "aligned"
    changes_dir = video_dir / "changes"
    transforms_dir = video_dir / "transforms"

    log.info(f"Starting pipeline for: {video_id}")
    log.info(f"Video: {video_path}")
    log.info(f"Output: {video_dir}")

    results = {"video_id": video_id, "video_path": str(video_path), "output_dir": str(video_dir)}

    # Step 1: Temporal Segmentation
    if not skip_segmentation:
        log.info("=" * 60)
        log.info("STEP 1: Temporal Segmentation")
        log.info("=" * 60)

        segments = segment_video(video_path, config=config)
        results["segments"] = segments
        results["segment_count"] = len(segments)

        # Save segments
        segments_metadata_dir = video_dir / "segments_metadata"
        segments_metadata_dir.mkdir(parents=True, exist_ok=True)
        save_json({"segments": segments}, segments_metadata_dir / "segments.json")
    else:
        log.info("Skipping temporal segmentation")

    # Step 2: Frame Extraction and Quality Filtering
    if not skip_extraction:
        log.info("=" * 60)
        log.info("STEP 2: Frame Extraction and Quality Filtering")
        log.info("=" * 60)

        # Load segments if not done in step 1
        if "segments" not in results:
            segments_file = video_dir / "segments_metadata" / "segments.json"
            import json

            with open(segments_file) as f:
                segments_data = json.load(f)
                results["segments"] = segments_data["segments"]

        segment_frames = []

        for i, segment in enumerate(results["segments"]):
            log.info(f"Processing segment {i}/{len(results['segments'])}")

            segment_output_dir = segments_dir / f"segment_{i}"
            extraction_result = extract_and_filter(
                video_path=video_path, output_dir=segment_output_dir, segment=segment, config=config
            )

            segment_frames.append(extraction_result["frames"])

        results["segment_frames"] = segment_frames
    else:
        log.info("Skipping frame extraction")

    # Step 3: COLMAP 3D Reconstruction
    if not skip_reconstruction:
        log.info("=" * 60)
        log.info("STEP 3: COLMAP 3D Reconstruction")
        log.info("=" * 60)

        # Check COLMAP installation
        if not check_colmap_installed():
            raise RuntimeError("COLMAP not installed")

        pointclouds = []

        for i in range(len(results.get("segments", []))):
            log.info(f"Reconstructing segment {i}")

            frame_dir = segments_dir / f"segment_{i}" / "selected"
            if not frame_dir.exists():
                log.warning(f"Frame directory not found: {frame_dir}, skipping")
                continue

            colmap_output = segments_dir / f"segment_{i}" / "colmap"
            reconstruction_result = reconstruct_segment(frame_dir=frame_dir, output_dir=colmap_output, config=config)

            pointclouds.append(reconstruction_result["pointcloud"])

        results["pointclouds"] = pointclouds
    else:
        log.info("Skipping COLMAP reconstruction")

    # Step 4: Point Cloud Alignment
    if not skip_alignment:
        log.info("=" * 60)
        log.info("STEP 4: Point Cloud Alignment (ICP)")
        log.info("=" * 60)

        # Load point clouds if not done in step 3
        if "pointclouds" not in results:
            pointclouds = []
            for i in range(len(results.get("segments", []))):
                pc_path = segments_dir / f"segment_{i}" / "colmap" / "pointcloud.ply"
                if pc_path.exists():
                    pointclouds.append(pc_path)
            results["pointclouds"] = pointclouds

        # Determine reference segment
        reference_method = config.get("alignment", {}).get("reference", "segment_0")

        if reference_method == "largest":
            # Choose segment with most points
            import open3d as o3d

            sizes = [len(o3d.io.read_point_cloud(str(pc)).points) for pc in results["pointclouds"]]
            reference_index = sizes.index(max(sizes))
            log.info(f"Using largest segment as reference: segment_{reference_index}")
        else:
            # Use first segment or specified index
            reference_index = 0
            log.info(f"Using segment_0 as reference")

        alignment_result = align_all_segments(
            segment_clouds=results["pointclouds"], reference_index=reference_index, output_dir=aligned_dir, config=config
        )

        results["aligned_clouds"] = alignment_result["aligned_clouds"]
        results["reference_cloud"] = alignment_result["reference"]
    else:
        log.info("Skipping point cloud alignment")

    # Step 5: Change Detection
    if not skip_changes:
        log.info("=" * 60)
        log.info("STEP 5: Change Detection")
        log.info("=" * 60)

        # Load aligned clouds if not done in step 4
        if "aligned_clouds" not in results:
            aligned_clouds = sorted(aligned_dir.glob("segment_*_aligned.ply"))
            results["aligned_clouds"] = aligned_clouds

        change_results = []

        for i in range(len(results["aligned_clouds"]) - 1):
            log.info(f"Detecting changes: segment {i} → segment {i+1}")

            cloud_a = results["aligned_clouds"][i]
            cloud_b = results["aligned_clouds"][i + 1]

            change_output = changes_dir / f"segment_{i}_to_{i+1}"
            change_result = detect_and_cluster_changes(
                cloud_a_path=cloud_a, cloud_b_path=cloud_b, output_dir=change_output, config=config
            )

            change_results.append(change_result)

        results["change_results"] = change_results
    else:
        log.info("Skipping change detection")

    # Step 6: Export for Viewer
    log.info("=" * 60)
    log.info("STEP 6: Export Timeline and Assets")
    log.info("=" * 60)

    viewer_dir = video_dir / "viewer"
    viewer_output = export_for_viewer(video_id=video_id, derived_dir=video_dir, viewer_output_dir=viewer_dir)

    results["viewer_dir"] = viewer_output

    # Save final results
    save_json(results, video_dir / "pipeline_results.json")

    log.info("=" * 60)
    log.info("Pipeline Complete!")
    log.info("=" * 60)
    log.info(f"Results saved to: {video_dir}")
    log.info(f"Viewer assets: {viewer_output}")

    return results


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(description="Temporal Construction World Model Pipeline")

    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--video-id", required=True, help="Unique identifier for this video")
    parser.add_argument("--config", help="Path to config YAML file")

    # Skip options
    parser.add_argument("--skip-segmentation", action="store_true", help="Skip temporal segmentation")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip frame extraction")
    parser.add_argument("--skip-reconstruction", action="store_true", help="Skip COLMAP reconstruction")
    parser.add_argument("--skip-alignment", action="store_true", help="Skip point cloud alignment")
    parser.add_argument("--skip-changes", action="store_true", help="Skip change detection")

    args = parser.parse_args()

    # Load config
    config = None
    if args.config:
        config = load_config(args.config)

    # Run pipeline
    run_pipeline(
        video_path=args.video,
        video_id=args.video_id,
        config=config,
        skip_segmentation=args.skip_segmentation,
        skip_extraction=args.skip_extraction,
        skip_reconstruction=args.skip_reconstruction,
        skip_alignment=args.skip_alignment,
        skip_changes=args.skip_changes,
    )


if __name__ == "__main__":
    main()
