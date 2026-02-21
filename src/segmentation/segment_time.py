"""Temporal segmentation: split video into construction phase segments."""

import subprocess
from pathlib import Path
from typing import Literal

import cv2

from src.utils import get_logger, load_config, save_json

log = get_logger(__name__)


def segment_video_scenedetect(
    video_path: Path | str,
    threshold: float = 27.0,
    min_scene_len: int = 30,
    output_dir: Path | str | None = None,
) -> list[dict[str, int | float]]:
    """Segment video using PySceneDetect content-aware detection.

    Args:
        video_path: Path to input video
        threshold: Scene change sensitivity (lower = more sensitive)
        min_scene_len: Minimum frames per segment
        output_dir: Output directory for segment metadata

    Returns:
        List of segments: [{"start_frame": int, "end_frame": int, "duration_sec": float}, ...]
    """
    video_path = Path(video_path)
    log.info(f"Segmenting video: {video_path.name}")

    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_segments"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use scenedetect CLI (requires: pip install scenedetect[opencv])
    stats_file = output_dir / "scene_stats.csv"

    cmd = [
        "scenedetect",
        "--input",
        str(video_path),
        "--output",
        str(output_dir),
        "--stats",
        str(stats_file),
        "detect-content",
        "--threshold",
        str(threshold),
        "--min-scene-len",
        str(min_scene_len),
        "list-scenes",
    ]

    log.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log.info(result.stdout)
    except subprocess.CalledProcessError as e:
        log.error(f"SceneDetect failed: {e.stderr}")
        raise

    # Parse scene list from output
    segments = _parse_scenedetect_output(output_dir / "scene_list.csv", video_path)

    # Save segments metadata
    save_json({"segments": segments, "method": "scenedetect", "threshold": threshold}, output_dir / "segments.json")

    log.info(f"Found {len(segments)} segments")
    return segments


def segment_video_manual(
    video_path: Path | str, segment_times: list[tuple[float, float]]
) -> list[dict[str, int | float]]:
    """Manually segment video by providing time ranges.

    Args:
        video_path: Path to input video
        segment_times: List of (start_sec, end_sec) tuples

    Returns:
        List of segments with frame numbers
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    segments = []
    for i, (start_sec, end_sec) in enumerate(segment_times):
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        segments.append(
            {
                "segment_id": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_sec,
                "end_time": end_sec,
                "duration_sec": end_sec - start_sec,
            }
        )

    log.info(f"Created {len(segments)} manual segments")
    return segments


def segment_video(
    video_path: Path | str,
    method: Literal["scenedetect", "manual"] = "scenedetect",
    config: dict | None = None,
    **kwargs,
) -> list[dict[str, int | float]]:
    """Segment video into temporal chunks.

    Args:
        video_path: Path to input video
        method: Segmentation method ("scenedetect" or "manual")
        config: Configuration dict (optional, will load from config.yaml if not provided)
        **kwargs: Method-specific parameters

    Returns:
        List of segment metadata dictionaries
    """
    if config is None:
        config = load_config()

    seg_config = config.get("segmentation", {})

    if method == "scenedetect":
        threshold = kwargs.get("threshold", seg_config.get("threshold", 27.0))
        min_scene_len = kwargs.get("min_scene_len", seg_config.get("min_scene_len", 30))
        return segment_video_scenedetect(video_path, threshold=threshold, min_scene_len=min_scene_len)

    elif method == "manual":
        segment_times = kwargs.get("segment_times")
        if segment_times is None:
            raise ValueError("Manual segmentation requires 'segment_times' parameter")
        return segment_video_manual(video_path, segment_times=segment_times)

    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def _parse_scenedetect_output(scene_list_csv: Path, video_path: Path) -> list[dict[str, int | float]]:
    """Parse scenedetect CSV output into segment metadata."""
    if not scene_list_csv.exists():
        log.warning(f"Scene list not found: {scene_list_csv}, treating video as single segment")
        # Fallback: treat entire video as one segment
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        return [
            {
                "segment_id": 0,
                "start_frame": 0,
                "end_frame": total_frames,
                "start_time": 0.0,
                "end_time": total_frames / fps,
                "duration_sec": total_frames / fps,
            }
        ]

    segments = []
    with open(scene_list_csv) as f:
        lines = f.readlines()

    # Skip header
    for i, line in enumerate(lines[1:], start=0):
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue

        # CSV format: Scene Number, Start Frame, End Frame, Start Time, End Time, ...
        start_frame = int(parts[1])
        end_frame = int(parts[2])
        start_time = float(parts[3])
        end_time = float(parts[4])

        segments.append(
            {
                "segment_id": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": end_time - start_time,
            }
        )

    return segments
