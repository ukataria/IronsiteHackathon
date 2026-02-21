"""Frame extraction with quality filtering and intelligent selection."""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import compute_blur_score, compute_image_similarity, get_logger, load_config, save_image, save_json

log = get_logger(__name__)


def extract_frames(
    video_path: Path | str,
    output_dir: Path | str,
    fps: float = 2.5,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[Path]:
    """Extract frames from video at specified FPS.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract
        start_frame: Optional start frame (for segment processing)
        end_frame: Optional end frame (for segment processing)

    Returns:
        List of paths to extracted frames
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = total_frames

    # Calculate frame interval
    frame_interval = int(video_fps / fps)

    log.info(f"Extracting frames from {video_path.name} at {fps} FPS (interval={frame_interval})")
    log.info(f"Video FPS: {video_fps}, Range: [{start_frame}, {end_frame}]")

    extracted_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    frame_counter = 0

    with tqdm(total=(end_frame - start_frame) // frame_interval) as pbar:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame - start_frame) % frame_interval == 0:
                frame_path = output_dir / f"frame_{frame_counter:04d}.png"
                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append(frame_path)
                frame_counter += 1
                pbar.update(1)

            current_frame += 1

    cap.release()
    log.info(f"Extracted {len(extracted_frames)} frames")
    return extracted_frames


def filter_blurry_frames(
    frame_paths: list[Path], blur_threshold: float = 100.0
) -> tuple[list[Path], dict[str, float]]:
    """Filter out blurry frames using Laplacian variance.

    Args:
        frame_paths: List of frame paths to filter
        blur_threshold: Minimum blur score to keep frame

    Returns:
        Tuple of (filtered_frames, blur_scores)
    """
    log.info(f"Filtering blurry frames (threshold={blur_threshold})")

    blur_scores = {}
    filtered_frames = []

    for frame_path in tqdm(frame_paths, desc="Computing blur scores"):
        img = cv2.imread(str(frame_path))
        score = compute_blur_score(img)
        blur_scores[frame_path.name] = score

        if score >= blur_threshold:
            filtered_frames.append(frame_path)

    log.info(f"Kept {len(filtered_frames)}/{len(frame_paths)} frames after blur filtering")
    return filtered_frames, blur_scores


def deduplicate_frames(
    frame_paths: list[Path], similarity_threshold: float = 0.95
) -> tuple[list[Path], dict[str, float]]:
    """Remove near-duplicate frames using SSIM.

    Args:
        frame_paths: List of frame paths to deduplicate
        similarity_threshold: SSIM threshold (frames above this are considered duplicates)

    Returns:
        Tuple of (unique_frames, similarity_scores)
    """
    log.info(f"Deduplicating frames (threshold={similarity_threshold})")

    if len(frame_paths) == 0:
        return [], {}

    unique_frames = [frame_paths[0]]
    similarity_scores = {}

    prev_img = cv2.imread(str(frame_paths[0]))

    for frame_path in tqdm(frame_paths[1:], desc="Checking similarity"):
        img = cv2.imread(str(frame_path))
        similarity = compute_image_similarity(prev_img, img)
        similarity_scores[frame_path.name] = similarity

        if similarity < similarity_threshold:
            unique_frames.append(frame_path)
            prev_img = img

    log.info(f"Kept {len(unique_frames)}/{len(frame_paths)} frames after deduplication")
    return unique_frames, similarity_scores


def select_best_frames(
    frame_paths: list[Path],
    blur_scores: dict[str, float],
    max_frames: int = 800,
) -> list[Path]:
    """Select best N frames based on quality scores.

    Args:
        frame_paths: List of candidate frames
        blur_scores: Blur score for each frame
        max_frames: Maximum number of frames to select

    Returns:
        List of selected frame paths
    """
    if len(frame_paths) <= max_frames:
        return frame_paths

    log.info(f"Selecting top {max_frames} frames by quality")

    # Sort by blur score (higher is better)
    scored_frames = [(path, blur_scores.get(path.name, 0.0)) for path in frame_paths]
    scored_frames.sort(key=lambda x: x[1], reverse=True)

    selected = [path for path, _ in scored_frames[:max_frames]]
    log.info(f"Selected {len(selected)} frames")
    return selected


def extract_and_filter(
    video_path: Path | str,
    output_dir: Path | str,
    segment: dict | None = None,
    config: dict | None = None,
) -> dict[str, list[Path] | dict]:
    """Complete extraction pipeline: extract → filter blur → deduplicate → select best.

    Args:
        video_path: Path to input video
        output_dir: Output directory for frames
        segment: Optional segment metadata (start_frame, end_frame)
        config: Configuration dict

    Returns:
        Dictionary with extracted frames and metadata
    """
    if config is None:
        config = load_config()

    extract_config = config.get("extraction", {})
    fps = extract_config.get("fps", 2.5)
    blur_threshold = extract_config.get("blur_threshold", 100.0)
    similarity_threshold = extract_config.get("similarity_threshold", 0.95)
    max_frames = extract_config.get("max_frames", 800)

    output_dir = Path(output_dir)

    # Extract frames
    start_frame = segment.get("start_frame") if segment else None
    end_frame = segment.get("end_frame") if segment else None

    frames = extract_frames(video_path, output_dir / "raw", fps=fps, start_frame=start_frame, end_frame=end_frame)

    # Filter blurry frames
    frames, blur_scores = filter_blurry_frames(frames, blur_threshold=blur_threshold)

    # Deduplicate
    frames, similarity_scores = deduplicate_frames(frames, similarity_threshold=similarity_threshold)

    # Select best frames
    selected_frames = select_best_frames(frames, blur_scores, max_frames=max_frames)

    # Copy selected frames to final directory
    final_dir = output_dir / "selected"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_frames = []
    for i, frame_path in enumerate(selected_frames):
        final_path = final_dir / f"frame_{i:04d}.png"
        img = cv2.imread(str(frame_path))
        cv2.imwrite(str(final_path), img)
        final_frames.append(final_path)

    # Save metadata
    metadata = {
        "total_extracted": len(frames),
        "after_blur_filter": len(frames),
        "after_dedup": len(selected_frames),
        "final_selected": len(final_frames),
        "blur_scores": {k: float(v) for k, v in blur_scores.items()},
        "similarity_scores": {k: float(v) for k, v in similarity_scores.items()},
    }

    save_json(metadata, output_dir / "extraction_metadata.json")

    log.info(f"Extraction complete: {len(final_frames)} frames selected")

    return {"frames": final_frames, "metadata": metadata}
