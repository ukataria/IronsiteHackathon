"""Frame extraction and keyframe selection from construction video clips."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.utils import FRAMES_DIR, frame_stem, save_image

log = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    clip_id: str,
    fps_sample: float = 1.0,
    max_frames: int = 300,
) -> list[Path]:
    """Extract frames from a video at a given sample rate.

    Args:
        video_path: Path to the source video file.
        clip_id: Short identifier used in output filenames.
        fps_sample: How many frames per second to sample (default 1.0).
        max_frames: Hard cap on total extracted frames.

    Returns:
        List of saved frame paths.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(video_fps / fps_sample))

    log.info(f"{clip_id}: {total_frames} frames @ {video_fps:.1f}fps, sampling every {interval} frames")

    saved: list[Path] = []
    frame_idx = 0
    sample_count = 0

    while cap.isOpened() and sample_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            stem = frame_stem(clip_id, sample_count)
            out_path = FRAMES_DIR / f"{stem}.png"
            save_image(frame, out_path)
            saved.append(out_path)
            sample_count += 1

        frame_idx += 1

    cap.release()
    log.info(f"{clip_id}: saved {len(saved)} frames to {FRAMES_DIR}")
    return saved


def sharpness_score(frame: np.ndarray) -> float:
    """Compute Laplacian variance as a sharpness proxy (higher = sharper)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(frame: np.ndarray) -> float:
    """Mean brightness 0-255 (avoid too dark or blown-out frames)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def frame_quality_score(frame: np.ndarray) -> float:
    """Combined quality score: sharpness weighted, penalize extreme brightness."""
    sharp = sharpness_score(frame)
    bright = brightness_score(frame)
    # Penalize frames that are very dark (<40) or very bright (>220)
    brightness_penalty = max(0.0, 40 - bright) + max(0.0, bright - 220)
    return sharp - brightness_penalty * 0.5


def select_keyframes(
    frame_paths: list[Path],
    top_n: int = 8,
    min_gap: int = 5,
) -> list[Path]:
    """Select the top-N sharpest, well-lit frames with minimum spacing.

    Args:
        frame_paths: All extracted frame paths (sorted by frame number).
        top_n: How many hero frames to select.
        min_gap: Minimum number of frames between selected frames.

    Returns:
        List of selected hero frame paths.
    """
    scored: list[tuple[float, int, Path]] = []
    for i, p in enumerate(frame_paths):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        score = frame_quality_score(frame)
        scored.append((score, i, p))

    scored.sort(reverse=True)

    selected: list[tuple[float, int, Path]] = []
    for score, idx, path in scored:
        if len(selected) >= top_n:
            break
        # Enforce minimum gap between picks
        if all(abs(idx - s_idx) >= min_gap for _, s_idx, _ in selected):
            selected.append((score, idx, path))
            log.info(f"  Keyframe: {path.name} (score={score:.1f})")

    # Return sorted by frame index
    selected.sort(key=lambda x: x[1])
    return [p for _, _, p in selected]


def process_video(
    video_path: str | Path,
    clip_id: str | None = None,
    fps_sample: float = 1.0,
    top_n: int = 8,
) -> dict[str, list[Path]]:
    """Full pipeline: extract frames, score, select keyframes.

    Returns:
        Dict with keys 'all_frames' and 'keyframes'.
    """
    video_path = Path(video_path)
    if clip_id is None:
        clip_id = video_path.stem

    all_frames = extract_frames(video_path, clip_id, fps_sample=fps_sample)
    keyframes = select_keyframes(all_frames, top_n=top_n)
    log.info(f"{clip_id}: {len(keyframes)} keyframes selected from {len(all_frames)} total")
    return {"all_frames": all_frames, "keyframes": keyframes}
