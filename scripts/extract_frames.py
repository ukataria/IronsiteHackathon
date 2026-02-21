"""Extract stable, well-lit frames from Ironsite video clips."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

RAW_DIR = Path("data/raw")
FRAMES_DIR = Path("data/frames")

# Minimum Laplacian variance to consider a frame "sharp enough"
BLUR_THRESHOLD = 100.0
# Sample one frame every N seconds
SAMPLE_INTERVAL_SEC = 2.0


def compute_blur_score(frame: np.ndarray) -> float:
    """Return Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_interval_sec: float = SAMPLE_INTERVAL_SEC,
    blur_threshold: float = BLUR_THRESHOLD,
) -> list[str]:
    """
    Extract stable frames from a video file.
    Skips blurry frames (low Laplacian variance).

    Returns list of saved frame paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(fps * sample_interval_sec)
    video_id = Path(video_path).stem

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = []
    frame_idx = 0
    saved_count = 0

    with tqdm(total=total_frames, desc=f"Extracting {video_id}", unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval_frames == 0:
                blur = compute_blur_score(frame)
                if blur >= blur_threshold:
                    out_name = f"{video_id}_f{frame_idx:06d}.jpg"
                    out_path = str(Path(output_dir) / out_name)
                    cv2.imwrite(out_path, frame)
                    saved.append(out_path)
                    saved_count += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"{video_id}: saved {saved_count} frames from {frame_idx} total.")
    return saved


def main() -> None:
    output_dir = sys.argv[1] if len(sys.argv) > 1 else str(FRAMES_DIR)
    video_paths = list(RAW_DIR.glob("*.mp4")) + list(RAW_DIR.glob("*.mov")) + list(RAW_DIR.glob("*.avi"))

    if not video_paths:
        print(f"No video files found in {RAW_DIR}. Place .mp4/.mov/.avi files there.")
        sys.exit(1)

    total_saved = 0
    for vp in sorted(video_paths):
        saved = extract_frames(str(vp), output_dir)
        total_saved += len(saved)

    print(f"\nTotal frames saved: {total_saved} → {output_dir}")


if __name__ == "__main__":
    main()
