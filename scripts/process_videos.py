"""Batch process all video clips: extract frames, run depth + segmentation.

Run this on Vast.ai with a GPU for fast parallel processing.
Usage: uv run python scripts/process_videos.py [--fps 1.0] [--top-n 8]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils import ensure_dirs, DATA
from src.video.extract import process_video
from src.depth.estimate import batch_estimate
from src.segmentation.segment import run_segmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = DATA / "raw"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process all videos in data/raw/")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--top-n", type=int, default=8, help="Hero frames per clip")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth estimation")
    parser.add_argument("--skip-seg", action="store_true", help="Skip segmentation")
    parser.add_argument("--clip-id", default=None, help="Process only this clip ID (video stem)")
    args = parser.parse_args()

    ensure_dirs()

    video_files = [f for f in RAW_DIR.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        log.error(f"No videos found in {RAW_DIR}. Add your .mp4/.mov files there.")
        return

    if args.clip_id:
        video_files = [f for f in video_files if f.stem == args.clip_id]
        if not video_files:
            log.error(f"No video with stem '{args.clip_id}' found.")
            return

    log.info(f"Found {len(video_files)} video(s) to process")

    all_keyframes = []
    for video_path in video_files:
        log.info(f"\n{'='*60}")
        log.info(f"Processing: {video_path.name}")

        result = process_video(
            video_path,
            clip_id=video_path.stem,
            fps_sample=args.fps,
            top_n=args.top_n,
        )
        keyframes = result["keyframes"]
        all_keyframes.extend(keyframes)

        if not args.skip_depth:
            log.info(f"Running depth estimation on {len(keyframes)} keyframes")
            batch_estimate(keyframes)

        if not args.skip_seg:
            log.info(f"Running segmentation on {len(keyframes)} keyframes")
            for kf in keyframes:
                run_segmentation(kf, stem=kf.stem)

    log.info(f"\n{'='*60}")
    log.info(f"Done. {len(all_keyframes)} total keyframes processed.")
    log.info("Next step: uv run python pipeline.py frame <frame_path>")
    log.info("Or run LLM scene description: src/scene/describe.py")


if __name__ == "__main__":
    main()
