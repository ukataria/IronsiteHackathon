"""Interactively review and select the best hero frames for the demo.

Usage: uv run python scripts/select_hero_frames.py
Prints a ranked list of frames by quality score with paths you can copy.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

from src.utils import FRAMES_DIR, COMPOSITES_DIR
from src.video.extract import frame_quality_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def rank_all_frames(top_n: int = 20) -> list[tuple[float, Path]]:
    """Score every frame in data/frames/ and return the top_n."""
    frame_paths = sorted(FRAMES_DIR.glob("*.png"))
    if not frame_paths:
        log.error(f"No frames in {FRAMES_DIR}. Run process_videos.py first.")
        return []

    log.info(f"Scoring {len(frame_paths)} frames...")
    scored = []
    for p in frame_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        score = frame_quality_score(frame)
        scored.append((score, p))

    scored.sort(reverse=True)
    return scored[:top_n]


def print_hero_report(top_n: int = 20) -> None:
    """Print ranked hero frames to stdout."""
    ranked = rank_all_frames(top_n)
    if not ranked:
        return

    print(f"\n{'='*70}")
    print(f"  TOP {len(ranked)} HERO FRAME CANDIDATES")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Score':>8}  {'Frame'}")
    print(f"{'-'*70}")

    for rank, (score, path) in enumerate(ranked, 1):
        has_composite = (COMPOSITES_DIR / f"{path.stem}_composite.png").exists()
        status = " [composited]" if has_composite else ""
        print(f"{rank:<6} {score:>8.1f}  {path}{status}")

    print(f"\nCopy these stems into your hero frames list:")
    stems = [p.stem for _, p in ranked[:8]]
    for s in stems:
        print(f"  {s}")


if __name__ == "__main__":
    print_hero_report()
