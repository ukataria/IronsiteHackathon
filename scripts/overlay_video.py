"""Apply a static ghost overlay to every frame of a video clip.

Option C: one great overlay applied to the whole clip. Works best when
the camera is relatively stable. Output is a new MP4 with the ghost baked in.

Usage:
    uv run python scripts/overlay_video.py \\
        --video data/raw/01_production_masonry.mp4 \\
        --overlay data/overlays/01_production_masonry_frame_0130_walls.png \\
        --mask data/segments/01_production_masonry_frame_0130_walls.png \\
        --alpha 0.45 \\
        --output data/composites/ghost_video.mp4

    # Or apply ALL available layers for a stem:
    uv run python scripts/overlay_video.py \\
        --video data/raw/01_production_masonry.mp4 \\
        --stem 01_production_masonry_frame_0130 \\
        --output data/composites/ghost_video.mp4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from src.utils import OVERLAYS_DIR, SEGMENTS_DIR, COMPOSITES_DIR, ensure_dirs
from src.composite.blend import alpha_blend, apply_ghost_tint, LAYER_ORDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_layers(stem: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load all available overlay+mask pairs for a given frame stem."""
    layers = []
    for layer_name in LAYER_ORDER:
        overlay_path = OVERLAYS_DIR / f"{stem}_{layer_name}.png"
        mask_path = SEGMENTS_DIR / f"{stem}_{layer_name}.png"
        if overlay_path.exists() and mask_path.exists():
            overlay = cv2.imread(str(overlay_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            layers.append((overlay, mask, layer_name))
            log.info(f"Loaded layer: {layer_name}")
    return layers


def composite_frame(
    frame: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray, str]],
    alpha: float,
    tint: bool = True,
) -> np.ndarray:
    """Apply all overlay layers to a single video frame."""
    result = frame.copy()
    h, w = frame.shape[:2]

    for overlay, mask, layer_name in layers:
        # Resize to match frame if needed
        if overlay.shape[:2] != (h, w):
            overlay = cv2.resize(overlay, (w, h))
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if tint:
            overlay = apply_ghost_tint(overlay, mask)

        result = alpha_blend(result, overlay, mask, alpha=alpha)

    return result


def overlay_video(
    video_path: str | Path,
    layers: list[tuple[np.ndarray, np.ndarray, str]],
    output_path: str | Path,
    alpha: float = 0.45,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    tint: bool = True,
) -> Path:
    """Write a new video with ghost overlay composited on every frame.

    Args:
        video_path: Input video file.
        layers: List of (overlay, mask, name) tuples.
        output_path: Where to save the output MP4.
        alpha: Ghost opacity (0=transparent, 1=opaque).
        start_sec: Start time in seconds (trim clip).
        end_sec: End time in seconds (None = full video).
        tint: Apply ghost blue tint.

    Returns:
        Path to output video.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec else total_frames

    log.info(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")
    log.info(f"Processing frames {start_frame}–{end_frame} ({(end_frame-start_frame)/fps:.1f}s)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    written = 0

    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        composited = composite_frame(frame, layers, alpha=alpha, tint=tint)
        writer.write(composited)
        written += 1

        if written % 30 == 0:
            pct = (frame_idx - start_frame) / (end_frame - start_frame) * 100
            log.info(f"  {pct:.0f}% ({written} frames written)")

        frame_idx += 1

    cap.release()
    writer.release()
    log.info(f"Done. Output: {output_path} ({written} frames)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ghost overlay to a video clip")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=COMPOSITES_DIR / "ghost_video.mp4")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--no-tint", action="store_true")

    # Layer source: either a stem (auto-loads all layers) or explicit overlay+mask
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stem", type=str, help="Frame stem to auto-load all layers from")
    group.add_argument("--overlay", type=Path, help="Single overlay image path")

    parser.add_argument("--mask", type=Path, help="Mask path (required with --overlay)")
    args = parser.parse_args()

    ensure_dirs()

    if args.stem:
        layers = load_layers(args.stem)
        if not layers:
            log.error(f"No overlay layers found for stem: {args.stem}")
            log.error(f"Expected files in {OVERLAYS_DIR}/ and {SEGMENTS_DIR}/")
            return
    else:
        if not args.mask:
            parser.error("--mask is required when using --overlay")
        overlay = cv2.imread(str(args.overlay))
        mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        layers = [(overlay, mask, "overlay")]

    overlay_video(
        video_path=args.video,
        layers=layers,
        output_path=args.output,
        alpha=args.alpha,
        start_sec=args.start,
        end_sec=args.end,
        tint=not args.no_tint,
    )


if __name__ == "__main__":
    main()
