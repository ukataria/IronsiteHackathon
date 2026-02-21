"""End-to-end Ghost Blueprint pipeline: frame → depth → segment → describe → inpaint → composite."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils import ensure_dirs, frame_stem, FRAMES_DIR, DEPTH_DIR, SEGMENTS_DIR, SCENES_DIR, OVERLAYS_DIR, COMPOSITES_DIR
from src.video.extract import process_video
from src.depth.estimate import estimate_and_save, batch_estimate
from src.segmentation.segment import run_segmentation
from src.scene.describe import describe_scene
from src.scene.predict import predict_finished_state, extract_inpaint_prompts
from src.inpaint.generate import inpaint_all_layers
from src.composite.blend import composite_layers, make_side_by_side

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_LAYER_ALPHAS = {
    "walls": 0.45,
    "floor": 0.4,
    "ceiling": 0.4,
    "electrical": 0.6,
    "plumbing": 0.5,
    "hvac": 0.5,
    "fixtures": 0.55,
}


def run_frame(
    frame_path: Path,
    depth_path: Path | None = None,
    mask_paths: dict[str, Path] | None = None,
    layer_alphas: dict[str, float] | None = None,
    use_controlnet: bool = False,
    seed: int = 42,
) -> dict:
    """Run the full pipeline on a single pre-extracted frame.

    Args:
        frame_path: Path to input frame PNG.
        depth_path: Optional pre-computed depth PNG (skips depth estimation).
        mask_paths: Optional pre-computed masks (skips segmentation).
        layer_alphas: Per-layer alpha overrides.
        use_controlnet: Use depth-conditioned ControlNet inpainting.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with paths to all outputs.
    """
    ensure_dirs()
    stem = frame_path.stem
    alphas = {**DEFAULT_LAYER_ALPHAS, **(layer_alphas or {})}

    # -- Step 1: Depth estimation --
    if depth_path is None:
        log.info(f"[{stem}] Step 1: Depth estimation")
        depth_npy_path, depth_png_path = estimate_and_save(frame_path, stem=stem)
    else:
        depth_npy_path = depth_path
        depth_png_path = depth_path

    # -- Step 2: Segmentation --
    if mask_paths is None:
        log.info(f"[{stem}] Step 2: Segmentation")
        mask_paths = run_segmentation(frame_path, stem=stem)

    # -- Step 3: Scene understanding --
    log.info(f"[{stem}] Step 3: Scene description")
    scene = describe_scene(frame_path, depth_path=depth_png_path)

    log.info(f"[{stem}] Step 3b: Future state prediction")
    future = predict_finished_state(scene, stem=stem)
    prompts = extract_inpaint_prompts(future)

    # -- Step 4: Inpainting --
    log.info(f"[{stem}] Step 4: Inpainting layers")
    if use_controlnet and depth_npy_path.exists():
        from src.inpaint.controlnet import generate_depth_conditioned
        overlay_paths = {}
        for layer_name, mask_path in mask_paths.items():
            overlay_paths[layer_name] = generate_depth_conditioned(
                frame_path=frame_path,
                mask_path=mask_path,
                depth_stem=stem,
                prompt=prompts.get(layer_name, f"finished {layer_name}, photorealistic, 8k"),
                stem=stem,
                layer_name=layer_name,
                seed=seed,
            )
    else:
        overlay_paths = inpaint_all_layers(
            frame_path=frame_path,
            mask_paths=mask_paths,
            prompts=prompts,
            stem=stem,
            seed=seed,
        )

    # -- Step 5: Compositing --
    log.info(f"[{stem}] Step 5: Compositing")
    composite_path = composite_layers(
        base_path=frame_path,
        overlay_paths=overlay_paths,
        mask_paths=mask_paths,
        layer_alphas=alphas,
        stem=stem,
    )

    sbs_path = make_side_by_side(frame_path, composite_path, stem=stem)

    result = {
        "stem": stem,
        "frame": frame_path,
        "depth_npy": depth_npy_path,
        "depth_png": depth_png_path,
        "masks": mask_paths,
        "scene": scene,
        "future": future,
        "overlays": overlay_paths,
        "composite": composite_path,
        "side_by_side": sbs_path,
    }
    log.info(f"[{stem}] Pipeline complete. Composite: {composite_path}")
    return result


def run_video(
    video_path: str | Path,
    clip_id: str | None = None,
    fps_sample: float = 1.0,
    top_n: int = 8,
    use_controlnet: bool = False,
    seed: int = 42,
) -> list[dict]:
    """Run the full pipeline on a video file.

    Extracts frames, selects keyframes, runs pipeline on each.
    """
    ensure_dirs()
    video_path = Path(video_path)
    if clip_id is None:
        clip_id = video_path.stem

    log.info(f"Processing video: {video_path}")
    extracted = process_video(video_path, clip_id=clip_id, fps_sample=fps_sample, top_n=top_n)
    keyframes = extracted["keyframes"]

    log.info(f"Running pipeline on {len(keyframes)} keyframes")
    results = []
    for kf in keyframes:
        result = run_frame(kf, use_controlnet=use_controlnet, seed=seed)
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost Blueprint — X-Ray Vision for Construction")
    sub = parser.add_subparsers(dest="command")

    # video command
    p_video = sub.add_parser("video", help="Process a video file end-to-end")
    p_video.add_argument("video_path", type=Path)
    p_video.add_argument("--clip-id", default=None)
    p_video.add_argument("--fps", type=float, default=1.0)
    p_video.add_argument("--top-n", type=int, default=8)
    p_video.add_argument("--controlnet", action="store_true")
    p_video.add_argument("--seed", type=int, default=42)

    # frame command
    p_frame = sub.add_parser("frame", help="Process a single frame image")
    p_frame.add_argument("frame_path", type=Path)
    p_frame.add_argument("--controlnet", action="store_true")
    p_frame.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "video":
        run_video(args.video_path, args.clip_id, args.fps, args.top_n, args.controlnet, args.seed)
    elif args.command == "frame":
        run_frame(args.frame_path, use_controlnet=args.controlnet, seed=args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
