"""LLM scene understanding — current state description."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.llm import claude_vision, gemini_vision
from src.prompts import SCENE_DESCRIBE_PROMPT
from src.utils import SCENES_DIR, save_json, load_json

log = logging.getLogger(__name__)


def describe_scene(
    frame_path: str | Path,
    depth_path: str | Path | None = None,
    use_gemini: bool = False,
) -> dict:
    """Generate structured scene description from a frame.

    Args:
        frame_path: Path to the construction frame image.
        depth_path: Optional path to depth visualization PNG.
        use_gemini: Use Gemini instead of Claude (fallback).

    Returns:
        Parsed scene description dict.
    """
    frame_path = Path(frame_path)
    stem = frame_path.stem
    out_path = SCENES_DIR / f"{stem}_scene.json"

    if out_path.exists():
        log.info(f"Scene cache hit: {out_path}")
        return load_json(out_path)

    # Build prompt — optionally mention depth map availability
    prompt = SCENE_DESCRIBE_PROMPT
    if depth_path:
        prompt = (
            "You are being given a construction frame image. "
            "A depth map is also available (brighter = closer to camera).\n\n"
            + SCENE_DESCRIBE_PROMPT
        )

    if use_gemini:
        result = gemini_vision(prompt, frame_path)
    else:
        result = claude_vision(prompt, frame_path)

    if isinstance(result, str):
        log.warning(f"LLM returned raw string for {stem}, wrapping.")
        result = {"raw": result}

    save_json(result, out_path)
    log.info(f"Scene saved: {out_path}")
    return result


def batch_describe(
    frame_paths: list[Path],
    use_gemini: bool = False,
) -> list[dict]:
    """Describe multiple frames sequentially (use asyncio version for speed)."""
    results = []
    for i, fp in enumerate(frame_paths):
        log.info(f"Describing [{i+1}/{len(frame_paths)}]: {fp.name}")
        result = describe_scene(fp, use_gemini=use_gemini)
        results.append(result)
    return results


async def batch_describe_async(
    frame_paths: list[Path],
    use_gemini: bool = False,
) -> list[dict]:
    """Describe multiple frames in parallel using asyncio."""
    import asyncio

    async def _describe_one(fp: Path) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, describe_scene, fp, None, use_gemini)

    return await asyncio.gather(*[_describe_one(fp) for fp in frame_paths])
