"""LLM future state prediction — what will this space look like when finished."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.llm import claude_text
from src.prompts import FUTURE_STATE_PROMPT
from src.utils import SCENES_DIR, save_json, load_json

log = logging.getLogger(__name__)


def predict_finished_state(scene_dict: dict, stem: str) -> dict:
    """Given a scene description, predict the finished state.

    Args:
        scene_dict: Structured scene description from describe_scene().
        stem: Frame stem used for caching.

    Returns:
        Finished state prediction dict with inpaint_prompts.
    """
    out_path = SCENES_DIR / f"{stem}_future.json"
    if out_path.exists():
        log.info(f"Future state cache hit: {out_path}")
        return load_json(out_path)

    scene_json_str = json.dumps(scene_dict, indent=2)
    prompt = FUTURE_STATE_PROMPT.format(scene_json=scene_json_str)
    result = claude_text(prompt)

    if isinstance(result, str):
        log.warning(f"LLM returned raw string for {stem} future state, wrapping.")
        result = {"raw": result}

    save_json(result, out_path)
    log.info(f"Future state saved: {out_path}")
    return result


def extract_inpaint_prompts(future_state: dict) -> dict[str, str]:
    """Pull out per-layer inpaint prompts from future state prediction.

    Falls back to sensible defaults if LLM didn't provide them.
    """
    from src.prompts import (
        INPAINT_WALL_PROMPT,
        INPAINT_FLOOR_PROMPT,
        INPAINT_CEILING_PROMPT,
        INPAINT_ELECTRICAL_PROMPT,
    )

    defaults = {
        "walls": INPAINT_WALL_PROMPT,
        "floor": INPAINT_FLOOR_PROMPT,
        "ceiling": INPAINT_CEILING_PROMPT,
        "electrical": INPAINT_ELECTRICAL_PROMPT,
    }

    llm_prompts = future_state.get("inpaint_prompts", {})
    return {layer: llm_prompts.get(layer, default) for layer, default in defaults.items()}


def get_electrical_placements(future_state: dict) -> list[dict]:
    """Extract electrical element placement info (type, height, location)."""
    layers = future_state.get("layers", {})
    return layers.get("electrical", [])
