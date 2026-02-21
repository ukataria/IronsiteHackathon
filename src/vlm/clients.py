"""Thin VLM API wrappers for Claude and GPT-4o with caching."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.utils import cached_api_call, get_image_id, image_to_base64, load_json, save_json, setup_logger
from src.vlm.prompts import (
    ANCHOR_CALIBRATED_PROMPT_TEMPLATE,
    BASELINE_PROMPT_TEMPLATE,
    DEPTH_AUGMENTED_PROMPT_TEMPLATE,
    INSPECTION_SYSTEM_PROMPT,
    STANDARDS_BLOCK,
    build_calibration_summary,
    format_measurements_block,
)

load_dotenv()

logger = setup_logger("vlm")

# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


def call_claude(
    prompt: str,
    image_path: str | None = None,
    system: str = INSPECTION_SYSTEM_PROMPT,
    cache_key: str = "",
    model: str = "claude-sonnet-4-6",
) -> str:
    """Call Claude API with disk caching. Returns response text."""
    import anthropic

    image_b64 = image_to_base64(image_path) if image_path else None

    def _call() -> str:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        content: list = []
        if image_path:
            suffix = Path(image_path).suffix.lower()
            media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": image_b64},
            })
        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    try:
        return cached_api_call(
            prompt=f"claude:{model}:{cache_key}:{prompt}",
            call_fn=_call,
            image_b64=image_b64,
        )
    except Exception as e:
        logger.error(f"Claude call failed [{cache_key}]: {e}")
        return f"ERROR: Claude call failed — {e}"


# ---------------------------------------------------------------------------
# GPT-4o
# ---------------------------------------------------------------------------


def call_gpt4o(
    prompt: str,
    image_path: str | None = None,
    system: str = INSPECTION_SYSTEM_PROMPT,
    cache_key: str = "",
    model: str = "gpt-4o",
) -> str:
    """Call OpenAI GPT-4o API with disk caching. Returns response text."""
    import openai

    image_b64 = image_to_base64(image_path) if image_path else None

    def _call() -> str:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        user_content: list = []
        if image_path:
            suffix = Path(image_path).suffix.lower()
            media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
            })
        user_content.append({"type": "text", "text": prompt})

        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or ""

    try:
        return cached_api_call(
            prompt=f"openai:{model}:{cache_key}:{prompt}",
            call_fn=_call,
            image_b64=image_b64,
        )
    except Exception as e:
        logger.error(f"GPT-4o call failed [{cache_key}]: {e}")
        return f"ERROR: GPT-4o call failed — {e}"


# ---------------------------------------------------------------------------
# Unified inspection runner
# ---------------------------------------------------------------------------


def run_inspection(
    image_path: str,
    measurements_json_path: str,
    depth_png_path: str | None,
    condition: str,
    vlm: str = "claude",
    output_dir: str = "data/results",
    question: str = "What deficiencies exist in this construction work? Provide a full inspection report.",
) -> dict:
    """
    Run one VLM condition on one image.

    condition: 'baseline' | 'depth' | 'anchor_calibrated'
    vlm: 'claude' | 'gpt4o'

    Saves: {output_dir}/{image_id}_{condition}_{vlm}.json
    Returns the result dict.
    """
    image_id = get_image_id(image_path)
    cache_key = f"{image_id}_{condition}_{vlm}"

    measurements = load_json(measurements_json_path) if Path(measurements_json_path).exists() else {}

    # Build prompt per condition
    if condition == "baseline":
        prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)
        primary_image = image_path
        secondary_image = None

    elif condition == "depth":
        prompt = DEPTH_AUGMENTED_PROMPT_TEMPLATE.format(question=question)
        primary_image = image_path
        secondary_image = depth_png_path  # passed separately below

    elif condition == "anchor_calibrated":
        mblock = format_measurements_block(measurements)
        cal_summary = build_calibration_summary(measurements)
        prompt = ANCHOR_CALIBRATED_PROMPT_TEMPLATE.format(
            question=question,
            calibration_summary=cal_summary,
            measurements_block=mblock,
            standards_block=STANDARDS_BLOCK,
        )
        primary_image = image_path
        secondary_image = None

    else:
        raise ValueError(f"Unknown condition: {condition}. Use 'baseline', 'depth', or 'anchor_calibrated'.")

    # Select VLM
    caller = call_claude if vlm == "claude" else call_gpt4o

    # For depth condition, we need to send two images — send depth map as the primary for now
    # and include a note; ideally both images would be in one call
    if condition == "depth" and secondary_image and Path(secondary_image).exists():
        response = caller(prompt, secondary_image, cache_key=f"{cache_key}_depth")
    else:
        response = caller(prompt, primary_image, cache_key=cache_key)

    result = {
        "image_id": image_id,
        "condition": condition,
        "vlm": vlm,
        "question": question,
        "response": response,
        "measurements_used": measurements,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / f"{image_id}_{condition}_{vlm}.json")
    save_json(result, out_path)
    logger.info(f"Inspection saved → {out_path}")
    return result
