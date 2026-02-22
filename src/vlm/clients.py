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
    ELEMENT_DIMENSIONS_BLOCK,
    INSPECTION_SYSTEM_PROMPT,
    STANDARDS_BLOCK,
    build_calibration_summary,
    build_chat_opening_prompt,
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
    secondary_image_path: str | None = None,
    system: str = INSPECTION_SYSTEM_PROMPT,
    cache_key: str = "",
    model: str = "claude-sonnet-4-6",
) -> str:
    """Call Claude API with disk caching. Supports up to two images. Returns response text."""
    import anthropic

    image_b64 = image_to_base64(image_path) if image_path else None
    secondary_b64 = image_to_base64(secondary_image_path) if secondary_image_path else None

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
        if secondary_image_path and secondary_b64:
            suffix = Path(secondary_image_path).suffix.lower()
            media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": secondary_b64},
            })
        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    secondary_key = f":{secondary_b64[:8]}" if secondary_b64 else ""
    try:
        return cached_api_call(
            prompt=f"claude:{model}:{cache_key}{secondary_key}:{prompt}",
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
# Ollama (local, no API key)
# ---------------------------------------------------------------------------


def call_ollama(
    prompt: str,
    image_path: str | None = None,
    system: str = INSPECTION_SYSTEM_PROMPT,
    cache_key: str = "",
    model: str = "llava",
) -> str:
    """Call a local Ollama model via its OpenAI-compatible API. Returns response text."""
    import openai

    image_b64 = image_to_base64(image_path) if image_path else None

    def _call() -> str:
        client = openai.OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )

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
            prompt=f"ollama:{model}:{cache_key}:{prompt}",
            call_fn=_call,
            image_b64=image_b64,
        )
    except Exception as e:
        logger.error(f"Ollama call failed [{cache_key}]: {e}")
        return f"ERROR: Ollama call failed — {e}"


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

    depth_exists = depth_png_path and Path(depth_png_path).exists()

    # Build prompt per condition
    if condition == "baseline":
        prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)
        primary_image = image_path
        secondary_image = None

    elif condition == "depth":
        prompt = DEPTH_AUGMENTED_PROMPT_TEMPLATE.format(question=question)
        primary_image = image_path
        secondary_image = depth_png_path if depth_exists else None

    elif condition == "anchor_calibrated":
        mblock = format_measurements_block(measurements)
        cal_summary = build_calibration_summary(measurements)
        prompt = ANCHOR_CALIBRATED_PROMPT_TEMPLATE.format(
            question=question,
            dimensions_block=ELEMENT_DIMENSIONS_BLOCK,
            calibration_summary=cal_summary,
            measurements_block=mblock,
            standards_block=STANDARDS_BLOCK,
        )
        primary_image = image_path
        secondary_image = depth_png_path if depth_exists else None

    else:
        raise ValueError(f"Unknown condition: {condition}. Use 'baseline', 'depth', or 'anchor_calibrated'.")

    # Select VLM caller — only Claude supports secondary_image natively here
    if vlm == "claude":
        response = call_claude(prompt, primary_image, secondary_image, cache_key=cache_key)
    else:
        # GPT-4o: send primary image only (depth as separate call not yet supported)
        response = call_gpt4o(prompt, primary_image, cache_key=cache_key)

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


# ---------------------------------------------------------------------------
# Multi-turn chat session
# ---------------------------------------------------------------------------


class InspectionSession:
    """
    Multi-turn Claude chat session grounded in a construction image + depth map + measurements.

    Usage:
        session = InspectionSession(image_path, measurements, depth_png_path=depth_path)
        report  = session.start()          # initial inspection report
        answer  = session.ask("Where exactly is the non-compliant mortar joint?")
        answer2 = session.ask("What would remediation look like?")
    """

    def __init__(
        self,
        image_path: str,
        measurements: dict,
        depth_png_path: str | None = None,
        model: str = "claude-sonnet-4-6",
        system: str = INSPECTION_SYSTEM_PROMPT,
    ) -> None:
        self.image_path = image_path
        self.measurements = measurements
        self.depth_png_path = depth_png_path
        self.model = model
        self.system = system
        self.history: list[dict] = []

        suffix = Path(image_path).suffix.lower()
        self._media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        self._image_b64 = image_to_base64(image_path)

        self._depth_b64: str | None = None
        if depth_png_path and Path(depth_png_path).exists():
            self._depth_b64 = image_to_base64(depth_png_path)

    def start(
        self,
        question: str = "Provide a full inspection report.",
    ) -> str:
        """
        Send the opening turn: original image + depth map (if available) + measurements + question.
        Returns the initial inspection report.
        """
        if self.history:
            raise RuntimeError("Session already started. Use ask() for follow-up questions.")

        opening_prompt = build_chat_opening_prompt(self.measurements, question)
        content: list = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self._media_type,
                    "data": self._image_b64,
                },
            },
        ]
        if self._depth_b64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": self._depth_b64,
                },
            })
        content.append({"type": "text", "text": opening_prompt})

        self.history.append({"role": "user", "content": content})
        return self._send()

    def ask(self, question: str) -> str:
        """Send a follow-up question. The image and context stay in history."""
        if not self.history:
            raise RuntimeError("Session not started. Call start() first.")
        self.history.append({"role": "user", "content": question})
        return self._send()

    def _send(self) -> str:
        """POST current history to Claude, append response, return text."""
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.system,
                messages=self.history,
            )
            text = response.content[0].text
        except Exception as e:
            logger.error(f"Claude chat failed: {e}")
            text = f"ERROR: {e}"
        self.history.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        """Clear conversation history so start() can be called again."""
        self.history.clear()
