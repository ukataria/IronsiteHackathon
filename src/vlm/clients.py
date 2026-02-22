"""Thin VLM API wrappers for Claude and GPT-4o with caching."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.utils import (
    cached_api_call,
    draw_scale_bar,
    get_image_id,
    image_to_base64,
    load_image,
    load_json,
    save_image,
    save_json,
    setup_logger,
)
from src.vlm.prompts import (
    ANCHOR_CALIBRATED_PROMPT_TEMPLATE,
    BASELINE_PROMPT_TEMPLATE,
    DEPTH_AUGMENTED_PROMPT_TEMPLATE,
    ELEMENT_DIMENSIONS_BLOCK,
    INSPECTION_SYSTEM_PROMPT,
    STANDARDS_BLOCK,
    build_calibration_summary,
    build_chat_opening_prompt,
    build_few_shot_messages,
    build_reference_objects_block,
    format_measurements_block,
    pick_bar_inches,
)

load_dotenv()

logger = setup_logger("vlm")

# ---------------------------------------------------------------------------
# Few-shot example — loaded once, reused for every anchor_calibrated call
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLE_PNG = Path("data/few_shot/example_scale_bar.png")
_FEW_SHOT_REF_IMAGE = Path("test-image-10.jpg")
_FEW_SHOT_REF_MEAS = Path("data/measurements/test-image-10_measurements.json")
_few_shot_b64_cache: str | None = None


def _get_few_shot_image_b64() -> str | None:
    """
    Return base64 of the few-shot example image (test-image-10 + scale bar).
    Generates and saves to data/few_shot/ on first call; cached in memory thereafter.
    Returns None if the reference files are missing.
    """
    global _few_shot_b64_cache
    if _few_shot_b64_cache is not None:
        return _few_shot_b64_cache

    if not _FEW_SHOT_EXAMPLE_PNG.exists():
        if not _FEW_SHOT_REF_IMAGE.exists() or not _FEW_SHOT_REF_MEAS.exists():
            logger.warning("Few-shot reference files missing — skipping few-shot.")
            return None
        try:
            meas = load_json(str(_FEW_SHOT_REF_MEAS))
            ppi = meas.get("scale_pixels_per_inch", 0.0)
            bar_in = pick_bar_inches(meas)
            img = load_image(str(_FEW_SHOT_REF_IMAGE))
            img_with_bar = draw_scale_bar(img, ppi, bar_in)
            _FEW_SHOT_EXAMPLE_PNG.parent.mkdir(parents=True, exist_ok=True)
            save_image(img_with_bar, str(_FEW_SHOT_EXAMPLE_PNG))
            logger.info(f"Few-shot example generated -> {_FEW_SHOT_EXAMPLE_PNG}")
        except Exception as e:
            logger.warning(f"Few-shot example generation failed: {e}")
            return None

    try:
        _few_shot_b64_cache = image_to_base64(str(_FEW_SHOT_EXAMPLE_PNG))
        return _few_shot_b64_cache
    except Exception as e:
        logger.warning(f"Few-shot image load failed: {e}")
        return None

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
    few_shot_messages: list[dict] | None = None,
) -> str:
    """Call Claude API with disk caching. Supports two images and few-shot turns."""
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

        # Few-shot turns go before the actual user message
        messages = list(few_shot_messages) if few_shot_messages else []
        messages.append({"role": "user", "content": content})

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    secondary_key = f":{secondary_b64[:8]}" if secondary_b64 else ""
    fs_key = ":fs" if few_shot_messages else ""
    try:
        return cached_api_call(
            prompt=f"claude:{model}:{cache_key}{secondary_key}{fs_key}:{prompt}",
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
        bar_inches = pick_bar_inches(measurements)
        mblock = format_measurements_block(measurements)
        cal_summary = build_calibration_summary(measurements)
        ref_block = build_reference_objects_block(measurements, bar_inches)
        prompt = ANCHOR_CALIBRATED_PROMPT_TEMPLATE.format(
            question=question,
            dimensions_block=ELEMENT_DIMENSIONS_BLOCK,
            calibration_summary=cal_summary,
            measurements_block=mblock,
            standards_block=STANDARDS_BLOCK,
            reference_objects_block=ref_block,
        )
        # Draw scale bar on a copy of the image so Claude has an in-image ruler
        ppi = measurements.get("scale_pixels_per_inch", 0.0)
        if ppi > 0:
            try:
                img_arr = load_image(image_path)
                img_with_bar = draw_scale_bar(img_arr, ppi, bar_inches)
                bar_img_path = str(Path(output_dir) / f"{image_id}_scale_bar.png")
                save_image(img_with_bar, bar_img_path)
                primary_image = bar_img_path
            except Exception as e:
                logger.warning(f"Scale bar draw failed, using original: {e}")
                primary_image = image_path
        else:
            primary_image = image_path
        secondary_image = depth_png_path if depth_exists else None

    else:
        raise ValueError(f"Unknown condition: {condition}. Use 'baseline', 'depth', or 'anchor_calibrated'.")

    # Build few-shot turns for anchor_calibrated (teaches scale-bar distance reasoning)
    few_shot_msgs: list[dict] | None = None
    if condition == "anchor_calibrated" and vlm == "claude":
        fs_b64 = _get_few_shot_image_b64()
        if fs_b64:
            few_shot_msgs = build_few_shot_messages(fs_b64)

    # Select VLM caller — only Claude supports secondary_image and few-shot natively here
    if vlm == "claude":
        response = call_claude(
            prompt, primary_image, secondary_image,
            cache_key=cache_key, few_shot_messages=few_shot_msgs,
        )
    else:
        # GPT-4o: send primary image only (depth/few-shot not yet supported)
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
        self._bar_inches = pick_bar_inches(measurements)

        # Always send as PNG so scale bar overlay renders cleanly
        self._media_type = "image/png"

        # Draw scale bar on the image in memory; fall back to original on error
        ppi = measurements.get("scale_pixels_per_inch", 0.0)
        try:
            if ppi > 0:
                img_arr = load_image(image_path)
                img_with_bar = draw_scale_bar(img_arr, ppi, self._bar_inches)
                import cv2
                _, buf = cv2.imencode(".png", cv2.cvtColor(img_with_bar, cv2.COLOR_RGB2BGR))
                import base64
                self._image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            else:
                self._image_b64 = image_to_base64(image_path)
        except Exception:
            self._image_b64 = image_to_base64(image_path)

        self._depth_b64: str | None = None
        if depth_png_path and Path(depth_png_path).exists():
            self._depth_b64 = image_to_base64(depth_png_path)

        # Preload few-shot example for scale-bar distance reasoning
        fs_b64 = _get_few_shot_image_b64()
        self._few_shot_messages: list[dict] | None = (
            build_few_shot_messages(fs_b64) if fs_b64 else None
        )

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

        opening_prompt = build_chat_opening_prompt(self.measurements, question, self._bar_inches)
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

        # Prepend few-shot turns so the model has a distance-reasoning example
        # before it sees the actual inspection image
        if self._few_shot_messages:
            self.history.extend(self._few_shot_messages)
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
