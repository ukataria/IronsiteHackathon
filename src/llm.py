"""Thin API wrappers for Claude and Gemini with prompt caching."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import os

from src.utils import load_cache, save_cache, prompt_hash, image_to_base64

load_dotenv()
log = logging.getLogger(__name__)

_anthropic_client: anthropic.Anthropic | None = None
_gemini_configured = False


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _ensure_gemini() -> None:
    global _gemini_configured
    if not _gemini_configured:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        _gemini_configured = True


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

def claude_vision(
    prompt: str,
    image_path: str | Path,
    model: str = "claude-opus-4-6",
    use_cache: bool = True,
) -> dict | str:
    """Call Claude with an image. Returns parsed JSON dict or raw string."""
    key = prompt_hash(prompt, str(image_path))
    if use_cache:
        cached = load_cache(key)
        if cached:
            log.info(f"Cache hit: {key}")
            return cached

    client = _get_anthropic()
    img_b64 = image_to_base64(image_path)
    suffix = Path(image_path).suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = response.content[0].text
            log.info(f"Claude tokens — in: {response.usage.input_tokens} out: {response.usage.output_tokens}")
            result = _try_parse_json(text)
            if use_cache:
                save_cache(key, result if isinstance(result, dict) else {"raw": result})
            return result
        except Exception as e:
            log.warning(f"Claude attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(2)

    return {"error": "Claude call failed after 2 attempts"}


def claude_text(prompt: str, model: str = "claude-opus-4-6", use_cache: bool = True) -> dict | str:
    """Call Claude with text only."""
    key = prompt_hash(prompt)
    if use_cache:
        cached = load_cache(key)
        if cached:
            log.info(f"Cache hit: {key}")
            return cached

    client = _get_anthropic()
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            log.info(f"Claude tokens — in: {response.usage.input_tokens} out: {response.usage.output_tokens}")
            result = _try_parse_json(text)
            if use_cache:
                save_cache(key, result if isinstance(result, dict) else {"raw": result})
            return result
        except Exception as e:
            log.warning(f"Claude attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(2)

    return {"error": "Claude call failed after 2 attempts"}


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def gemini_vision(
    prompt: str,
    image_path: str | Path,
    model: str = "gemini-2.0-flash",
    use_cache: bool = True,
) -> dict | str:
    """Call Gemini with an image."""
    key = prompt_hash(prompt, f"gemini_{image_path}")
    if use_cache:
        cached = load_cache(key)
        if cached:
            log.info(f"Cache hit: {key}")
            return cached

    _ensure_gemini()
    import PIL.Image
    img = PIL.Image.open(str(image_path))
    m = genai.GenerativeModel(model)

    for attempt in range(2):
        try:
            response = m.generate_content([prompt, img])
            text = response.text
            result = _try_parse_json(text)
            if use_cache:
                save_cache(key, result if isinstance(result, dict) else {"raw": result})
            return result
        except Exception as e:
            log.warning(f"Gemini attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(2)

    return {"error": "Gemini call failed after 2 attempts"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> dict | str:
    """Try to extract and parse JSON from a response string."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON block within the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return text
