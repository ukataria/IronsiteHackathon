"""SD 1.5 Inpainting pipeline — generate finished-state overlays per element layer."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from src.prompts import NEGATIVE_PROMPT
from src.utils import OVERLAYS_DIR

log = logging.getLogger(__name__)

_inpaint_pipe = None


def _get_pipe():
    """Lazy-load SD 1.5 Inpainting pipeline (~2GB, fits on constrained disks)."""
    global _inpaint_pipe
    if _inpaint_pipe is not None:
        return _inpaint_pipe

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    log.info(f"Loading SD 1.5 Inpainting on {device}")
    _inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
    ).to(device)
    _inpaint_pipe.enable_attention_slicing()
    log.info("SD 1.5 Inpainting loaded.")
    return _inpaint_pipe


def inpaint_layer(
    frame_path: str | Path,
    mask_path: str | Path,
    prompt: str,
    stem: str,
    layer_name: str,
    strength: float = 0.99,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 30,
    seed: int = 42,
) -> Path:
    """Run inpainting on one element layer.

    Args:
        frame_path: Original construction frame.
        mask_path: Binary mask (white=inpaint, black=keep).
        prompt: Text description of what should appear in the masked region.
        stem: Frame stem for output naming.
        layer_name: e.g. 'walls', 'floor', 'ceiling'.
        strength: How much to deviate from original (0.99 = near-full repaint).
        guidance_scale: CFG scale (higher = more prompt adherent).
        num_inference_steps: Diffusion steps.
        seed: RNG seed for reproducibility.

    Returns:
        Path to saved overlay image.
    """
    import torch

    out_path = OVERLAYS_DIR / f"{stem}_{layer_name}.png"
    if out_path.exists():
        log.info(f"Overlay cache hit: {out_path}")
        return out_path

    pipe = _get_pipe()

    image = PILImage.open(str(frame_path)).convert("RGB")
    mask = PILImage.open(str(mask_path)).convert("L")

    # SD 1.5 inpainting wants 512x512 — resize, inpaint, resize back
    orig_size = image.size
    target_size = (512, 512)
    image_resized = image.resize(target_size, PILImage.LANCZOS)
    mask_resized = mask.resize(target_size, PILImage.NEAREST)

    generator = torch.Generator().manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image_resized,
        mask_image=mask_resized,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    result = result.resize(orig_size, PILImage.LANCZOS)
    result.save(str(out_path))
    log.info(f"Overlay saved: {out_path}")
    return out_path


def inpaint_all_layers(
    frame_path: str | Path,
    mask_paths: dict[str, Path],
    prompts: dict[str, str],
    stem: str,
    seed: int = 42,
) -> dict[str, Path]:
    """Run inpainting for all available layers.

    Args:
        frame_path: Original frame.
        mask_paths: Dict of layer_name → mask path.
        prompts: Dict of layer_name → inpaint prompt.
        stem: Frame stem.
        seed: Shared seed across layers for consistency.

    Returns:
        Dict of layer_name → overlay path.
    """
    results = {}
    for layer_name, mask_path in mask_paths.items():
        prompt = prompts.get(layer_name, f"finished {layer_name}, photorealistic, 8k")
        log.info(f"Inpainting layer: {layer_name}")
        overlay_path = inpaint_layer(
            frame_path=frame_path,
            mask_path=mask_path,
            prompt=prompt,
            stem=stem,
            layer_name=layer_name,
            seed=seed,
        )
        results[layer_name] = overlay_path
    return results
