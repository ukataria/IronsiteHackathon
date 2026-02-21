"""ControlNet depth-conditioned generation for spatially accurate overlays."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from src.prompts import NEGATIVE_PROMPT
from src.utils import OVERLAYS_DIR, load_depth

log = logging.getLogger(__name__)

_controlnet_pipe = None


def _get_controlnet_pipe():
    """Lazy-load ControlNet + SDXL pipeline."""
    global _controlnet_pipe
    if _controlnet_pipe is not None:
        return _controlnet_pipe

    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    log.info(f"Loading ControlNet depth + SDXL on {device}")

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=dtype,
    )

    _controlnet_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    ).to(device)

    _controlnet_pipe.enable_attention_slicing()
    log.info("ControlNet pipeline loaded.")
    return _controlnet_pipe


def depth_to_controlnet_image(depth: np.ndarray) -> PILImage.Image:
    """Normalize and convert depth array to a ControlNet-compatible RGB image."""
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    # ControlNet depth expects RGB where all channels are the same
    depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
    return PILImage.fromarray(depth_rgb)


def generate_depth_conditioned(
    frame_path: str | Path,
    mask_path: str | Path,
    depth_stem: str,
    prompt: str,
    stem: str,
    layer_name: str,
    controlnet_conditioning_scale: float = 0.8,
    strength: float = 0.95,
    guidance_scale: float = 8.5,
    num_inference_steps: int = 30,
    seed: int = 42,
) -> Path:
    """Generate a finished-state overlay conditioned on the depth map.

    This ensures the generated content respects the 3D geometry of the scene.

    Args:
        frame_path: Original construction frame.
        mask_path: Binary inpainting mask.
        depth_stem: Stem to load depth .npy from data/depth/.
        prompt: Text prompt for the finished element.
        stem: Frame stem for output naming.
        layer_name: Layer name (walls, floor, ceiling, etc.).
        controlnet_conditioning_scale: How strongly depth influences generation.
        strength: Inpainting strength.
        guidance_scale: CFG scale.
        num_inference_steps: Diffusion steps.
        seed: RNG seed.

    Returns:
        Path to saved depth-conditioned overlay.
    """
    import torch

    out_path = OVERLAYS_DIR / f"{stem}_{layer_name}_depth.png"
    if out_path.exists():
        log.info(f"Depth-conditioned overlay cache hit: {out_path}")
        return out_path

    pipe = _get_controlnet_pipe()
    depth = load_depth(depth_stem)

    image = PILImage.open(str(frame_path)).convert("RGB")
    mask = PILImage.open(str(mask_path)).convert("L")
    depth_image = depth_to_controlnet_image(depth)

    orig_size = image.size
    target_size = (1024, 1024)
    image_r = image.resize(target_size, PILImage.LANCZOS)
    mask_r = mask.resize(target_size, PILImage.NEAREST)
    depth_r = depth_image.resize(target_size, PILImage.LANCZOS)

    generator = torch.Generator().manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image_r,
        mask_image=mask_r,
        control_image=depth_r,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    result = result.resize(orig_size, PILImage.LANCZOS)
    result.save(str(out_path))
    log.info(f"Depth-conditioned overlay saved: {out_path}")
    return out_path
