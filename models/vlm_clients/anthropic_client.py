"""Anthropic Claude VLM client for depth estimation queries."""

import os
import base64
from typing import Dict, Any, Optional
from anthropic import Anthropic
from PIL import Image
import io


class AnthropicVLMClient:
    """Client for querying Anthropic Claude vision models about depth/distance."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        """
        Initialize Anthropic client.

        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
            api_key: Anthropic API key (defaults to env var ANTHROPIC_API_KEY)
        """
        self.model = model
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def encode_image(self, image_path: str) -> tuple[str, str]:
        """
        Encode image to base64 string and determine media type.

        Returns:
            Tuple of (base64_string, media_type)
        """
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        # Determine media type from file extension
        if image_path.lower().endswith('.png'):
            media_type = "image/png"
        elif image_path.lower().endswith(('.jpg', '.jpeg')):
            media_type = "image/jpeg"
        elif image_path.lower().endswith('.webp'):
            media_type = "image/webp"
        elif image_path.lower().endswith('.gif'):
            media_type = "image/gif"
        else:
            media_type = "image/jpeg"  # default

        return base64.b64encode(image_data).decode("utf-8"), media_type

    def query_point_depth(
        self,
        image_path: str,
        pixel_coords: tuple[int, int],
        intrinsics: Optional[Dict[str, float]] = None,
        prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the VLM for depth at a specific pixel coordinate.

        Args:
            image_path: Path to RGB image
            pixel_coords: (u, v) pixel coordinates (origin top-left)
            intrinsics: Camera intrinsics dict with keys fx, fy, cx, cy
            prompt_template: Custom prompt template (optional)

        Returns:
            Dict with keys: predicted_depth (float or None), raw_response (str)
        """
        u, v = pixel_coords

        # Default prompt
        if prompt_template is None:
            prompt = f"""You are analyzing an indoor scene image.
What is the approximate distance in meters from the camera to the point at pixel coordinates ({u}, {v})?
Pixel coordinates are measured from the top-left corner (origin at top-left).

Please provide ONLY a single number representing the distance in meters.
Examples: "2.5" or "0.8" or "4.2"

Your answer (single number only):"""
        else:
            prompt = prompt_template.format(u=u, v=v, **intrinsics) if intrinsics else prompt_template.format(u=u, v=v)

        # Encode image
        base64_image, media_type = self.encode_image(image_path)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            raw_response = response.content[0].text.strip()

            # Try to parse numeric answer
            try:
                # Extract first number from response
                import re
                numbers = re.findall(r'\d+\.?\d*', raw_response)
                predicted_depth = float(numbers[0]) if numbers else None
            except (ValueError, IndexError):
                predicted_depth = None

            return {
                "predicted_depth": predicted_depth,
                "raw_response": raw_response,
                "model": self.model
            }

        except Exception as e:
            return {
                "predicted_depth": None,
                "raw_response": f"Error: {str(e)}",
                "model": self.model
            }
