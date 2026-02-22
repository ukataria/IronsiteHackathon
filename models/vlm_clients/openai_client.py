"""OpenAI VLM client for depth estimation queries."""

import os
import base64
from typing import Dict, Any, Optional
from openai import OpenAI
from PIL import Image
import io


class OpenAIVLMClient:
    """Client for querying OpenAI vision models about depth/distance."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize OpenAI client.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4-turbo")
            api_key: OpenAI API key (defaults to env var OPENAI_API_KEY)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

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

TASK: Estimate the approximate distance in meters from the camera to the point at pixel coordinates ({u}, {v}).
Pixel coordinates are measured from the top-left corner (origin at top-left).

CRITICAL REQUIREMENTS:
- You MUST provide a numeric estimate - refusal is not acceptable
- Even if you're uncertain, make your best guess based on visual cues
- Base your estimate on: object sizes, perspective, typical indoor dimensions
- If the image appears synthetic or unrealistic, still provide your best estimate

OUTPUT FORMAT:
- Respond with ONLY a single number in meters
- NO explanations, NO refusals, NO disclaimers
- Just the number
- Examples: "2.5" or "0.8" or "4.2"

Your distance estimate (number only):"""
        else:
            prompt = prompt_template.format(u=u, v=v, **intrinsics) if intrinsics else prompt_template.format(u=u, v=v)

        # Encode image
        base64_image = self.encode_image(image_path)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=10,  # Reduced to force short numeric responses
                temperature=0.0
            )

            raw_response = response.choices[0].message.content.strip()

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
