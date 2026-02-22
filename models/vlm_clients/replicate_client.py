"""
Replicate VLM Client

Supports various vision models via Replicate API.
"""

import os
import base64
from typing import Tuple


class ReplicateVLMClient:
    """Client for Replicate-hosted vision models."""

    def __init__(self, model: str = "meta/meta-llama-3.2-90b-vision-instruct"):
        """
        Initialize Replicate client.

        Args:
            model: Model identifier (e.g., "meta/meta-llama-3.2-90b-vision-instruct")
        """
        try:
            import replicate
        except ImportError:
            raise ImportError("replicate package not installed. Run: pip install replicate")

        api_key = os.getenv("REPLICATE_API_TOKEN")
        if not api_key:
            raise ValueError("REPLICATE_API_TOKEN environment variable not set")

        self.client = replicate.Client(api_token=api_key)
        self.model = model

    def query_distance(
        self,
        image_path: str,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        prompt: str
    ) -> dict:
        """
        Query distance between two points.

        Args:
            image_path: Path to image
            point1: (x, y) first point
            point2: (x, y) second point
            prompt: Text prompt

        Returns:
            Dict with predicted_distance, raw_response, model
        """
        # Encode image as data URI
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        image_uri = f"data:image/jpeg;base64,{image_b64}"

        try:
            output = self.client.run(
                self.model,
                input={
                    "image": image_uri,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.0
                }
            )

            # Replicate returns a generator, join the output
            if hasattr(output, '__iter__'):
                raw_response = "".join(output).strip()
            else:
                raw_response = str(output).strip()

            # Parse numeric response
            try:
                import re
                numbers = re.findall(r"[-+]?\d*\.?\d+", raw_response)
                if numbers:
                    predicted_distance = float(numbers[0])
                else:
                    predicted_distance = None
            except:
                predicted_distance = None

            return {
                "predicted_distance": predicted_distance,
                "raw_response": raw_response,
                "model": self.model
            }

        except Exception as e:
            print(f"Replicate API error: {e}")
            return {
                "predicted_distance": None,
                "raw_response": f"Error: {str(e)}",
                "model": self.model
            }
