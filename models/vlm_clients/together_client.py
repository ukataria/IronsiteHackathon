"""
Together AI VLM Client

Supports Llama 3.2 Vision and other models via Together AI API.
"""

import os
import base64
from typing import Tuple


class TogetherVLMClient:
    """Client for Together AI vision models."""

    def __init__(self, model: str = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"):
        """
        Initialize Together AI client.

        Args:
            model: Model name (e.g., "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo")
        """
        self.model = model
        self.client = None
        self.error_msg = None

        try:
            import together
        except ImportError:
            self.error_msg = "together package not installed. Run: pip install together"
            print(f"⚠️  {self.error_msg}")
            return

        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            self.error_msg = "TOGETHER_API_KEY environment variable not set"
            print(f"⚠️  {self.error_msg}")
            return

        try:
            self.client = together.Together(api_key=api_key)
            print(f"✓ Together AI client initialized with {model}")
        except Exception as e:
            self.error_msg = f"Failed to initialize Together client: {e}"
            print(f"⚠️  {self.error_msg}")

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
        # Check if client is initialized
        if self.client is None:
            return {
                "predicted_distance": None,
                "raw_response": f"Client not initialized: {self.error_msg}",
                "model": self.model
            }

        # Encode image
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

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
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0.0
            )

            raw_response = response.choices[0].message.content.strip()

            # Parse numeric response
            try:
                # Extract first number from response
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
            print(f"Together AI API error: {e}")
            return {
                "predicted_distance": None,
                "raw_response": f"Error: {str(e)}",
                "model": self.model
            }
