"""
Google Gemini VLM Client
"""

import os
import base64
from typing import Tuple, Dict
import google.generativeai as genai


class GeminiVLMClient:
    """Client for Google Gemini models."""

    def __init__(self, model: str = "gemini-1.5-pro"):
        """
        Initialize Gemini client.

        Args:
            model: Gemini model name
        """
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    def query_distance(
        self,
        image_path: str,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        prompt: str
    ) -> Dict:
        """
        Query Gemini for distance estimation.

        Args:
            image_path: Path to marked image
            point1: (u, v) coordinates of point A
            point2: (u, v) coordinates of point B
            prompt: Prompt text

        Returns:
            Dict with predicted_distance, raw_response, model
        """
        try:
            # Load image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Create image part
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_data
            }

            # Query model
            response = self.client.generate_content([prompt, image_part])

            raw_response = response.text.strip()

            # Parse numeric response
            import re
            numbers = re.findall(r'\d+\.?\d*', raw_response)
            predicted_distance = float(numbers[0]) if numbers else None

            return {
                "predicted_distance": predicted_distance,
                "raw_response": raw_response,
                "model": self.model
            }

        except Exception as e:
            return {
                "predicted_distance": None,
                "raw_response": f"Error: {str(e)}",
                "model": self.model
            }
