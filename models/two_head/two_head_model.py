"""
Two-Head Model: Complete PreCheck Architecture

Combines:
1. Perception Head (depth + anchors)
2. Measurement Head (calibrated measurements)
3. Reasoning Head (VLM with injected measurements)
"""

import numpy as np
from typing import Dict, List, Optional
from PIL import Image
import json

from .perception_head import PerceptionHead, AnchorDetection
from .measurement_head import MeasurementHead, Measurement, DepthPlane
from models.vlm_clients.openai_client import OpenAIVLMClient
from models.vlm_clients.anthropic_client import AnthropicVLMClient


class TwoHeadModel:
    """
    Complete two-head architecture for construction inspection.

    Architecture:
        Image → Perception Head → Measurement Head → Reasoning Head → Inspection Report
    """

    def __init__(
        self,
        vlm_provider: str = "anthropic",  # "openai" or "anthropic"
        vlm_model: str = "claude-sonnet-4-20250514",
        depth_model: str = "depth_anything_v2",
        anchor_detector: str = "simple_heuristic",
        device: str = "cpu"
    ):
        """
        Initialize two-head model.

        Args:
            vlm_provider: Which VLM provider to use
            vlm_model: Specific VLM model
            depth_model: Depth estimation model
            anchor_detector: Anchor detection method
            device: Device for inference
        """
        # Perception head (frozen)
        self.perception = PerceptionHead(
            depth_model=depth_model,
            anchor_detector=anchor_detector,
            device=device
        )

        # Measurement head (can be learned later, but starts as pure math)
        self.measurement = MeasurementHead()

        # Reasoning head (VLM)
        if vlm_provider == "openai":
            self.reasoning = OpenAIVLMClient(model=vlm_model)
        elif vlm_provider == "anthropic":
            self.reasoning = AnthropicVLMClient(model=vlm_model)
        else:
            raise ValueError(f"Unknown VLM provider: {vlm_provider}")

        self.vlm_provider = vlm_provider

    def _format_measurements_for_prompt(
        self,
        planes: List[DepthPlane],
        measurements: List[Measurement]
    ) -> str:
        """
        Format calibrated measurements for VLM prompt.

        Follows the structure from Technical.md Stage 6.

        Args:
            planes: Calibrated depth planes
            measurements: Extracted measurements

        Returns:
            Formatted measurement text
        """
        lines = []

        # Calibration info
        if planes:
            primary_plane = planes[0]  # Use first/closest plane
            lines.append("CALIBRATED SPATIAL MEASUREMENTS (via anchor calibration):")
            lines.append(f"Calibration anchor: {primary_plane.anchors[0].class_name}")
            lines.append(f"Calibration confidence: {primary_plane.confidence:.2f} ({len(primary_plane.anchors)} anchors)")
            lines.append(f"Scale: {primary_plane.pixels_per_inch:.1f} pixels per inch at primary plane")
            lines.append("")

        # Element measurements
        if measurements:
            lines.append("Element measurements:")
            for i, m in enumerate(measurements):
                if m.measurement_type == "spacing":
                    lines.append(f"  Spacing {i+1}: {m.value_inches:.1f} inches ({m.from_object} → {m.to_object})")
            lines.append("")

        # Construction standards (from Technical.md)
        lines.append("APPLICABLE STANDARDS:")
        lines.append("  Stud spacing: 16\" on center (tolerance: ±0.5\")")
        lines.append("  Outlet box height: 12\" to center (tolerance: ±1\")")
        lines.append("")

        return "\n".join(lines)

    def query_with_measurements(
        self,
        image_path: str,
        planes: List[DepthPlane],
        measurements: List[Measurement],
        question: str = "Are there any code violations or deficiencies in this framing?"
    ) -> Dict:
        """
        Query VLM with injected calibrated measurements.

        Args:
            image_path: Path to construction image
            planes: Calibrated depth planes
            measurements: Extracted measurements
            question: Inspection question

        Returns:
            VLM response dict
        """
        # Format measurements
        measurements_text = self._format_measurements_for_prompt(planes, measurements)

        # Build prompt (follows Technical.md Stage 6 template)
        prompt = f"""You are a construction inspection AI. You have been provided with an image of construction work and calibrated spatial measurements extracted from that image using known reference dimensions.

{measurements_text}

TASK: {question}

Generate an inspection report with:
1. Per-element pass/fail with measurements
2. List of any deficiencies with precise locations
3. Overall pass/fail recommendation

Respond in JSON format:
{{
  "elements": [
    {{"name": "...", "status": "pass/fail", "measurement": "...", "notes": "..."}}
  ],
  "deficiencies": [
    {{"description": "...", "location": "...", "severity": "critical/major/minor"}}
  ],
  "overall": "pass/fail",
  "summary": "..."
}}"""

        # Query VLM
        # For OpenAI/Anthropic clients, we need to adapt the interface
        # They currently expect pixel coords for depth estimation
        # We'll use a simpler text-only query here

        from anthropic import Anthropic
        from openai import OpenAI
        import base64

        # Encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        if self.vlm_provider == "anthropic":
            client = Anthropic()
            response = client.messages.create(
                model=self.reasoning.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
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
            raw_response = response.content[0].text

        else:  # OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.reasoning.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
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
            raw_response = response.choices[0].message.content

        # Try to parse JSON response
        try:
            parsed_response = json.loads(raw_response)
        except:
            parsed_response = {"raw": raw_response}

        return {
            "response": parsed_response,
            "raw_response": raw_response,
            "measurements_provided": measurements_text,
            "num_measurements": len(measurements),
            "calibration_confidence": planes[0].confidence if planes else 0.0
        }

    def run(
        self,
        image: np.ndarray,
        image_path: str,
        question: str = "Are there any code violations or deficiencies in this framing?"
    ) -> Dict:
        """
        Run complete two-head pipeline.

        Args:
            image: RGB image array (H, W, 3)
            image_path: Path to image (for VLM)
            question: Inspection question

        Returns:
            Complete analysis including measurements and VLM response
        """
        # Step 1: Perception Head
        depth_map, anchors = self.perception.run(image)

        # Step 2: Measurement Head
        planes, measurements = self.measurement.run(anchors)

        # Step 3: Reasoning Head
        vlm_result = self.query_with_measurements(image_path, planes, measurements, question)

        return {
            "perception": {
                "num_anchors": len(anchors),
                "anchors": [
                    {
                        "class": a.class_name,
                        "confidence": a.confidence,
                        "pixel_width": a.pixel_width,
                        "known_width_inches": a.known_width_inches,
                        "depth": a.depth_value
                    }
                    for a in anchors
                ],
                "depth_map_shape": depth_map.shape
            },
            "measurement": {
                "num_planes": len(planes),
                "num_measurements": len(measurements),
                "planes": [
                    {
                        "depth": p.depth_value,
                        "num_anchors": len(p.anchors),
                        "pixels_per_inch": p.pixels_per_inch,
                        "confidence": p.confidence
                    }
                    for p in planes
                ],
                "measurements": [
                    {
                        "type": m.measurement_type,
                        "value_inches": m.value_inches,
                        "confidence": m.confidence,
                        "from": m.from_object,
                        "to": m.to_object
                    }
                    for m in measurements
                ]
            },
            "reasoning": vlm_result
        }
