"""
HuggingFace VLM Client for open-source models
Supports: InternVL3, Qwen2.5-VL, LLaVA-OneVision, Pixtral, Molmo, Kimi-VL, MiniCPM-V
"""

import os
import base64
from typing import Tuple, Dict
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class HuggingFaceVLMClient:
    """Client for HuggingFace-hosted VLMs."""

    # Model configs
    MODEL_CONFIGS = {
        # Open research models
        "internvl3": {
            "hf_id": "OpenGVLab/InternVL3-8B",
            "processor": "auto",
            "chat_template": True
        },
        "qwen2.5-vl": {
            "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "processor": "Qwen/Qwen2.5-VL-7B-Instruct",
            "chat_template": True
        },
        "llava-onevision": {
            "hf_id": "lmms-lab/llava-onevision-qwen2-7b-ov",
            "processor": "auto",
            "chat_template": True
        },
        "pixtral": {
            "hf_id": "mistral-community/pixtral-12b",
            "processor": "auto",
            "chat_template": True
        },
        "molmo": {
            "hf_id": "allenai/Molmo-7B-D-0924",
            "processor": "auto",
            "chat_template": True
        },
        "kimi-vl": {
            "hf_id": "Pro/Kimi-VL-7B",
            "processor": "auto",
            "chat_template": True
        },
        # Edge deployable
        "qwen-vl-7b": {
            "hf_id": "Qwen/Qwen-VL-Chat",
            "processor": "Qwen/Qwen-VL-Chat",
            "chat_template": True
        },
        "minicpm-v": {
            "hf_id": "openbmb/MiniCPM-V-2_6",
            "processor": "auto",
            "chat_template": True
        },
        "phi-multimodal": {
            "hf_id": "microsoft/Phi-3-vision-128k-instruct",
            "processor": "auto",
            "chat_template": True
        }
    }

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize HuggingFace VLM client.

        Args:
            model_name: Short name from MODEL_CONFIGS
            device: Device for inference
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        config = self.MODEL_CONFIGS[model_name]
        self.hf_id = config["hf_id"]
        self.device = device

        print(f"Loading {model_name} from {self.hf_id}...")

        # Load model and processor
        try:
            self.model = AutoModel.from_pretrained(
                self.hf_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(device)

            processor_id = config["processor"] if config["processor"] != "auto" else self.hf_id
            self.processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)

            print(f"✓ Loaded {model_name}")

        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            print(f"  Falling back to API mode or skipping")
            self.model = None
            self.processor = None

    def query_distance(
        self,
        image_path: str,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        prompt: str
    ) -> Dict:
        """
        Query HF VLM for distance estimation.

        Args:
            image_path: Path to marked image
            point1: (u, v) coordinates of point A
            point2: (u, v) coordinates of point B
            prompt: Prompt text

        Returns:
            Dict with predicted_distance, raw_response, model
        """
        if self.model is None or self.processor is None:
            return {
                "predicted_distance": None,
                "raw_response": f"Model {self.model_name} not loaded",
                "model": self.hf_id
            }

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False
                )

            # Decode response
            raw_response = self.processor.decode(outputs[0], skip_special_tokens=True)
            raw_response = raw_response.strip()

            # Parse numeric response
            import re
            numbers = re.findall(r'\d+\.?\d*', raw_response)
            predicted_distance = float(numbers[0]) if numbers else None

            return {
                "predicted_distance": predicted_distance,
                "raw_response": raw_response,
                "model": self.hf_id
            }

        except Exception as e:
            return {
                "predicted_distance": None,
                "raw_response": f"Error: {str(e)}",
                "model": self.hf_id
            }
