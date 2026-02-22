#!/usr/bin/env python3
"""Quick test script to verify VLM clients are working."""

import os
from dotenv import load_dotenv
from models.vlm_clients.openai_client import OpenAIVLMClient
from models.vlm_clients.anthropic_client import AnthropicVLMClient

# Load API keys
load_dotenv()

print("Testing VLM Client Setup")
print("=" * 60)

# Check API keys
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

if openai_key:
    print(f"✓ OpenAI API key found (starts with: {openai_key[:10]}...)")
else:
    print("✗ OpenAI API key not found")

if anthropic_key:
    print(f"✓ Anthropic API key found (starts with: {anthropic_key[:10]}...)")
else:
    print("✗ Anthropic API key not found")

print("\n" + "=" * 60)

# Try initializing clients
try:
    openai_client = OpenAIVLMClient(model="gpt-4o")
    print("✓ OpenAI client initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize OpenAI client: {e}")

try:
    anthropic_client = AnthropicVLMClient(model="claude-3-5-sonnet-20241022")
    print("✓ Anthropic client initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize Anthropic client: {e}")

print("\n" + "=" * 60)
print("\nSetup verification complete!")
print("\nNext steps:")
print("1. Create data/processed/frames.csv with your ARKitScenes data")
print("2. Run: python eval/runners/eval_vlm_points.py --frames_csv data/processed/frames.csv --model_type openai")
