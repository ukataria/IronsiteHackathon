#!/usr/bin/env python3
"""Test which Anthropic models are available with the current API key."""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# Common Claude model IDs to try
models_to_test = [
    # Claude 3.5 Sonnet variants
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",

    # Claude 3 Opus
    "claude-3-opus-20240229",

    # Claude 3 Sonnet
    "claude-3-sonnet-20240229",

    # Claude 3 Haiku
    "claude-3-haiku-20240307",
]

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Testing Anthropic Model Access")
print("=" * 60)

for model in models_to_test:
    print(f"\nTesting: {model}")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": "Hi"
                }
            ]
        )
        print(f"  ✓ SUCCESS - Response: {response.content[0].text}")
    except Exception as e:
        error_str = str(e)
        if "not_found_error" in error_str:
            print(f"  ✗ FAILED - Model not found (404)")
        elif "permission" in error_str.lower():
            print(f"  ✗ FAILED - Permission denied")
        else:
            print(f"  ✗ FAILED - {error_str[:100]}")

print("\n" + "=" * 60)
print("\nNote: Models that succeed can be used for vision tasks.")
print("Vision requires adding image content blocks to messages.")
