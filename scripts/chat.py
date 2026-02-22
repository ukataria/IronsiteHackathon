# scripts/chat.py
# Usage:
#   uv run python scripts/chat.py <image>
#   uv run python scripts/chat.py <image> ollama
#   uv run python scripts/chat.py <image> ollama:gemma3:27b-it-q8_0
import json, sys
from src.vlm.clients import InspectionSession

image = sys.argv[1]
measurements = json.load(open(f"data/measurements/{__import__('pathlib').Path(image).stem}_measurements.json"))

# Parse optional "provider" or "provider:model" argument
provider, model = "claude", None
if len(sys.argv) > 2:
    parts = sys.argv[2].split(":", 1)
    provider = parts[0]
    model = parts[1] if len(parts) > 1 else None

session = InspectionSession(image, measurements, provider=provider, model=model)
print(f"[chat] provider={provider}  model={session.model}")

started = False
while True:
    q = input("\nYou: ").strip()
    if not q:
        continue
    if q.lower() in ("exit", "quit", "q"):
        break
    if not started:
        print("\nClaude:", session.start(q))
        started = True
    else:
        print("\nClaude:", session.ask(q))
