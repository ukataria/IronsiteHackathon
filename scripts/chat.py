# scripts/chat.py
import json, sys
from src.vlm.clients import InspectionSession

image = sys.argv[1]
measurements = json.load(open(f"data/measurements/{__import__('pathlib').Path(image).stem}_measurements.json"))

session = InspectionSession(image, measurements)

print("Starting inspection...\n")
print(session.start())

while True:
    q = input("\nYou: ").strip()
    if q.lower() in ("exit", "quit", "q"):
        break
    print("\nClaude:", session.ask(q))
