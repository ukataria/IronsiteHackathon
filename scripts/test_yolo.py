"""Quick YOLO test — run a model on an image and show/save annotated result."""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Model options (downloaded automatically on first use):
#   yolov8n.pt  ~6MB   fastest, least accurate
#   yolov8s.pt  ~22MB  fast, decent
#   yolov8m.pt  ~50MB  good balance
#   yolov8x.pt  ~130MB slowest, most accurate
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONF = 0.25   # minimum confidence threshold


def run_yolo(image_path: str, model_name: str = DEFAULT_MODEL, conf: float = DEFAULT_CONF) -> None:
    """Run YOLO on image_path, print detections, and save annotated output."""
    model = YOLO("../yolo_weights/finetune_5_weights/best.pt")
    results = model(image_path, conf=conf)
    r = results[0]

    # Print detections
    print(f"\nImage: {image_path}")
    print(f"Model: {model_name}  |  conf threshold: {conf}")
    print(f"Detected {len(r.boxes)} objects:\n")
    for box in r.boxes:
        cls_name = model.names[int(box.cls)]
        confidence = float(box.conf)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        print(f"  [{cls_name}]  conf={confidence:.2f}  box=({x1},{y1}) → ({x2},{y2})")

    # Save annotated image
    image_id = Path(image_path).stem
    out_path = f"data/detections/{image_id}_yolo_annotated.png"
    Path("data/detections").mkdir(parents=True, exist_ok=True)

    annotated = r.plot()                          # BGR numpy array with boxes drawn
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/test_yolo.py <image_path> [model] [conf]")
        print("  model: yolov8n.pt (default), yolov8s.pt, yolov8m.pt, yolov8x.pt")
        print("  conf:  confidence threshold, default 0.25")
        sys.exit(1)

    img = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CONF

    run_yolo(img, model, conf)
