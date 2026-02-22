"""Run YOLO on a video file or webcam and display annotated frames live.

Usage:
    uv run python scripts/yolo_video.py data/raw/clip.mp4         # video file
    uv run python scripts/yolo_video.py 0                          # webcam

Controls:
    q / ESC  — quit
    s        — save current annotated frame to data/detections/
    SPACE    — pause / resume

command: uv run python scripts/yolo_video.py brick.mp4.mp4 ../yolo_weights/finetune_2_weights/best.pt 0.6
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_MODEL = "../yolo_weights/finetune__3weights/best.pt"
DEFAULT_CONF = 0.25
WINDOW_NAME = "Ghost Blueprint — YOLO Live"


def run_yolo_video(
    source: str | int,
    model_name: str = DEFAULT_MODEL,
    conf: float = DEFAULT_CONF,
) -> None:
    """Stream YOLO detections over a video source (file path or webcam index)."""
    model = YOLO(model_name)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))  # ms to wait per frame so playback is real-time

    print(f"Source : {source}")
    print(f"Model  : {model_name}  |  conf: {conf}")
    print("Press  q/ESC=quit  s=save frame  SPACE=pause")

    paused = False
    frame_idx = 0
    save_dir = Path("data/detections")
    save_dir.mkdir(parents=True, exist_ok=True)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break
            frame_idx += 1

            results = model(frame, conf=conf, verbose=False)
            annotated = results[0].plot()  # BGR numpy array with boxes drawn

            n = len(results[0].boxes)
            cv2.putText(
                annotated,
                f"frame {frame_idx}  |  {n} detections",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow(WINDOW_NAME, annotated)
        key = cv2.waitKey(1 if paused else delay) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("s"):
            out = save_dir / f"frame_{frame_idx:05d}_yolo.png"
            cv2.imwrite(str(out), annotated)
            print(f"Saved → {out}")
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    raw = sys.argv[1]
    source: str | int = int(raw) if raw.isdigit() else raw
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CONF

    run_yolo_video(source, model, conf)
