"""Batch-process multiple images through the full PreCheck pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

FRAMES_DIR = "data/frames"


def already_processed(image_id: str) -> bool:
    """Check if this image has already been fully processed."""
    measurements_path = Path("data/measurements") / f"{image_id}_measurements.json"
    return measurements_path.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run PreCheck pipeline on all frames")
    parser.add_argument("--image-dir", default=FRAMES_DIR, help="Directory of input images")
    parser.add_argument("--vlm", default="claude", choices=["claude", "gpt4o"])
    parser.add_argument("--device", default="cpu", help="torch device: cpu | cuda | mps")
    parser.add_argument("--model-size", default="small", choices=["small", "base", "large"],
                        help="Depth Anything V2 model size")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM stage (faster for debugging)")
    parser.add_argument("--force", action="store_true", help="Reprocess even if output already exists")
    args = parser.parse_args()

    image_paths = (
        list(Path(args.image_dir).glob("*.jpg"))
        + list(Path(args.image_dir).glob("*.jpeg"))
        + list(Path(args.image_dir).glob("*.png"))
    )

    if not image_paths:
        print(f"No images found in {args.image_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {args.image_dir}")

    from pipeline import run_pipeline

    success, skipped, failed = 0, 0, 0
    last_ppi: float = 0.0  # carry forward last valid scale

    for img_path in tqdm(sorted(image_paths), desc="Processing images"):
        image_id = img_path.stem

        if not args.force and already_processed(image_id):
            # Keep last_ppi in sync even for skipped frames
            cal_path = Path("data/calibrations") / f"{image_id}_calibration.json"
            if cal_path.exists():
                import json
                ppi = json.loads(cal_path.read_text()).get("primary_pixels_per_inch", 0.0)
                if ppi > 0:
                    last_ppi = ppi
            tqdm.write(f"  SKIP {image_id} (already processed)")
            skipped += 1
            continue

        try:
            result = run_pipeline(
                str(img_path),
                vlm=args.vlm,
                device=args.device,
                model_size=args.model_size,
                skip_vlm=args.skip_vlm,
                fallback_ppi=last_ppi,
            )
            # Update carry-forward scale if this frame produced a good one
            cal = result.get("stages", {}).get("calibration", {})
            ppi = cal.get("pixels_per_inch", 0.0)
            if ppi > 0:
                last_ppi = ppi
            success += 1
        except Exception as e:
            tqdm.write(f"  FAIL {image_id}: {e}")
            failed += 1

    print(f"\nBatch complete: {success} ok | {skipped} skipped | {failed} failed")


if __name__ == "__main__":
    main()
