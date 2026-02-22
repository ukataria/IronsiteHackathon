"""
Balance the OUTLET COCO dataset to match the Bricks dataset size per split.

Reads:  OUTLET.v1i.coco/{train,valid,test}/_annotations.coco.json
Reads:  Bricks.v1i.coco/{train,valid,test}/_annotations.coco.json  (for target counts)
Writes: OUTLET.v1i.coco.balanced/{train,valid,test}/_annotations.coco.json

Usage:
    uv run python scripts/balance_outlet_dataset.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

BRICKS_DIR = Path("Bricks.v1i.coco")
OUTLET_DIR = Path("OUTLET.v1i.coco")
OUT_DIR = Path("OUTLET.v1i.coco.balanced")
SPLITS = ["train", "valid", "test"]


def subsample_coco(
    coco: dict,
    n: int,
    rng: random.Random,
) -> dict:
    """Return a new COCO dict with exactly n randomly sampled images and their annotations."""
    images = coco["images"]
    if n >= len(images):
        return coco  # nothing to do

    sampled_images = rng.sample(images, n)
    sampled_ids = {img["id"] for img in sampled_images}
    sampled_annotations = [a for a in coco["annotations"] if a["image_id"] in sampled_ids]

    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": sampled_images,
        "annotations": sampled_annotations,
    }


def main(seed: int) -> None:
    rng = random.Random(seed)

    for split in SPLITS:
        bricks_json = BRICKS_DIR / split / "_annotations.coco.json"
        outlet_json = OUTLET_DIR / split / "_annotations.coco.json"
        out_json = OUT_DIR / split / "_annotations.coco.json"

        bricks = json.loads(bricks_json.read_text())
        outlet = json.loads(outlet_json.read_text())

        target = len(bricks["images"])
        before = len(outlet["images"])

        balanced = subsample_coco(outlet, target, rng)

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(balanced, indent=2))

        after = len(balanced["images"])
        ann_after = len(balanced["annotations"])
        print(
            f"[{split}] Outlet {before} → {after} images "
            f"({ann_after} annotations) to match Bricks {target}"
        )

        # Copy the corresponding images into the balanced dir
        outlet_img_dir = OUTLET_DIR / split / "images"
        out_img_dir = OUT_DIR / split / "images"
        if outlet_img_dir.exists():
            out_img_dir.mkdir(parents=True, exist_ok=True)
            sampled_filenames = {img["file_name"] for img in balanced["images"]}
            copied = 0
            for fname in sampled_filenames:
                src = outlet_img_dir / fname
                dst = out_img_dir / fname
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1
            print(f"         Copied {copied} images → {out_img_dir}")

    print(f"\nBalanced dataset written to: {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args.seed)
