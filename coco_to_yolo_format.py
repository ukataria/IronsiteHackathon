"""
COCO to YOLOv8 Format Converter
================================
Converts a COCO-format dataset into YOLOv8 format with automatic
train/valid/test splitting.

Handles two scenarios:
  1. All data in a single folder (e.g. just "train/") — splits it automatically
  2. Pre-split folders (train/valid/test) — converts each as-is

COCO input (single folder):
    dataset/
    └── train/
        ├── _annotations.coco.json
        └── *.jpg / *.png

COCO input (pre-split):
    dataset/
    ├── train/
    ├── valid/
    └── test/

YOLOv8 output:
    dataset_yolo/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Usage:
    python coco_to_yolo.py --input /path/to/coco_dataset --output /path/to/yolo_dataset

    # Custom split ratios (default: 80/15/5):
    python coco_to_yolo.py --input ./dataset --output ./yolo_dataset --train 0.8 --valid 0.15 --test 0.05

    # No test set (just train/valid):
    python coco_to_yolo.py --input ./dataset --output ./yolo_dataset --train 0.85 --valid 0.15 --test 0.0

    # Reproducible split with a seed:
    python coco_to_yolo.py --input ./dataset --output ./yolo_dataset --seed 42

    # Cap to 500 images max (randomly sampled):
    python coco_to_yolo.py --input ./dataset --output ./yolo_dataset --max 500
"""

import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x_min, y_min, width, height] (absolute pixels)
    to YOLO format [cx, cy, w, h] (normalized 0-1).
    """
    x_min, y_min, w, h = [float(v) for v in bbox]
    img_width, img_height = float(img_width), float(img_height)
    cx = (x_min + w / 2) / img_width
    cy = (y_min + h / 2) / img_height
    nw = w / img_width
    nh = h / img_height

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return cx, cy, nw, nh


def load_coco_annotations(coco_dir):
    """
    Find and load the COCO annotation JSON from a directory.
    Returns (coco_data, ann_file_path) or (None, None) if not found.
    """
    coco_dir = Path(coco_dir)
    for candidate in ["_annotations.coco.json", "annotations.json", "instances_default.json"]:
        ann_file = coco_dir / candidate
        if ann_file.exists():
            with open(ann_file, "r") as f:
                return json.load(f), ann_file
    return None, None


def find_image_source(file_name, search_dirs):
    """Search multiple directories for an image file."""
    for d in search_dirs:
        candidate = Path(d) / file_name
        if candidate.exists():
            return candidate
    return None


def stratified_split(images, ann_by_image, cat_id_to_yolo_id, train_ratio, valid_ratio, test_ratio, seed):
    """
    Split images into train/valid/test with stratification by class.
    Ensures each split has a representative proportion of each class.
    Images with no annotations go proportionally into each split.
    """
    rng = random.Random(seed)

    # Determine the "primary class" of each image (most frequent class in that image)
    # This is a simple stratification heuristic
    class_to_images = defaultdict(list)
    no_class_images = []

    for img_id, img_info in images.items():
        anns = ann_by_image.get(img_id, [])
        if not anns:
            no_class_images.append(img_id)
            continue

        # Count classes in this image
        class_counts = defaultdict(int)
        for ann in anns:
            yolo_id = cat_id_to_yolo_id.get(ann["category_id"])
            if yolo_id is not None:
                class_counts[yolo_id] += 1

        if class_counts:
            # Assign to the least-represented class (helps balance minority classes)
            primary_class = min(class_counts, key=lambda c: len(class_to_images[c]))
            class_to_images[primary_class].append(img_id)
        else:
            no_class_images.append(img_id)

    train_ids, valid_ids, test_ids = [], [], []

    # Split each class group proportionally
    for class_id in sorted(class_to_images.keys()):
        img_ids = class_to_images[class_id]
        rng.shuffle(img_ids)
        n = len(img_ids)
        n_train = max(1, round(n * train_ratio))  # at least 1 in train
        n_valid = max(1, round(n * valid_ratio)) if valid_ratio > 0 else 0
        # Ensure we don't exceed total
        if n_train + n_valid > n:
            n_valid = n - n_train
        n_test = n - n_train - n_valid

        train_ids.extend(img_ids[:n_train])
        valid_ids.extend(img_ids[n_train:n_train + n_valid])
        test_ids.extend(img_ids[n_train + n_valid:])

    # Split no-class (negative) images proportionally
    rng.shuffle(no_class_images)
    n = len(no_class_images)
    n_train = round(n * train_ratio)
    n_valid = round(n * valid_ratio)
    train_ids.extend(no_class_images[:n_train])
    valid_ids.extend(no_class_images[n_train:n_train + n_valid])
    test_ids.extend(no_class_images[n_train + n_valid:])

    return train_ids, valid_ids, test_ids


def write_yolo_split(split_name, img_ids, images, ann_by_image, cat_id_to_yolo_id,
                     image_search_dirs, output_dir):
    """Write images and labels for one split to disk."""
    if not img_ids:
        return 0, 0

    img_out = Path(output_dir) / split_name / "images"
    lbl_out = Path(output_dir) / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for img_id in img_ids:
        img_info = images[img_id]
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_img = find_image_source(file_name, image_search_dirs)
        if src_img is None:
            skipped += 1
            continue

        shutil.copy2(src_img, img_out / file_name)

        # Write label file
        label_name = Path(file_name).stem + ".txt"
        annotations = ann_by_image.get(img_id, [])

        with open(lbl_out / label_name, "w") as lf:
            for ann in annotations:
                yolo_id = cat_id_to_yolo_id.get(ann["category_id"])
                if yolo_id is None:
                    continue
                cx, cy, nw, nh = convert_coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                lf.write(f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        converted += 1

    return converted, skipped


def convert_split(coco_dir, output_dir, split_name):
    """
    Convert one pre-existing split (train/valid/test) from COCO to YOLO format.
    Returns the category list if annotations are found, else None.
    """
    coco_split = Path(coco_dir) / split_name
    if not coco_split.exists():
        return None

    coco_data, ann_file = load_coco_annotations(coco_split)
    if coco_data is None:
        print(f"  Skipping '{split_name}' — no annotation JSON found.")
        return None

    print(f"  Converting '{split_name}' using {ann_file.name}...")

    categories = sorted(coco_data["categories"], key=lambda c: c["id"])
    cat_id_to_yolo_id = {cat["id"]: idx for idx, cat in enumerate(categories)}
    images = {img["id"]: img for img in coco_data["images"]}

    ann_by_image = {}
    for ann in coco_data["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    search_dirs = [coco_split, coco_split / "images"]
    converted, skipped = write_yolo_split(
        split_name, list(images.keys()), images, ann_by_image,
        cat_id_to_yolo_id, search_dirs, output_dir
    )

    print(f"    {converted} images converted, {skipped} skipped (not found).")
    return categories


def convert_and_split_single_folder(coco_dir, output_dir, source_split,
                                     train_ratio, valid_ratio, test_ratio, seed,
                                     max_images=None):
    """
    Load annotations from a single COCO folder, then split into train/valid/test.
    Returns the category list or None.
    """
    coco_split = Path(coco_dir) / source_split
    coco_data, ann_file = load_coco_annotations(coco_split)
    if coco_data is None:
        return None

    print(f"  Loading annotations from '{source_split}/{ann_file.name}'...")

    categories = sorted(coco_data["categories"], key=lambda c: c["id"])
    cat_id_to_yolo_id = {cat["id"]: idx for idx, cat in enumerate(categories)}
    images = {img["id"]: img for img in coco_data["images"]}

    ann_by_image = {}
    for ann in coco_data["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Cap dataset size if --max is set
    if max_images is not None and len(images) > max_images:
        rng = random.Random(seed)
        sampled_ids = rng.sample(list(images.keys()), max_images)
        images = {img_id: images[img_id] for img_id in sampled_ids}
        # Filter annotations to only include sampled images
        ann_by_image = {img_id: ann_by_image[img_id] for img_id in sampled_ids if img_id in ann_by_image}
        print(f"  Capped to {max_images} images (from {len(coco_data['images'])} total)")

    # Count per-class totals for logging
    class_counts = defaultdict(int)
    for img_id, anns in ann_by_image.items():
        for ann in anns:
            yolo_id = cat_id_to_yolo_id.get(ann["category_id"])
            if yolo_id is not None:
                class_counts[yolo_id] += 1

    print(f"  Total images: {len(images)}")
    for idx, cat in enumerate(categories):
        print(f"    Class '{cat['name']}': {class_counts.get(idx, 0)} annotations")

    # Stratified split
    print(f"\n  Splitting with ratios — train: {train_ratio}, valid: {valid_ratio}, test: {test_ratio}")
    print(f"  Random seed: {seed}")

    train_ids, valid_ids, test_ids = stratified_split(
        images, ann_by_image, cat_id_to_yolo_id,
        train_ratio, valid_ratio, test_ratio, seed
    )

    search_dirs = [coco_split, coco_split / "images"]

    for split_name, ids in [("train", train_ids), ("valid", valid_ids), ("test", test_ids)]:
        if not ids:
            print(f"\n  '{split_name}': skipped (0 images)")
            continue
        converted, skipped = write_yolo_split(
            split_name, ids, images, ann_by_image,
            cat_id_to_yolo_id, search_dirs, output_dir
        )
        print(f"\n  '{split_name}': {converted} images converted, {skipped} skipped")

        # Per-class breakdown for this split
        split_class_counts = defaultdict(int)
        for img_id in ids:
            for ann in ann_by_image.get(img_id, []):
                yolo_id = cat_id_to_yolo_id.get(ann["category_id"])
                if yolo_id is not None:
                    split_class_counts[yolo_id] += 1
        for idx, cat in enumerate(categories):
            print(f"    Class '{cat['name']}': {split_class_counts.get(idx, 0)} annotations")

    return categories


def create_data_yaml(output_dir, categories):
    """Generate the data.yaml file YOLOv8 expects."""
    output = Path(output_dir)
    names = {idx: cat["name"] for idx, cat in enumerate(categories)}

    yaml_content = f"""# YOLOv8 Dataset Config — auto-generated by coco_to_yolo.py
path: {output.resolve()}
train: train/images
val: valid/images
"""

    if (output / "test" / "images").exists():
        yaml_content += "test: test/images\n"

    yaml_content += f"\nnc: {len(names)}\n"
    yaml_content += f"names: {names}\n"

    yaml_path = output / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n  data.yaml written to {yaml_path}")
    return yaml_path


def detect_dataset_structure(coco_dir):
    """
    Detect whether the dataset has pre-existing splits or is a single folder.
    Returns ('pre_split', [splits]) or ('single', source_folder).
    """
    coco_dir = Path(coco_dir)
    existing_splits = []

    for split in ["train", "valid", "validation", "val", "test"]:
        split_dir = coco_dir / split
        if split_dir.exists():
            coco_data, _ = load_coco_annotations(split_dir)
            if coco_data is not None:
                existing_splits.append(split)

    if len(existing_splits) >= 2:
        return "pre_split", existing_splits
    elif len(existing_splits) == 1:
        return "single", existing_splits[0]
    else:
        # Check if annotations are in the root folder itself
        coco_data, _ = load_coco_annotations(coco_dir)
        if coco_data is not None:
            return "single_root", "."
        return "none", None


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO dataset to YOLOv8 format with automatic train/valid/test splitting."
    )
    parser.add_argument("--input", required=True, help="Path to COCO dataset root folder")
    parser.add_argument("--output", required=True, help="Path for YOLOv8 output folder")
    parser.add_argument("--train", type=float, default=0.80, help="Train split ratio (default: 0.80)")
    parser.add_argument("--valid", type=float, default=0.15, help="Valid split ratio (default: 0.15)")
    parser.add_argument("--test", type=float, default=0.05, help="Test split ratio (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--max", type=int, default=None, help="Max total images to use (randomly sampled before splitting)")
    args = parser.parse_args()

    # Validate split ratios
    total = args.train + args.valid + args.test
    if abs(total - 1.0) > 0.01:
        print(f"Error: Split ratios must sum to 1.0, got {total:.2f}")
        return

    print(f"\nCOCO → YOLOv8 Converter")
    print(f"{'='*40}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Detect dataset structure
    structure, info = detect_dataset_structure(args.input)

    if structure == "none":
        print("\nError: No COCO annotations found in any folder!")
        return

    elif structure == "pre_split":
        print(f"\nDetected pre-split dataset with splits: {info}")
        print("Converting each split as-is...\n")
        categories = None
        for split in info:
            # Normalize split names
            out_split = "valid" if split in ["validation", "val"] else split
            result = convert_split(args.input, args.output, split)
            if result is not None:
                categories = result

    elif structure in ("single", "single_root"):
        source = info
        print(f"\nDetected single folder: '{source}'")
        print("Will split into train/valid/test automatically.\n")

        if structure == "single_root":
            # Annotations in root — treat root as the source
            categories = convert_and_split_single_folder(
                args.input, args.output, ".",
                args.train, args.valid, args.test, args.seed,
                max_images=args.max
            )
        else:
            categories = convert_and_split_single_folder(
                args.input, args.output, source,
                args.train, args.valid, args.test, args.seed,
                max_images=args.max
            )

    if categories is None:
        print("\nError: No annotations could be processed!")
        return

    categories = sorted(categories, key=lambda c: c["id"])
    create_data_yaml(args.output, categories)

    print(f"\nDone! Your YOLOv8 dataset is at: {args.output}")
    print(f"Use it in training with:")
    print(f'  model.train(data="{Path(args.output).resolve()}/data.yaml", ...)')


if __name__ == "__main__":
    main()