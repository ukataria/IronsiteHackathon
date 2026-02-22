"""
YOLO Fine-Tuning Script with Multi-Dataset Merging
====================================================
Fine-tunes a YOLO model (YOLOv8 or YOLO26) on one or more YOLOv8-format datasets.

If multiple datasets are provided, they are merged into a single unified dataset
with remapped class IDs and a combined data.yaml.

Usage:
    # Single dataset
    python finetune_yolo.py --data ./Bricks.yolo/data.yaml

    # Multiple datasets merged together
    python finetune_yolo.py --data ./Bricks.yolo/data.yaml ./Outlets.yolo/data.yaml

    # Use YOLO26 instead of YOLOv8
    python finetune_yolo.py --data ./Bricks.yolo/data.yaml ./Outlets.yolo/data.yaml --model yolo26m.pt

    # Custom training params
    python finetune_yolo.py --data ./merged/data.yaml --model yolo26m.pt --epochs 100 --batch 8 --imgsz 640
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO


def load_data_yaml(yaml_path):
    """Load a data.yaml and resolve paths relative to its location."""
    yaml_path = Path(yaml_path).resolve()
    
    # Try multiple encodings in case file has non-UTF-8 characters
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            with open(yaml_path, "r", encoding=encoding) as f:
                data = yaml.safe_load(f)
            break
        except (UnicodeDecodeError, yaml.YAMLError):
            continue
    else:
        raise ValueError(f"Could not decode {yaml_path} with any supported encoding")

    base_dir = yaml_path.parent
    # If 'path' is specified, use it as base; otherwise use yaml's parent dir
    if "path" in data and data["path"]:
        candidate = Path(data["path"])
        if not candidate.is_absolute():
            candidate = yaml_path.parent / candidate
        candidate = candidate.resolve()
        # Use the path from yaml only if it actually exists;
        # otherwise fall back to yaml's parent dir (handles cross-platform moves)
        if candidate.exists():
            base_dir = candidate
        else:
            print(f"  Warning: path '{data['path']}' in {yaml_path.name} not found, using {base_dir} instead")

    data["_base_dir"] = base_dir
    data["_yaml_path"] = yaml_path
    return data


def merge_datasets(yaml_paths, output_dir, max_per_split: int | None = None, max_per_dataset: list[int | None] | None = None):
    """
    Merge multiple YOLO datasets into one unified dataset.

    - Combines all class names into a single list (deduplicating by name)
    - Remaps class IDs in label files accordingly
    - Copies images and remapped labels into a single dataset
    - Generates a unified data.yaml

    Returns the path to the merged data.yaml.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build unified class list ──
    unified_names = []  # ordered list of unique class names
    name_to_id = {}     # class_name -> unified_id

    datasets = []
    for yp in yaml_paths:
        data = load_data_yaml(yp)
        datasets.append(data)

        # data["names"] can be a dict {0: "brick", 1: "outlet"} or list ["brick", "outlet"]
        names = data.get("names", {})
        if isinstance(names, dict):
            names = {int(k): v for k, v in names.items()}
            sorted_names = [names[k] for k in sorted(names.keys())]
        else:
            sorted_names = list(names)

        for class_name in sorted_names:
            if class_name not in name_to_id:
                name_to_id[class_name] = len(unified_names)
                unified_names.append(class_name)

        data["_sorted_names"] = sorted_names

    print(f"\nUnified class list ({len(unified_names)} classes):")
    for i, name in enumerate(unified_names):
        print(f"  {i}: {name}")

    # ── Step 2: Build class ID remap for each dataset ──
    for data in datasets:
        # old_id -> new_id
        remap = {}
        for old_id, class_name in enumerate(data["_sorted_names"]):
            remap[old_id] = name_to_id[class_name]
        data["_remap"] = remap

    # ── Step 3: Copy images and remap labels ──
    for split in ["train", "valid", "test"]:
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # Collect all candidates first so we can sample before copying
        img_candidates = []

        for ds_idx, data in enumerate(datasets):
            base_dir = data["_base_dir"]
            remap = data["_remap"]
            yaml_dir = data["_yaml_path"].parent

            img_dir_rel = data.get(split) or data.get("val" if split == "valid" else split)
            if img_dir_rel is None:
                continue

            # Try multiple base directories in case of cross-platform moves
            path_candidates = [
                Path(base_dir) / img_dir_rel,
                yaml_dir / img_dir_rel,
                yaml_dir / split / "images",
                yaml_dir / split,
            ]

            img_dir = None
            for c in path_candidates:
                if c.exists() and any(c.iterdir()):
                    img_dir = c
                    break

            if img_dir is None:
                continue

            # Infer labels dir from images dir
            lbl_dir = Path(str(img_dir).replace("/images", "/labels").replace("\\images", "\\labels"))
            if not lbl_dir.exists():
                lbl_dir = img_dir.parent / "labels"

            ds_files = [
                f for f in img_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            ]
            ds_max = max_per_dataset[ds_idx] if max_per_dataset and ds_idx < len(max_per_dataset) else None
            if ds_max and len(ds_files) > ds_max:
                total_ds = len(ds_files)
                ds_files = random.sample(ds_files, ds_max)
                print(f"    ds{ds_idx} {split}: sampled {ds_max} / {total_ds} images")
            for img_file in ds_files:
                img_candidates.append((img_file, ds_idx, remap, lbl_dir))

        # Random subset if dataset exceeds max_per_split
        total_found = len(img_candidates)
        if total_found == 0:
            continue
        if max_per_split and total_found > max_per_split:
            img_candidates = random.sample(img_candidates, max_per_split)
            print(f"  {split}: sampled {max_per_split} / {total_found} images")
        else:
            print(f"  {split}: {total_found} images merged")

        # Copy sampled images and remap labels
        for img_file, ds_idx, remap, lbl_dir in img_candidates:
            new_name = f"ds{ds_idx}_{img_file.name}"
            shutil.copy2(img_file, img_out / new_name)

            label_file = lbl_dir / (img_file.stem + ".txt")
            new_label = lbl_out / (f"ds{ds_idx}_{img_file.stem}.txt")

            if label_file.exists():
                with open(label_file, "r") as f:
                    lines = f.readlines()

                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_id = int(parts[0])
                        new_id = remap.get(old_id, old_id)
                        remapped_lines.append(f"{new_id} {' '.join(parts[1:])}\n")

                with open(new_label, "w") as f:
                    f.writelines(remapped_lines)
            else:
                new_label.touch()

    # ── Step 4: Write unified data.yaml ──
    unified_yaml = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(unified_names),
        "names": {i: name for i, name in enumerate(unified_names)},
    }

    if (output_dir / "test" / "images").exists() and any((output_dir / "test" / "images").iterdir()):
        unified_yaml["test"] = "test/images"

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(unified_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Merged data.yaml written to {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on one or more datasets.")
    parser.add_argument("--data", nargs="+", required=True,
                        help="Path(s) to data.yaml file(s). If multiple, datasets are merged.")
    parser.add_argument("--model", type=str, default="yolo26m.pt",
                        help="Pretrained model (default: yolo26m.pt). Options: yolo26n/s/m/l.pt or yolov8n/s/m/l/x.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--device", type=str, default="0", help="Device: 0 for GPU, cpu, or mps (default: 0)")
    parser.add_argument("--project", type=str, default="runs", help="Output project directory (default: runs)")
    parser.add_argument("--name", type=str, default="finetune", help="Run name (default: finetune)")
    parser.add_argument("--merge-dir", type=str, default="merged_dataset",
                        help="Directory for merged dataset (default: merged_dataset)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images per split total. If exceeded, a random subset is used.")
    parser.add_argument("--max-per-dataset", type=int, nargs="+", default=None,
                        help="Max images per dataset per split (one value per --data arg, e.g. --max-per-dataset 500 300).")
    args = parser.parse_args()

    # ── Resolve dataset ──
    if len(args.data) > 1 or args.max_images or args.max_per_dataset:
        print(f"\nMerging {len(args.data)} datasets...")
        data_yaml = merge_datasets(args.data, args.merge_dir, max_per_split=args.max_images, max_per_dataset=args.max_per_dataset)
    else:
        data_yaml = Path(args.data[0]).resolve()
        print(f"\nUsing single dataset: {data_yaml}")

    # ── Load model ──
    print(f"\nLoading pretrained model: {args.model}")
    model = YOLO(args.model)

    # ── Train ──
    print(f"\nStarting fine-tuning...")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device:    {args.device}")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        lr0=0.001,
        lrf=0.01,
        augment=True,
        device=int(args.device) if args.device.isdigit() else args.device,
        project=args.project,
        name=args.name,
    )

    # ── Evaluate ──
    print("\nEvaluating on validation set...")
    metrics = model.val()
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    # ── Print results ──
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nDone! Best weights saved to: {best_weights}")
    print(f"\nTo run inference:")
    print(f'  model = YOLO("{best_weights}")')
    print(f'  results = model.predict("image.jpg", conf=0.25)')

    # ── Export (optional, commented out) ──
    # best_model = YOLO(best_weights)
    # best_model.export(format="onnx")


if __name__ == "__main__":
    main()