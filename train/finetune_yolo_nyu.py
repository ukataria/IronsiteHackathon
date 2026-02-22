"""
Fine-tune YOLO on NYU Depth V2 Dataset

Uses NYU semantic labels to create training data for anchor detection.
"""

import numpy as np
from pathlib import Path
import yaml
from PIL import Image
import shutil
from tqdm import tqdm


# NYU label mappings (from NYU Depth V2 dataset)
# Label ID -> Object name
NYU_LABELS = {
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floor_mat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "refrigerator",
    25: "television",
    26: "paper",
    27: "towel",
    28: "shower_curtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "night_stand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
    38: "otherstructure",
    39: "otherfurniture",
    40: "otherprop"
}

# Objects with known/standard dimensions (good for anchors)
ANCHOR_OBJECTS = {
    "door": 32.0,           # Standard interior door width (inches)
    "refrigerator": 33.0,   # Standard fridge width
    "bed": 54.0,            # Standard queen bed width (NOT length which is 80")
    "desk": 30.0,           # Standard desk depth (more consistent than width)
    "table": 30.0,          # Dining table height (most consistent dimension)
    "counter": 36.0,        # Standard counter height (NOT depth)
    "cabinet": 30.0,        # Upper cabinet height (12-18", base 30-36", using 30")
    "window": 36.0,         # Average window width
    "bathtub": 32.0,        # Standard tub width (NOT length which is 60")
    "sink": 22.0,           # Standard bathroom sink width
    "toilet": 14.0,         # Standard toilet bowl width (front to back is 27-30")
}


def create_yolo_dataset(nyu_data_dir: str, output_dir: str, train_split: float = 0.8):
    """
    Create YOLO-format dataset from NYU labels.

    Args:
        nyu_data_dir: Path to NYU extracted data
        output_dir: Output directory for YOLO dataset
        train_split: Fraction of data for training (rest for validation)
    """
    nyu_path = Path(nyu_data_dir)
    output_path = Path(output_dir)

    # Create YOLO directory structure
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Get list of images
    rgb_files = sorted((nyu_path / "rgb").glob("*.jpg"))
    n_train = int(len(rgb_files) * train_split)

    print(f"Creating YOLO dataset from {len(rgb_files)} NYU images")
    print(f"  Train: {n_train} images")
    print(f"  Val: {len(rgb_files) - n_train} images")

    # Class mapping: only keep anchor objects
    anchor_class_ids = {name: idx for idx, name in enumerate(ANCHOR_OBJECTS.keys())}

    for img_idx, rgb_file in enumerate(tqdm(rgb_files, desc="Processing images")):
        # Determine split
        split = "train" if img_idx < n_train else "val"

        # Load semantic label
        label_file = nyu_path / "labels" / f"{rgb_file.stem}.npy"
        if not label_file.exists():
            continue

        labels = np.load(label_file)

        # Load RGB to get image dimensions
        img = Image.open(rgb_file)
        img_width, img_height = img.size

        # Extract bounding boxes from semantic labels
        yolo_annotations = []

        for label_id, label_name in NYU_LABELS.items():
            if label_name not in ANCHOR_OBJECTS:
                continue  # Skip non-anchor objects

            # Find all pixels with this label
            mask = (labels == label_id)

            if not mask.any():
                continue

            # Get bounding box
            rows, cols = np.where(mask)

            if len(rows) == 0:
                continue

            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()

            # Skip very small objects (likely noise)
            if (x_max - x_min) < 20 or (y_max - y_min) < 20:
                continue

            # Convert to YOLO format (normalized center_x, center_y, width, height)
            center_x = ((x_min + x_max) / 2) / img_width
            center_y = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            class_idx = anchor_class_ids[label_name]

            yolo_annotations.append(f"{class_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

        # Skip images with no anchor objects
        if not yolo_annotations:
            continue

        # Copy image
        dst_img = output_path / "images" / split / rgb_file.name
        shutil.copy(rgb_file, dst_img)

        # Write YOLO label file
        label_txt = output_path / "labels" / split / f"{rgb_file.stem}.txt"
        with open(label_txt, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    # Create YOLO config file
    config = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": anchor_class_ids
    }

    config_file = output_path / "nyu_anchors.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    print(f"\n✓ Dataset created at: {output_path}")
    print(f"✓ Config file: {config_file}")
    print(f"✓ Classes: {list(ANCHOR_OBJECTS.keys())}")

    return str(config_file)


def train_yolo(config_file: str, epochs: int = 50, batch_size: int = 16, device: str = "cuda"):
    """
    Train YOLO on NYU anchor detection.

    Args:
        config_file: Path to YOLO config YAML
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
    """
    from ultralytics import YOLO

    print(f"\nTraining YOLO on NYU anchors...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")

    # Load pretrained YOLOv8n and fine-tune
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data=config_file,
        epochs=epochs,
        batch=batch_size,
        device=device,
        imgsz=640,
        patience=10,  # Early stopping
        save=True,
        project="runs/nyu_anchor_detection",
        name="yolov8n_nyu",
        exist_ok=True,
        verbose=True
    )

    print(f"\n✓ Training complete!")
    print(f"✓ Best weights: runs/nyu_anchor_detection/yolov8n_nyu/weights/best.pt")

    return results


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune YOLO on NYU Depth V2")
    parser.add_argument(
        "--nyu_data_dir",
        type=str,
        default="data/nyu_depth_v2/extracted",
        help="Path to NYU extracted data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/yolo_nyu_dataset",
        help="Output directory for YOLO dataset"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--skip_dataset_creation",
        action="store_true",
        help="Skip dataset creation (use existing)"
    )

    args = parser.parse_args()

    # Step 1: Create dataset from NYU labels
    if not args.skip_dataset_creation:
        config_file = create_yolo_dataset(
            args.nyu_data_dir,
            args.output_dir
        )
    else:
        config_file = Path(args.output_dir) / "nyu_anchors.yaml"
        print(f"Using existing dataset: {config_file}")

    # Step 2: Train YOLO
    train_yolo(
        str(config_file),
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nTo use the fine-tuned model:")
    print("  1. Update models/anchor_detection.py:")
    print("     model_path = 'runs/nyu_anchor_detection/yolov8n_nyu/weights/best.pt'")
    print("  2. Re-run benchmark:")
    print("     python eval/nyu_distance/benchmark_all_models.py --models two-head-claude")
    print()


if __name__ == "__main__":
    main()
