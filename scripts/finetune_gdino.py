"""
Fine-tune GroundingDINO (grounding-dino-base) on Bricks + Outlet COCO datasets.

Datasets used:
  Bricks.v1i.coco/{train,valid}/          — 143 train / 38 val images
  OUTLET.v1i.coco.balanced/{train,valid}/ — 143 train / 38 val images (balanced)
  Images for balanced outlet read from OUTLET.v1i.coco/ (original image dir).

Classes → text phrases:
  Bricks-Masonry / brick      → "brick"
  Electrical-Outlet / OUTLET  → "electrical outlet"

Text prompt (fixed): "brick . electrical outlet ."

Usage (GPU recommended):
    uv run python scripts/finetune_gdino.py
    uv run python scripts/finetune_gdino.py --epochs 20 --batch-size 4 --lr 2e-5
    uv run python scripts/finetune_gdino.py --output-dir data/gdino_finetuned --freeze-backbone
"""

from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# ---------------------------------------------------------------------------
# Class → phrase mapping and text prompt
# ---------------------------------------------------------------------------

CLASS_TO_PHRASE: dict[str, str] = {
    "Bricks-Masonry": "brick",
    "brick": "brick",
    "Electrical-Outlet": "electrical outlet",
    "OUTLET": "electrical outlet",
}

PHRASES: list[str] = ["brick", "electrical outlet"]
TEXT_PROMPT: str = " . ".join(PHRASES) + " ."

MODEL_ID = "IDEA-Research/grounding-dino-base"


# ---------------------------------------------------------------------------
# Token span utilities
# ---------------------------------------------------------------------------


def compute_phrase_token_spans(
    processor: Any,
    text: str,
    phrases: list[str],
) -> dict[str, list[int]]:
    """
    Find which token indices in the tokenized `text` correspond to each phrase.
    Returns {phrase: [token_idx, ...]} using character-offset matching.
    """
    encoding = processor.tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]

    spans: dict[str, list[int]] = {}
    for phrase in phrases:
        char_start = text.find(phrase)
        if char_start == -1:
            spans[phrase] = []
            continue
        char_end = char_start + len(phrase)
        spans[phrase] = [
            i for i, (ts, te) in enumerate(offsets)
            if ts >= char_start and te <= char_end and ts < te
        ]
    return spans


def build_positive_map(
    phrase_token_spans: dict[str, list[int]],
    phrase: str,
    seq_len: int,
) -> torch.Tensor:
    """
    Binary (seq_len,) tensor: 1.0 at token positions for the phrase, 0.0 elsewhere.
    Normalized so the loss treats single-token and multi-token phrases equally.
    """
    pos_map = torch.zeros(seq_len)
    for tok_idx in phrase_token_spans.get(phrase, []):
        if tok_idx < seq_len:
            pos_map[tok_idx] = 1.0
    n = pos_map.sum()
    if n > 0:
        pos_map = pos_map / n
    return pos_map


# ---------------------------------------------------------------------------
# COCO Dataset
# ---------------------------------------------------------------------------


class COCODetectionDataset(Dataset):
    """
    COCO-format dataset for GroundingDINO fine-tuning.

    Args:
        ann_file:  path to _annotations.coco.json
        img_dir:   directory containing the image files (flat, no subdir)
        processor: HuggingFace GroundingDINO processor
        phrase_token_spans: output of compute_phrase_token_spans()
        seq_len:   text sequence length (from processor tokenization)
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        processor: Any,
        phrase_token_spans: dict[str, list[int]],
        seq_len: int,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.phrase_token_spans = phrase_token_spans
        self.seq_len = seq_len

        data = json.loads(Path(ann_file).read_text())
        self.id_to_image: dict[int, dict] = {img["id"]: img for img in data["images"]}
        self.id_to_catname: dict[int, str] = {
            cat["id"]: cat["name"] for cat in data["categories"]
        }

        # Group annotations by image; skip unknown classes
        self.anns_by_image: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            cat_name = self.id_to_catname.get(ann["category_id"], "")
            if cat_name not in CLASS_TO_PHRASE:
                continue
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # Only keep images that have at least one valid annotation
        self.image_ids = [
            img_id for img_id in self.id_to_image
            if self.anns_by_image.get(img_id)
        ]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.image_ids[idx]
        img_info = self.id_to_image[img_id]
        img_path = self.img_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        boxes: list[list[float]] = []
        class_labels: list[int] = []
        positive_maps: list[torch.Tensor] = []

        for ann in self.anns_by_image.get(img_id, []):
            cat_name = self.id_to_catname[ann["category_id"]]
            phrase = CLASS_TO_PHRASE[cat_name]
            phrase_idx = PHRASES.index(phrase)

            # COCO [x, y, w, h] absolute → normalized cxcywh
            x, y, w, h = ann["bbox"]
            cx = max(0.0, min(1.0, (x + w / 2) / img_w))
            cy = max(0.0, min(1.0, (y + h / 2) / img_h))
            nw = max(0.0, min(1.0, w / img_w))
            nh = max(0.0, min(1.0, h / img_h))

            boxes.append([cx, cy, nw, nh])
            class_labels.append(phrase_idx)
            positive_maps.append(
                build_positive_map(self.phrase_token_spans, phrase, self.seq_len)
            )

        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
            "positive_maps": torch.stack(positive_maps) if positive_maps else torch.zeros((0, self.seq_len)),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_fn(batch: list[dict], processor: Any) -> dict:
    """Collate a batch: run processor on images + fixed text, attach labels."""
    images = [item["image"] for item in batch]
    encoding = processor(
        images=images,
        text=[TEXT_PROMPT] * len(images),
        return_tensors="pt",
        padding=True,
    )
    encoding["labels"] = [
        {
            "boxes": item["boxes"],
            "class_labels": item["class_labels"],
            "positive_map": item["positive_maps"],
        }
        for item in batch
    ]
    return encoding


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def freeze_backbones(model: nn.Module) -> None:
    """Freeze the vision (Swin) and text (BERT) backbones; train fusion + head."""
    frozen = 0
    for name, param in model.named_parameters():
        if "backbone" in name or "text_backbone" in name or "bert" in name.lower():
            param.requires_grad = False
            frozen += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Frozen {frozen:,} params. "
        f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Train / eval loop
# ---------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for batch_idx, batch in enumerate(loader):
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k != "labels"
            }
            labels = [
                {k: v.to(device) for k, v in lbl.items()}
                for lbl in batch["labels"]
            ]

            try:
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
            except Exception as e:
                print(f"  [batch {batch_idx}] forward error: {e}")
                continue

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GroundingDINO on Bricks + Outlet")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU step")
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak learning rate")
    parser.add_argument("--output-dir", type=str, default="data/gdino_finetuned")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="Freeze vision + text backbones (recommended for small dataset)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Processor + token spans ----
    print(f"Loading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Tokenize the fixed text prompt to get actual seq_len used by model
    _enc = processor(
        images=Image.new("RGB", (224, 224)),
        text=TEXT_PROMPT,
        return_tensors="pt",
    )
    seq_len: int = _enc["input_ids"].shape[1]
    print(f"Text prompt: '{TEXT_PROMPT}'  →  seq_len={seq_len}")

    phrase_token_spans = compute_phrase_token_spans(processor, TEXT_PROMPT, PHRASES)
    print(f"Phrase token spans: {phrase_token_spans}")

    # ---- Datasets ----
    def make_ds(ann_file: str, img_dir: str) -> COCODetectionDataset:
        return COCODetectionDataset(ann_file, img_dir, processor, phrase_token_spans, seq_len)

    train_ds = ConcatDataset([
        make_ds(
            "Bricks.v1i.coco/train/_annotations.coco.json",
            "Bricks.v1i.coco/train",
        ),
        make_ds(
            # Balanced JSON; images still live in original OUTLET dir
            "OUTLET.v1i.coco.balanced/train/_annotations.coco.json",
            "OUTLET.v1i.coco/train",
        ),
    ])
    val_ds = ConcatDataset([
        make_ds(
            "Bricks.v1i.coco/valid/_annotations.coco.json",
            "Bricks.v1i.coco/valid",
        ),
        make_ds(
            "OUTLET.v1i.coco.balanced/valid/_annotations.coco.json",
            "OUTLET.v1i.coco/valid",
        ),
    ])

    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    _collate = partial(collate_fn, processor=processor)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    print(f"Loading model from {MODEL_ID}...")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    if args.freeze_backbone:
        freeze_backbones(model)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    # ---- Training loop ----
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, None, device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:>2}/{args.epochs} — "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = output_dir / "best"
            model.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))
            print(f"  → New best checkpoint saved (val_loss={best_val_loss:.4f})")

    # ---- Save final ----
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir}/best/  and  {output_dir}/final/")


if __name__ == "__main__":
    main()
