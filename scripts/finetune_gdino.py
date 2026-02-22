"""
Fine-tune GroundingDINO following the learnopencv.com approach.

Uses the official GroundingDINO package with:
  - build_captions_and_token_span  for correct caption / token-span construction
  - create_positive_map_from_span  for positive maps used in the Hungarian matcher
  - HungarianMatcher + SetCriterion (ported from Asad-Ismail/Grounding-Dino-FineTuning)
  - nested_tensor_from_tensor_list for image batching

Setup (run once):
    # Install GroundingDINO package
    pip install git+https://github.com/IDEA-Research/GroundingDINO.git

    # Download SwinT weights
    mkdir -p weights
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \\
         -O weights/groundingdino_swint_ogc.pth

Usage:
    uv run python scripts/finetune_gdino.py
    uv run python scripts/finetune_gdino.py --epochs 20 --lr 2e-4
    uv run python scripts/finetune_gdino.py --use-lora --epochs 50 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
import groundingdino
import torch
import torch.nn as nn
import torch.nn.functional as F
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.train import load_image, load_model
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Paths and class mapping
# ---------------------------------------------------------------------------

WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
CONFIG_PATH = str(Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py")

CLASS_TO_PHRASE: dict[str, str] = {
    "Bricks-Masonry": "brick",
    "brick": "brick",
    "Electrical-Outlet": "electrical outlet",
    "OUTLET": "electrical outlet",
}

# Extra distractor categories added to captions during training to prevent overfitting
EXTRA_NEGATIVE_CLASSES: list[str] = ["person", "window", "door", "pipe", "cable"]


# ---------------------------------------------------------------------------
# COCO JSON → GroundingDINO Dataset
# ---------------------------------------------------------------------------


class COCOGroundingDataset(Dataset):
    """
    Reads a COCO JSON annotation file and produces targets in the format
    expected by GroundingDINOTrainer:
      image:  (3, H, W) float tensor (GroundingDINO preprocessed)
      target: {
          boxes:          (N, 4) float32 – [cx, cy, w, h] in pixel coords
          size:           (2,)   int      – [H, W]
          str_cls_lst:    list[str]        – class name per box (positive)
          all_categories: list[str]        – positive + negative categories
          caption:        str              – text query built from all_categories
          cat2tokenspan:  dict             – {category: [(start, end), ...]}
          labels:         (N,) int64       – index of each box into str_cls_lst
      }
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        negative_sampling_rate: float = 0.5,
        extra_classes: list[str] | None = None,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.negative_sampling_rate = negative_sampling_rate

        data = json.loads(Path(ann_file).read_text())
        id_to_image = {img["id"]: img for img in data["images"]}
        id_to_catname = {cat["id"]: cat["name"] for cat in data["categories"]}

        # Group annotations by image; skip unknown classes
        anns_by_image: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            cat_name = id_to_catname.get(ann["category_id"], "")
            if cat_name not in CLASS_TO_PHRASE:
                continue
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # Build per-image records
        self.records: list[dict] = []
        for img_id, img_info in id_to_image.items():
            anns = anns_by_image.get(img_id)
            if not anns:
                continue
            self.records.append({"img_info": img_info, "anns": anns})

        # All unique positive phrases for negative sampling
        self.all_phrases: list[str] = list(
            {CLASS_TO_PHRASE[id_to_catname[a["category_id"]]] for rec in self.records for a in rec["anns"]}
        )
        if extra_classes:
            self.all_phrases = list(set(self.all_phrases + extra_classes))

        self._id_to_catname = id_to_catname

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        rec = self.records[idx]
        img_info = rec["img_info"]
        anns = rec["anns"]

        img_path = self.img_dir / img_info["file_name"]
        image_source, image = load_image(str(img_path))  # image: (3, H, W) float
        h, w = image_source.shape[:2]

        # Positive phrase per box (preserving order)
        str_cls_lst: list[str] = []
        boxes: list[list[float]] = []
        for ann in anns:
            cat_name = self._id_to_catname[ann["category_id"]]
            phrase = CLASS_TO_PHRASE[cat_name]
            x, y, bw, bh = ann["bbox"]
            cx, cy = x + bw / 2, y + bh / 2
            boxes.append([cx, cy, bw, bh])
            str_cls_lst.append(phrase)

        # Negative sampling: add phrases not present in this image
        positive_set = set(str_cls_lst)
        candidates = [p for p in self.all_phrases if p not in positive_set]
        n_neg = max(1, int(len(str_cls_lst) * self.negative_sampling_rate))
        if candidates and self.negative_sampling_rate > 0:
            n_neg = min(n_neg, len(candidates))
            negatives = random.sample(candidates, n_neg)
        else:
            negatives = []

        all_categories = str_cls_lst + negatives  # positives first, negatives after
        caption, cat2tokenspan = build_captions_and_token_span(all_categories, force_lowercase=True)

        # Class index = position in str_cls_lst (unique phrases in order)
        phrase_to_idx = {p: i for i, p in enumerate(dict.fromkeys(str_cls_lst))}
        labels = torch.tensor([phrase_to_idx[p] for p in str_cls_lst], dtype=torch.int64)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "size": torch.as_tensor([h, w]),
            "str_cls_lst": str_cls_lst,
            "all_categories": all_categories,
            "caption": caption,
            "cat2tokenspan": cat2tokenspan,
            "labels": labels,
        }
        return image, target


# ---------------------------------------------------------------------------
# Hungarian Matcher (ported from Asad-Ismail/Grounding-Dino-FineTuning)
# ---------------------------------------------------------------------------


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.eps = 1e-6

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]) -> tuple[list, torch.Tensor]:
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid().clamp(self.eps, 1 - self.eps)
        pred_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Normalize target boxes and collect token spans
        tgt_boxes_list, token_spans_list = [], []
        for t in targets:
            h, w = t["size"]
            scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=pred_bbox.device)
            tgt_boxes_list.append(t["boxes"].to(pred_bbox.device) / scale)
            token_spans_list.append([t["cat2tokenspan"][cls] for cls in t["str_cls_lst"]])

        tgt_bbox = torch.cat(tgt_boxes_list)

        # Build positive maps from token spans
        pos_maps = []
        for spans in token_spans_list:
            pm = create_positive_map_from_span(
                outputs["tokenized"], spans, max_text_len=256
            ).to(out_prob.device)
            pos_maps.append(pm)
        tgt_labels = torch.cat(pos_maps, dim=0)

        # Classification cost
        norm_tgt = tgt_labels / (tgt_labels.sum(dim=1, keepdim=True) + self.eps)
        cost_class = -(out_prob.unsqueeze(1) * norm_tgt.unsqueeze(0).float()).sum(-1)

        # Box costs
        cost_bbox = torch.cdist(pred_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        ).clamp(min=-1.0)

        C = (self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(t["boxes"]) for t in targets]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in [linear_sum_assignment(c[b]) for b, c in enumerate(C.split(sizes, -1))]
        ]
        return indices, tgt_labels


# ---------------------------------------------------------------------------
# SetCriterion (ported from Asad-Ismail/Grounding-Dino-FineTuning)
# ---------------------------------------------------------------------------


def _sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    text_mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eos_coef: float = 0.1,
) -> torch.Tensor:
    """Focal loss for grounded detection token prediction."""
    p = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    focal_w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - p_t).pow(gamma)

    # Apply masks: valid positions (matched) weighted normally, background weighted by eos_coef
    weight = torch.where(valid_mask, torch.ones_like(ce), torch.full_like(ce, eos_coef))
    # Only score tokens within the text (not padding)
    text_mask_exp = text_mask.unsqueeze(1).expand_as(pred)
    loss = (focal_w * ce * weight * text_mask_exp).sum()
    normalizer = text_mask_exp.float().sum().clamp(min=1)
    return loss / normalizer


class SetCriterion(nn.Module):
    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        eos_coef: float = 0.1,
        max_txt_len: int = 256,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.max_txt_len = max_txt_len

    def _loss_labels(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list,
        tgt_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bs, nq, nc = outputs["pred_logits"].shape
        device = outputs["pred_logits"].device

        # Official GDINO package uses text_token_mask; fall back to tokenizer attention_mask
        if "text_token_mask" in outputs:
            raw_mask = outputs["text_token_mask"]          # (bs, L)
        elif "text_mask" in outputs:
            raw_mask = outputs["text_mask"]                # (bs, nc) legacy
        else:
            raw_mask = outputs["tokenized"]["attention_mask"]  # (bs, L)

        raw_mask = raw_mask.to(device)
        L = raw_mask.shape[1]
        if L < nc:
            text_mask = F.pad(raw_mask.float(), (0, nc - L)).bool()
        else:
            text_mask = raw_mask[:, :nc].bool()

        target_labels = torch.zeros(bs, nq, nc, device=device)
        valid_mask = torch.zeros(bs, nq, nc, dtype=torch.bool, device=device)

        offset = 0
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            n_tgt = len(targets[b]["boxes"])
            batch_tgt = tgt_labels[offset : offset + n_tgt]
            target_labels[b, pred_idx] = batch_tgt[tgt_idx]
            valid_mask[b, pred_idx] = text_mask[b]
            offset += n_tgt

        loss = _sigmoid_focal_loss(
            outputs["pred_logits"], target_labels, valid_mask, text_mask, eos_coef=self.eos_coef
        )
        return {"loss_ce": loss}

    def _loss_boxes(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list,
    ) -> dict[str, torch.Tensor]:
        pred_boxes = outputs["pred_boxes"]
        device = pred_boxes.device

        batch_idx = torch.cat([torch.full((len(i),), b, dtype=torch.long) for b, (i, _) in enumerate(indices)])
        src_idx = torch.cat([i for i, _ in indices])
        src_boxes = pred_boxes[batch_idx, src_idx]

        tgt_boxes_norm: list[torch.Tensor] = []
        for t, (_, j) in zip(targets, indices):
            h, w = t["size"]
            scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=device)
            tgt_boxes_norm.append(t["boxes"].to(device)[j] / scale)
        tgt_boxes = torch.cat(tgt_boxes_norm)

        n = max(tgt_boxes.shape[0], 1)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / n
        loss_giou = (
            1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))
        ).sum() / n
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        indices, tgt_labels = self.matcher(outputs, targets)
        losses: dict[str, torch.Tensor] = {}
        losses.update(self._loss_labels(outputs, targets, indices, tgt_labels))
        losses.update(self._loss_boxes(outputs, targets, indices))
        return losses


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_fn(batch: list[tuple]) -> tuple:
    images, targets = zip(*batch)
    images = nested_tensor_from_tensor_list(list(images))
    return images, list(targets)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class GroundingDINOTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-5,
        num_epochs: int = 15,
        steps_per_epoch: int = 1,
        use_amp: bool = True,
        grad_accum_steps: int = 4,
        max_grad_norm: float = 5.0,
        weight_dict: dict | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm

        if weight_dict is None:
            weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        self.weight_dict = weight_dict

        matcher = HungarianMatcher(
            cost_class=weight_dict["loss_ce"],
            cost_bbox=weight_dict["loss_bbox"],
            cost_giou=weight_dict["loss_giou"],
        )
        self.criterion = SetCriterion(matcher, weight_dict).to(device)

        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )
        total_steps = steps_per_epoch * num_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def _prepare_batch(self, batch: tuple) -> tuple:
        images, targets = batch
        images = images.to(self.device)
        captions = [t["caption"] for t in targets]
        for t in targets:
            t["boxes"] = t["boxes"].to(self.device)
            t["size"] = t["size"].to(self.device)
        return images, targets, captions

    def train_step(self, batch: tuple, step_idx: int) -> dict[str, float]:
        self.model.train()
        images, targets, captions = self._prepare_batch(batch)

        try:
            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                outputs = self.model(images, captions=captions)
                # On first step, print available output keys for debugging
                if step_idx == 0:
                    print(f"  [debug] outputs keys: {list(outputs.keys())}")
                loss_dict = self.criterion(outputs, targets)
                if step_idx == 0:
                    print(f"  [debug] loss_dict: { {k: round(v.item(), 4) for k, v in loss_dict.items()} }")
                total_loss = sum(
                    loss_dict[k] * self.weight_dict.get(k, 1.0) for k in loss_dict
                ) / self.grad_accum_steps
        except Exception as e:
            print(f"  [step {step_idx}] forward error: {e}")
            return {}

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        is_step = (step_idx + 1) % self.grad_accum_steps == 0
        if is_step:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = defaultdict(float)
        n = 0
        for batch in val_loader:
            images, targets, captions = self._prepare_batch(batch)
            try:
                with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                    outputs = self.model(images, captions=captions)
                    loss_dict = self.criterion(outputs, targets)
            except Exception as e:
                print(f"  [val] forward error: {e}")
                continue
            if n == 0:
                print(f"  [val debug] loss_dict: { {k: round(v.item(), 4) for k, v in loss_dict.items()} }")
            for k, v in loss_dict.items():
                totals[k] += v.item()
            totals["total_loss"] += sum(
                loss_dict[k].item() * self.weight_dict.get(k, 1.0) for k in loss_dict
            )
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    def save(self, path: str, epoch: int, losses: dict) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "losses": losses,
            },
            path,
        )


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------


def freeze_backbones(model: nn.Module) -> None:
    """Freeze Swin vision backbone and BERT text backbone; train fusion + head."""
    for name, param in model.named_parameters():
        if "backbone" in name or "bert" in name.lower():
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=str, default="data/gdino_finetuned")
    parser.add_argument("--no-freeze", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--neg-rate", type=float, default=0.5,
                        help="Negative category sampling rate (0=off)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}  |  AMP: {use_amp}  |  grad_accum: {args.grad_accum}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Datasets ----
    extra = EXTRA_NEGATIVE_CLASSES

    def make_ds(ann: str, img: str) -> COCOGroundingDataset:
        return COCOGroundingDataset(ann, img, negative_sampling_rate=args.neg_rate, extra_classes=extra)

    train_ds = ConcatDataset([
        make_ds("Bricks.v1i.coco/train/_annotations.coco.json", "Bricks.v1i.coco/train"),
        make_ds("OUTLET.v1i.coco.balanced/train/_annotations.coco.json", "OUTLET.v1i.coco/train"),
    ])
    val_ds = ConcatDataset([
        make_ds("Bricks.v1i.coco/valid/_annotations.coco.json", "Bricks.v1i.coco/valid"),
        make_ds("OUTLET.v1i.coco.balanced/valid/_annotations.coco.json", "OUTLET.v1i.coco/valid"),
    ])
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # ---- Model ----
    print(f"Loading model from {args.weights}...")
    model = load_model(CONFIG_PATH, args.weights)
    if not args.no_freeze:
        freeze_backbones(model)

    steps_per_epoch = len(train_ds) // args.batch_size
    trainer = GroundingDINOTrainer(
        model=model,
        device=device,
        lr=args.lr,
        num_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        use_amp=use_amp,
        grad_accum_steps=args.grad_accum,
    )
    trainer.optimizer.zero_grad()

    # ---- Training loop ----
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_losses: dict[str, list[float]] = defaultdict(list)

        for step_idx, batch in enumerate(train_loader):
            losses = trainer.train_step(batch, step_idx)
            for k, v in losses.items():
                epoch_losses[k].append(v)

            if step_idx % 20 == 0 and losses:
                loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
                print(f"  [{epoch}/{args.epochs}  step {step_idx}/{len(train_loader)}]  {loss_str}")

        avg_train = {k: sum(v) / len(v) for k, v in epoch_losses.items() if v}
        val_losses = trainer.validate(val_loader)
        val_total = val_losses.get("total_loss", 0.0)

        print(
            f"Epoch {epoch:>2}/{args.epochs} — "
            + "  ".join(f"train_{k}={v:.4f}" for k, v in avg_train.items())
            + f"  val_total={val_total:.4f}"
        )

        if val_total < best_val_loss:
            best_val_loss = val_total
            trainer.save(str(output_dir / "best.pth"), epoch, val_losses)
            print(f"  → Saved best checkpoint (val={best_val_loss:.4f})")

    trainer.save(str(output_dir / "final.pth"), args.epochs, {})
    print(f"\nDone. Best val_total={best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir}/best.pth  and  {output_dir}/final.pth")


if __name__ == "__main__":
    main()
