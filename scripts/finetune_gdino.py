"""
Fine-tune GroundingDINO using HuggingFace transformers.

No groundingdino package required — weights download automatically from HuggingFace Hub.
Only needs: torch, transformers, scipy, PIL (all already in pyproject.toml).

Usage:
    uv run python scripts/finetune_gdino.py
    uv run python scripts/finetune_gdino.py --epochs 20 --lr 1e-5 --grad-accum 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoProcessor, GroundingDinoForObjectDetection

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "IDEA-Research/grounding-dino-base"
MAX_TEXT_LEN = 256

CLASS_TO_PHRASE: dict[str, str] = {
    "Bricks-Masonry": "brick",
    "brick": "brick",
    "Electrical-Outlet": "electrical outlet",
    "OUTLET": "electrical outlet",
}

EXTRA_NEGATIVE_CLASSES: list[str] = ["person", "window", "door", "pipe", "cable"]


# ---------------------------------------------------------------------------
# Helper functions (replace groundingdino.util.*)
# ---------------------------------------------------------------------------


def build_captions_and_token_span(
    cat_list: list[str], force_lowercase: bool = True
) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    """Build a '. '-joined caption and character-level span per category."""
    cat2span: dict[str, list[tuple[int, int]]] = {}
    caption = ""
    for cat in cat_list:
        phrase = cat.lower() if force_lowercase else cat
        start = len(caption)
        caption += phrase
        end = len(caption)
        cat2span[cat] = [(start, end)]
        caption += " . "
    caption = caption.rstrip(" .")
    return caption, cat2span


def create_positive_map_from_span(
    encoding,
    token_spans: list[list[tuple[int, int]]],
    max_text_len: int = MAX_TEXT_LEN,
) -> torch.Tensor:
    """Return (N, max_text_len) float tensor with 1s at token positions for each span."""
    pos_map = torch.zeros(len(token_spans), max_text_len)
    for j, spans in enumerate(token_spans):
        for start_char, end_char in spans:
            for char_pos in range(start_char, end_char):
                tok = encoding.char_to_token(char_pos)
                if tok is not None and tok < max_text_len:
                    pos_map[j, tok] = 1.0
    row_sums = pos_map.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return pos_map / row_sums


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[cx, cy, w, h] → [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise GIoU matrix (N, M) for xyxy boxes."""
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    ex1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    ey1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    ex2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    ey2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0)
    return iou - (enc - union) / enc.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class COCOGroundingDataset(Dataset):
    """
    Reads a COCO JSON file and returns (PIL.Image, target_dict) pairs.

    target = {
        boxes:          (N, 4) float32 – [cx, cy, w, h] pixel coords
        size:           (2,) int        – [H, W]
        str_cls_lst:    list[str]       – positive phrase per box
        all_categories: list[str]       – positives + negatives
        caption:        str
        cat2tokenspan:  dict
        labels:         (N,) int64
        encoding:       BatchEncoding   – per-caption tokenization for positive map
    }
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        processor: AutoProcessor,
        negative_sampling_rate: float = 0.5,
        extra_classes: list[str] | None = None,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.negative_sampling_rate = negative_sampling_rate

        data = json.loads(Path(ann_file).read_text())
        id_to_image = {img["id"]: img for img in data["images"]}
        id_to_catname = {cat["id"]: cat["name"] for cat in data["categories"]}

        anns_by_image: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            cat_name = id_to_catname.get(ann["category_id"], "")
            if cat_name not in CLASS_TO_PHRASE:
                continue
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.records: list[dict] = []
        for img_id, img_info in id_to_image.items():
            anns = anns_by_image.get(img_id)
            if not anns:
                continue
            self.records.append({"img_info": img_info, "anns": anns})

        self.all_phrases: list[str] = list(
            {CLASS_TO_PHRASE[id_to_catname[a["category_id"]]] for rec in self.records for a in rec["anns"]}
        )
        if extra_classes:
            self.all_phrases = list(set(self.all_phrases + extra_classes))

        self._id_to_catname = id_to_catname

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict]:
        rec = self.records[idx]
        img_info = rec["img_info"]
        anns = rec["anns"]

        img_path = self.img_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        str_cls_lst: list[str] = []
        boxes: list[list[float]] = []
        for ann in anns:
            cat_name = self._id_to_catname[ann["category_id"]]
            phrase = CLASS_TO_PHRASE[cat_name]
            x, y, bw, bh = ann["bbox"]
            boxes.append([x + bw / 2, y + bh / 2, bw, bh])
            str_cls_lst.append(phrase)

        positive_set = set(str_cls_lst)
        candidates = [p for p in self.all_phrases if p not in positive_set]
        if candidates and self.negative_sampling_rate > 0:
            n_neg = min(max(1, int(len(str_cls_lst) * self.negative_sampling_rate)), len(candidates))
            negatives = random.sample(candidates, n_neg)
        else:
            negatives = []

        all_categories = str_cls_lst + negatives
        caption, cat2tokenspan = build_captions_and_token_span(all_categories, force_lowercase=True)

        # Tokenize with same truncation settings used in the model forward
        encoding = self.processor.tokenizer(
            caption,
            max_length=MAX_TEXT_LEN,
            truncation=True,
        )

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
            "encoding": encoding,
        }
        return image, target


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def make_collate_fn(processor: AutoProcessor):
    """Returns a collate_fn that batches PIL images + captions through the HF processor."""

    def collate_fn(batch: list[tuple]) -> tuple[dict, list[dict]]:
        images, targets = zip(*batch)
        captions = [t["caption"] for t in targets]
        inputs = processor(
            images=list(images),
            text=captions,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_TEXT_LEN,
            truncation=True,
        )
        return dict(inputs), list(targets)

    return collate_fn


# ---------------------------------------------------------------------------
# Hungarian Matcher
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

        # Normalize target boxes per image
        tgt_boxes_list: list[torch.Tensor] = []
        for t in targets:
            h, w = t["size"]
            scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=pred_bbox.device)
            tgt_boxes_list.append(t["boxes"].to(pred_bbox.device) / scale)
        tgt_bbox = torch.cat(tgt_boxes_list)

        # Build positive maps using per-target encoding (no shared tokenized needed)
        pos_maps: list[torch.Tensor] = []
        for t in targets:
            spans = [t["cat2tokenspan"][cls] for cls in t["str_cls_lst"]]
            pm = create_positive_map_from_span(t["encoding"], spans, max_text_len=MAX_TEXT_LEN)
            pos_maps.append(pm.to(out_prob.device))
        tgt_labels = torch.cat(pos_maps, dim=0)

        norm_tgt = tgt_labels / (tgt_labels.sum(dim=1, keepdim=True) + self.eps)
        cost_class = -(out_prob.unsqueeze(1) * norm_tgt.unsqueeze(0).float()).sum(-1)
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
# SetCriterion
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
    p = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    focal_w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - p_t).pow(gamma)
    weight = torch.where(valid_mask, torch.ones_like(ce), torch.full_like(ce, eos_coef))
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
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

    def _loss_labels(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list,
        tgt_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bs, nq, nc = outputs["pred_logits"].shape
        device = outputs["pred_logits"].device

        # text_mask: (bs, nc) — which token positions are real (not padding)
        raw_mask = outputs["text_mask"].to(device)  # (bs, L) from processor attention_mask
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
            target_labels[b, pred_idx] = batch_tgt[tgt_idx].to(device)
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
            total_steps=max(total_steps, 1),
            pct_start=0.1,
            anneal_strategy="cos",
        )
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def _prepare_batch(self, batch: tuple) -> tuple[dict, list[dict]]:
        inputs, targets = batch
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        for t in targets:
            t["boxes"] = t["boxes"].to(self.device)
            t["size"] = t["size"].to(self.device)
        return inputs, targets

    def _forward(self, inputs: dict, step_idx: int = -1) -> dict:
        outputs = self.model(**inputs)
        if step_idx == 0:
            print(f"  [debug] pred_logits: {outputs.pred_logits.shape}  pred_boxes: {outputs.pred_boxes.shape}")
        return {
            "pred_logits": outputs.pred_logits,   # (bs, nq, text_len)
            "pred_boxes": outputs.pred_boxes,       # (bs, nq, 4)
            "text_mask": inputs["attention_mask"],  # (bs, text_len)
        }

    def train_step(self, batch: tuple, step_idx: int) -> dict[str, float]:
        self.model.train()
        inputs, targets = self._prepare_batch(batch)

        try:
            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                output_dict = self._forward(inputs, targets, step_idx)
                loss_dict = self.criterion(output_dict, targets)
                if step_idx == 0:
                    print(f"  [debug] loss_dict: { {k: round(v.item(), 4) for k, v in loss_dict.items()} }")
                total_loss = sum(
                    loss_dict[k] * self.weight_dict.get(k, 1.0) for k in loss_dict
                ) / self.grad_accum_steps
        except Exception as e:
            import traceback
            print(f"  [step {step_idx}] forward error: {e}")
            traceback.print_exc()
            return {}

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (step_idx + 1) % self.grad_accum_steps == 0:
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
            inputs, targets = self._prepare_batch(batch)
            try:
                with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                    output_dict = self._forward(inputs, targets)
                    loss_dict = self.criterion(output_dict, targets)
            except Exception as e:
                print(f"  [val] forward error: {e}")
                continue
            if n == 0:
                print(f"  [val debug] { {k: round(v.item(), 4) for k, v in loss_dict.items()} }")
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
                "losses": losses,
            },
            path,
        )


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------


def freeze_backbones(model: nn.Module) -> None:
    """Freeze Swin image backbone and BERT text encoder; train fusion + detection head."""
    for name, param in model.named_parameters():
        if "backbone" in name or "language_model" in name:
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
    parser.add_argument("--output-dir", type=str, default="data/gdino_finetuned")
    parser.add_argument("--no-freeze", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--neg-rate", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 is safest with HF BatchEncoding in targets")
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

    # ---- Model + processor (auto-downloaded from HuggingFace Hub) ----
    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID)
    if not args.no_freeze:
        freeze_backbones(model)

    # ---- Datasets ----
    def make_ds(ann: str, img: str) -> COCOGroundingDataset:
        return COCOGroundingDataset(
            ann, img, processor,
            negative_sampling_rate=args.neg_rate,
            extra_classes=EXTRA_NEGATIVE_CLASSES,
        )

    train_ds = ConcatDataset([
        make_ds("Bricks.v1i.coco/train/_annotations.coco.json", "Bricks.v1i.coco/train"),
        make_ds("OUTLET.v1i.coco.balanced/train/_annotations.coco.json", "OUTLET.v1i.coco/train"),
    ])
    val_ds = ConcatDataset([
        make_ds("Bricks.v1i.coco/valid/_annotations.coco.json", "Bricks.v1i.coco/valid"),
        make_ds("OUTLET.v1i.coco.balanced/valid/_annotations.coco.json", "OUTLET.v1i.coco/valid"),
    ])
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    steps_per_epoch = max(len(train_ds) // args.batch_size, 1)
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
            print(f"  → Saved best (val={best_val_loss:.4f})")

    trainer.save(str(output_dir / "final.pth"), args.epochs, {})
    print(f"\nDone. Best val_total={best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir}/best.pth  and  {output_dir}/final.pth")


if __name__ == "__main__":
    main()
