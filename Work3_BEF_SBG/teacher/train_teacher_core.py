from __future__ import annotations

import argparse
import csv
import os
import time
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from Work3_BEF_SBG.hetero_teachers.common import (
    ISICTeacherDataset,
    dice_score,
    save_teacher_checkpoint,
    seed_everything,
    segmentation_boundary_loss,
)
from Work3_BEF_SBG.hetero_teachers.factory import build_teacher_for_training


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--teacher_type", required=True, choices=["cnn", "sam_adapter"])
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--boundary_weight", type=float, default=0.35)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--image_size", type=int, default=0)
    parser.add_argument("--sam_type", default="vit_b")
    parser.add_argument("--sam_base_checkpoint", default="")
    parser.add_argument("--sam_unfreeze_last_blocks", type=int, default=0)
    parser.add_argument("--sam_adapter_bottleneck", type=int, default=64)
    return parser


def build_loaders(args, image_size: int, sam_input: bool):
    train_dataset = ISICTeacherDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        list_txt=args.train_list,
        image_size=image_size,
        augment=True,
        sam_input=sam_input,
    )
    val_dataset = ISICTeacherDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        list_txt=args.val_list,
        image_size=image_size,
        augment=False,
        sam_input=sam_input,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) >= args.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    batches = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        boundary = batch["boundary"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            mask_logits, boundary_logits = model(image)
            loss, _ = segmentation_boundary_loss(
                mask_logits,
                boundary_logits,
                mask,
                boundary,
                boundary_weight=args.boundary_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            dice = dice_score(torch.sigmoid(mask_logits), mask)
        total_loss += float(loss.detach())
        total_dice += float(dice.detach())
        batches += 1
    return total_loss / max(batches, 1), total_dice / max(batches, 1)


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    batches = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        boundary = batch["boundary"].to(device, non_blocking=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            mask_logits, boundary_logits = model(image)
            loss, _ = segmentation_boundary_loss(
                mask_logits,
                boundary_logits,
                mask,
                boundary,
                boundary_weight=args.boundary_weight,
            )
        dice = dice_score(torch.sigmoid(mask_logits), mask)
        total_loss += float(loss)
        total_dice += float(dice)
        batches += 1
    return total_loss / max(batches, 1), total_dice / max(batches, 1)


def main(args=None):
    parser = add_common_arguments(argparse.ArgumentParser())
    args = parser.parse_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    model, metadata = build_teacher_for_training(
        teacher_type=args.teacher_type,
        pretrained=not args.no_pretrained,
        sam_type=args.sam_type,
        sam_base_checkpoint=args.sam_base_checkpoint or None,
        sam_freeze_encoder=True,
        sam_unfreeze_last_blocks=args.sam_unfreeze_last_blocks,
        sam_adapter_bottleneck=args.sam_adapter_bottleneck,
    )
    image_size = args.image_size or int(metadata.image_size)
    metadata.image_size = image_size
    sam_input = args.teacher_type == "sam_adapter"
    model = model.to(device)

    train_loader, val_loader = build_loaders(args, image_size=image_size, sam_input=sam_input)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_lr,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    history_path = os.path.join(args.save_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "lr", "train_loss", "train_dice", "val_loss", "val_dice", "seconds"])

    best_dice = -1.0
    best_path = os.path.join(args.save_dir, f"{args.teacher_type}-best.pth")
    last_path = os.path.join(args.save_dir, f"{args.teacher_type}-last.pth")

    print("[Teacher Training]")
    print("  type:", args.teacher_type)
    print("  train samples:", len(train_loader.dataset))
    print("  val samples:", len(val_loader.dataset))
    print("  image size:", image_size)
    print("  trainable parameters:", sum(p.numel() for p in trainable))

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args
        )
        val_loss, val_dice = validate(model, val_loader, device, args)
        elapsed = time.time() - start
        current_lr = float(optimizer.param_groups[0]["lr"])

        save_teacher_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_dice=max(best_dice, val_dice),
            metadata=metadata,
        )
        if val_dice > best_dice:
            best_dice = val_dice
            save_teacher_checkpoint(
                best_path,
                model,
                None,
                epoch,
                best_dice=best_dice,
                metadata=metadata,
            )

        with open(history_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                epoch,
                f"{current_lr:.8e}",
                f"{train_loss:.6f}",
                f"{train_dice:.6f}",
                f"{val_loss:.6f}",
                f"{val_dice:.6f}",
                f"{elapsed:.2f}",
            ])
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"lr={current_lr:.3e} "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"best={best_dice:.4f} time={elapsed:.1f}s",
            flush=True,
        )
        scheduler.step()

    print("[DONE]")
    print("  best:", best_path)
    print("  last:", last_path)
    print("  best_dice:", best_dice)


if __name__ == "__main__":
    main()
