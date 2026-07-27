from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from Work3_BEF_SBG.hetero_teachers.common import (
    IMG_EXTS,
    ImageOnlyDataset,
    collate_image_only,
    find_existing_file,
    list_images,
    read_id_list,
    save_gray,
)
from Work3_BEF_SBG.hetero_teachers.factory import (
    build_teacher_from_checkpoint,
    teacher_expects_sam_input,
)


def resolve_images(image_dir: str, list_txt: Optional[str]) -> List[str]:
    if not list_txt:
        return list_images(image_dir)
    paths: List[str] = []
    for stem in read_id_list(list_txt):
        path = find_existing_file(image_dir, stem, IMG_EXTS)
        if path is None:
            print(f"[WARN] missing image: {stem}")
            continue
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_type", required=True, choices=["bgdnet", "cnn", "sam_adapter"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--list_txt", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--bgdnet_root", default="/data/zjy_work/BGDNet")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, metadata = build_teacher_from_checkpoint(
        checkpoint_path=args.checkpoint,
        teacher_type=args.teacher_type,
        device=device,
        bgdnet_root=args.bgdnet_root,
        strict=True,
    )
    image_size = args.image_size or int(metadata.get("image_size", 352))
    sam_input = teacher_expects_sam_input(metadata)
    image_paths = resolve_images(args.image_dir, args.list_txt or None)
    if not image_paths:
        raise RuntimeError("No input images found.")

    mask_dir = os.path.join(args.out_dir, "pred_masks")
    boundary_dir = os.path.join(args.out_dir, "pred_boundaries")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(boundary_dir, exist_ok=True)

    dataset = ImageOnlyDataset(image_paths, image_size=image_size, sam_input=sam_input)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_image_only,
    )

    processed = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with autocast(enabled=args.amp and device.type == "cuda"):
                output = model(images)
            if isinstance(output, (tuple, list)):
                mask_logits = output[0]
                boundary_logits = output[1] if len(output) > 1 else None
            else:
                mask_logits = output
                boundary_logits = None

            for index, stem in enumerate(batch["stem"]):
                height, width = batch["original_size"][index]
                mask_prob = torch.sigmoid(
                    F.interpolate(
                        mask_logits[index : index + 1],
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )[0, 0].float().cpu().numpy()
                save_gray(mask_prob, os.path.join(mask_dir, stem + ".png"))

                if boundary_logits is not None:
                    boundary_prob = torch.sigmoid(
                        F.interpolate(
                            boundary_logits[index : index + 1],
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        )
                    )[0, 0].float().cpu().numpy()
                    save_gray(boundary_prob, os.path.join(boundary_dir, stem + ".png"))
                processed += 1
                if processed % 50 == 0 or processed == len(dataset):
                    print(f"[{processed}/{len(dataset)}]", flush=True)

    print("[DONE] teacher directory inference")
    print("  teacher_type:", args.teacher_type)
    print("  checkpoint:", args.checkpoint)
    print("  out_dir:", args.out_dir)
    print("  images:", processed)


if __name__ == "__main__":
    main()
