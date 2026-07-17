import os
import argparse
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def read_id_list(list_txt: str) -> List[str]:
    ids = []
    with open(list_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def find_existing_file(dir_path: str, stem: str, exts: List[str]) -> Optional[str]:
    for ext in exts:
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return p
    return None


def find_mask(mask_dir: str, stem: str) -> Optional[str]:
    p = find_existing_file(mask_dir, stem + "_segmentation", MASK_EXTS)
    if p is not None:
        return p

    p = find_existing_file(mask_dir, stem, MASK_EXTS)
    if p is not None:
        return p

    return None


def read_rgb(path: str, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return img


def read_gray_as_rgb(path: str, size: int, color_map: bool = False) -> Image.Image:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, np.uint8)

    if color_map:
        arr_color = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
        arr_color = cv2.cvtColor(arr_color, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr_color)

    return Image.fromarray(arr).convert("RGB")


def add_title(img: Image.Image, title: str, header_h: int = 30) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + header_h), color=(255, 255, 255))
    canvas.paste(img, (0, header_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    draw.text((6, 7), title, fill=(0, 0, 0), font=font)
    return canvas


def concat_h(images: List[Image.Image]) -> Image.Image:
    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]

    canvas = Image.new("RGB", (sum(widths), max(heights)), color=(255, 255, 255))

    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    return canvas


def parse_samples(samples: str) -> Optional[List[str]]:
    if samples is None or samples.strip() == "":
        return None
    return [s.strip() for s in samples.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default="/data/zjy_work/ISIC2018/train/Images",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/data/zjy_work/ISIC2018/train/Masks",
    )
    parser.add_argument(
        "--pred_mask_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--feedback_dir",
        type=str,
        required=True,
        help="Output dir of build_boundary_feedback.py",
    )
    parser.add_argument(
        "--list_txt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/zjy_work/Work3_BEF_SBG/results/visual_feedback",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="",
        help="Comma-separated sample stems. If empty, use first max_samples from list_txt.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=224,
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ids = read_id_list(args.list_txt)
    user_samples = parse_samples(args.samples)

    if user_samples is not None:
        ids = user_samples
    else:
        ids = ids[: args.max_samples]

    for stem in ids:
        img_path = find_existing_file(args.image_dir, stem, IMG_EXTS)
        gt_path = find_mask(args.mask_dir, stem)
        pred_path = find_existing_file(args.pred_mask_dir, stem, MASK_EXTS)

        error_path = find_existing_file(os.path.join(args.feedback_dir, "error_abs"), stem, MASK_EXTS)
        diff_path = find_existing_file(os.path.join(args.feedback_dir, "difficulty"), stem, MASK_EXTS)
        prior_path = find_existing_file(os.path.join(args.feedback_dir, "adaptive_boundary_prior"), stem, MASK_EXTS)

        missing = []
        for name, path in [
            ("image", img_path),
            ("gt", gt_path),
            ("teacher_pred", pred_path),
            ("error_abs", error_path),
            ("difficulty", diff_path),
            ("adaptive_prior", prior_path),
        ]:
            if path is None:
                missing.append(name)

        if len(missing) > 0:
            print(f"[WARN] skip {stem}, missing: {missing}")
            continue

        tile = args.tile_size

        panels = [
            add_title(read_rgb(img_path, tile), "Image"),
            add_title(read_gray_as_rgb(gt_path, tile, color_map=False), "GT"),
            add_title(read_gray_as_rgb(pred_path, tile, color_map=False), "Teacher Pred"),
            add_title(read_gray_as_rgb(error_path, tile, color_map=True), "Error map"),
            add_title(read_gray_as_rgb(diff_path, tile, color_map=True), "Difficulty map"),
            add_title(read_gray_as_rgb(prior_path, tile, color_map=True), "Adaptive Prior"),
        ]

        collage = concat_h(panels)
        out_path = os.path.join(args.out_dir, f"{stem}_feedback.png")
        collage.save(out_path)

        print(f"[SAVE] {out_path}")

    print(f"[DONE] visual feedback saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
