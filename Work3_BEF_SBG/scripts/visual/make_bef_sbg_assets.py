#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_bef_sbg_figure_assets.py

Purpose:
    Generate real visual assets for the BEF-SBG method figure.

Required completed steps:
    Step 03: OOF teacher prediction
    Step 04: boundary feedback construction
    Step 09: BEF-SBG diffusion generation, optional for generated samples
    Step 10: generated sample scoring, optional for reliability card
    Step 13: final BGDNet-BEF testing, optional for final model output

Example:
python /data/zjy_work/Work3_BEF_SBG/scripts/visual/make_bef_sbg_assets.py \
  --ratio_tag 5p \
  --seed 0 \
  --max_samples 3 \
  --out_dir /data/zjy_work/Work3_BEF_SBG/results/figure_assets_bef_sbg/ISIC2018_5p

python /data/zjy_work/Work3_BEF_SBG/scripts/visual/make_bef_sbg_assets.py \
    --ratio_tag 5p \
    --seed 0 \
    --samples ISIC_0000000,ISIC_0000025,ISIC_0000100 \
    --out_dir /data/zjy_work/Work3_BEF_SBG/results/figure_assets_bef_sbg/ISIC2018_5p


If --samples is empty, the script will automatically choose available cases
from low_5p_all.txt.
"""

import os
import csv
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


# ============================================================
# Basic file helpers
# ============================================================

def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def strip_stem(x: str) -> str:
    stem = os.path.splitext(os.path.basename(x.strip()))[0]
    if stem.endswith("_segmentation"):
        stem = stem.replace("_segmentation", "")
    return stem


def read_id_list(list_txt: str) -> List[str]:
    ids = []
    with open(list_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(strip_stem(s))
    return ids


def find_existing_file(dir_path: str, stem: str, exts: List[str]) -> Optional[str]:
    if dir_path is None or not os.path.isdir(dir_path):
        return None
    for ext in exts:
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return p
    return None


def find_mask(mask_dir: str, stem: str) -> Optional[str]:
    p = find_existing_file(mask_dir, stem + "_segmentation", MASK_EXTS)
    if p is not None:
        return p
    return find_existing_file(mask_dir, stem, MASK_EXTS)


def list_images(dir_path: str) -> List[str]:
    if dir_path is None or not os.path.isdir(dir_path):
        return []
    files = []
    for ext in IMG_EXTS + MASK_EXTS:
        files.extend(Path(dir_path).glob(f"*{ext}"))
    return sorted([str(p) for p in files])


# ============================================================
# Image loading and visualization
# ============================================================

def minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)


def read_rgb(path: str, size: Optional[int] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def read_gray_float(path: str, size: Optional[int] = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def save_rgb(arr: np.ndarray, path: str):
    mkdir(os.path.dirname(path))
    arr = np.asarray(arr)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_gray_white(mask: np.ndarray, path: str):
    m = minmax(mask)
    out = (m * 255.0).round().astype(np.uint8)
    save_rgb(np.stack([out, out, out], axis=-1), path)


def heatmap_on_black(
    x: np.ndarray,
    cmap_name: str = "inferno",
    gamma: float = 0.75
) -> np.ndarray:
    """
    Convert a [0,1] map to a publication-style heatmap on black background.
    Good for boundary error / uncertainty / adaptive prior.
    """
    x = minmax(x)
    x_show = np.power(x, gamma)
    cmap = plt.get_cmap(cmap_name)
    color = cmap(x_show)[..., :3]
    alpha = x_show[..., None]
    out = color * alpha
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def overlay_heatmap(
    rgb: np.ndarray,
    heat: np.ndarray,
    cmap_name: str = "inferno",
    alpha: float = 0.55,
    gamma: float = 0.75
) -> np.ndarray:
    rgb_f = np.asarray(rgb, dtype=np.float32) / 255.0
    heat = minmax(heat)
    heat_show = np.power(heat, gamma)
    cmap = plt.get_cmap(cmap_name)
    color = cmap(heat_show)[..., :3]
    alpha_map = alpha * heat_show[..., None]
    out = rgb_f * (1.0 - alpha_map) + color * alpha_map
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def add_title(img: np.ndarray, title: str, header_h: int = 34) -> Image.Image:
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGB")
    w, h = pil.size
    canvas = Image.new("RGB", (w, h + header_h), "white")
    canvas.paste(pil, (0, header_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    draw.text((8, 8), title, fill=(20, 35, 60), font=font)
    return canvas


def concat_h(items: List[Image.Image], gap: int = 8) -> Image.Image:
    if len(items) == 0:
        raise ValueError("concat_h receives empty list")
    widths = [im.size[0] for im in items]
    heights = [im.size[1] for im in items]
    canvas = Image.new("RGB", (sum(widths) + gap * (len(items) - 1), max(heights)), "white")
    x = 0
    for im in items:
        canvas.paste(im, (x, 0))
        x += im.size[0] + gap
    return canvas


def concat_v(items: List[Image.Image], gap: int = 12) -> Image.Image:
    if len(items) == 0:
        raise ValueError("concat_v receives empty list")
    widths = [im.size[0] for im in items]
    heights = [im.size[1] for im in items]
    canvas = Image.new("RGB", (max(widths), sum(heights) + gap * (len(items) - 1)), "white")
    y = 0
    for im in items:
        canvas.paste(im, (0, y))
        y += im.size[1] + gap
    return canvas


# ============================================================
# Generated sample and scoring helpers
# ============================================================

def find_generated_by_index(gen_subdir: str, original_stem: str, index: int) -> Optional[str]:
    """
    Step 09 generated outputs are often named like:
        id-000000_s-00_idx-0.png

    This function first tries original_stem, then tries id-{index:06} prefix.
    """
    direct = find_existing_file(gen_subdir, original_stem, IMG_EXTS + MASK_EXTS)
    if direct is not None:
        return direct

    if not os.path.isdir(gen_subdir):
        return None

    prefix = f"id-{index:06}"
    candidates = []
    for p in Path(gen_subdir).glob(prefix + "*.png"):
        candidates.append(str(p))
    for p in Path(gen_subdir).glob(prefix + "*.jpg"):
        candidates.append(str(p))

    if len(candidates) == 0:
        return None
    return sorted(candidates)[0]


def load_score_csv(score_csv: str) -> Dict[str, Dict[str, str]]:
    """
    Key by generated image basename stem.
    """
    score = {}
    if score_csv is None or not os.path.exists(score_csv):
        return score

    with open(score_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row.get("image_path", "")
            if image_path:
                score[Path(image_path).stem] = row
    return score


def make_reliability_card(
    record: Optional[Dict[str, str]],
    size: Tuple[int, int] = (320, 256)
) -> np.ndarray:
    """
    Create a simple reliability-weight card as visual material.
    """
    w, h = size
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font_title = ImageFont.load_default()
        font = ImageFont.load_default()

    draw.text((16, 14), "Reliability scoring", fill=(20, 35, 60), font=font_title)

    if record is None:
        draw.text((16, 60), "No scoring CSV item", fill=(120, 120, 120), font=font)
        return np.asarray(canvas)

    def safe_float(key: str, default: float = 0.0) -> float:
        try:
            x = float(record.get(key, default))
            if not math.isfinite(x):
                return default
            return x
        except Exception:
            return default

    values = [
        ("Dice consistency", safe_float("consistency_dice")),
        ("Boundary difficulty", safe_float("boundary_difficulty")),
        ("Sample weight", safe_float("weight")),
    ]

    accepted = str(record.get("accepted", "0"))
    y = 58
    for name, value in values:
        draw.text((16, y), name, fill=(40, 40, 40), font=font)
        bar_x, bar_y = 16, y + 24
        bar_w, bar_h = w - 70, 15
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=(220, 220, 220))
        fill_w = int(bar_w * max(0.0, min(value, 1.5)) / 1.5)
        draw.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=(235, 130, 25))
        draw.text((bar_x + bar_w + 8, bar_y - 3), f"{value:.3f}", fill=(40, 40, 40), font=font)
        y += 58

    status = "Accepted" if accepted == "1" else "Rejected / low confidence"
    fill = (40, 120, 60) if accepted == "1" else (160, 60, 50)
    draw.text((16, h - 32), status, fill=fill, font=font_title)

    return np.asarray(canvas)


# ============================================================
# Main asset generation
# ============================================================

def resolve_paths(args):
    work_root = args.work_root
    args.split_dir = args.split_dir or f"{work_root}/splits/ISIC2018_seed{args.seed}"
    args.list_txt = args.list_txt or f"{args.split_dir}/low_{args.ratio_tag}_all.txt"

    args.oof_dir = args.oof_dir or f"{work_root}/results/oof_teacher/ISIC2018_{args.ratio_tag}"
    args.feedback_dir = args.feedback_dir or f"{work_root}/results/boundary_feedback/ISIC2018_{args.ratio_tag}"

    args.gen_dir = args.gen_dir or f"{work_root}/results/generated_bef_sbg/{args.ratio_tag}"
    args.score_csv = args.score_csv or f"{work_root}/results/generated_bef_sbg_scores_{args.ratio_tag}.csv"

    args.final_eval_dir = args.final_eval_dir or f"{work_root}/results/eval_final_bef_{args.ratio_tag}"

    args.train_image_dir = args.train_image_dir or f"{args.isic_train_root}/Images"
    args.train_mask_dir = args.train_mask_dir or f"{args.isic_train_root}/Masks"

    args.test_image_dir = args.test_image_dir or f"{args.isic_test_root}/Images"
    args.test_mask_dir = args.test_mask_dir or f"{args.isic_test_root}/Masks"

    args.out_dir = args.out_dir or f"{work_root}/results/figure_assets_bef_sbg/ISIC2018_{args.ratio_tag}"
    mkdir(args.out_dir)


def choose_stems(args, all_ids: List[str]) -> List[Tuple[str, int]]:
    if args.samples.strip():
        wanted = [strip_stem(x) for x in args.samples.split(",") if x.strip()]
        index_map = {s: i for i, s in enumerate(all_ids)}
        return [(s, index_map.get(s, -1)) for s in wanted]

    chosen = []
    pred_dir = os.path.join(args.oof_dir, "pred_masks")
    fb_required = [
        os.path.join(args.feedback_dir, "error_abs"),
        os.path.join(args.feedback_dir, "entropy"),
        os.path.join(args.feedback_dir, "soft_boundary_gt"),
        os.path.join(args.feedback_dir, "difficulty"),
        os.path.join(args.feedback_dir, "adaptive_boundary_prior"),
    ]

    for idx, stem in enumerate(all_ids):
        img = find_existing_file(args.train_image_dir, stem, IMG_EXTS)
        gt = find_mask(args.train_mask_dir, stem)
        pred = find_existing_file(pred_dir, stem, MASK_EXTS)
        maps_ok = all(find_existing_file(d, stem, MASK_EXTS) is not None for d in fb_required)
        if img and gt and pred and maps_ok:
            chosen.append((stem, idx))
        if len(chosen) >= args.max_samples:
            break

    return chosen


def get_final_prediction(args, prefer_stem: str) -> Tuple[Optional[str], Optional[str]]:
    final_mask_dir = os.path.join(args.final_eval_dir, "masks")
    final_bnd_dir = os.path.join(args.final_eval_dir, "boundaries")

    pred = find_existing_file(final_mask_dir, prefer_stem, MASK_EXTS)
    bnd = find_existing_file(final_bnd_dir, prefer_stem, MASK_EXTS)

    if pred is not None:
        return pred, bnd

    all_preds = list_images(final_mask_dir)
    if len(all_preds) == 0:
        return None, None

    pred = all_preds[0]
    stem = Path(pred).stem
    bnd = find_existing_file(final_bnd_dir, stem, MASK_EXTS)
    return pred, bnd


def process_one_case(args, stem: str, index: int, score_map: Dict[str, Dict[str, str]]):
    tile = args.tile_size
    case_dir = os.path.join(args.out_dir, stem)
    mkdir(case_dir)

    image_path = find_existing_file(args.train_image_dir, stem, IMG_EXTS)
    mask_path = find_mask(args.train_mask_dir, stem)
    pred_path = find_existing_file(os.path.join(args.oof_dir, "pred_masks"), stem, MASK_EXTS)

    error_path = find_existing_file(os.path.join(args.feedback_dir, "error_abs"), stem, MASK_EXTS)
    entropy_path = find_existing_file(os.path.join(args.feedback_dir, "entropy"), stem, MASK_EXTS)
    soft_path = find_existing_file(os.path.join(args.feedback_dir, "soft_boundary_gt"), stem, MASK_EXTS)
    diff_path = find_existing_file(os.path.join(args.feedback_dir, "difficulty"), stem, MASK_EXTS)
    prior_path = find_existing_file(os.path.join(args.feedback_dir, "adaptive_boundary_prior"), stem, MASK_EXTS)

    required = {
        "image": image_path,
        "gt_mask": mask_path,
        "teacher_oof_pred": pred_path,
        "boundary_error": error_path,
        "uncertainty_entropy": entropy_path,
        "soft_boundary": soft_path,
        "difficulty": diff_path,
        "adaptive_prior": prior_path,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        print(f"[WARN] skip {stem}, missing: {missing}")
        return

    # ----------------------------
    # A/B: teacher feedback assets
    # ----------------------------
    rgb = read_rgb(image_path, tile)
    gt = read_gray_float(mask_path, tile)
    pred_prob = read_gray_float(pred_path, tile)
    pred_mask = (pred_prob >= 0.5).astype(np.float32)

    error = read_gray_float(error_path, tile)
    entropy = read_gray_float(entropy_path, tile)
    soft = read_gray_float(soft_path, tile)
    difficulty = read_gray_float(diff_path, tile)
    prior = read_gray_float(prior_path, tile)

    save_rgb(rgb, os.path.join(case_dir, "01_low_label_image.png"))
    save_gray_white(gt, os.path.join(case_dir, "02_gt_mask_white.png"))
    save_rgb(heatmap_on_black(pred_prob, "turbo"), os.path.join(case_dir, "03_teacher_oof_prediction_prob_heatmap.png"))
    save_gray_white(pred_mask, os.path.join(case_dir, "04_teacher_oof_prediction_mask.png"))

    save_rgb(heatmap_on_black(error, "inferno"), os.path.join(case_dir, "05_boundary_error_heatmap.png"))
    save_rgb(heatmap_on_black(entropy, "turbo"), os.path.join(case_dir, "06_uncertainty_entropy_heatmap.png"))
    save_rgb(heatmap_on_black(soft, "Blues"), os.path.join(case_dir, "07_soft_boundary_band.png"))
    save_rgb(heatmap_on_black(difficulty, "magma"), os.path.join(case_dir, "08_difficulty_heatmap.png"))
    save_rgb(heatmap_on_black(prior, "inferno"), os.path.join(case_dir, "09_adaptive_soft_boundary_prior.png"))

    save_rgb(overlay_heatmap(rgb, error, "inferno"), os.path.join(case_dir, "10_boundary_error_overlay.png"))
    save_rgb(overlay_heatmap(rgb, entropy, "turbo"), os.path.join(case_dir, "11_uncertainty_overlay.png"))
    save_rgb(overlay_heatmap(rgb, prior, "inferno"), os.path.join(case_dir, "12_adaptive_prior_overlay.png"))

    panel_a = concat_h([
        add_title(rgb, "Image"),
        add_title(np.stack([(gt * 255).astype(np.uint8)] * 3, axis=-1), "GT"),
        add_title(np.stack([(pred_mask * 255).astype(np.uint8)] * 3, axis=-1), "Teacher OOF Pred"),
        add_title(heatmap_on_black(error, "inferno"), "Boundary Error"),
        add_title(heatmap_on_black(entropy, "turbo"), "Uncertainty"),
    ])
    panel_a.save(os.path.join(case_dir, "panel_A_error_aware_feedback_mining.png"))

    panel_b = concat_h([
        add_title(heatmap_on_black(soft, "Blues"), "Soft Boundary"),
        add_title(heatmap_on_black(error, "inferno"), "Boundary Error"),
        add_title(heatmap_on_black(entropy, "turbo"), "Uncertainty"),
        add_title(heatmap_on_black(difficulty, "magma"), "Difficulty"),
        add_title(heatmap_on_black(prior, "inferno"), "Adaptive Prior"),
    ])
    panel_b.save(os.path.join(case_dir, "panel_B_adaptive_prior_construction.png"))

    # ----------------------------
    # C: generated sample + weight
    # ----------------------------
    gen_img = find_generated_by_index(os.path.join(args.gen_dir, "images"), stem, index)
    gen_mask = None
    gen_prior = None
    weight_card = None

    if gen_img is not None:
        gen_stem = Path(gen_img).stem
        gen_mask = find_existing_file(os.path.join(args.gen_dir, "masks"), gen_stem, MASK_EXTS)
        gen_prior = find_existing_file(os.path.join(args.gen_dir, "boundary_prior"), gen_stem, MASK_EXTS)

        gen_rgb = read_rgb(gen_img, tile)
        save_rgb(gen_rgb, os.path.join(case_dir, "13_generated_image.png"))

        if gen_mask is not None:
            gen_m = read_gray_float(gen_mask, tile)
            save_gray_white(gen_m, os.path.join(case_dir, "14_generated_mask.png"))

        if gen_prior is not None:
            gen_p = read_gray_float(gen_prior, tile)
            save_rgb(heatmap_on_black(gen_p, "inferno"), os.path.join(case_dir, "15_generated_boundary_prior.png"))

        score_record = score_map.get(gen_stem)
        weight_card = make_reliability_card(score_record, size=(tile, tile))
        save_rgb(weight_card, os.path.join(case_dir, "16_reliability_weight_card.png"))
    else:
        print(f"[INFO] no generated image found for {stem}, index={index}")

    # ----------------------------
    # Final BGDNet-BEF output from Step 13
    # ----------------------------
    final_pred, final_bnd = get_final_prediction(args, prefer_stem=stem)

    final_pred_img = None
    final_bnd_img = None

    if final_pred is not None:
        final_pred_prob = read_gray_float(final_pred, tile)
        final_pred_img = np.stack([((final_pred_prob >= 0.5) * 255).astype(np.uint8)] * 3, axis=-1)
        save_rgb(final_pred_img, os.path.join(case_dir, "17_final_bgdnet_bef_pred_mask.png"))

    if final_bnd is not None:
        final_bnd_prob = read_gray_float(final_bnd, tile)
        final_bnd_img = heatmap_on_black(final_bnd_prob, "inferno")
        save_rgb(final_bnd_img, os.path.join(case_dir, "18_final_bgdnet_bef_boundary.png"))

    panel_c_items = [
        add_title(heatmap_on_black(prior, "inferno"), "Adaptive Prior"),
        add_title(np.stack([(gt * 255).astype(np.uint8)] * 3, axis=-1), "Mask Condition"),
    ]

    if gen_img is not None:
        panel_c_items.append(add_title(read_rgb(gen_img, tile), "Generated Image"))

    if weight_card is not None:
        panel_c_items.append(add_title(weight_card, "Reliability Weight"))

    if final_pred_img is not None:
        panel_c_items.append(add_title(final_pred_img, "Final BGDNet"))

    panel_c = concat_h(panel_c_items)
    panel_c.save(os.path.join(case_dir, "panel_C_generation_and_weighted_training.png"))

    panel_full = concat_v([
        panel_a,
        panel_b,
        panel_c,
    ])
    panel_full.save(os.path.join(case_dir, "panel_full_BEF_SBG_assets.png"))

    print(f"[SAVE] {case_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--work_root", type=str, default="/data/zjy_work/Work3_BEF_SBG")
    parser.add_argument("--isic_train_root", type=str, default="/data/zjy_work/ISIC2018/train")
    parser.add_argument("--isic_test_root", type=str, default="/data/zjy_work/ISIC2018/test")

    parser.add_argument("--ratio_tag", type=str, default="5p")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--split_dir", type=str, default=None)
    parser.add_argument("--list_txt", type=str, default=None)

    parser.add_argument("--train_image_dir", type=str, default=None)
    parser.add_argument("--train_mask_dir", type=str, default=None)
    parser.add_argument("--test_image_dir", type=str, default=None)
    parser.add_argument("--test_mask_dir", type=str, default=None)

    parser.add_argument("--oof_dir", type=str, default=None)
    parser.add_argument("--feedback_dir", type=str, default=None)
    parser.add_argument("--gen_dir", type=str, default=None)
    parser.add_argument("--score_csv", type=str, default=None)
    parser.add_argument("--final_eval_dir", type=str, default=None)

    parser.add_argument("--samples", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()
    resolve_paths(args)

    print("============================================================")
    print("[BEF-SBG Figure Asset Generator]")
    print(f"ratio_tag       = {args.ratio_tag}")
    print(f"seed            = {args.seed}")
    print(f"list_txt        = {args.list_txt}")
    print(f"oof_dir         = {args.oof_dir}")
    print(f"feedback_dir    = {args.feedback_dir}")
    print(f"gen_dir         = {args.gen_dir}")
    print(f"score_csv       = {args.score_csv}")
    print(f"final_eval_dir  = {args.final_eval_dir}")
    print(f"out_dir         = {args.out_dir}")
    print("============================================================")

    if not os.path.exists(args.list_txt):
        raise FileNotFoundError(f"list_txt not found: {args.list_txt}")

    all_ids = read_id_list(args.list_txt)
    chosen = choose_stems(args, all_ids)

    if len(chosen) == 0:
        raise RuntimeError(
            "No valid samples found. Please check Step 03 and Step 04 outputs:\n"
            f"  OOF pred masks: {args.oof_dir}/pred_masks\n"
            f"  Feedback dir:   {args.feedback_dir}"
        )

    score_map = load_score_csv(args.score_csv)

    for stem, index in chosen:
        process_one_case(args, stem, index, score_map)

    print("[DONE] figure assets generated.")


if __name__ == "__main__":
    main()
