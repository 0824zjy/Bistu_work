import os
import csv
import math
import argparse
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def find_existing_file(dir_path: Optional[str], stem: str, exts: List[str]) -> Optional[str]:
    """
    Find file by stem under dir_path.
    """
    if dir_path is None:
        return None

    for ext in exts:
        path = os.path.join(dir_path, stem + ext)
        if os.path.exists(path):
            return path

    return None


def find_mask(mask_dir: str, stem: str) -> Optional[str]:
    """
    ISIC masks may be named:
      stem_segmentation.png
      stem.png
    """
    path = find_existing_file(mask_dir, stem + "_segmentation", IMG_EXTS)
    if path is not None:
        return path

    path = find_existing_file(mask_dir, stem, IMG_EXTS)
    if path is not None:
        return path

    return None


def read_list_txt(list_txt: str) -> List[str]:
    """
    Read image ids from txt.

    Compatible with:
      ISIC_0000000
      ISIC_0000000.jpg
      /path/to/ISIC_0000000.jpg
      ISIC_0000000_segmentation.png
    """
    ids = []

    with open(list_txt, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()

            if not item or item.startswith("#"):
                continue

            stem = os.path.splitext(os.path.basename(item))[0]

            if stem.endswith("_segmentation"):
                stem = stem.replace("_segmentation", "")

            ids.append(stem)

    return ids


def read_gray_float(path: str, resize_to=None) -> np.ndarray:
    """
    Read grayscale image as float32 map in [0, 1].

    This function supports:
      1. Binary mask saved as 0/255;
      2. Soft probability map saved as 0~255;
      3. Already-normalized float-like image.

    Args:
        path: image path.
        resize_to: optional (width, height).
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)

    if resize_to is not None:
        arr = cv2.resize(arr, resize_to, interpolation=cv2.INTER_LINEAR)

    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)

    if arr.max() > 1.0:
        arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)

    return arr


def save_gray_png(arr: np.ndarray, path: str):
    """
    Save float map in [0,1] to uint8 grayscale png.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)

    out = (arr * 255.0).round().astype(np.uint8)
    cv2.imwrite(path, out)


def hard_boundary_from_mask(mask_bin: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Build 1-pixel-ish hard boundary from binary GT mask using morphology gradient.
    """
    mask_bin = (mask_bin > 0.5).astype(np.uint8)

    k = int(kernel_size)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    kernel = np.ones((k, k), np.uint8)

    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    eroded = cv2.erode(mask_bin, kernel, iterations=1)

    boundary = dilated.astype(np.float32) - eroded.astype(np.float32)
    boundary = np.clip(boundary, 0.0, 1.0).astype(np.float32)

    return boundary


def soft_boundary_from_hard(
    hard_boundary: np.ndarray,
    radius: int = 12,
    tau: float = 4.0,
) -> np.ndarray:
    """
    Build soft boundary prior from hard boundary by outward dilation with exponential decay.

    soft_boundary = max over shells:
        exp(-r / tau)
    """
    hard = (hard_boundary > 0.5).astype(np.float32)

    soft = hard.copy()
    previous = hard.copy()

    radius = int(radius)
    tau = max(float(tau), 1e-6)

    for r in range(1, radius + 1):
        k = 2 * r + 1
        kernel = np.ones((k, k), np.uint8)

        dilated = cv2.dilate(hard, kernel, iterations=1)
        shell = np.clip(dilated - previous, 0.0, 1.0)

        shell_weight = math.exp(-float(r) / tau)
        soft = np.maximum(soft, shell * shell_weight)

        previous = dilated

    soft = np.nan_to_num(soft, nan=0.0, posinf=1.0, neginf=0.0)
    soft = np.clip(soft, 0.0, 1.0).astype(np.float32)

    return soft


def safe_binary_entropy(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Numerically stable binary entropy:
        H(p) = -p log(p) - (1-p) log(1-p)

    Direct entropy on p=0 or p=1 causes:
        0 * log(0) = nan

    Therefore probability is clipped to [eps, 1-eps].
    """
    prob = np.asarray(prob, dtype=np.float32)
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, eps, 1.0 - eps)

    entropy = -prob * np.log(prob) - (1.0 - prob) * np.log(1.0 - prob)

    entropy = np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    entropy = np.clip(entropy, 0.0, 0.6931472).astype(np.float32)

    return entropy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)

    # In the current OOF results, pred_masks are actually soft probability maps saved as 0~255 png.
    parser.add_argument("--pred_mask_dir", type=str, required=True)

    # Kept for compatibility with previous shell script.
    # This file version does not rely on pred_boundary_dir.
    parser.add_argument("--pred_boundary_dir", type=str, default=None)
    # Optional heterogeneous-teacher uncertainty maps.
    parser.add_argument("--pred_variance_dir", type=str, default=None)
    parser.add_argument("--pred_reliability_dir", type=str, default=None)

    parser.add_argument("--list_txt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--tau0", type=float, default=4.0)
    parser.add_argument("--kernel", type=int, default=3)

    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--lambda_u", type=float, default=0.75)
    parser.add_argument("--lambda_epi", type=float, default=0.75)
    parser.add_argument("--reliability_floor", type=float, default=0.05)

    # More emphasis on false-negative boundary errors.
    # In lesion segmentation, missing lesion pixels usually hurts Dice more than
    # slightly over-segmenting background pixels.
    parser.add_argument("--w_fn", type=float, default=1.5)
    parser.add_argument("--w_fp", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    out_hard_dir = os.path.join(args.out_dir, "hard_boundary")
    out_soft_dir = os.path.join(args.out_dir, "soft_boundary_gt")
    out_error_abs_dir = os.path.join(args.out_dir, "error_abs")
    out_error_fn_dir = os.path.join(args.out_dir, "error_fn")
    out_error_fp_dir = os.path.join(args.out_dir, "error_fp")
    out_entropy_dir = os.path.join(args.out_dir, "entropy")
    out_variance_dir = os.path.join(args.out_dir, "epistemic_variance")
    out_reliability_dir = os.path.join(args.out_dir, "teacher_reliability")
    out_difficulty_dir = os.path.join(args.out_dir, "difficulty")
    out_prior_dir = os.path.join(args.out_dir, "adaptive_boundary_prior")

    for d in [
        out_hard_dir,
        out_soft_dir,
        out_error_abs_dir,
        out_error_fn_dir,
        out_error_fp_dir,
        out_entropy_dir,
        out_variance_dir,
        out_reliability_dir,
        out_difficulty_dir,
        out_prior_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    stems = read_list_txt(args.list_txt)

    rows = []
    miss_mask = 0
    miss_pred = 0
    nan_count = 0

    for idx, stem in enumerate(stems):
        mask_path = find_mask(args.mask_dir, stem)
        pred_path = find_existing_file(args.pred_mask_dir, stem, IMG_EXTS)
        variance_path = (
            find_existing_file(args.pred_variance_dir, stem, IMG_EXTS)
            if args.pred_variance_dir
            else None
        )
        reliability_path = (
            find_existing_file(args.pred_reliability_dir, stem, IMG_EXTS)
            if args.pred_reliability_dir
            else None
        )

        if mask_path is None:
            miss_mask += 1
            print(f"[WARN] missing GT mask: {stem}")
            continue

        if pred_path is None:
            miss_pred += 1
            print(f"[WARN] missing pred probability map: {stem}")
            continue

        gt = read_gray_float(mask_path)
        height, width = gt.shape

        gt_bin = (gt >= 0.5).astype(np.float32)

        # Current pred_masks are soft probability maps saved as 0~255 grayscale.
        pred_prob = read_gray_float(pred_path, resize_to=(width, height))
        pred_prob = np.nan_to_num(pred_prob, nan=0.0, posinf=1.0, neginf=0.0)
        pred_prob = np.clip(pred_prob, 0.0, 1.0).astype(np.float32)

        if variance_path is not None:
            epistemic_variance = read_gray_float(variance_path, resize_to=(width, height))
        else:
            epistemic_variance = np.zeros_like(pred_prob, dtype=np.float32)
        epistemic_variance = np.clip(
            np.nan_to_num(epistemic_variance, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        ).astype(np.float32)

        if reliability_path is not None:
            teacher_reliability = read_gray_float(reliability_path, resize_to=(width, height))
        else:
            teacher_reliability = np.ones_like(pred_prob, dtype=np.float32)
        teacher_reliability = np.clip(
            np.nan_to_num(teacher_reliability, nan=0.0, posinf=1.0, neginf=0.0),
            float(args.reliability_floor),
            1.0,
        ).astype(np.float32)

        # Binarized prediction is used only for error maps.
        pred_bin = (pred_prob >= 0.5).astype(np.float32)

        # GT boundary priors.
        hard_boundary = hard_boundary_from_mask(gt_bin, kernel_size=args.kernel)
        soft_boundary = soft_boundary_from_hard(
            hard_boundary,
            radius=args.radius,
            tau=args.tau0,
        )

        # Prediction error maps.
        error_abs = np.abs(pred_bin - gt_bin).astype(np.float32)
        error_fn = ((gt_bin > 0.5) & (pred_bin < 0.5)).astype(np.float32)
        error_fp = ((gt_bin < 0.5) & (pred_bin > 0.5)).astype(np.float32)

        # Entropy must be computed from probability map, not from binary mask.
        entropy = safe_binary_entropy(pred_prob)

        # Raw difficulty combines teacher hard errors and uncertainty.
        # We assign a larger weight to false negatives because missing lesion boundary
        # pixels usually has a stronger negative effect on Dice than mild over-segmentation.
        difficulty_raw = (
            float(args.w_fn) * error_fn
            + float(args.w_fp) * error_fp
            + float(args.lambda_u) * (entropy / 0.6931472)
            + float(args.lambda_epi) * epistemic_variance
        )

        difficulty_raw = np.nan_to_num(
            difficulty_raw,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        # Normalize per image before applying soft boundary.
        # This avoids a few very hard images dominating the adaptive prior scale.
        max_diff = float(np.max(difficulty_raw))
        if max_diff > 1e-8:
            difficulty_raw = difficulty_raw / max_diff

        difficulty_raw = np.clip(difficulty_raw, 0.0, 1.0).astype(np.float32)
        # Boundary-focused difficulty.
        # Reliability prevents pure teacher noise from dominating while retaining
        # high-variance regions as hard examples through the epistemic term above.
        reliability_gate = 0.5 + 0.5 * teacher_reliability
        difficulty = difficulty_raw * soft_boundary * reliability_gate
        difficulty = np.nan_to_num(
            difficulty,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        difficulty = np.clip(difficulty, 0.0, 1.0).astype(np.float32)

        # Adaptive soft boundary prior.
        adaptive_prior = soft_boundary * (1.0 + float(args.gamma) * difficulty)
        adaptive_prior = np.nan_to_num(
            adaptive_prior,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        max_v = float(np.max(adaptive_prior))
        if max_v > 1e-8:
            adaptive_prior = adaptive_prior / max_v

        adaptive_prior = np.clip(adaptive_prior, 0.0, 1.0).astype(np.float32)

        # Final finite-value check.
        has_nan = False
        for map_name, map_value in [
            ("entropy", entropy),
            ("difficulty", difficulty),
            ("adaptive_prior", adaptive_prior),
        ]:
            if not np.isfinite(map_value).all():
                print(f"[WARN] non-finite value detected in {map_name}: {stem}")
                has_nan = True

        if has_nan:
            nan_count += 1

        out_name = stem + ".png"

        save_gray_png(hard_boundary, os.path.join(out_hard_dir, out_name))
        save_gray_png(soft_boundary, os.path.join(out_soft_dir, out_name))
        save_gray_png(error_abs, os.path.join(out_error_abs_dir, out_name))
        save_gray_png(error_fn, os.path.join(out_error_fn_dir, out_name))
        save_gray_png(error_fp, os.path.join(out_error_fp_dir, out_name))
        save_gray_png(entropy / 0.6931472, os.path.join(out_entropy_dir, out_name))
        save_gray_png(epistemic_variance, os.path.join(out_variance_dir, out_name))
        save_gray_png(teacher_reliability, os.path.join(out_reliability_dir, out_name))
        save_gray_png(difficulty, os.path.join(out_difficulty_dir, out_name))
        save_gray_png(adaptive_prior, os.path.join(out_prior_dir, out_name))

        mean_error = float(np.nanmean(error_abs))
        mean_entropy = float(np.nanmean(entropy))
        mean_variance = float(np.nanmean(epistemic_variance))
        mean_reliability = float(np.nanmean(teacher_reliability))
        mean_difficulty = float(np.nanmean(difficulty))
        boundary_area = float(np.sum(hard_boundary > 0.5))

        mean_error = 0.0 if not np.isfinite(mean_error) else mean_error
        mean_entropy = 0.0 if not np.isfinite(mean_entropy) else mean_entropy
        mean_variance = 0.0 if not np.isfinite(mean_variance) else mean_variance
        mean_reliability = 0.0 if not np.isfinite(mean_reliability) else mean_reliability
        mean_difficulty = 0.0 if not np.isfinite(mean_difficulty) else mean_difficulty
        boundary_area = 0.0 if not np.isfinite(boundary_area) else boundary_area

        rows.append({
            "image_name": stem,
            "mean_error": f"{mean_error:.6f}",
            "mean_entropy": f"{mean_entropy:.6f}",
            "mean_variance": f"{mean_variance:.6f}",
            "mean_reliability": f"{mean_reliability:.6f}",
            "mean_difficulty": f"{mean_difficulty:.6f}",
            "boundary_area": f"{boundary_area:.1f}",
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(stems):
            print(
                f"[{idx + 1}/{len(stems)}] "
                f"rows={len(rows)}, "
                f"miss_mask={miss_mask}, "
                f"miss_pred={miss_pred}, "
                f"nan_count={nan_count}"
            )

    summary_csv = os.path.join(args.out_dir, "feedback_summary.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "mean_error",
                "mean_entropy",
                "mean_variance",
                "mean_reliability",
                "mean_difficulty",
                "boundary_area",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("[DONE] build boundary feedback finished.")
    print(f"  list_txt      = {args.list_txt}")
    print(f"  pred_mask_dir = {args.pred_mask_dir}")
    print(f"  out_dir       = {args.out_dir}")
    print(f"  summary_csv   = {summary_csv}")
    print(f"  rows          = {len(rows)}")
    print(f"  miss_mask     = {miss_mask}")
    print(f"  miss_pred     = {miss_pred}")
    print(f"  nan_count     = {nan_count}")


if __name__ == "__main__":
    main()
