from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from Work3_BEF_SBG.hetero_teachers.common import (
    IMG_EXTS,
    build_soft_boundary,
    find_existing_file,
    finite_array,
    normalized_entropy,
    read_gray,
    save_gray,
)


def dice_iou(prediction: np.ndarray, target: np.ndarray):
    pred = (prediction >= 0.5).astype(np.float32)
    gt = (target >= 0.5).astype(np.float32)
    intersection = float((pred * gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    union = pred_sum + gt_sum - intersection
    return (
        (2.0 * intersection + 1.0) / (pred_sum + gt_sum + 1.0),
        (intersection + 1.0) / (union + 1.0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_image_dir", required=True)
    parser.add_argument("--gen_mask_dir", required=True)
    parser.add_argument("--gen_prior_dir", required=True)
    parser.add_argument("--fused_prediction_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_reliability_dir", required=True)
    parser.add_argument("--out_hardness_dir", required=True)
    parser.add_argument("--cons_threshold", type=float, default=0.80)
    parser.add_argument("--beta_hard", type=float, default=0.5)
    parser.add_argument("--agreement_gamma", type=float, default=1.0)
    parser.add_argument("--reliability_floor", type=float, default=0.05)
    parser.add_argument("--fallback_radius", type=int, default=12)
    parser.add_argument("--fallback_tau", type=float, default=4.0)
    args = parser.parse_args()

    pred_mask_dir = os.path.join(args.fused_prediction_dir, "pred_masks")
    variance_dir = os.path.join(args.fused_prediction_dir, "pred_variance")
    entropy_dir = os.path.join(args.fused_prediction_dir, "pred_entropy")
    fused_reliability_dir = os.path.join(args.fused_prediction_dir, "pred_reliability")
    for directory in [pred_mask_dir, variance_dir, entropy_dir, fused_reliability_dir]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(directory)
    for directory in [args.out_reliability_dir, args.out_hardness_dir]:
        os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    names = sorted(
        name for name in os.listdir(args.gen_image_dir)
        if os.path.splitext(name)[1].lower() in IMG_EXTS
    )
    rows: List[Dict[str, object]] = []
    accepted_count = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as jsonl_handle:
        for index, name in enumerate(names):
            stem = os.path.splitext(name)[0]
            image_path = os.path.join(args.gen_image_dir, name)
            mask_path = find_existing_file(args.gen_mask_dir, stem, IMG_EXTS)
            pred_path = find_existing_file(pred_mask_dir, stem, IMG_EXTS)
            variance_path = find_existing_file(variance_dir, stem, IMG_EXTS)
            entropy_path = find_existing_file(entropy_dir, stem, IMG_EXTS)
            base_reliability_path = find_existing_file(fused_reliability_dir, stem, IMG_EXTS)
            prior_path = find_existing_file(args.gen_prior_dir, stem, IMG_EXTS)
            if mask_path is None or pred_path is None:
                print(f"[WARN] missing mask/prediction: {stem}")
                continue

            probability = read_gray(pred_path)
            height, width = probability.shape
            target = read_gray(mask_path)
            if target.shape != probability.shape:
                target = cv2.resize(target, (width, height), interpolation=cv2.INTER_NEAREST)
            target = (target >= 0.5).astype(np.float32)

            variance = read_gray(variance_path) if variance_path else np.zeros_like(probability)
            entropy = read_gray(entropy_path) if entropy_path else normalized_entropy(probability)
            base_reliability = (
                read_gray(base_reliability_path)
                if base_reliability_path
                else np.ones_like(probability)
            )
            if variance.shape != probability.shape:
                variance = cv2.resize(variance, (width, height), interpolation=cv2.INTER_LINEAR)
            if entropy.shape != probability.shape:
                entropy = cv2.resize(entropy, (width, height), interpolation=cv2.INTER_LINEAR)
            if base_reliability.shape != probability.shape:
                base_reliability = cv2.resize(
                    base_reliability, (width, height), interpolation=cv2.INTER_LINEAR
                )

            agreement = finite_array(1.0 - np.abs(probability - target), 0.0, 1.0)
            reliability = np.power(agreement, float(args.agreement_gamma)) * base_reliability
            reliability = finite_array(reliability, args.reliability_floor, 1.0)

            if prior_path:
                prior = read_gray(prior_path)
                if prior.shape != probability.shape:
                    prior = cv2.resize(prior, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                prior = build_soft_boundary(target, args.fallback_radius, args.fallback_tau)
            uncertainty = finite_array(0.5 * entropy + 0.5 * variance, 0.0, 1.0)
            hardness = finite_array(prior * uncertainty * reliability, 0.0, 1.0)
            if float(hardness.max()) > 1e-8:
                hardness /= float(hardness.max())

            dice, iou = dice_iou(probability, target)
            prior_sum = float(prior.sum())
            boundary_reliability = (
                float((reliability * prior).sum() / prior_sum)
                if prior_sum > 1e-8 else float(reliability.mean())
            )
            boundary_difficulty = (
                float((hardness * prior).sum() / prior_sum)
                if prior_sum > 1e-8 else float(hardness.mean())
            )
            accepted = int(dice >= args.cons_threshold)
            weight = 0.0
            if accepted:
                weight = dice * (1.0 + args.beta_hard * boundary_difficulty)
                weight *= 0.5 + 0.5 * boundary_reliability
                weight = float(np.clip(weight, 0.0, 1.5))

            reliability_path = os.path.join(args.out_reliability_dir, stem + ".png")
            hardness_path = os.path.join(args.out_hardness_dir, stem + ".png")
            save_gray(reliability, reliability_path)
            save_gray(hardness, hardness_path)

            row = {
                "image_path": image_path,
                "mask_path": mask_path,
                "prior_path": prior_path or "",
                "reliability_path": reliability_path,
                "hardness_path": hardness_path,
                "consistency_dice": dice,
                "consistency_iou": iou,
                "mean_entropy": float(entropy.mean()),
                "mean_variance": float(variance.mean()),
                "boundary_difficulty": boundary_difficulty,
                "boundary_reliability": boundary_reliability,
                "weight": weight,
                "accepted": accepted,
            }
            rows.append(row)
            if accepted:
                accepted_count += 1
                jsonl_handle.write(json.dumps({
                    "image": image_path,
                    "mask": mask_path,
                    "reliability": reliability_path,
                    "hardness": hardness_path,
                    "boundary_prior": prior_path or "",
                    "weight": weight,
                    "source": "bef_sbg",
                }, ensure_ascii=False, allow_nan=False) + "\n")

            if (index + 1) % 50 == 0 or index + 1 == len(names):
                print(f"[{index + 1}/{len(names)}] accepted={accepted_count}", flush=True)

    fieldnames = list(rows[0].keys()) if rows else ["image_path"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("[DONE] generated samples scored with heterogeneous teacher predictions")
    print("  records:", len(rows))
    print("  accepted:", accepted_count)
    print("  csv:", args.out_csv)
    print("  jsonl:", args.out_jsonl)


if __name__ == "__main__":
    main()
