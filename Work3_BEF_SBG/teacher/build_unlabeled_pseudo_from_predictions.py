from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

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
    largest_component_fill,
    list_images,
    normalized_entropy,
    read_gray,
    save_gray,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled_image_dir", required=True)
    parser.add_argument("--fused_prediction_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--pseudo_threshold", type=float, default=0.5)
    parser.add_argument("--min_mean_reliability", type=float, default=0.70)
    parser.add_argument("--min_boundary_reliability", type=float, default=0.55)
    parser.add_argument("--min_foreground_ratio", type=float, default=0.005)
    parser.add_argument("--max_foreground_ratio", type=float, default=0.80)
    parser.add_argument("--boundary_radius", type=int, default=12)
    parser.add_argument("--boundary_tau", type=float, default=4.0)
    args = parser.parse_args()

    pred_dir = os.path.join(args.fused_prediction_dir, "pred_masks")
    variance_dir = os.path.join(args.fused_prediction_dir, "pred_variance")
    entropy_dir = os.path.join(args.fused_prediction_dir, "pred_entropy")
    reliability_dir = os.path.join(args.fused_prediction_dir, "pred_reliability")
    for directory in [pred_dir, variance_dir, entropy_dir, reliability_dir]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(directory)

    mask_out = os.path.join(args.out_root, "pseudo_masks")
    reliability_out = os.path.join(args.out_root, "reliability")
    hardness_out = os.path.join(args.out_root, "hardness")
    prior_out = os.path.join(args.out_root, "boundary_prior")
    for directory in [mask_out, reliability_out, hardness_out, prior_out]:
        os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    images = list_images(args.unlabeled_image_dir)
    accepted = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as handle:
        for index, image_path in enumerate(images):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            paths = {
                "prob": find_existing_file(pred_dir, stem, IMG_EXTS),
                "variance": find_existing_file(variance_dir, stem, IMG_EXTS),
                "entropy": find_existing_file(entropy_dir, stem, IMG_EXTS),
                "reliability": find_existing_file(reliability_dir, stem, IMG_EXTS),
            }
            if paths["prob"] is None:
                print(f"[WARN] missing fused prediction: {stem}")
                continue
            probability = read_gray(paths["prob"])
            variance = read_gray(paths["variance"]) if paths["variance"] else np.zeros_like(probability)
            entropy = read_gray(paths["entropy"]) if paths["entropy"] else normalized_entropy(probability)
            reliability = read_gray(paths["reliability"]) if paths["reliability"] else np.ones_like(probability)

            pseudo = largest_component_fill(probability >= args.pseudo_threshold)
            foreground_ratio = float(pseudo.mean())
            prior = build_soft_boundary(pseudo, args.boundary_radius, args.boundary_tau)
            uncertainty = finite_array(0.5 * entropy + 0.5 * variance, 0.0, 1.0)
            hardness = finite_array(prior * uncertainty * reliability, 0.0, 1.0)
            if float(hardness.max()) > 1e-8:
                hardness /= float(hardness.max())

            mean_reliability = float(reliability.mean())
            prior_sum = float(prior.sum())
            boundary_reliability = (
                float((reliability * prior).sum() / prior_sum)
                if prior_sum > 1e-8 else mean_reliability
            )
            valid = (
                args.min_foreground_ratio <= foreground_ratio <= args.max_foreground_ratio
                and mean_reliability >= args.min_mean_reliability
                and boundary_reliability >= args.min_boundary_reliability
            )
            if not valid:
                continue

            mask_path = os.path.join(mask_out, stem + ".png")
            reliability_path = os.path.join(reliability_out, stem + ".png")
            hardness_path = os.path.join(hardness_out, stem + ".png")
            prior_path = os.path.join(prior_out, stem + ".png")
            save_gray(pseudo, mask_path)
            save_gray(reliability, reliability_path)
            save_gray(hardness, hardness_path)
            save_gray(prior, prior_path)
            sample_weight = float(np.clip(0.5 * mean_reliability + 0.5 * boundary_reliability, 0.0, 1.0))
            handle.write(json.dumps({
                "image": image_path,
                "mask": mask_path,
                "reliability": reliability_path,
                "hardness": hardness_path,
                "boundary_prior": prior_path,
                "weight": sample_weight,
                "source": "pseudo_real",
                "mean_reliability": mean_reliability,
                "boundary_reliability": boundary_reliability,
                "foreground_ratio": foreground_ratio,
            }, ensure_ascii=False, allow_nan=False) + "\n")
            accepted += 1

            if (index + 1) % 50 == 0 or index + 1 == len(images):
                print(f"[{index + 1}/{len(images)}] accepted={accepted}", flush=True)

    print("[DONE] heterogeneous pseudo labels")
    print("  total:", len(images))
    print("  accepted:", accepted)
    print("  jsonl:", args.out_jsonl)


if __name__ == "__main__":
    main()
