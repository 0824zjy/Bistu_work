from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from Work3_BEF_SBG.hetero_teachers.common import (
    IMG_EXTS,
    build_boundary,
    find_existing_file,
    find_mask,
    finite_array,
    normalized_entropy,
    read_gray,
    read_id_list,
    save_gray,
)


def load_sources(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("sources", [])
    if not isinstance(payload, list) or not payload:
        raise ValueError("Prediction config must contain a non-empty sources list.")
    result: List[Dict[str, Any]] = []
    for index, source in enumerate(payload):
        if not isinstance(source, dict):
            raise TypeError(f"Invalid source at index {index}")
        mask_dir = source.get("pred_mask_dir")
        if not mask_dir or not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Invalid pred_mask_dir: {mask_dir}")
        item = dict(source)
        item["weight"] = max(float(item.get("weight", 1.0)), 0.0)
        item["name"] = str(item.get("name", f"source{index}"))
        result.append(item)
    if sum(item["weight"] for item in result) <= 0:
        raise ValueError("At least one source weight must be positive.")
    return result


def weighted_stats(arrays: Sequence[np.ndarray], weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.stack(arrays, axis=0).astype(np.float32)
    weight = np.asarray(weights, dtype=np.float32)
    weight = weight / max(float(weight.sum()), 1e-8)
    mean = np.tensordot(weight, stack, axes=(0, 0))
    second = np.tensordot(weight, stack * stack, axes=(0, 0))
    variance = np.maximum(second - mean * mean, 0.0)
    return finite_array(mean, 0.0, 1.0), finite_array(variance, 0.0, 0.25)


def dice_iou(prediction: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
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
    parser.add_argument("--sources_json", required=True)
    parser.add_argument("--list_txt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--entropy_lambda", type=float, default=1.0)
    parser.add_argument("--variance_lambda", type=float, default=2.0)
    parser.add_argument("--confidence_gamma", type=float, default=1.0)
    parser.add_argument("--reliability_floor", type=float, default=0.02)
    args = parser.parse_args()

    sources = load_sources(args.sources_json)
    stems = read_id_list(args.list_txt)
    directories = {
        "pred_masks": os.path.join(args.out_dir, "pred_masks"),
        "pred_boundaries": os.path.join(args.out_dir, "pred_boundaries"),
        "pred_variance": os.path.join(args.out_dir, "pred_variance"),
        "pred_entropy": os.path.join(args.out_dir, "pred_entropy"),
        "pred_reliability": os.path.join(args.out_dir, "pred_reliability"),
    }
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    missing = 0
    for index, stem in enumerate(stems):
        mask_arrays: List[np.ndarray] = []
        mask_weights: List[float] = []
        boundary_arrays: List[np.ndarray] = []
        boundary_weights: List[float] = []
        shape: Optional[Tuple[int, int]] = None

        for source in sources:
            path = find_existing_file(source["pred_mask_dir"], stem, IMG_EXTS)
            if path is None:
                print(f"[WARN] {source['name']} missing prediction: {stem}")
                continue
            probability = read_gray(path)
            if shape is None:
                shape = probability.shape
            elif probability.shape != shape:
                probability = cv2.resize(
                    probability,
                    (shape[1], shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            mask_arrays.append(probability)
            mask_weights.append(source["weight"])

            boundary_dir = source.get("pred_boundary_dir")
            if boundary_dir:
                boundary_path = find_existing_file(boundary_dir, stem, IMG_EXTS)
                if boundary_path is not None:
                    boundary_probability = read_gray(boundary_path)
                    if boundary_probability.shape != shape:
                        boundary_probability = cv2.resize(
                            boundary_probability,
                            (shape[1], shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    boundary_arrays.append(boundary_probability)
                    boundary_weights.append(source["weight"])

        if not mask_arrays:
            missing += 1
            continue

        mean_probability, variance = weighted_stats(mask_arrays, mask_weights)
        entropy = normalized_entropy(mean_probability)
        variance_norm = finite_array(variance / 0.25, 0.0, 1.0)
        confidence = finite_array(np.abs(mean_probability - 0.5) * 2.0, 0.0, 1.0)
        reliability = np.power(confidence, float(args.confidence_gamma))
        reliability *= np.exp(
            -float(args.entropy_lambda) * entropy
            -float(args.variance_lambda) * variance_norm
        )
        reliability = finite_array(reliability, args.reliability_floor, 1.0)

        if boundary_arrays:
            boundary_probability, _ = weighted_stats(boundary_arrays, boundary_weights)
        else:
            boundary_probability = build_boundary(mean_probability >= 0.5, kernel_size=3)

        save_gray(mean_probability, os.path.join(directories["pred_masks"], stem + ".png"))
        save_gray(boundary_probability, os.path.join(directories["pred_boundaries"], stem + ".png"))
        save_gray(variance_norm, os.path.join(directories["pred_variance"], stem + ".png"))
        save_gray(entropy, os.path.join(directories["pred_entropy"], stem + ".png"))
        save_gray(reliability, os.path.join(directories["pred_reliability"], stem + ".png"))

        row: Dict[str, Any] = {
            "stem": stem,
            "n_predictions": len(mask_arrays),
            "mean_entropy": float(entropy.mean()),
            "mean_variance": float(variance_norm.mean()),
            "mean_reliability": float(reliability.mean()),
        }
        if args.mask_dir:
            mask_path = find_mask(args.mask_dir, stem)
            if mask_path is not None:
                target = read_gray(mask_path)
                if target.shape != mean_probability.shape:
                    target = cv2.resize(
                        target,
                        (mean_probability.shape[1], mean_probability.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                dice, iou = dice_iou(mean_probability, target)
                row["dice"] = dice
                row["iou"] = iou
        rows.append(row)

        if (index + 1) % 50 == 0 or index + 1 == len(stems):
            print(f"[{index + 1}/{len(stems)}] written={len(rows)} missing={missing}", flush=True)

    csv_path = os.path.join(args.out_dir, "heterogeneous_oof_summary.csv")
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["stem"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[DONE] heterogeneous predictions fused")
    print("  sources:", len(sources))
    print("  samples:", len(rows))
    print("  missing:", missing)
    print("  out_dir:", args.out_dir)


if __name__ == "__main__":
    main()
