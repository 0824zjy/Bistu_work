import os
import sys
import csv
import argparse
from typing import Dict

sys.path.insert(0, "/data/zjy_work/BGDNet")

import cv2
import numpy as np
import pandas as pd

import torch

# Work3 BEF-SBG cuDNN safe switch
import os
if os.environ.get("BGDNET_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    print("[WARN] cuDNN disabled by BGDNET_DISABLE_CUDNN=1")
else:
    torch.backends.cudnn.enabled = True

torch.backends.cudnn.benchmark = False

import torch.nn.functional as F

from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
    binary_fill_holes,
)
from models.BGDNet import BGDNet
from utils.dataloader_BGDiff import test_dataset


def load_state_dict_safely(model, pth_path: str):
    ckpt = torch.load(pth_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    boundary = mask ^ eroded
    return boundary.astype(bool)


def surface_distances(pred: np.ndarray, gt: np.ndarray):
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    pred_b = extract_boundary(pred)
    gt_b = extract_boundary(gt)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return np.array([0.0], dtype=np.float32)

    if pred_b.sum() == 0 or gt_b.sum() == 0:
        h, w = pred.shape
        diag = np.sqrt(h * h + w * w)
        return np.array([diag], dtype=np.float32)

    dt_gt = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)

    d_pred_to_gt = dt_gt[pred_b]
    d_gt_to_pred = dt_pred[gt_b]

    dists = np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype(np.float32)

    if dists.size == 0:
        return np.array([0.0], dtype=np.float32)

    return dists


def compute_hd95_assd(pred: np.ndarray, gt: np.ndarray):
    dists = surface_distances(pred, gt)
    hd95 = float(np.percentile(dists, 95))
    assd = float(np.mean(dists))
    return hd95, assd

def resize_np_nearest(mask: np.ndarray, size_hw):
    """
    Resize binary mask with nearest interpolation.

    Args:
        mask: H,W
        size_hw: (H,W)
    """
    h, w = size_hw
    return cv2.resize(
        mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )


def binarize_prob(prob: np.ndarray, threshold: float):
    return (prob >= threshold).astype(np.uint8)

@torch.no_grad()
def predict_with_tta(model, image: torch.Tensor):
    """Four-way flip TTA for mask, boundary and signed-distance heads."""
    logits_m = []
    logits_b = []
    logits_d = []

    variants = [
        (image, ()),
        (torch.flip(image, dims=[3]), (3,)),
        (torch.flip(image, dims=[2]), (2,)),
        (torch.flip(image, dims=[2, 3]), (2, 3)),
    ]

    for aug_image, flip_dims in variants:
        output = model(aug_image)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError("BGDNet must return at least mask and boundary logits.")
        pm, pb = output[0], output[1]
        pd = output[2] if len(output) >= 3 else None

        if flip_dims:
            pm = torch.flip(pm, dims=list(flip_dims))
            pb = torch.flip(pb, dims=list(flip_dims))
            if pd is not None:
                pd = torch.flip(pd, dims=list(flip_dims))

        logits_m.append(pm)
        logits_b.append(pb)
        if pd is not None:
            logits_d.append(pd)

    pred_m = torch.stack(logits_m, dim=0).mean(dim=0)
    pred_b = torch.stack(logits_b, dim=0).mean(dim=0)
    pred_d = torch.stack(logits_d, dim=0).mean(dim=0) if logits_d else None
    return pred_m, pred_b, pred_d


def unpack_model_outputs(output):
    if not isinstance(output, (tuple, list)):
        return output, None, None
    pred_m = output[0]
    pred_b = output[1] if len(output) >= 2 else None
    pred_d = output[2] if len(output) >= 3 else None
    return pred_m, pred_b, pred_d


def fuse_mask_and_distance(
    mask_logits: torch.Tensor,
    distance_logits: torch.Tensor,
    fusion_weight: float,
    distance_temperature: float,
):
    mask_prob = torch.sigmoid(mask_logits)
    if distance_logits is None or float(fusion_weight) <= 0.0:
        return mask_prob
    distance_prob = torch.sigmoid(
        distance_logits * float(distance_temperature)
    )
    w = float(max(0.0, min(float(fusion_weight), 1.0)))
    return (1.0 - w) * mask_prob + w * distance_prob


def keep_largest_component(mask: np.ndarray):
    """
    Keep largest connected component and fill holes.
    This is usually valid for ISIC lesion segmentation because each image
    typically contains one main lesion region.
    """
    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask.astype(np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]

    if areas.size == 0:
        return mask.astype(np.uint8)

    largest_label = 1 + int(np.argmax(areas))

    out = (labels == largest_label).astype(np.uint8)
    out = binary_fill_holes(out > 0).astype(np.uint8)

    return out

def boundary_f1_iou(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2):
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    pred_b = extract_boundary(pred)
    gt_b = extract_boundary(gt)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0, 1.0

    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0, 0.0

    structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)

    gt_dil = binary_dilation(gt_b, structure=structure)
    pred_dil = binary_dilation(pred_b, structure=structure)

    pred_match = pred_b & gt_dil
    gt_match = gt_b & pred_dil

    precision = pred_match.sum() / (pred_b.sum() + 1e-8)
    recall = gt_match.sum() / (gt_b.sum() + 1e-8)

    bf1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    pred_band = pred_dil
    gt_band = gt_dil
    inter = np.logical_and(pred_band, gt_band).sum()
    union = np.logical_or(pred_band, gt_band).sum()

    biou = inter / (union + 1e-8)

    return float(bf1), float(biou)


def compute_seg_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    tp = float((pred * gt).sum())
    fp = float((pred * (1.0 - gt)).sum())
    fn = float(((1.0 - pred) * gt).sum())
    tn = float(((1.0 - pred) * (1.0 - gt)).sum())

    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    union = pred_sum + gt_sum - tp

    smooth = 1.0

    empty = (pred_sum == 0 and gt_sum == 0)

    dice = (2.0 * tp + smooth) / (pred_sum + gt_sum + smooth)
    iou = (tp + smooth) / (union + smooth)

    if empty:
        sensitivity = 1.0
        specificity = 1.0
        precision = 1.0
        f1 = 1.0
        mcc = 1.0
    else:
        sensitivity = tp / (tp + fn + 1e-8) if tp > 0 else 0.0
        precision = tp / (tp + fp + 1e-8) if tp > 0 else 0.0
        specificity = tn / (tn + fp + 1e-8) if tn > 0 else 0.0
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity + 1e-8) if tp > 0 else 0.0

        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn + smooth) / (denom + smooth)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def save_prob_png(prob_tensor: torch.Tensor, path: str):
    arr = prob_tensor.squeeze().detach().float().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    img = (arr * 255.0).round().astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pth_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, default="/data/zjy_work/ISIC2018/test/")
    parser.add_argument("--test_list", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="/data/zjy_work/Work3_BEF_SBG/results/eval_final_bef/")
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--boundary_tolerance", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary_threshold", type=float, default=0.5)
    parser.add_argument("--distance_fusion_weight", type=float, default=0.20)
    parser.add_argument("--distance_temperature", type=float, default=4.0)

    parser.add_argument("--tta", action="store_true", help="Enable flip TTA during testing.")
    parser.add_argument("--postprocess_lcc", action="store_true", help="Keep largest connected component and fill holes.")

    args = parser.parse_args()

    device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)

    mask_dir = os.path.join(args.out_dir, "masks")
    bnd_dir = os.path.join(args.out_dir, "boundaries")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(bnd_dir, exist_ok=True)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if os.environ.get("BGDNET_ENABLE_DISTANCE_HEAD", "0") != "1":
        raise RuntimeError(
            "Set BGDNET_ENABLE_DISTANCE_HEAD=1 when testing a CHFS checkpoint."
        )

    model = BGDNet(num_classes=1).to(device)
    load_state_dict_safely(model, args.pth_path)
    model.eval()

    image_root = os.path.join(args.test_data_path, "Images")
    gt_root = os.path.join(args.test_data_path, "Masks")

    loader = test_dataset(
        image_root=image_root,
        gt_root=gt_root,
        testsize=args.testsize,
        list_txt=args.test_list,
        mode="isic",
    )

    if loader.size == 0:
        raise RuntimeError("test_dataset size == 0. Please check test path and test_list.")

    per_records = []

    metric_keys = [
        "dice",
        "iou",
        "sensitivity",
        "specificity",
        "accuracy",
        "precision",
        "f1",
        "mcc",

        "hd95",
        "assd",
        "normalized_hd95",
        "normalized_assd",

        "mask_contour_f1",
        "mask_contour_iou",

        "mask_contour_f1_352",
        "mask_contour_iou_352",

        "boundary_head_f1",
        "boundary_head_iou",
    ]


    sums = {k: 0.0 for k in metric_keys}

    with torch.no_grad():
        for _ in range(loader.size):
            image, gt_pil, name = loader.load_data()

            gt = np.asarray(gt_pil, np.float32)
            gt = gt / (gt.max() + 1e-8)
            gt_bin = (gt >= 0.5).astype(np.uint8)

            image = image.to(device, non_blocking=True)

            if args.tta:
                pred_m, pred_b, pred_d = predict_with_tta(model, image)
            else:
                pred_m, pred_b, pred_d = unpack_model_outputs(model(image))

            pred_m = F.interpolate(pred_m, size=gt.shape, mode="bilinear", align_corners=False)
            pred_b = F.interpolate(pred_b, size=gt.shape, mode="bilinear", align_corners=False)
            if pred_d is not None:
                pred_d = F.interpolate(
                    pred_d,
                    size=gt.shape,
                    mode="bilinear",
                    align_corners=False,
                )

            pm_prob = fuse_mask_and_distance(
                pred_m,
                pred_d,
                fusion_weight=args.distance_fusion_weight,
                distance_temperature=args.distance_temperature,
            )
            pb_prob = torch.sigmoid(pred_b)

            pred_prob = pm_prob.squeeze().detach().float().cpu().numpy()
            boundary_prob = pb_prob.squeeze().detach().float().cpu().numpy()

            pred_bin = binarize_prob(pred_prob, args.threshold)

            if args.postprocess_lcc:
                pred_bin = keep_largest_component(pred_bin)
            # -------------------------
            # Main segmentation metrics on original GT size
            # -------------------------
            seg_m = compute_seg_metrics(pred_bin, gt_bin)

            hd95, assd = compute_hd95_assd(pred_bin, gt_bin)

            diag = float(np.sqrt(gt_bin.shape[0] ** 2 + gt_bin.shape[1] ** 2))
            normalized_hd95 = hd95 / (diag + 1e-8)
            normalized_assd = assd / (diag + 1e-8)

            # Scale tolerance for original image size
            scale = max(gt_bin.shape) / float(args.testsize)
            scaled_tolerance = max(
                1,
                int(round(args.boundary_tolerance * scale)),
            )

            mask_cf1, mask_ciou = boundary_f1_iou(
                pred_bin,
                gt_bin,
                tolerance=scaled_tolerance,
            )

            # -------------------------
            # Boundary metrics on fixed 352x352 scale
            # -------------------------
            gt_352 = resize_np_nearest(gt_bin, (args.testsize, args.testsize))
            pred_352 = resize_np_nearest(pred_bin, (args.testsize, args.testsize))

            mask_cf1_352, mask_ciou_352 = boundary_f1_iou(
                pred_352,
                gt_352,
                tolerance=args.boundary_tolerance,
            )

            # -------------------------
            # Boundary head metrics
            # -------------------------
            boundary_pred_bin = binarize_prob(boundary_prob, args.boundary_threshold)
            gt_boundary_bin = extract_boundary(gt_bin).astype(np.uint8)

            boundary_head_f1, boundary_head_iou = boundary_f1_iou(
                boundary_pred_bin,
                gt_boundary_bin,
                tolerance=scaled_tolerance,
            )

            record = {
                "image_name": name,

                "dice": seg_m["dice"],
                "iou": seg_m["iou"],
                "sensitivity": seg_m["sensitivity"],
                "specificity": seg_m["specificity"],
                "accuracy": seg_m["accuracy"],
                "precision": seg_m["precision"],
                "f1": seg_m["f1"],
                "mcc": seg_m["mcc"],

                "hd95": hd95,
                "assd": assd,
                "normalized_hd95": normalized_hd95,
                "normalized_assd": normalized_assd,

                "mask_contour_f1": mask_cf1,
                "mask_contour_iou": mask_ciou,

                "mask_contour_f1_352": mask_cf1_352,
                "mask_contour_iou_352": mask_ciou_352,

                "boundary_head_f1": boundary_head_f1,
                "boundary_head_iou": boundary_head_iou,
            }


            for k in metric_keys:
                sums[k] += float(record[k])

            per_records.append(record)

            out_name = os.path.splitext(name)[0] + ".png"
            save_prob_png(pm_prob, os.path.join(mask_dir, out_name))
            save_prob_png(pb_prob, os.path.join(bnd_dir, out_name))

    n = len(per_records)
    summary = {k: sums[k] / max(n, 1) for k in metric_keys}

    per_df = pd.DataFrame(per_records)
    per_csv = os.path.join(args.out_dir, "per_image_metrics.csv")
    per_df.to_csv(per_csv, index=False)

    summary_df = pd.DataFrame([{
        "Threshold": args.threshold,
        "BoundaryThreshold": args.boundary_threshold,

        "Sensitivity": round(summary["sensitivity"], 6),
        "Specificity": round(summary["specificity"], 6),
        "Accuracy": round(summary["accuracy"], 6),
        "Precision": round(summary["precision"], 6),
        "F1": round(summary["f1"], 6),
        "MCC": round(summary["mcc"], 6),

        "Jacard": round(summary["iou"], 6),
        "Dice": round(summary["dice"], 6),

        "HD95": round(summary["hd95"], 6),
        "ASSD": round(summary["assd"], 6),
        "Normalized-HD95": round(summary["normalized_hd95"], 6),
        "Normalized-ASSD": round(summary["normalized_assd"], 6),

        "Mask-Contour-F1": round(summary["mask_contour_f1"], 6),
        "Mask-Contour-IoU": round(summary["mask_contour_iou"], 6),

        "Mask-Contour-F1-352": round(summary["mask_contour_f1_352"], 6),
        "Mask-Contour-IoU-352": round(summary["mask_contour_iou_352"], 6),

        "Boundary-Head-F1": round(summary["boundary_head_f1"], 6),
        "Boundary-Head-IoU": round(summary["boundary_head_iou"], 6),
    }])


    summary_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("*****************************************************")
    print(f"Threshold:             {args.threshold:.3f}")
    print(f"Boundary threshold:    {args.boundary_threshold:.3f}")
    print(f"Dice:                  {summary['dice']:.6f}")
    print(f"IoU/Jacard:            {summary['iou']:.6f}")
    print(f"HD95:                  {summary['hd95']:.6f}")
    print(f"ASSD:                  {summary['assd']:.6f}")
    print(f"Normalized-HD95:       {summary['normalized_hd95']:.6f}")
    print(f"Normalized-ASSD:       {summary['normalized_assd']:.6f}")
    print(f"Mask-Contour-F1:       {summary['mask_contour_f1']:.6f}")
    print(f"Mask-Contour-IoU:      {summary['mask_contour_iou']:.6f}")
    print(f"Mask-Contour-F1-352:   {summary['mask_contour_f1_352']:.6f}")
    print(f"Mask-Contour-IoU-352:  {summary['mask_contour_iou_352']:.6f}")
    print(f"Boundary-Head-F1:      {summary['boundary_head_f1']:.6f}")
    print(f"Boundary-Head-IoU:     {summary['boundary_head_iou']:.6f}")
    print("*****************************************************")



if __name__ == "__main__":
    main()
