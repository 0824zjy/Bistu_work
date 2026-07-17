import os
import sys
import csv
import json
import math
import argparse
from typing import List, Optional, Dict

sys.path.insert(0, "/data/zjy_work/BGDNet")

import cv2
import numpy as np
from PIL import Image

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
import torchvision.transforms as transforms

from models.BGDNet import BGDNet


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def finite_float(x, default=0.0):
    """
    Convert x to a finite python float.
    NaN/Inf/invalid values are replaced by default.
    """
    try:
        y = float(x)
    except Exception:
        return float(default)

    if not math.isfinite(y):
        return float(default)

    return float(y)


def finite_array(x, default=0.0, clip_min=None, clip_max=None):
    """
    Replace NaN/Inf in numpy array and optionally clip.
    """
    arr = np.asarray(x, np.float32)
    arr = np.nan_to_num(
        arr,
        nan=float(default),
        posinf=float(default),
        neginf=float(default),
    )

    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else float(clip_min)
        hi = np.inf if clip_max is None else float(clip_max)
        arr = np.clip(arr, lo, hi)

    return arr.astype(np.float32)


def safe_weight_from_metrics(consistency_dice, boundary_difficulty, cons_threshold, beta_hard):
    """
    Return a safe finite weight and accepted flag.
    Any non-finite metric is rejected.
    """
    consistency_dice = finite_float(consistency_dice, default=0.0)
    boundary_difficulty = finite_float(boundary_difficulty, default=0.0)

    if consistency_dice < float(cons_threshold):
        return 0.0, 0

    weight = consistency_dice * (1.0 + float(beta_hard) * boundary_difficulty)
    weight = finite_float(weight, default=0.0)
    weight = float(np.clip(weight, 0.0, 1.5))

    if (not math.isfinite(weight)) or weight <= 0.0:
        return 0.0, 0

    return weight, 1


def save_jsonl_item(f, item):
    """
    allow_nan=False is important:
    if any NaN still leaks here, json.dumps will raise immediately.
    """
    f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")

def find_existing_file(dir_path: str, stem: str, exts: List[str]) -> Optional[str]:
    for ext in exts:
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return p
    return None


def list_images(image_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(image_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(image_dir, fn))
    return sorted(files)


def load_state_dict_safely(model: torch.nn.Module, pth_path: str):
    ckpt = torch.load(pth_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)


def read_gray_float(path: str, resize_to=None) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, np.float32)

    if resize_to is not None:
        arr = cv2.resize(arr, resize_to, interpolation=cv2.INTER_LINEAR)

    arr = finite_array(arr, default=0.0)

    maxv = finite_float(arr.max(), default=0.0)
    if maxv > 1.0:
        arr = arr / 255.0

    arr = finite_array(arr, default=0.0, clip_min=0.0, clip_max=1.0)
    return arr.astype(np.float32)



def build_hard_boundary(mask_bin: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    m = (mask_bin > 0.5).astype(np.uint8)
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1

    kernel = np.ones((k, k), np.uint8)
    dilation = cv2.dilate(m, kernel, iterations=1)
    erosion = cv2.erode(m, kernel, iterations=1)

    boundary = np.clip(dilation.astype(np.float32) - erosion.astype(np.float32), 0.0, 1.0)
    return boundary.astype(np.float32)


def build_soft_boundary_from_mask(mask_bin: np.ndarray, radius: int = 12, tau: float = 4.0) -> np.ndarray:
    hard = build_hard_boundary(mask_bin, kernel_size=3)
    soft = hard.copy()
    prev = hard.copy()

    for r in range(1, int(radius) + 1):
        k = 2 * r + 1
        kernel = np.ones((k, k), np.uint8)

        dilated = cv2.dilate(hard, kernel, iterations=1)
        shell = np.clip(dilated - prev, 0.0, 1.0)

        weight = math.exp(-float(r) / max(float(tau), 1e-6))
        soft = np.maximum(soft, shell * weight)
        prev = dilated

    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, smooth: float = 1.0):
    pred = (pred_bin > 0.5).astype(np.float32)
    gt = (gt_bin > 0.5).astype(np.float32)

    tp = float((pred * gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    union = pred_sum + gt_sum - tp

    dice = (2.0 * tp + smooth) / (pred_sum + gt_sum + smooth)
    iou = (tp + smooth) / (union + smooth)

    return float(dice), float(iou)


def binary_entropy(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = finite_array(p, default=0.0, clip_min=0.0, clip_max=1.0)
    p = np.clip(p, eps, 1.0 - eps)

    h = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    h = finite_array(h, default=0.0, clip_min=0.0, clip_max=None)

    return h.astype(np.float32)



def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


@torch.no_grad()
def predict_prob(model, image_path: str, img_size: int, device: torch.device) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # W, H

    trans = build_transform(img_size)
    x = trans(img).unsqueeze(0).to(device)

    out = model(x)
    pred_m = out[0] if isinstance(out, (tuple, list)) else out

    pred_m = F.interpolate(
        pred_m,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False,
    )

    prob = torch.sigmoid(pred_m).squeeze().detach().cpu().numpy()
    prob = finite_array(prob, default=0.0, clip_min=0.0, clip_max=1.0)
    return prob

@torch.no_grad()
def predict_prob_ensemble(models, image_path: str, img_size: int, device: torch.device) -> np.ndarray:
    """
    Average probability predictions from multiple quality models.
    """
    probs = []

    for model in models:
        prob = predict_prob(
            model=model,
            image_path=image_path,
            img_size=img_size,
            device=device,
        )
        prob = finite_array(prob, default=0.0, clip_min=0.0, clip_max=1.0)
        probs.append(prob)

    if len(probs) == 0:
        raise RuntimeError("No quality models available for ensemble prediction.")

    avg_prob = np.mean(probs, axis=0).astype(np.float32)
    avg_prob = finite_array(avg_prob, default=0.0, clip_min=0.0, clip_max=1.0)

    return avg_prob


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gen_image_dir", type=str, required=True)
    parser.add_argument("--gen_mask_dir", type=str, required=True)
    parser.add_argument("--gen_prior_dir", type=str, required=True)
    parser.add_argument("--quality_model_pth", type=str, required=True)

    # Optional comma-separated checkpoint list for ensemble quality scoring.
    # If provided, this will override single-model scoring.
    parser.add_argument("--quality_model_pths", type=str, default="")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_size", type=int, default=352)
    parser.add_argument("--cons_threshold", type=float, default=0.75)
    parser.add_argument("--beta_hard", type=float, default=0.5)

    parser.add_argument("--fallback_radius", type=int, default=12)
    parser.add_argument("--fallback_tau", type=float, default=4.0)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    device = torch.device(args.device)

    quality_paths = []

    if args.quality_model_pths is not None and args.quality_model_pths.strip() != "":
        quality_paths = [
            p.strip()
            for p in args.quality_model_pths.split(",")
            if p.strip()
        ]
    else:
        quality_paths = [args.quality_model_pth]

    if len(quality_paths) == 0:
        raise RuntimeError("No quality model checkpoint provided.")

    models = []

    print("[Quality Models]")
    for pth in quality_paths:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Missing quality model checkpoint: {pth}")

        print(f"  {pth}")

        m = BGDNet(num_classes=1)
        load_state_dict_safely(m, pth)
        m.to(device)
        m.eval()
        models.append(m)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    image_paths = list_images(args.gen_image_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No generated images found in {args.gen_image_dir}")

    records = []

    accepted_count = 0
    missing_mask = 0
    missing_prior = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as jf:
        for idx, image_path in enumerate(image_paths):
            name = os.path.basename(image_path)
            stem = os.path.splitext(name)[0]

            mask_path = find_existing_file(args.gen_mask_dir, stem, MASK_EXTS)
            prior_path = find_existing_file(args.gen_prior_dir, stem, MASK_EXTS)

            if mask_path is None:
                missing_mask += 1
                print(f"[WARN] missing generated mask for {name}, skip.")
                continue

            pred_prob = predict_prob_ensemble(
                models=models,
                image_path=image_path,
                img_size=args.img_size,
                device=device,
            )
            pred_prob = finite_array(pred_prob, default=0.0, clip_min=0.0, clip_max=1.0)
            H, W = pred_prob.shape

            mask = read_gray_float(mask_path, resize_to=(W, H))
            mask = finite_array(mask, default=0.0, clip_min=0.0, clip_max=1.0)
            mask_bin = (mask > 0.5).astype(np.float32)

            if prior_path is not None:
                prior = read_gray_float(prior_path, resize_to=(W, H))
            else:
                missing_prior += 1
                prior = build_soft_boundary_from_mask(
                    mask_bin,
                    radius=args.fallback_radius,
                    tau=args.fallback_tau,
                )

            prior = finite_array(prior, default=0.0, clip_min=0.0, clip_max=1.0)

            pred_bin = (pred_prob >= 0.5).astype(np.float32)

            consistency_dice, consistency_iou = dice_iou(pred_bin, mask_bin)
            consistency_dice = finite_float(consistency_dice, default=0.0)
            consistency_iou = finite_float(consistency_iou, default=0.0)

            entropy = binary_entropy(pred_prob)
            difficulty_map = finite_array(entropy * prior, default=0.0, clip_min=0.0, clip_max=None)
            boundary_difficulty = finite_float(difficulty_map.mean(), default=0.0)

            weight, accepted = safe_weight_from_metrics(
                consistency_dice=consistency_dice,
                boundary_difficulty=boundary_difficulty,
                cons_threshold=args.cons_threshold,
                beta_hard=args.beta_hard,
            )


            record = {
                "image_path": image_path,
                "mask_path": mask_path,
                "prior_path": prior_path if prior_path is not None else "",
                "consistency_dice": f"{consistency_dice:.6f}",
                "consistency_iou": f"{consistency_iou:.6f}",
                "boundary_difficulty": f"{boundary_difficulty:.6f}",
                "weight": f"{weight:.6f}",
                "accepted": accepted,
            }
            records.append(record)

            if accepted == 1:
                accepted_count += 1
                save_jsonl_item(jf, {
                    "image": image_path,
                    "mask": mask_path,
                    "weight": weight,
                    "source": "bef_sbg",
                })

            if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
                print(
                    f"[{idx + 1}/{len(image_paths)}] "
                    f"accepted={accepted_count}, "
                    f"missing_mask={missing_mask}, "
                    f"missing_prior={missing_prior}"
                )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as cf:
        fieldnames = [
            "image_path",
            "mask_path",
            "prior_path",
            "consistency_dice",
            "consistency_iou",
            "boundary_difficulty",
            "weight",
            "accepted",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("[DONE] generated sample scoring finished.")
    print(f"  total images   = {len(image_paths)}")
    print(f"  records        = {len(records)}")
    print(f"  accepted       = {accepted_count}")
    print(f"  missing_mask   = {missing_mask}")
    print(f"  missing_prior  = {missing_prior}")
    print(f"  csv            = {args.out_csv}")
    print(f"  jsonl accepted = {args.out_jsonl}")


if __name__ == "__main__":
    main()
