import os
import sys
import argparse

sys.path.insert(0, "/data/zjy_work/BGDNet")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.BGDNet import BGDNet
from utils.dataloader_BGDiff import test_dataset


def load_state_dict_safely(model, path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if any(k.startswith("module.") for k in ckpt):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)


def unpack(output):
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise RuntimeError("CHFS model must return mask, boundary and distance logits.")
    return output[0], output[1], output[2]


def keep_largest_component(mask):
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_path", required=True)
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--distance_fusion_weight", type=float, default=0.2)
    parser.add_argument("--distance_temperature", type=float, default=4.0)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--postprocess_lcc", action="store_true")
    args = parser.parse_args()

    os.environ["BGDNET_ENABLE_DISTANCE_HEAD"] = "1"
    device = torch.device(args.device)
    model = BGDNet(num_classes=1).to(device)
    load_state_dict_safely(model, args.pth_path)
    model.eval()

    image_root = os.path.join(args.test_data_path, "Images")
    mask_root = os.path.join(args.test_data_path, "Masks")
    loader = test_dataset(
        image_root=image_root,
        gt_root=mask_root,
        testsize=args.testsize,
        list_txt=None,
        mode="isic",
    )
    if loader.size == 0:
        raise RuntimeError("No test images found.")

    mask_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    dice_sum = 0.0
    with torch.inference_mode():
        for index in range(loader.size):
            image, gt_pil, name = loader.load_data()
            gt = np.asarray(gt_pil, np.float32)
            gt = gt / (gt.max() + 1e-8)
            gt_bin = (gt >= 0.5).astype(np.uint8)

            image = image.to(device, non_blocking=True)
            pred_m, _, pred_d = unpack(model(image))
            pred_m = F.interpolate(
                pred_m,
                size=gt.shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_d = F.interpolate(
                pred_d,
                size=gt.shape,
                mode="bilinear",
                align_corners=False,
            )

            mask_prob = torch.sigmoid(pred_m)
            distance_prob = torch.sigmoid(
                pred_d * float(args.distance_temperature)
            )
            w = float(np.clip(args.distance_fusion_weight, 0.0, 1.0))
            prob = (1.0 - w) * mask_prob + w * distance_prob
            prob_np = prob.squeeze().cpu().numpy()
            pred_bin = (prob_np >= args.threshold).astype(np.uint8)
            if args.postprocess_lcc:
                pred_bin = keep_largest_component(pred_bin)

            inter = float((pred_bin * gt_bin).sum())
            dice = (2.0 * inter + 1.0) / (
                float(pred_bin.sum()) + float(gt_bin.sum()) + 1.0
            )
            dice_sum += dice

            out = np.round(np.clip(prob_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            cv2.imwrite(
                os.path.join(mask_dir, os.path.splitext(name)[0] + ".png"),
                out,
            )

            if (index + 1) % args.print_freq == 0 or index + 1 == loader.size:
                print(
                    f"[{index + 1}/{loader.size}] "
                    f"mean_dice={dice_sum / (index + 1):.6f}"
                )

    print(f"[DONE] Dice={dice_sum / loader.size:.6f}")


if __name__ == "__main__":
    main()
