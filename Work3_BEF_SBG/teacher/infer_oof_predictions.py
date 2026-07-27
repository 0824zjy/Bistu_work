import os
import sys
import csv
import argparse
from typing import Dict

import cv2
import numpy as np
import torch

# ============================================================
# Work3 BEF-SBG cuDNN safe switch
# ============================================================

disable_cudnn = (
    os.environ.get(
        "BGDNET_DISABLE_CUDNN",
        "1",
    ).strip().lower()
    in ["1", "true", "yes", "y", "on"]
)

if disable_cudnn:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(
        "[WARN] cuDNN disabled by "
        "BGDNET_DISABLE_CUDNN=1"
    )
else:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False


import torch.nn.functional as F


def extract_checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be dict, got {type(checkpoint)}"
        )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif (
        "model" in checkpoint
        and isinstance(checkpoint["model"], dict)
    ):
        state_dict = checkpoint["model"]

    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


def assert_standard_bgdnet_checkpoint(
    state_dict,
    checkpoint_path,
):
    key = "swin.patch_embed.proj.weight"

    if key not in state_dict:
        raise RuntimeError(
            f"Checkpoint does not contain {key}: "
            f"{checkpoint_path}"
        )

    actual_shape = tuple(state_dict[key].shape)
    expected_shape = (96, 3, 4, 4)

    if actual_shape != expected_shape:
        raise RuntimeError(
            "\nOld or incompatible BGDNet checkpoint detected.\n"
            f"checkpoint: {checkpoint_path}\n"
            f"actual patch_embed shape: {actual_shape}\n"
            f"expected standard shape: {expected_shape}\n"
            "This pipeline only accepts the new standard BGDNet."
        )


def load_state_dict_safely(
    model,
    pth_path,
    device="cpu",
):
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {pth_path}"
        )

    checkpoint = torch.load(
        pth_path,
        map_location=device,
    )

    state_dict = extract_checkpoint_state_dict(
        checkpoint
    )

    assert_standard_bgdnet_checkpoint(
        state_dict,
        pth_path,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "[CHECKPOINT OK] standard BGDNet loaded: "
        f"{pth_path}"
    )

@torch.no_grad()
def compute_metrics_gpu(pred_prob: torch.Tensor, gt_np: np.ndarray, device: torch.device) -> Dict[str, float]:
    """
    pred_prob: [1,1,H,W], sigmoid probability, GPU tensor
    gt_np: H x W, numpy, [0,1]
    """
    gt = torch.as_tensor(gt_np, device=device, dtype=torch.float32)
    gt = gt.unsqueeze(0).unsqueeze(0)
    gt = (gt >= 0.5).float()

    pred = (pred_prob >= 0.5).float()

    tp = (pred * gt).sum()
    fp = (pred * (1.0 - gt)).sum()
    fn = ((1.0 - pred) * gt).sum()

    pred_sum = pred.sum()
    gt_sum = gt.sum()
    union = pred_sum + gt_sum - tp

    smooth = 1.0
    dice = (2.0 * tp + smooth) / (pred_sum + gt_sum + smooth)
    iou = (tp + smooth) / (union + smooth)

    return {
        "dice": float(dice.detach().cpu().item()),
        "iou": float(iou.detach().cpu().item()),
    }


def save_prob_png(prob_tensor: torch.Tensor, save_path: str):
    """
    保存 sigmoid 概率图为 0-255 PNG。
    不做 min-max 归一化，保留概率含义。
    """
    prob = prob_tensor.squeeze().detach().float().cpu().numpy()
    prob = np.clip(prob, 0.0, 1.0)
    img = (prob * 255.0).round().astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

def save_prob_npz(
    mask_prob,
    boundary_prob,
    save_path,
):
    mask = (
        mask_prob
        .squeeze()
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float16)
    )

    boundary = (
        boundary_prob
        .squeeze()
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float16)
    )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    np.savez_compressed(
        save_path,
        mask_prob=mask,
        boundary_prob=boundary,
    )
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bgdnet_root",
        type=str,
        default="/data/zjy_work/BGDNet",
        help="BGDNet project root",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/data/zjy_work/ISIC2018/train",
        help="ISIC2018 train root, containing Images/ and Masks/",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="/data/zjy_work/Work3_BEF_SBG/splits",
    )
    parser.add_argument(
        "--ratio_tag",
        type=str,
        default="5p",
        help="5p, 10p, 20p, 100p",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default=None,
        help="Teacher checkpoint root. Default: Work3 teacher/checkpoints/ISIC2018_{ratio_tag}",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="BGDNet-best.pth",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output dir. Default: Work3 results/oof_teacher/ISIC2018_{ratio_tag}",
    )
    parser.add_argument(
        "--testsize",
        type=int,
        default=352,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    args = parser.parse_args()

    if args.ckpt_root is None:
        args.ckpt_root = f"/data/zjy_work/Work3_BEF_SBG/teacher/checkpoints/ISIC2018_{args.ratio_tag}"

    if args.out_dir is None:
        args.out_dir = f"/data/zjy_work/Work3_BEF_SBG/results/oof_teacher/ISIC2018_{args.ratio_tag}"

    pred_mask_dir = os.path.join(args.out_dir, "pred_masks")
    pred_bnd_dir = os.path.join(args.out_dir, "pred_boundaries")
    os.makedirs(pred_mask_dir, exist_ok=True)
    os.makedirs(pred_bnd_dir, exist_ok=True)

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device was requested, "
                "but torch.cuda.is_available() is False."
            )

        device = torch.device(args.device)

        if device.type == "cuda":
            # CUDA_VISIBLE_DEVICES=3 后，物理 GPU 3
            # 在当前 Python 进程中编号为 cuda:0。
            torch.cuda.set_device(device)

            # Do not enable cuDNN benchmark in this environment.
            torch.backends.cudnn.benchmark = False

            if disable_cudnn:
                torch.backends.cudnn.enabled = False
                torch.backends.cudnn.deterministic = True

            print(
                f"[Device] requested={args.device}, "
                f"resolved={device}, "
                f"name={torch.cuda.get_device_name(device)}",
                flush=True,
            )

            print(
                f"[cuDNN] enabled={torch.backends.cudnn.enabled}, "
                f"benchmark={torch.backends.cudnn.benchmark}, "
                f"deterministic="
                f"{torch.backends.cudnn.deterministic}",
                flush=True,
            )
        else:
            print(
                f"[Device] using CPU: {device}",
                flush=True,
            )


    # 动态加入 BGDNet 工程路径
    if args.bgdnet_root not in sys.path:
        sys.path.insert(0, args.bgdnet_root)

    from models.BGDNet import BGDNet
    from utils.dataloader_BGDiff import test_dataset

    image_root = os.path.join(args.test_data_path, "Images")
    gt_root = os.path.join(args.test_data_path, "Masks")

    records = []

    for fold in range(args.n_folds):
        val_list = os.path.join(args.split_dir, f"low_{args.ratio_tag}_fold{fold}_val.txt")
        ckpt_path = os.path.join(args.ckpt_root, f"fold{fold}", args.ckpt_name)

        if not os.path.exists(val_list):
            raise FileNotFoundError(f"Missing val list: {val_list}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        print("============================================================")
        print(f"[Fold {fold}]")
        print(f"  val_list = {val_list}")
        print(f"  ckpt     = {ckpt_path}")
        print("============================================================")

        # ------------------------------------------------------------
        # Create model on CPU first.
        # No external backbone weights are needed because a complete
        # teacher checkpoint will be loaded below.
        # ------------------------------------------------------------
        model = BGDNet(
            num_classes=args.num_classes,
        )

        # Always load checkpoint on CPU first.
        # The complete model will be moved to GPU afterwards.
        load_state_dict_safely(
            model=model,
            pth_path=ckpt_path,
            device="cpu",
        )

        # Verify that this is the new standard BGDNet.
        patch_shape = tuple(
            model.swin.patch_embed.proj.weight.shape
        )

        expected_patch_shape = (96, 3, 4, 4)

        if patch_shape != expected_patch_shape:
            raise RuntimeError(
                f"Fold {fold} loaded an incompatible BGDNet.\n"
                f"checkpoint: {ckpt_path}\n"
                f"patch_embed shape: {patch_shape}\n"
                f"expected shape: {expected_patch_shape}"
            )

        # Important: move the whole model to the inference device.
        model = model.to(device)
        model.eval()

        model_device = next(model.parameters()).device

        if model_device != device:
            raise RuntimeError(
                f"Model device mismatch after model.to(device): "
                f"model={model_device}, expected={device}"
            )

        print(
            f"[Model Ready] "
            f"fold={fold}, "
            f"architecture=standard_swin96_patch4, "
            f"patch_embed={patch_shape}, "
            f"device={model_device}",
            flush=True,
        )


        loader = test_dataset(
            image_root=image_root,
            gt_root=gt_root,
            testsize=args.testsize,
            list_txt=val_list,
            mode="isic",
        )

        if loader.size == 0:
            print(f"[WARN] fold={fold} val loader size is 0, skip.")
            continue

        for _ in range(loader.size):
            image, gt_pil, name = loader.load_data()

            gt = np.asarray(gt_pil, np.float32)
            gt = gt / (gt.max() + 1e-8)

            image = image.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            model_device = next(model.parameters()).device

            if image.device != model_device:
                raise RuntimeError(
                    f"Device mismatch before forward: "
                    f"image={image.device}, "
                    f"model={model_device}"
                )

            with torch.inference_mode():
                out = model(image)

                if isinstance(out, (tuple, list)):
                    pred_m, pred_b = out[0], out[1]
                else:
                    pred_m = out
                    pred_b = torch.zeros_like(pred_m)

                pred_m = F.interpolate(
                    pred_m,
                    size=gt.shape,
                    mode="bilinear",
                    align_corners=False,
                )
                pred_b = F.interpolate(
                    pred_b,
                    size=gt.shape,
                    mode="bilinear",
                    align_corners=False,
                )

                pm_prob = torch.sigmoid(pred_m)
                pb_prob = torch.sigmoid(pred_b)

            metrics = compute_metrics_gpu(pm_prob, gt, device)

            stem = os.path.splitext(name)[0]
            out_name = stem + ".png"

            save_prob_png(pm_prob, os.path.join(pred_mask_dir, out_name))
            save_prob_png(pb_prob, os.path.join(pred_bnd_dir, out_name))

            records.append({
                "fold": fold,
                "image_name": name,
                "stem": stem,
                "pred_mask": out_name,
                "pred_boundary": out_name,
                "dice": f"{metrics['dice']:.6f}",
                "iou": f"{metrics['iou']:.6f}",
            })

        del model
        del loader

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        print(
            f"[Fold {fold}] inference completed.",
            flush=True,
        )

    csv_path = os.path.join(args.out_dir, "per_image_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "fold",
            "image_name",
            "stem",
            "pred_mask",
            "pred_boundary",
            "dice",
            "iou",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    if len(records) > 0:
        dice_mean = np.mean([float(r["dice"]) for r in records])
        iou_mean = np.mean([float(r["iou"]) for r in records])
        print("*****************************************************")
        print(f"OOF samples: {len(records)}")
        print(f"Mean Dice: {dice_mean:.6f}")
        print(f"Mean IoU : {iou_mean:.6f}")
        print("*****************************************************")

    print(f"[DONE] OOF teacher predictions saved to: {args.out_dir}")
    print(f"[DONE] per-image metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
