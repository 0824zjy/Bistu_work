import os
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from BGDNet.models.BGDNet_or import BGDNet
from utils.dataloader_BGDiff import test_dataset


def load_state_dict_safely(model, pth_path: str):
    sd = torch.load(pth_path, map_location="cpu")
    # 兼容 DataParallel 保存的 "module." 前缀
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


@torch.no_grad()
def compute_metrics_per_image_gpu(pred_prob: torch.Tensor, gt_np, smooth: float = 1.0, device="cuda:0"):
    """
    pred_prob: (1,1,H,W) GPU 上 sigmoid 后概率
    gt_np:     (H,W) numpy (float/uint8等均可)
    返回：dict(各指标 python float)
    """
    # gt -> GPU bool
    gt = torch.as_tensor(gt_np, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    gt = (gt >= 0.5)

    # pred -> GPU bool
    pred = (pred_prob >= 0.5)

    pred_f = pred.float()
    gt_f = gt.float()

    # confusion terms
    tp = (pred_f * gt_f).sum(dim=(1, 2, 3))
    fp = (pred_f * (1.0 - gt_f)).sum(dim=(1, 2, 3))
    fn = ((1.0 - pred_f) * gt_f).sum(dim=(1, 2, 3))
    tn = ((1.0 - pred_f) * (1.0 - gt_f)).sum(dim=(1, 2, 3))

    pred_sum = pred_f.sum(dim=(1, 2, 3))
    gt_sum = gt_f.sum(dim=(1, 2, 3))
    union = pred_sum + gt_sum - tp

    # Dice / IoU（与你原来 smooth=1 的写法一致）
    dice = (2.0 * tp + smooth) / (pred_sum + gt_sum + smooth)
    iou = (tp + smooth) / (union + smooth)

    # 全空特殊情况：与你原 get_metrics 的逻辑对齐
    empty = (gt_sum == 0) & (pred_sum == 0)

    sensitivity = torch.where(
        empty, torch.ones_like(tp),
        torch.where(tp == 0, torch.zeros_like(tp), tp / (tp + fn + 1e-8))
    )
    precision = torch.where(
        empty, torch.ones_like(tp),
        torch.where(tp == 0, torch.zeros_like(tp), tp / (tp + fp + 1e-8))
    )
    f1 = torch.where(
        empty, torch.ones_like(tp),
        torch.where(tp == 0, torch.zeros_like(tp),
                    2.0 * precision * sensitivity / (precision + sensitivity + 1e-8))
    )
    specificity = torch.where(
        empty, torch.ones_like(tp),
        torch.where(tn == 0, torch.zeros_like(tp), tn / (tn + fp + 1e-8))
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    # MCC：保持你原来 +smooth 的风格
    mcc = torch.where(
        empty, torch.ones_like(tp),
        (tp * tn - fp * fn + smooth) /
        (torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + smooth)
    )

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "sensitivity": float(sensitivity.item()),
        "specificity": float(specificity.item()),
        "accuracy": float(accuracy.item()),
        "precision": float(precision.item()),
        "f1": float(f1.item()),
        "mcc": float(mcc.item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--pth_path", type=str, required=True)

    parser.add_argument("--test_data_path", type=str, default="/data/zjy_work/ISIC2018/test/",
                        help="测试集根路径，需包含 Images/ Masks/")
    parser.add_argument("--test_list", type=str, default=None,
                        help="测试集txt（行=ISIC_00xxxxxx stem）")

    parser.add_argument("--out_dir", type=str, default="./model_out/ISIC2018_eval/BGDNet/",
                        help="输出目录：会自动创建 masks/ boundaries/ 并保存csv")
    parser.add_argument("--save_csv", type=str, default="per_image_metrics.csv",
                        help="逐图指标 CSV 文件名（保存到 out_dir 下）")
    parser.add_argument("--save_summary_csv", type=str, default="summary_metrics.csv",
                        help="汇总指标 CSV 文件名（保存到 out_dir 下）")

    # 额外：可选开关
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true", help="启用 AMP 混合精度推理（通常更快）")
    args = parser.parse_args()

    device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    mask_dir = os.path.join(args.out_dir, "masks")
    bnd_dir = os.path.join(args.out_dir, "boundaries")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(bnd_dir, exist_ok=True)

    # 推理加速常用设置
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # build model
    model = BGDNet(num_classes=args.num_classes).to(device)
    load_state_dict_safely(model, args.pth_path)
    model.eval()

    # build test loader (你提供的 test_dataset，一次读一张)
    data_path = args.test_data_path
    image_root = os.path.join(data_path, "Images")
    gt_root = os.path.join(data_path, "Masks")

    print(f"Evaluating: {data_path}")
    test_loader = test_dataset(
        image_root=image_root,
        gt_root=gt_root,
        testsize=args.testsize,
        list_txt=args.test_list,
        mode="isic",
    )
    num1 = test_loader.size
    if num1 == 0:
        raise RuntimeError("test_loader.size == 0，请检查：test_list 内容、Images/Masks 路径、以及 dataloader 的 isic mask 规则。")

    # 逐图记录
    per_image_records = []

    # 逐图平均的累计（与你原始代码的平均方式一致）
    sum_dice = 0.0
    sum_iou = 0.0
    sum_sens = 0.0
    sum_spec = 0.0
    sum_acc = 0.0
    sum_prec = 0.0
    sum_f1 = 0.0
    sum_mcc = 0.0

    for _ in range(num1):
        image, gt, name = test_loader.load_data()

        # gt 处理成 [0,1]
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.to(device, non_blocking=True)

        # 推理：inference_mode + (可选) autocast
        with torch.inference_mode():
            if args.amp and device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    pm, pb = model(image)
                    pm = F.interpolate(pm, size=gt.shape, mode='bilinear', align_corners=False)
                    pb = F.interpolate(pb, size=gt.shape, mode='bilinear', align_corners=False)
                    pm_prob = torch.sigmoid(pm)
                    pb_prob = torch.sigmoid(pb)
            else:
                pm, pb = model(image)
                pm = F.interpolate(pm, size=gt.shape, mode='bilinear', align_corners=False)
                pb = F.interpolate(pb, size=gt.shape, mode='bilinear', align_corners=False)
                pm_prob = torch.sigmoid(pm)
                pb_prob = torch.sigmoid(pb)

        # GPU 指标
        m = compute_metrics_per_image_gpu(pm_prob, gt, smooth=1.0, device=str(device))

        sum_dice += m["dice"]
        sum_iou  += m["iou"]
        sum_sens += m["sensitivity"]
        sum_spec += m["specificity"]
        sum_acc  += m["accuracy"]
        sum_prec += m["precision"]
        sum_f1   += m["f1"]
        sum_mcc  += m["mcc"]

        per_image_records.append({
            "image_name": name,
            "dice": float(f"{m['dice']:.4f}"),
            "iou":  float(f"{m['iou']:.4f}")
        })

        # 保存 mask 概率图（和你原来一致：归一化后存 0-255）
        pm_vis = pm_prob.squeeze(0).squeeze(0)  # (H,W) GPU
        pm_vis = (pm_vis - pm_vis.min()) / (pm_vis.max() - pm_vis.min() + 1e-8)
        pm_vis = (pm_vis * 255.0).to(torch.uint8).detach().cpu().numpy()
        cv2.imwrite(os.path.join(mask_dir, name), pm_vis)

        # 保存 boundary 概率图
        pb_vis = pb_prob.squeeze(0).squeeze(0)
        pb_vis = (pb_vis - pb_vis.min()) / (pb_vis.max() - pb_vis.min() + 1e-8)
        pb_vis = (pb_vis * 255.0).to(torch.uint8).detach().cpu().numpy()
        cv2.imwrite(os.path.join(bnd_dir, name), pb_vis)

    # 汇总（逐图平均）
    dice_mean = sum_dice / num1
    iou_mean = sum_iou / num1
    sensitivity = sum_sens / num1
    specificity = sum_spec / num1
    accuracy = sum_acc / num1
    precision = sum_prec / num1
    f1 = sum_f1 / num1
    mcc = sum_mcc / num1

    print('*****************************************************')
    print('Dice Score: ' + str(dice_mean))
    print('Jacard Score: ' + str(iou_mean))
    print('Finish!')
    print('*****************************************************')

    # per-image csv
    per_csv_path = os.path.join(args.out_dir, args.save_csv)
    df = pd.DataFrame(per_image_records)
    df.to_csv(per_csv_path, index=False)
    print(f"Per-image metrics saved to: {per_csv_path}")

    # summary csv
    summary_df = pd.DataFrame([{
        "Sensitivity": float(f"{sensitivity:.6f}"),
        "Specificity": float(f"{specificity:.6f}"),
        "Accuracy":   float(f"{accuracy:.6f}"),
        "Precision":  float(f"{precision:.6f}"),
        "F1":         float(f"{f1:.6f}"),
        "MCC":        float(f"{mcc:.6f}"),
        "Jacard":     float(f"{iou_mean:.6f}"),
        "Dice":       float(f"{dice_mean:.6f}")
    }])

    summary_csv_path = os.path.join(args.out_dir, args.save_summary_csv)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary metrics saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()
