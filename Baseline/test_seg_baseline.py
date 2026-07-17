import os
import argparse

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from scipy.ndimage import (
    distance_transform_edt,
    binary_erosion,
    generate_binary_structure,
)

from dataloader_seg_baseline import SegTestDataset
from seg_baselines import get_model, extract_logits


SUPPORTED_MODES = [
    "isic",
    "ph2",
    "paired",
    "auto",
]


def load_state_dict_safely(
    model,
    pth_path: str,
    device="cpu",
):
    """
    安全加载模型权重。

    支持以下保存格式：

    1. 直接保存 state_dict：
        torch.save(model.state_dict(), path)

    2. checkpoint 字典：
        {
            "state_dict": model.state_dict()
        }

    3. checkpoint 字典：
        {
            "model_state_dict": model.state_dict()
        }

    4. DataParallel 保存的权重：
        module.xxx
    """
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(
            f"找不到模型权重文件: {pth_path}"
        )

    checkpoint = torch.load(
        pth_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "model" in checkpoint and isinstance(
            checkpoint["model"],
            dict,
        ):
            state_dict = checkpoint["model"]

        else:
            # 训练代码使用：
            # torch.save(model.state_dict(), path)
            state_dict = checkpoint
    else:
        raise TypeError(
            "权重文件格式错误，预期为 state_dict 或 checkpoint 字典，"
            f"实际类型为 {type(checkpoint)}"
        )

    # 去除 DataParallel 的 module. 前缀
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        cleaned_state_dict[new_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    print(
        f"[INFO] Successfully loaded checkpoint: "
        f"{pth_path}"
    )


def compute_hd95_binary(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing=None,
) -> float:
    """
    计算二值分割结果的 HD95。

    Args:
        pred_mask:
            [H, W]，bool 或 0/1 numpy 数组。

        gt_mask:
            [H, W]，bool 或 0/1 numpy 数组。

        spacing:
            像素间距，例如：
                (spacing_y, spacing_x)

            如果为 None，则 HD95 单位为像素。

    Returns:
        HD95，float。
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"HD95 输入尺寸不一致: "
            f"pred={pred_mask.shape}, gt={gt_mask.shape}"
        )

    pred_sum = int(pred_mask.sum())
    gt_sum = int(gt_mask.sum())

    # 两者都为空，认为完全匹配
    if pred_sum == 0 and gt_sum == 0:
        return 0.0

    # 只有一个为空时，用图像对角线作为惩罚值
    if pred_sum == 0 or gt_sum == 0:
        height, width = gt_mask.shape

        if spacing is None:
            return float(
                np.sqrt(
                    height ** 2
                    + width ** 2
                )
            )

        spacing_y, spacing_x = spacing

        return float(
            np.sqrt(
                (height * spacing_y) ** 2
                + (width * spacing_x) ** 2
            )
        )

    structure = generate_binary_structure(
        rank=2,
        connectivity=1,
    )

    pred_surface = pred_mask ^ binary_erosion(
        pred_mask,
        structure=structure,
        border_value=0,
    )

    gt_surface = gt_mask ^ binary_erosion(
        gt_mask,
        structure=structure,
        border_value=0,
    )

    # distance_transform_edt 在输入为 0 的位置表示目标点。
    # 因此 ~gt_surface 会让 gt 边界位置为 0。
    dt_to_gt = distance_transform_edt(
        ~gt_surface,
        sampling=spacing,
    )

    dt_to_pred = distance_transform_edt(
        ~pred_surface,
        sampling=spacing,
    )

    pred_to_gt_distances = dt_to_gt[
        pred_surface
    ]

    gt_to_pred_distances = dt_to_pred[
        gt_surface
    ]

    all_surface_distances = np.concatenate(
        [
            pred_to_gt_distances,
            gt_to_pred_distances,
        ],
        axis=0,
    )

    if all_surface_distances.size == 0:
        return 0.0

    hd95 = np.percentile(
        all_surface_distances,
        95,
    )

    return float(hd95)


@torch.no_grad()
def compute_metrics_per_image_gpu(
    pred_prob: torch.Tensor,
    gt_np: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1.0,
    device="cuda:0",
    hd95_spacing=None,
):
    """
    计算单张图片的分割指标。

    Args:
        pred_prob:
            sigmoid 后概率图，形状为 [1, 1, H, W]。

        gt_np:
            原始 GT，形状为 [H, W]。

        threshold:
            预测二值化阈值。

        smooth:
            Dice、IoU 和 MCC 中的平滑项。

        device:
            计算指标使用的设备。

        hd95_spacing:
            None 或 (spacing_y, spacing_x)。

    Returns:
        包含各项指标的字典。
    """
    if pred_prob.ndim != 4:
        raise ValueError(
            f"pred_prob 应为 [B,C,H,W]，"
            f"当前 shape={tuple(pred_prob.shape)}"
        )

    if pred_prob.shape[0] != 1:
        raise ValueError(
            f"测试阶段 batch size 应为 1，"
            f"当前 batch={pred_prob.shape[0]}"
        )

    if pred_prob.shape[1] != 1:
        raise ValueError(
            "当前指标实现用于二分类单通道分割，"
            f"当前输出通道数={pred_prob.shape[1]}"
        )

    gt = torch.as_tensor(
        gt_np,
        device=device,
        dtype=torch.float32,
    )

    gt_max = gt.max()

    if gt_max > 0:
        gt = gt / gt_max

    gt = gt.unsqueeze(0).unsqueeze(0)
    gt = gt >= 0.5

    pred = pred_prob >= threshold

    if pred.shape != gt.shape:
        raise ValueError(
            f"预测和 GT 尺寸不一致: "
            f"pred={tuple(pred.shape)}, "
            f"gt={tuple(gt.shape)}"
        )

    pred_f = pred.float()
    gt_f = gt.float()

    tp = (
        pred_f * gt_f
    ).sum(dim=(1, 2, 3))

    fp = (
        pred_f * (1.0 - gt_f)
    ).sum(dim=(1, 2, 3))

    fn = (
        (1.0 - pred_f) * gt_f
    ).sum(dim=(1, 2, 3))

    tn = (
        (1.0 - pred_f) * (1.0 - gt_f)
    ).sum(dim=(1, 2, 3))

    pred_sum = pred_f.sum(
        dim=(1, 2, 3)
    )

    gt_sum = gt_f.sum(
        dim=(1, 2, 3)
    )

    union = (
        pred_sum
        + gt_sum
        - tp
    )

    dice = (
        2.0 * tp
        + smooth
    ) / (
        pred_sum
        + gt_sum
        + smooth
    )

    iou = (
        tp
        + smooth
    ) / (
        union
        + smooth
    )

    both_empty = (
        (gt_sum == 0)
        & (pred_sum == 0)
    )

    sensitivity = torch.where(
        both_empty,
        torch.ones_like(tp),
        tp / (
            tp
            + fn
            + 1e-8
        ),
    )

    precision = torch.where(
        both_empty,
        torch.ones_like(tp),
        tp / (
            tp
            + fp
            + 1e-8
        ),
    )

    f1 = torch.where(
        both_empty,
        torch.ones_like(tp),
        2.0
        * precision
        * sensitivity
        / (
            precision
            + sensitivity
            + 1e-8
        ),
    )

    specificity = torch.where(
        both_empty,
        torch.ones_like(tp),
        tn / (
            tn
            + fp
            + 1e-8
        ),
    )

    accuracy = (
        tp
        + tn
    ) / (
        tp
        + tn
        + fp
        + fn
        + 1e-8
    )

    mcc_denominator = torch.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    mcc = torch.where(
        both_empty,
        torch.ones_like(tp),
        torch.where(
            mcc_denominator > 0,
            (
                tp * tn
                - fp * fn
            ) / (
                mcc_denominator
                + 1e-8
            ),
            torch.zeros_like(tp),
        ),
    )

    # HD95 在 CPU 上使用 numpy/scipy 计算
    pred_np = (
        pred[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )

    gt_np_float = gt_np.astype(
        np.float32
    )

    gt_np_max = gt_np_float.max()

    if gt_np_max > 0:
        gt_np_float = (
            gt_np_float
            / gt_np_max
        )

    gt_np_bin = (
        gt_np_float >= 0.5
    )

    hd95 = compute_hd95_binary(
        pred_mask=pred_np,
        gt_mask=gt_np_bin,
        spacing=hd95_spacing,
    )

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "sensitivity": float(
            sensitivity.item()
        ),
        "specificity": float(
            specificity.item()
        ),
        "accuracy": float(
            accuracy.item()
        ),
        "precision": float(
            precision.item()
        ),
        "f1": float(f1.item()),
        "mcc": float(mcc.item()),
        "hd95": float(hd95),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "通用二分类医学图像分割测试脚本，"
            "支持 ISIC、PH2 和 paired 数据集。"
        )
    )

    # =====================================================
    # 模型参数
    # =====================================================
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=[
            "unet",
            "deeplabv3",
            "deeplabv3_resnet50",
            "nnunet",
            "nnunet2d",
            "nnunet_2d",
        ],
        help="选择分割模型",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="二分类单通道分割设置为 1",
    )

    parser.add_argument(
        "--testsize",
        type=int,
        default=352,
        help="模型测试输入尺寸",
    )

    parser.add_argument(
        "--pth_path",
        type=str,
        required=True,
        help="模型权重路径",
    )

    # =====================================================
    # 测试数据
    # =====================================================
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/data/zjy_work/ISIC2018/test/",
        help=(
            "测试集根路径，"
            "目录中需包含 Images/ 和 Masks/"
        ),
    )

    parser.add_argument(
        "--test_list",
        type=str,
        default=None,
        help=(
            "测试集 ID 列表。"
            "每行可写 IMD003、IMD003.bmp 或完整图片路径。"
        ),
    )

    parser.add_argument(
        "--test_mode",
        type=str,
        default="isic",
        choices=SUPPORTED_MODES,
        help=(
            "测试数据的命名匹配模式："
            "isic、ph2、paired 或 auto"
        ),
    )

    parser.add_argument(
        "--mask_suffix",
        type=str,
        default="_segmentation",
        help=(
            "ISIC mask 后缀，默认 _segmentation。"
            "PH2 模式会自动使用 _lesion。"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="预测概率二值化阈值",
    )

    # =====================================================
    # 输出路径
    # =====================================================
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./model_out/ISIC2018_eval/baseline/",
        help="测试结果输出目录",
    )

    parser.add_argument(
        "--save_csv",
        type=str,
        default="per_image_metrics.csv",
        help="逐图指标 CSV 文件名",
    )

    parser.add_argument(
        "--save_summary_csv",
        type=str,
        default="summary_metrics.csv",
        help="汇总指标 CSV 文件名",
    )

    # =====================================================
    # 运行设备
    # =====================================================
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="启用自动混合精度推理",
    )

    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="每多少张图片打印一次测试进度",
    )

    # =====================================================
    # HD95
    # =====================================================
    parser.add_argument(
        "--hd95_spacing",
        type=float,
        nargs=2,
        default=None,
        metavar=(
            "SPACING_Y",
            "SPACING_X",
        ),
        help=(
            "HD95 像素间距，例如："
            "--hd95_spacing 0.5 0.5。"
            "默认单位为像素。"
        ),
    )

    return parser.parse_args()


def validate_args(args):
    """
    测试前参数检查。
    """
    if args.num_classes != 1:
        raise ValueError(
            "当前测试指标实现只支持二分类单通道输出，"
            f"--num_classes 应为 1，当前值为 {args.num_classes}"
        )

    if args.testsize <= 0:
        raise ValueError(
            f"--testsize 必须大于 0，"
            f"当前值为 {args.testsize}"
        )

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            f"--threshold 必须在 [0,1] 范围内，"
            f"当前值为 {args.threshold}"
        )

    if not os.path.isfile(args.pth_path):
        raise FileNotFoundError(
            f"找不到权重文件: {args.pth_path}"
        )

    if not os.path.isdir(args.test_data_path):
        raise FileNotFoundError(
            f"找不到测试集根目录: {args.test_data_path}"
        )

    image_root = os.path.join(
        args.test_data_path,
        "Images",
    )

    gt_root = os.path.join(
        args.test_data_path,
        "Masks",
    )

    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"找不到测试图片目录: {image_root}"
        )

    if not os.path.isdir(gt_root):
        raise FileNotFoundError(
            f"找不到测试掩码目录: {gt_root}"
        )

    if (
        args.test_list is not None
        and args.test_list != ""
        and not os.path.isfile(args.test_list)
    ):
        raise FileNotFoundError(
            f"找不到测试列表: {args.test_list}"
        )


def save_prediction_masks(
    prob: torch.Tensor,
    image_name: str,
    probability_mask_dir: str,
    binary_mask_dir: str,
    threshold: float,
):
    """
    保存概率预测图和二值预测图。

    概率预测图：
        sigmoid 概率乘 255。

    二值预测图：
        probability >= threshold 后保存为 0/255。
    """
    prob_np = (
        prob[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    prob_np = np.clip(
        prob_np,
        0.0,
        1.0,
    )

    probability_mask = (
        prob_np * 255.0
    ).astype(np.uint8)

    binary_mask = (
        prob_np >= threshold
    ).astype(np.uint8) * 255

    # 使用原图片 stem，统一保存成 PNG，
    # 避免 JPEG 压缩破坏 mask。
    image_stem = os.path.splitext(
        os.path.basename(image_name)
    )[0]

    probability_path = os.path.join(
        probability_mask_dir,
        f"{image_stem}.png",
    )

    binary_path = os.path.join(
        binary_mask_dir,
        f"{image_stem}.png",
    )

    probability_ok = cv2.imwrite(
        probability_path,
        probability_mask,
    )

    binary_ok = cv2.imwrite(
        binary_path,
        binary_mask,
    )

    if not probability_ok:
        raise RuntimeError(
            f"概率 mask 保存失败: {probability_path}"
        )

    if not binary_ok:
        raise RuntimeError(
            f"二值 mask 保存失败: {binary_path}"
        )


def main():
    args = parse_args()

    validate_args(args)

    device = torch.device(
        args.device
    )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 CUDA 设备，但当前 PyTorch 无法使用 CUDA。"
            )

        torch.backends.cudnn.benchmark = True

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    probability_mask_dir = os.path.join(
        args.out_dir,
        "probability_masks",
    )

    binary_mask_dir = os.path.join(
        args.out_dir,
        "binary_masks",
    )

    os.makedirs(
        probability_mask_dir,
        exist_ok=True,
    )

    os.makedirs(
        binary_mask_dir,
        exist_ok=True,
    )

    print(f"[INFO] Evaluating model: {args.model}")
    print(f"[INFO] pth_path: {args.pth_path}")
    print(f"[INFO] test_data_path: {args.test_data_path}")
    print(f"[INFO] test_list: {args.test_list}")
    print(f"[INFO] test_mode: {args.test_mode}")
    print(f"[INFO] threshold: {args.threshold}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] AMP: {args.amp}")

    if args.hd95_spacing is None:
        hd95_spacing = None

        print(
            "[INFO] HD95 spacing: None, "
            "unit = pixels"
        )
    else:
        hd95_spacing = tuple(
            args.hd95_spacing
        )

        print(
            f"[INFO] HD95 spacing: "
            f"{hd95_spacing}"
        )

    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained_backbone=False,
    ).to(device)

    load_state_dict_safely(
        model=model,
        pth_path=args.pth_path,
        device="cpu",
    )

    model.eval()

    image_root = os.path.join(
        args.test_data_path,
        "Images",
    )

    gt_root = os.path.join(
        args.test_data_path,
        "Masks",
    )

    test_loader = SegTestDataset(
        image_root=image_root,
        gt_root=gt_root,
        testsize=args.testsize,
        list_txt=args.test_list,
        mode=args.test_mode,
        mask_suffix=args.mask_suffix,
    )

    num_images = test_loader.size

    if num_images == 0:
        raise RuntimeError(
            "test_loader.size == 0。"
            "请检查 test_list、Images/Masks 路径、"
            "文件命名规则以及 --test_mode。"
        )

    print(
        f"[INFO] Number of test images: "
        f"{num_images}"
    )

    per_image_records = []

    metric_names = [
        "dice",
        "iou",
        "sensitivity",
        "specificity",
        "accuracy",
        "precision",
        "f1",
        "mcc",
        "hd95",
    ]

    sums = {
        metric_name: 0.0
        for metric_name in metric_names
    }

    inference_start_event = None
    inference_end_event = None

    if device.type == "cuda":
        inference_start_event = (
            torch.cuda.Event(
                enable_timing=True
            )
        )

        inference_end_event = (
            torch.cuda.Event(
                enable_timing=True
            )
        )

    total_inference_time_ms = 0.0

    for index in range(num_images):
        image, gt, name = (
            test_loader.load_data()
        )

        gt_np = np.asarray(
            gt,
            dtype=np.float32,
        )

        image = image.to(
            device,
            non_blocking=True,
        )

        if device.type == "cuda":
            inference_start_event.record()

        with torch.inference_mode():
            if (
                args.amp
                and device.type == "cuda"
            ):
                with torch.cuda.amp.autocast(
                    enabled=True
                ):
                    output = model(image)
                    logits = extract_logits(
                        output
                    )
            else:
                output = model(image)
                logits = extract_logits(
                    output
                )

            if logits.ndim != 4:
                raise ValueError(
                    f"模型输出应为 [B,C,H,W]，"
                    f"当前 shape={tuple(logits.shape)}"
                )

            if logits.shape[1] != 1:
                raise ValueError(
                    "当前测试脚本用于单通道二分类分割，"
                    f"模型输出通道数为 {logits.shape[1]}"
                )

            logits = F.interpolate(
                logits,
                size=gt_np.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            prob = torch.sigmoid(
                logits
            )

        if device.type == "cuda":
            inference_end_event.record()
            torch.cuda.synchronize(device)

            total_inference_time_ms += (
                inference_start_event.elapsed_time(
                    inference_end_event
                )
            )

        metrics = compute_metrics_per_image_gpu(
            pred_prob=prob,
            gt_np=gt_np,
            threshold=args.threshold,
            smooth=1.0,
            device=device,
            hd95_spacing=hd95_spacing,
        )

        for metric_name in metric_names:
            sums[metric_name] += (
                metrics[metric_name]
            )

        per_image_records.append(
            {
                "image_name": name,
                "dice": round(
                    metrics["dice"],
                    6,
                ),
                "iou": round(
                    metrics["iou"],
                    6,
                ),
                "hd95": round(
                    metrics["hd95"],
                    6,
                ),
                "sensitivity": round(
                    metrics["sensitivity"],
                    6,
                ),
                "specificity": round(
                    metrics["specificity"],
                    6,
                ),
                "accuracy": round(
                    metrics["accuracy"],
                    6,
                ),
                "precision": round(
                    metrics["precision"],
                    6,
                ),
                "f1": round(
                    metrics["f1"],
                    6,
                ),
                "mcc": round(
                    metrics["mcc"],
                    6,
                ),
            }
        )

        save_prediction_masks(
            prob=prob,
            image_name=name,
            probability_mask_dir=probability_mask_dir,
            binary_mask_dir=binary_mask_dir,
            threshold=args.threshold,
        )

        current_index = index + 1

        if (
            current_index % args.print_freq == 0
            or current_index == num_images
        ):
            print(
                f"[TEST] "
                f"[{current_index:04d}/{num_images:04d}] "
                f"name={name}, "
                f"Dice={metrics['dice']:.6f}, "
                f"IoU={metrics['iou']:.6f}, "
                f"HD95={metrics['hd95']:.6f}"
            )

    means = {
        metric_name: (
            sums[metric_name]
            / num_images
        )
        for metric_name in metric_names
    }

    if (
        device.type == "cuda"
        and num_images > 0
    ):
        average_inference_time_ms = (
            total_inference_time_ms
            / num_images
        )
    else:
        average_inference_time_ms = 0.0

    print("*****************************************************")
    print(f"Number of images: {num_images}")
    print(f"Dice Score: {means['dice']:.6f}")
    print(f"Jaccard Score: {means['iou']:.6f}")
    print(f"HD95: {means['hd95']:.6f}")
    print(
        f"Sensitivity: "
        f"{means['sensitivity']:.6f}"
    )
    print(
        f"Specificity: "
        f"{means['specificity']:.6f}"
    )
    print(
        f"Accuracy: "
        f"{means['accuracy']:.6f}"
    )
    print(
        f"Precision: "
        f"{means['precision']:.6f}"
    )
    print(f"F1: {means['f1']:.6f}")
    print(f"MCC: {means['mcc']:.6f}")

    if device.type == "cuda":
        print(
            f"Average inference time: "
            f"{average_inference_time_ms:.4f} ms/image"
        )

    print("Finish!")
    print("*****************************************************")

    # =====================================================
    # 保存逐图指标
    # =====================================================
    per_csv_path = os.path.join(
        args.out_dir,
        args.save_csv,
    )

    per_df = pd.DataFrame(
        per_image_records
    )

    per_df.to_csv(
        per_csv_path,
        index=False,
    )

    print(
        f"Per-image metrics saved to: "
        f"{per_csv_path}"
    )

    # =====================================================
    # 保存汇总指标
    # =====================================================
    summary_record = {
        "Model": args.model,
        "Checkpoint": args.pth_path,
        "TestMode": args.test_mode,
        "Threshold": args.threshold,
        "NumImages": num_images,
        "Sensitivity": round(
            means["sensitivity"],
            6,
        ),
        "Specificity": round(
            means["specificity"],
            6,
        ),
        "Accuracy": round(
            means["accuracy"],
            6,
        ),
        "Precision": round(
            means["precision"],
            6,
        ),
        "F1": round(
            means["f1"],
            6,
        ),
        "MCC": round(
            means["mcc"],
            6,
        ),
        "Jaccard": round(
            means["iou"],
            6,
        ),
        "Dice": round(
            means["dice"],
            6,
        ),
        "HD95": round(
            means["hd95"],
            6,
        ),
    }

    if device.type == "cuda":
        summary_record[
            "AvgInferenceTimeMs"
        ] = round(
            average_inference_time_ms,
            6,
        )

    summary_df = pd.DataFrame(
        [summary_record]
    )

    summary_csv_path = os.path.join(
        args.out_dir,
        args.save_summary_csv,
    )

    summary_df.to_csv(
        summary_csv_path,
        index=False,
    )

    print(
        f"Summary metrics saved to: "
        f"{summary_csv_path}"
    )

    print(
        f"Probability masks saved to: "
        f"{probability_mask_dir}"
    )

    print(
        f"Binary masks saved to: "
        f"{binary_mask_dir}"
    )


if __name__ == "__main__":
    main()
