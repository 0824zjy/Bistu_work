import os
import time
import argparse
import logging
from datetime import datetime

import numpy as np

import torch
import torch.nn.functional as F

from dataloader_seg_baseline import get_loader, SegTestDataset
from seg_baselines import get_model, extract_logits


def dice_loss_with_logits(logits, targets, eps=1e-7):
    """
    Dice loss。

    Args:
        logits:
            模型原始输出，未经过 sigmoid。
            形状通常为 [B, 1, H, W]。

        targets:
            二值掩码，取值范围为 [0, 1]。
            形状通常为 [B, 1, H, W]。

        eps:
            防止除零。

    Returns:
        标量 Dice loss。
    """
    probs = torch.sigmoid(logits)

    numerator = (
        2.0 * (probs * targets).sum(dim=(2, 3))
        + eps
    )

    denominator = (
        probs.pow(2).sum(dim=(2, 3))
        + targets.pow(2).sum(dim=(2, 3))
        + eps
    )

    dice_loss = 1.0 - numerator / denominator

    return dice_loss.mean()


def seg_loss(
    logits,
    targets,
    bce_weight=1.0,
    dice_weight=1.0,
):
    """
    BCEWithLogitsLoss + Dice loss。

    Args:
        logits:
            模型原始输出，未经过 sigmoid。

        targets:
            二值掩码，取值为 0 或 1。

        bce_weight:
            BCE loss 权重。

        dice_weight:
            Dice loss 权重。

    Returns:
        total_loss, bce_loss, dice_loss
    """
    if logits.ndim != 4:
        raise ValueError(
            f"logits 应为四维张量 [B,C,H,W]，"
            f"当前 shape={tuple(logits.shape)}"
        )

    if targets.ndim != 4:
        raise ValueError(
            f"targets 应为四维张量 [B,C,H,W]，"
            f"当前 shape={tuple(targets.shape)}"
        )

    if logits.shape[2:] != targets.shape[2:]:
        logits = F.interpolate(
            logits,
            size=targets.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
    )

    dice = dice_loss_with_logits(
        logits,
        targets,
    )

    loss = (
        bce_weight * bce
        + dice_weight * dice
    )

    return loss, bce, dice


def clip_gradient(optimizer, grad_clip):
    """
    按元素裁剪梯度。

    与原始训练代码保持一致。
    """
    if grad_clip <= 0:
        return

    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(
                    -grad_clip,
                    grad_clip,
                )


def adjust_lr(
    optimizer,
    init_lr,
    epoch,
    decay_rate=0.1,
    decay_epoch=200,
):
    """
    阶梯式学习率衰减。

    使用 (epoch - 1) // decay_epoch，使得：
        epoch 1~200   : init_lr
        epoch 201~400 : init_lr * decay_rate

    例如：
        init_lr=1e-4
        decay_rate=0.1
        decay_epoch=200
    """
    if decay_epoch <= 0:
        lr = init_lr
    else:
        decay = decay_rate ** ((epoch - 1) // decay_epoch)
        lr = init_lr * decay

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


class AvgMeter:
    """
    记录最近 num 个 iteration 的均值。
    """

    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.losses = []

    def update(self, val):
        if torch.is_tensor(val):
            val = val.detach().float().cpu().item()

        self.losses.append(float(val))

    def show(self):
        if len(self.losses) == 0:
            return 0.0

        start = max(
            len(self.losses) - self.num,
            0,
        )

        recent = self.losses[start:]

        return float(np.mean(recent))


@torch.no_grad()
def evaluate_dice(
    model,
    data_path,
    img_size,
    test_list=None,
    device="cuda:0",
    test_mode="isic",
    mask_suffix="_segmentation",
):
    """
    在测试集上计算二值 Dice。

    测试数据目录格式：

        data_path/
        ├── Images/
        └── Masks/

    测试流程：
        1. 输入图片 resize 到 img_size。
        2. GT 保持原始尺寸。
        3. 模型预测 resize 回 GT 原始尺寸。
        4. sigmoid 后以 0.5 为阈值二值化。
        5. 逐图计算 Dice，最后取平均值。

    Args:
        model:
            分割模型。

        data_path:
            测试集根目录。

        img_size:
            模型输入尺寸。

        test_list:
            测试样本 ID txt。

        device:
            torch.device 或设备字符串。

        test_mode:
            掩码匹配模式：
                isic
                ph2
                paired
                auto

        mask_suffix:
            ISIC 掩码后缀，默认 "_segmentation"。

    Returns:
        mean_dice:
            测试集平均 Dice。

        num_images:
            有效测试图片数量。
    """
    image_root = os.path.join(
        data_path,
        "Images",
    )

    gt_root = os.path.join(
        data_path,
        "Masks",
    )

    model.eval()

    test_loader = SegTestDataset(
        image_root=image_root,
        gt_root=gt_root,
        testsize=img_size,
        list_txt=test_list,
        mode=test_mode,
        mask_suffix=mask_suffix,
    )

    num_images = test_loader.size

    if num_images == 0:
        warning = (
            "[WARN] test_loader.size == 0，"
            "请检查 test_list、Images/Masks 路径、"
            "文件命名规则以及 test_mode"
        )
        print(warning)
        logging.warning(warning)

        return 0.0, 0

    dice_sum = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )

    smooth = 1.0

    for _ in range(num_images):
        image, gt, name = test_loader.load_data()

        gt_np = np.asarray(
            gt,
            dtype=np.float32,
        )

        gt_t = torch.from_numpy(gt_np).to(
            device=device,
            dtype=torch.float32,
        )

        # PIL 灰度 mask 通常范围为 0~255。
        # 对非空 mask 归一化到 0~1。
        gt_max = gt_t.max()

        if gt_max > 0:
            gt_t = gt_t / gt_max

        gt_t = (gt_t >= 0.5).float()

        image = image.to(
            device,
            non_blocking=True,
        )

        output = model(image)
        logits = extract_logits(output)

        if logits.ndim != 4:
            raise ValueError(
                f"模型输出 logits 应为 [B,C,H,W]，"
                f"当前 shape={tuple(logits.shape)}，"
                f"样本={name}"
            )

        logits = F.interpolate(
            logits,
            size=gt_t.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # 默认二分类分割使用第一个输出通道。
        prob = torch.sigmoid(logits)

        prob = prob[0, 0]
        pred = (prob >= 0.5).float()

        intersection = (
            pred * gt_t
        ).sum()

        dice = (
            2.0 * intersection
            + smooth
        ) / (
            pred.sum()
            + gt_t.sum()
            + smooth
        )

        dice_sum += dice

    mean_dice = (
        dice_sum / float(num_images)
    ).item()

    return mean_dice, num_images


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    epoch,
    args,
    device,
    scaler=None,
):
    """
    训练一个 epoch。
    """
    model.train()

    loss_meter = AvgMeter()
    bce_meter = AvgMeter()
    dice_meter = AvgMeter()

    max_memory_usage = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for i, pack in enumerate(
        train_loader,
        start=1,
    ):
        images, gts = pack

        images = images.to(
            device,
            non_blocking=True,
        )

        gts = gts.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        use_amp = (
            args.amp
            and device.type == "cuda"
        )

        if use_amp:
            with torch.cuda.amp.autocast(
                enabled=True,
            ):
                output = model(images)
                logits = extract_logits(output)

                loss, bce, dice = seg_loss(
                    logits=logits,
                    targets=gts,
                    bce_weight=args.bce_weight,
                    dice_weight=args.dice_weight,
                )

            scaler.scale(loss).backward()

            if args.clip > 0:
                scaler.unscale_(optimizer)

                clip_gradient(
                    optimizer,
                    args.clip,
                )

            scaler.step(optimizer)
            scaler.update()

        else:
            output = model(images)
            logits = extract_logits(output)

            loss, bce, dice = seg_loss(
                logits=logits,
                targets=gts,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
            )

            loss.backward()

            if args.clip > 0:
                clip_gradient(
                    optimizer,
                    args.clip,
                )

            optimizer.step()

        loss_meter.update(loss)
        bce_meter.update(bce)
        dice_meter.update(dice)

        if device.type == "cuda":
            current_memory = (
                torch.cuda.max_memory_allocated(device)
            )

            max_memory_usage = max(
                max_memory_usage,
                current_memory,
            )

        if (
            i % args.print_freq == 0
            or i == len(train_loader)
        ):
            info = (
                f"{datetime.now()} "
                f"Epoch [{epoch:03d}/{args.epoch:03d}], "
                f"Step [{i:04d}/{len(train_loader):04d}], "
                f"loss={loss_meter.show():.4f}, "
                f"bce={bce_meter.show():.4f}, "
                f"dice_loss={dice_meter.show():.4f}, "
                f"max_mem={max_memory_usage / (1024 ** 2):.2f} MB"
            )

            print(info)
            logging.info(info)

    return loss_meter.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "通用二分类医学图像分割训练脚本，"
            "支持 ISIC、PH2 和同名 paired 数据。"
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
        help="输出类别数，二分类分割使用 1",
    )

    parser.add_argument(
        "--pretrained_backbone",
        action="store_true",
        help="仅对 DeepLabv3 等 torchvision backbone 有效",
    )

    # =====================================================
    # 训练参数
    # =====================================================
    parser.add_argument(
        "--epoch",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=[
            "AdamW",
            "SGD",
        ],
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=352,
    )

    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="启用训练数据增强",
    )

    parser.add_argument(
        "--clip",
        type=float,
        default=0.5,
        help="梯度裁剪范围；小于等于 0 表示关闭",
    )

    parser.add_argument(
        "--decay_rate",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--decay_epoch",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--bce_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dice_weight",
        type=float,
        default=1.0,
    )

    # =====================================================
    # 数据路径
    # =====================================================
    parser.add_argument(
        "--train_path1",
        type=str,
        default="/data/zjy_work/ISIC2018/train/",
        help=(
            "第一个训练集根目录，"
            "目录内需包含 Images/ 和 Masks/"
        ),
    )

    parser.add_argument(
        "--train_path2",
        type=str,
        default="/data/zjy_work/ISIC2018/exp_mask2img_cn_only/",
        help=(
            "第二个训练集根目录，"
            "目录内需包含 images/ 和 masks/"
        ),
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default="/data/zjy_work/ISIC2018/test/",
        help=(
            "测试集根目录，"
            "目录内需包含 Images/ 和 Masks/"
        ),
    )

    parser.add_argument(
        "--train_list1",
        type=str,
        default=None,
        help="第一个训练数据源的样本 ID txt",
    )

    parser.add_argument(
        "--train_list2",
        type=str,
        default=None,
        help="第二个训练数据源的样本 ID txt",
    )

    parser.add_argument(
        "--test_list",
        type=str,
        default=None,
        help="测试集样本 ID txt",
    )

    # =====================================================
    # 数据集命名匹配模式
    # =====================================================
    parser.add_argument(
        "--train_mode1",
        type=str,
        default="isic",
        choices=[
            "isic",
            "ph2",
            "paired",
            "auto",
        ],
        help=(
            "第一个训练数据源的命名匹配模式。"
            "PH2 使用 ph2。"
        ),
    )

    parser.add_argument(
        "--train_mode2",
        type=str,
        default="paired",
        choices=[
            "isic",
            "ph2",
            "paired",
            "auto",
        ],
        help=(
            "第二个训练数据源的命名匹配模式。"
            "图片和 mask 同名时使用 paired。"
        ),
    )

    parser.add_argument(
        "--test_mode",
        type=str,
        default="isic",
        choices=[
            "isic",
            "ph2",
            "paired",
            "auto",
        ],
        help=(
            "测试数据集命名匹配模式。"
            "PH2 测试集使用 ph2。"
        ),
    )

    parser.add_argument(
        "--mask_suffix",
        type=str,
        default="_segmentation",
        help=(
            "ISIC 掩码后缀，默认 _segmentation。"
            "PH2 模式固定优先使用 _lesion。"
        ),
    )

    # =====================================================
    # 输出路径
    # =====================================================
    parser.add_argument(
        "--train_save",
        type=str,
        default="./model_out/ISIC2018/baseline/",
        help="模型和日志保存目录",
    )

    # =====================================================
    # 设备和运行参数
    # =====================================================
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="启用 CUDA 自动混合精度训练",
    )

    parser.add_argument(
        "--print_freq",
        type=int,
        default=50,
    )

    return parser.parse_args()


def validate_args(args):
    """
    在训练开始前检查必要参数。
    """
    if args.epoch <= 0:
        raise ValueError(
            f"--epoch 必须大于 0，当前值为 {args.epoch}"
        )

    if args.batchsize <= 0:
        raise ValueError(
            f"--batchsize 必须大于 0，当前值为 {args.batchsize}"
        )

    if args.img_size <= 0:
        raise ValueError(
            f"--img_size 必须大于 0，当前值为 {args.img_size}"
        )

    if args.num_classes != 1:
        raise ValueError(
            "当前损失函数和测试 Dice 实现用于二分类单通道分割，"
            f"因此 --num_classes 应设为 1，当前值为 {args.num_classes}"
        )

    if not os.path.isdir(args.train_path1):
        raise FileNotFoundError(
            f"找不到第一个训练集根目录: {args.train_path1}"
        )

    if not os.path.isdir(args.test_path):
        raise FileNotFoundError(
            f"找不到测试集根目录: {args.test_path}"
        )

    if (
        args.train_list1 is not None
        and args.train_list1 != ""
        and not os.path.isfile(args.train_list1)
    ):
        raise FileNotFoundError(
            f"找不到 train_list1: {args.train_list1}"
        )

    if (
        args.train_list2 is not None
        and args.train_list2 != ""
        and not os.path.isfile(args.train_list2)
    ):
        raise FileNotFoundError(
            f"找不到 train_list2: {args.train_list2}"
        )

    if (
        args.test_list is not None
        and args.test_list != ""
        and not os.path.isfile(args.test_list)
    ):
        raise FileNotFoundError(
            f"找不到 test_list: {args.test_list}"
        )


def create_optimizer(args, model):
    """
    创建优化器。
    """
    if args.optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )

    if args.optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

    raise ValueError(
        f"不支持的优化器: {args.optimizer}"
    )


def main():
    args = parse_args()

    validate_args(args)

    os.makedirs(
        args.train_save,
        exist_ok=True,
    )

    logging.basicConfig(
        filename=os.path.join(
            args.train_save,
            "train.log",
        ),
        format=(
            "[%(asctime)s-"
            "%(filename)s-"
            "%(levelname)s:"
            "%(message)s]"
        ),
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d %I:%M:%S %p",
    )

    device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 CUDA 设备，但当前 PyTorch 无法使用 CUDA。"
            )

        torch.backends.cudnn.benchmark = True

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] train_mode1: {args.train_mode1}")
    print(f"[INFO] train_mode2: {args.train_mode2}")
    print(f"[INFO] test_mode: {args.test_mode}")

    logging.info(f"Using device: {device}")
    logging.info(f"Model: {args.model}")
    logging.info(f"train_mode1: {args.train_mode1}")
    logging.info(f"train_mode2: {args.train_mode2}")
    logging.info(f"test_mode: {args.test_mode}")

    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    logging.info(model)

    optimizer = create_optimizer(
        args=args,
        model=model,
    )

    # =====================================================
    # 数据加载
    #
    # 第一个数据源：
    #     train_path1/Images
    #     train_path1/Masks
    #
    # 第二个数据源：
    #     train_path2/images
    #     train_path2/masks
    # =====================================================
    image_roots = [
        os.path.join(
            args.train_path1,
            "Images",
        )
    ]

    gt_roots = [
        os.path.join(
            args.train_path1,
            "Masks",
        )
    ]

    list_txts = [
        args.train_list1
    ]

    modes = [
        args.train_mode1
    ]

    use_second = (
        args.train_list2 is not None
        and args.train_list2 != ""
        and args.train_path2 is not None
        and args.train_path2 != ""
    )

    if use_second:
        second_image_root = os.path.join(
            args.train_path2,
            "images",
        )

        second_gt_root = os.path.join(
            args.train_path2,
            "masks",
        )

        if not os.path.isdir(second_image_root):
            raise FileNotFoundError(
                f"找不到第二个训练集图片目录: {second_image_root}"
            )

        if not os.path.isdir(second_gt_root):
            raise FileNotFoundError(
                f"找不到第二个训练集掩码目录: {second_gt_root}"
            )

        image_roots.append(
            second_image_root
        )

        gt_roots.append(
            second_gt_root
        )

        list_txts.append(
            args.train_list2
        )

        modes.append(
            args.train_mode2
        )

    train_loader = get_loader(
        image_roots=image_roots,
        gt_roots=gt_roots,
        batchsize=args.batchsize,
        trainsize=args.img_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        augmentation=args.augmentation,
        drop_last=True,
        list_txts=list_txts,
        modes=modes,
        mask_suffix=args.mask_suffix,
    )

    print(f"[INFO] image_roots = {image_roots}")
    print(f"[INFO] gt_roots    = {gt_roots}")
    print(f"[INFO] list_txts   = {list_txts}")
    print(f"[INFO] modes       = {modes}")
    print(f"[INFO] total_step  = {len(train_loader)}")

    logging.info(f"image_roots = {image_roots}")
    logging.info(f"gt_roots = {gt_roots}")
    logging.info(f"list_txts = {list_txts}")
    logging.info(f"modes = {modes}")
    logging.info(f"total_step = {len(train_loader)}")

    if len(train_loader) == 0:
        raise RuntimeError(
            "train_loader 长度为 0。"
            "可能是有效样本数小于 batchsize 且 drop_last=True，"
            "或者训练数据未正确匹配。"
        )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            args.amp
            and device.type == "cuda"
        )
    )

    best_dice = 0.0
    best_epoch = 0
    total_train_time = 0.0

    print(
        "#" * 20,
        f"Start Training {args.model}",
        "#" * 20,
    )

    logging.info(
        f"Start Training {args.model}"
    )

    for epoch in range(
        1,
        args.epoch + 1,
    ):
        current_lr = adjust_lr(
            optimizer=optimizer,
            init_lr=args.lr,
            epoch=epoch,
            decay_rate=args.decay_rate,
            decay_epoch=args.decay_epoch,
        )

        lr_info = (
            f"[LR] Epoch {epoch:03d}, "
            f"lr={current_lr:.8f}"
        )

        print(lr_info)
        logging.info(lr_info)

        start_time = time.time()

        train_loss = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            device=device,
            scaler=scaler,
        )

        epoch_time = (
            time.time() - start_time
        )

        total_train_time += epoch_time

        time_info = (
            f"[TIME] Epoch {epoch:03d}, "
            f"time={epoch_time:.2f}s, "
            f"total={total_train_time:.2f}s"
        )

        print(time_info)
        logging.info(time_info)

        # 每个 epoch 保存最新模型权重
        save_last = os.path.join(
            args.train_save,
            f"{args.model}-last.pth",
        )

        torch.save(
            model.state_dict(),
            save_last,
        )

        # 测试集评估
        dice, n_images = evaluate_dice(
            model=model,
            data_path=args.test_path,
            img_size=args.img_size,
            test_list=args.test_list,
            device=device,
            test_mode=args.test_mode,
            mask_suffix=args.mask_suffix,
        )

        if n_images > 0:
            eval_info = (
                f"[EVAL] Epoch {epoch:03d}, "
                f"train_loss={train_loss:.6f}, "
                f"Dice={dice:.6f}, "
                f"n={n_images}"
            )

            print(eval_info)
            logging.info(eval_info)

            if dice > best_dice:
                best_info = (
                    f"[BEST] Dice improved "
                    f"{best_dice:.6f} -> {dice:.6f}"
                )

                print(best_info)
                logging.info(best_info)

                best_dice = dice
                best_epoch = epoch

                save_best = os.path.join(
                    args.train_save,
                    f"{args.model}-best.pth",
                )

                torch.save(
                    model.state_dict(),
                    save_best,
                )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_time = (
        total_train_time / args.epoch
    )

    done_info_1 = (
        f"[DONE] Best Dice: {best_dice:.6f}"
    )

    done_info_2 = (
        f"[DONE] Best Epoch: {best_epoch}"
    )

    done_info_3 = (
        f"[DONE] Avg train time per epoch: "
        f"{avg_time:.2f}s"
    )

    print(done_info_1)
    print(done_info_2)
    print(done_info_3)

    logging.info(done_info_1)
    logging.info(done_info_2)
    logging.info(done_info_3)


if __name__ == "__main__":
    main()
