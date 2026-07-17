import os
import sys
import time
import argparse
import logging
import warnings
from datetime import datetime
from collections import Counter

sys.path.insert(0, "/data/zjy_work/BGDNet")
sys.path.insert(0, "/data/zjy_work/Work3_BEF_SBG/segmentation")

import numpy as np
import torch

# Work3 BEF-SBG cuDNN safe switch
import os
if os.environ.get("BGDNET_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    print("[WARN] cuDNN disabled by BGDNET_DISABLE_CUDNN=1")
else:
    torch.backends.cudnn.enabled = True

torch.backends.cudnn.benchmark = False

import torch.nn as nn
import torch.nn.functional as F

from models.BGDNet import BGDNet
from utils.dataloader_BGDiff import test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

from dataloader_BEF import get_weighted_loader


warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ["1", "true", "yes", "y", "on"]
def set_seed(seed: int = 0):
    """
    Reproducibility helper.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_bn_layers(model):
    """
    Freeze BatchNorm statistics and affine parameters.

    Important:
        model.train() will reset BN layers to train mode,
        so this function should be called after model.train()
        in every epoch when freeze_bn=True.
    """
    bn_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )

    for m in model.modules():
        if isinstance(m, bn_types):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def strip_module_prefix(state_dict):
    """
    Remove DataParallel 'module.' prefix if present.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    return state_dict


def load_resume_checkpoint(model, resume_path: str):
    """
    Load checkpoint for fine-tuning.

    Supports:
        1. plain state_dict
        2. {"state_dict": ...}
        3. {"model": ...}
        4. {"model_state_dict": ...}
    """
    if resume_path is None or str(resume_path).strip() == "":
        return

    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    print(f"[Resume] loading checkpoint from: {resume_path}")

    ckpt = torch.load(resume_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = strip_module_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)

    print("[Resume] loaded successfully.")


def get_save_state_dict(model):
    """
    Save clean state_dict without DataParallel 'module.' prefix.
    """
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()

def sanitize_weights(weights: torch.Tensor, max_weight: float = 1.5):
    """
    Remove NaN/Inf and clamp sample weights.
    """
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.clamp(weights, 0.0, max_weight)
    return weights


def soft_dice_loss_per_sample(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0):
    """
    Per-sample soft Dice loss.

    Args:
        logits:  [B,1,H,W]
        targets: [B,1,H,W]

    Returns:
        loss: [B]
    """
    probs = torch.sigmoid(logits)

    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2.0 * inter + eps) / (denom + eps)

    return 1.0 - dice

def focal_tversky_loss_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha_fp: float = 0.3,
    beta_fn: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1e-6,
):
    """
    Per-sample Focal Tversky loss.

    Args:
        logits:  [B,1,H,W]
        targets: [B,1,H,W]

    alpha_fp:
        weight for false positives.

    beta_fn:
        weight for false negatives.
        For lesion segmentation, missing lesion pixels often hurts Dice more,
        so beta_fn is usually larger than alpha_fp.

    gamma:
        focal exponent.
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()

    tp = (probs * targets).sum(dim=(1, 2, 3))
    fp = (probs * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - probs) * targets).sum(dim=(1, 2, 3))

    tversky = (tp + eps) / (
        tp
        + float(alpha_fp) * fp
        + float(beta_fn) * fn
        + eps
    )

    loss = torch.pow(1.0 - tversky, float(gamma))

    return loss

def sanitize_weights(weights, max_weight=1.5):
    weights = weights.float()
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.clamp(weights, 0.0, float(max_weight))
    return weights


def tensor_all_finite(x):
    return bool(torch.isfinite(x).all().item())


def model_has_nonfinite_params(model):
    for name, p in model.named_parameters():
        if p is not None and not torch.isfinite(p).all():
            print(f"[ERROR] non-finite model parameter detected: {name}")
            return True
    return False


def optimizer_has_nonfinite_grads(model):
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            print(f"[WARN] non-finite gradient detected: {name}")
            return True
    return False


def safe_meter_show(meter):
    if len(meter.losses) == 0:
        return 0.0
    v = meter.show()
    if torch.is_tensor(v):
        v = float(v.detach().cpu().item())
    else:
        v = float(v)
    if not np.isfinite(v):
        return 0.0
    return v

def weighted_joint_loss(
    pred_m,
    gts,
    pred_b,
    bnds,
    weights,
    alpha=1.0,
    beta=0.25,
    eps=1e-8,
    max_weight=1.5,
    use_tversky=True,
    seg_bce_w=0.5,
    seg_dice_w=1.0,
    seg_tversky_w=0.5,
    tversky_alpha_fp=0.3,
    tversky_beta_fn=0.7,
    tversky_gamma=0.75,
):
    """
    Final weighted loss.

    Segmentation:
        seg_bce_w * BCE
        + seg_dice_w * soft Dice
        + seg_tversky_w * Focal Tversky

    Boundary:
        0.5 * class-balanced BCE
        + 0.5 * boundary Dice

    Return:
        loss, seg_loss_mean, bnd_loss_mean, raw_loss_mean, valid_n
    """
    pred_m = F.interpolate(
        pred_m,
        size=gts.shape[2:],
        mode="bilinear",
        align_corners=False,
    )

    pred_b = F.interpolate(
        pred_b,
        size=bnds.shape[2:],
        mode="bilinear",
        align_corners=False,
    )

    # -------------------------
    # Segmentation BCE
    # -------------------------
    seg_bce_i = F.binary_cross_entropy_with_logits(
        pred_m,
        gts,
        reduction="none",
    ).mean(dim=(1, 2, 3))

    # -------------------------
    # Segmentation Dice
    # -------------------------
    seg_dice_i = soft_dice_loss_per_sample(pred_m, gts)

    seg_loss_i = (
        float(seg_bce_w) * seg_bce_i
        + float(seg_dice_w) * seg_dice_i
    )

    # -------------------------
    # Segmentation Focal Tversky
    # -------------------------
    if use_tversky:
        seg_tversky_i = focal_tversky_loss_per_sample(
            pred_m,
            gts,
            alpha_fp=float(tversky_alpha_fp),
            beta_fn=float(tversky_beta_fn),
            gamma=float(tversky_gamma),
        )

        seg_loss_i = seg_loss_i + float(seg_tversky_w) * seg_tversky_i

    # -------------------------
    # Boundary balanced BCE + Dice
    # -------------------------
    pos = bnds.sum(dim=(1, 2, 3))
    total = float(bnds[0].numel())
    neg = total - pos

    pos_weight = (neg / (pos + eps)).clamp(1.0, 20.0)
    pos_weight = pos_weight.view(-1, 1, 1, 1)

    bnd_bce_map = F.binary_cross_entropy_with_logits(
        pred_b,
        bnds,
        reduction="none",
    )

    pixel_weight = 1.0 + (pos_weight - 1.0) * bnds
    bnd_bce_i = (bnd_bce_map * pixel_weight).mean(dim=(1, 2, 3))

    bnd_dice_i = soft_dice_loss_per_sample(pred_b, bnds)

    bnd_loss_i = 0.5 * bnd_bce_i + 0.5 * bnd_dice_i

    # -------------------------
    # Sample weighted loss
    # -------------------------
    loss_i = float(alpha) * seg_loss_i + float(beta) * bnd_loss_i

    weights = sanitize_weights(weights, max_weight=max_weight).view(-1).to(
        device=loss_i.device,
        dtype=loss_i.dtype,
    )

    valid_mask = (
        torch.isfinite(loss_i)
        & torch.isfinite(weights)
        & (weights > 1e-8)
    )

    valid_n = int(valid_mask.sum().item())

    if valid_n == 0:
        return None, None, None, None, 0

    loss_i_valid = loss_i[valid_mask]
    weights_valid = weights[valid_mask]

    seg_loss_valid = seg_loss_i[valid_mask]
    bnd_loss_valid = bnd_loss_i[valid_mask]

    final_loss = (
        weights_valid * loss_i_valid
    ).sum() / (weights_valid.sum() + eps)

    return (
        final_loss,
        seg_loss_valid.detach().mean(),
        bnd_loss_valid.detach().mean(),
        loss_i_valid.detach().mean(),
        valid_n,
    )

@torch.no_grad()
def test(model, data_path, img_size, test_list=None):
    image_root = os.path.join(data_path, "Images")
    gt_root = os.path.join(data_path, "Masks")

    model.eval()

    test_loader = test_dataset(
        image_root=image_root,
        gt_root=gt_root,
        testsize=img_size,
        list_txt=test_list,
        mode="isic",
    )

    num1 = test_loader.size
    if num1 == 0:
        print("[WARN] test_loader.size == 0, please check test_list and paths.")
        return 0.0, 0

    DSC = torch.zeros((), device="cuda", dtype=torch.float32)
    smooth = 1.0

    for _ in range(num1):
        image, gt, name = test_loader.load_data()

        gt_np = np.asarray(gt, np.float32)
        gt_t = torch.from_numpy(gt_np).to(device="cuda", dtype=torch.float32)
        gt_t = gt_t / (gt_t.max() + 1e-8)

        image = image.cuda(non_blocking=True)

        out = model(image)
        res = out[0] if isinstance(out, (tuple, list)) else out

        res = F.interpolate(
            res,
            size=gt_t.shape,
            mode="bilinear",
            align_corners=False,
        )

        # Important:
        # Do NOT apply per-image min-max normalization here.
        # Keep the same evaluation rule as final test:
        # sigmoid(logits) >= 0.5
        res = torch.sigmoid(res).squeeze()

        input_bin = (res >= 0.5).float()
        target_bin = (gt_t >= 0.5).float()

        intersection = (input_bin * target_bin).sum()
        dice = (2.0 * intersection + smooth) / (input_bin.sum() + target_bin.sum() + smooth)

        DSC += dice

    return (DSC / num1).item(), num1


def train_one_epoch(train_loader, model, optimizer, epoch, args):
    model.train()

    if args.freeze_bn:
        freeze_bn_layers(model)

    loss_record = AvgMeter()
    loss_seg_meter = AvgMeter()
    loss_bnd_meter = AvgMeter()
    loss_raw_meter = AvgMeter()

    skipped_batches = 0
    skipped_nonfinite_input = 0
    skipped_nonfinite_pred = 0
    skipped_nonfinite_loss = 0
    skipped_nonfinite_grad = 0
    skipped_zero_weight = 0

    source_counter = Counter()

    time_before = time.time()
    max_memory_usage = 0
    total_step = len(train_loader)

    for i, pack in enumerate(train_loader, start=1):
        images, gts, bnds, weights, sources = pack

        images = images.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)
        bnds = bnds.cuda(non_blocking=True)

        # clean NaN/Inf weights and clamp them into a safe range
        weights = weights.cuda(non_blocking=True).float()
        weights = sanitize_weights(weights, max_weight=args.max_weight)

        if isinstance(sources, (list, tuple)):
            source_counter.update(list(sources))

        # 1. input guard
        if (
            not tensor_all_finite(images)
            or not tensor_all_finite(gts)
            or not tensor_all_finite(bnds)
            or not tensor_all_finite(weights)
        ):
            skipped_batches += 1
            skipped_nonfinite_input += 1
            optimizer.zero_grad(set_to_none=True)
            print(f"[WARN] skip non-finite input batch: epoch={epoch}, step={i}")
            continue

        if weights.sum().item() <= 1e-8:
            skipped_batches += 1
            skipped_zero_weight += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)

        # 2. forward
        pred_m, pred_b = model(images)

        if (not tensor_all_finite(pred_m)) or (not tensor_all_finite(pred_b)):
            skipped_batches += 1
            skipped_nonfinite_pred += 1
            optimizer.zero_grad(set_to_none=True)
            print(f"[WARN] skip non-finite prediction batch: epoch={epoch}, step={i}")
            continue

        loss, l_seg, l_bnd, l_raw, valid_n = weighted_joint_loss(
            pred_m=pred_m,
            gts=gts,
            pred_b=pred_b,
            bnds=bnds,
            weights=weights,
            alpha=args.alpha,
            beta=args.beta,
            max_weight=args.max_weight,
            use_tversky=args.use_tversky,
            seg_bce_w=args.seg_bce_w,
            seg_dice_w=args.seg_dice_w,
            seg_tversky_w=args.seg_tversky_w,
            tversky_alpha_fp=args.tversky_alpha_fp,
            tversky_beta_fn=args.tversky_beta_fn,
            tversky_gamma=args.tversky_gamma,
        )
        if loss is None or not torch.isfinite(loss).item():
            skipped_batches += 1
            skipped_nonfinite_loss += 1
            optimizer.zero_grad(set_to_none=True)
            print(f"[WARN] skip non-finite loss batch: epoch={epoch}, step={i}")
            continue

        # 4. backward
        loss.backward()

        if optimizer_has_nonfinite_grads(model):
            skipped_batches += 1
            skipped_nonfinite_grad += 1
            optimizer.zero_grad(set_to_none=True)
            print(f"[WARN] skip non-finite grad batch: epoch={epoch}, step={i}")
            continue

        clip_gradient(optimizer, args.clip)
        optimizer.step()

        # 5. model param guard: fail fast if model is already corrupted
        if model_has_nonfinite_params(model):
            raise RuntimeError(
                f"Model parameters became non-finite after optimizer.step(), "
                f"epoch={epoch}, step={i}. Stop training immediately."
            )

        loss_record.update(loss.detach(), images.size(0))
        loss_seg_meter.update(l_seg.detach(), images.size(0))
        loss_bnd_meter.update(l_bnd.detach(), images.size(0))
        loss_raw_meter.update(l_raw.detach(), images.size(0))

        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated()
            max_memory_usage = max(max_memory_usage, current_memory)

        if i % 50 == 0 or i == total_step:
            msg = (
                f"{datetime.now()} Epoch [{epoch:03d}/{args.epoch:03d}], "
                f"Step [{i:04d}/{total_step:04d}], "
                f"loss: {safe_meter_show(loss_record):.4f} "
                f"(seg {safe_meter_show(loss_seg_meter):.4f} | "
                f"bnd {safe_meter_show(loss_bnd_meter):.4f} | "
                f"raw {safe_meter_show(loss_raw_meter):.4f}), "
                f"valid_n={valid_n}, "
                f"skipped_batches={skipped_batches}, "
                f"skip_input={skipped_nonfinite_input}, "
                f"skip_pred={skipped_nonfinite_pred}, "
                f"skip_loss={skipped_nonfinite_loss}, "
                f"skip_grad={skipped_nonfinite_grad}, "
                f"skip_zero_w={skipped_zero_weight}, "
                f"sources={dict(source_counter)}, "
                f"max_mem={max_memory_usage / (1024 ** 2):.2f} MB"
            )
            print(msg)
            logging.info(msg)

    epoch_time = time.time() - time_before

    msg = (
        f"[Epoch {epoch}] train time: {epoch_time:.2f}s, "
        f"skipped_batches={skipped_batches}, "
        f"skip_input={skipped_nonfinite_input}, "
        f"skip_pred={skipped_nonfinite_pred}, "
        f"skip_loss={skipped_nonfinite_loss}, "
        f"skip_grad={skipped_nonfinite_grad}, "
        f"skip_zero_w={skipped_zero_weight}"
    )
    print(msg)
    logging.info(msg)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weighted_train_json", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--freeze_bn",
        type=str2bool,
        default=str2bool(os.environ.get("BGDNET_FREEZE_BN", "0")),
    )

    parser.add_argument("--max_weight", type=float, default=1.5)

    parser.add_argument("--use_tversky", type=str2bool, default=True)
    parser.add_argument("--seg_bce_w", type=float, default=0.5)
    parser.add_argument("--seg_dice_w", type=float, default=1.0)
    parser.add_argument("--seg_tversky_w", type=float, default=0.5)

    parser.add_argument("--tversky_alpha_fp", type=float, default=0.3)
    parser.add_argument("--tversky_beta_fn", type=float, default=0.7)
    parser.add_argument("--tversky_gamma", type=float, default=0.75)
    parser.add_argument("--test_path", type=str, default="/data/zjy_work/ISIC2018/test/")
    parser.add_argument("--test_list", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=352)

    parser.add_argument("--train_save", type=str, default="/data/zjy_work/Work3_BEF_SBG/results/final_bgdnet_bef/")
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--augmentation", type=str2bool, default=True)

    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--decay_epoch", type=int, default=200)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.train_save, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.train_save, "train.log"),
        format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d %I:%M:%S %p",
    )

    print("[Train BGDNet-BEF Config]")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
        logging.info(f"{k}: {v}")

    model = BGDNet(num_classes=1).cuda(0)

    # Load checkpoint before DataParallel wrapping.
    # This keeps checkpoint key names clean and avoids module. mismatch.
    load_resume_checkpoint(model, args.resume)

    if args.freeze_bn:
        freeze_bn_layers(model)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            weight_decay=1e-4,
            momentum=0.9,
        )

    train_loader = get_weighted_loader(
        weighted_jsonl=args.weighted_train_json,
        batchsize=args.batchsize,
        trainsize=args.img_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        augmentation=args.augmentation,
        drop_last=True,
    )

    best = 0.0
    total_train_time = 0.0

    print("#" * 20, "Start Training BGDNet-BEF", "#" * 20)
    logging.info("Start Training BGDNet-BEF")

    for epoch in range(1, args.epoch + 1):
        adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)

        time_start = time.time()
        train_one_epoch(train_loader, model, optimizer, epoch, args)
        total_train_time += time.time() - time_start

        if model_has_nonfinite_params(model):
            raise RuntimeError(
                f"Non-finite model parameters detected after epoch {epoch}. "
                f"Refuse to save corrupted checkpoint."
            )

        save_last = os.path.join(args.train_save, "BGDNet-BEF-last.pth")
        torch.save(get_save_state_dict(model), save_last)

        dataset_dice, n_images = test(
            model=model,
            data_path=args.test_path,
            img_size=args.img_size,
            test_list=args.test_list,
        )

        if n_images > 0:
            print(f"[Epoch {epoch}] Test Dice: {dataset_dice:.6f}")
            logging.info(f"[Epoch {epoch}] Test Dice: {dataset_dice:.6f}")

            if dataset_dice > best:
                print(f"######## Dice improved {best:.6f} -> {dataset_dice:.6f}")
                logging.info(f"######## Dice improved {best:.6f} -> {dataset_dice:.6f}")

                best = dataset_dice
                save_best = os.path.join(args.train_save, "BGDNet-BEF-best.pth")
                if model_has_nonfinite_params(model):
                    raise RuntimeError(
                        f"Non-finite model parameters detected before saving best checkpoint at epoch {epoch}."
                    )

                torch.save(get_save_state_dict(model), save_best)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    avg_time = total_train_time / max(args.epoch, 1)
    print(f"[DONE] avg train time: {avg_time:.2f}s")
    logging.info(f"[DONE] avg train time: {avg_time:.2f}s")
    print(f"[DONE] best dice: {best:.6f}")
    logging.info(f"[DONE] best dice: {best:.6f}")


if __name__ == "__main__":
    main()
