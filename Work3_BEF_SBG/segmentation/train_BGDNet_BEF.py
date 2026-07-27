import os
import sys
import time
import math
import copy
import argparse
import logging
import warnings
from datetime import datetime
from collections import Counter

sys.path.insert(0, "/data/zjy_work/BGDNet")
sys.path.insert(0, "/data/zjy_work/Work3_BEF_SBG/segmentation")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if os.environ.get("BGDNET_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    print("[WARN] cuDNN disabled by BGDNET_DISABLE_CUDNN=1")
else:
    torch.backends.cudnn.enabled = True

torch.backends.cudnn.benchmark = False

from models.BGDNet import BGDNet
from utils.dataloader_BGDiff import test_dataset
from utils.utils import clip_gradient, AvgMeter
from dataloader_BEF import get_weighted_loader

warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ["1", "true", "yes", "y", "on"]


def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_bn_layers(model):
    bn_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    for module in model.modules():
        if isinstance(module, bn_types):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def de_parallel(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def strip_module_prefix(state_dict):
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ["state_dict", "model_state_dict", "model"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def load_resume_checkpoint(model, resume_path: str):
    if not resume_path:
        return
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    state_dict = strip_module_prefix(
        extract_state_dict(torch.load(resume_path, map_location="cpu"))
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    allowed_missing_prefixes = (
        "distance_feature.",
        "distance_head.",
        "distance_to_mask.",
        "distance_to_boundary.",
    )
    bad_missing = [
        key for key in missing
        if not key.startswith(allowed_missing_prefixes)
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Checkpoint mismatch. "
            f"bad_missing={bad_missing}, unexpected={unexpected}"
        )
    print(f"[Resume] loaded: {resume_path}")
    if missing:
        print(f"[Resume] newly initialized distance keys: {len(missing)}")


def get_save_state_dict(model):
    return de_parallel(model).state_dict()


class ModelEMA:
    """Exponential moving average used for validation and final inference."""

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.ema = copy.deepcopy(de_parallel(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = de_parallel(model).state_dict()
        ema_state = self.ema.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(
                    model_value.to(dtype=ema_value.dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                ema_value.copy_(model_value)


def sanitize_weights(weights: torch.Tensor, max_weight: float = 1.5):
    weights = torch.nan_to_num(
        weights.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return weights.clamp(0.0, float(max_weight))


def tensor_all_finite(x):
    return bool(torch.isfinite(x).all().item())


def model_has_nonfinite_params(model):
    for name, p in model.named_parameters():
        if p is not None and not torch.isfinite(p).all():
            print(f"[ERROR] non-finite model parameter: {name}")
            return True
    return False


def optimizer_has_nonfinite_grads(model):
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            print(f"[WARN] non-finite gradient: {name}")
            return True
    return False


def safe_meter_show(meter):
    if not meter.losses:
        return 0.0
    value = meter.show()
    value = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    return value if np.isfinite(value) else 0.0


def unpack_model_outputs(output):
    if not isinstance(output, (tuple, list)):
        return output, None, None
    if len(output) == 2:
        return output[0], output[1], None
    if len(output) >= 3:
        return output[0], output[1], output[2]
    raise RuntimeError("Empty model output.")


def build_boundary_band(mask: torch.Tensor, kernel_size: int):
    mask = (mask >= 0.5).float()
    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    eroded = -F.max_pool2d(
        -mask,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return (dilated - eroded).clamp(0.0, 1.0)


def weighted_spatial_mean(value_map, pixel_weight, eps=1e-8):
    numerator = (value_map * pixel_weight).sum(dim=(1, 2, 3))
    denominator = pixel_weight.sum(dim=(1, 2, 3)) + eps
    return numerator / denominator


def weighted_dice_loss_per_sample(logits, targets, pixel_weight, eps=1.0):
    probs = torch.sigmoid(logits)
    inter = (probs * targets * pixel_weight).sum(dim=(1, 2, 3))
    denom = (
        (probs * pixel_weight).sum(dim=(1, 2, 3))
        + (targets * pixel_weight).sum(dim=(1, 2, 3))
    )
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def weighted_focal_tversky_per_sample(
    logits,
    targets,
    pixel_weight,
    alpha_fp=0.3,
    beta_fn=0.7,
    gamma=0.75,
    eps=1e-6,
):
    probs = torch.sigmoid(logits)
    tp = (probs * targets * pixel_weight).sum(dim=(1, 2, 3))
    fp = (probs * (1.0 - targets) * pixel_weight).sum(dim=(1, 2, 3))
    fn = ((1.0 - probs) * targets * pixel_weight).sum(dim=(1, 2, 3))
    tversky = (tp + eps) / (
        tp + float(alpha_fp) * fp + float(beta_fn) * fn + eps
    )
    return torch.pow(1.0 - tversky, float(gamma))


def linear_ramp(epoch, start_epoch, ramp_epochs):
    if epoch < int(start_epoch):
        return 0.0
    if int(ramp_epochs) <= 0:
        return 1.0
    return float(np.clip(
        (epoch - int(start_epoch) + 1) / float(ramp_epochs),
        0.0,
        1.0,
    ))


def source_masks(sources, device):
    real = torch.tensor(
        [1.0 if str(s) == "real" else 0.0 for s in sources],
        device=device,
        dtype=torch.float32,
    )
    pseudo = torch.tensor(
        [1.0 if str(s) == "pseudo_real" else 0.0 for s in sources],
        device=device,
        dtype=torch.float32,
    )
    synthetic = (1.0 - real - pseudo).clamp(0.0, 1.0)
    return real, synthetic, pseudo


def weighted_multitask_loss(
    pred_m,
    pred_b,
    pred_d,
    gts,
    bnds,
    distance_targets,
    reliability,
    hardness,
    sample_weights,
    sources,
    epoch,
    args,
):
    target_size = gts.shape[-2:]
    pred_m = F.interpolate(pred_m, size=target_size, mode="bilinear", align_corners=False)
    pred_b = F.interpolate(pred_b, size=target_size, mode="bilinear", align_corners=False)
    if pred_d is not None:
        pred_d = F.interpolate(pred_d, size=target_size, mode="bilinear", align_corners=False)

    reliability = reliability.clamp(0.0, 1.0)
    hardness = hardness.clamp(0.0, 1.0)

    real_mask, synthetic_mask, pseudo_mask = source_masks(sources, gts.device)
    synthetic_ramp = linear_ramp(
        epoch,
        args.synthetic_start_epoch,
        args.synthetic_warmup_epochs,
    )
    pseudo_ramp = linear_ramp(
        epoch,
        args.pseudo_start_epoch,
        args.pseudo_warmup_epochs,
    )
    hard_ramp = linear_ramp(
        epoch,
        args.hard_start_epoch,
        args.hard_ramp_epochs,
    )
    hard_gain = float(args.hard_gain_max) * hard_ramp

    # Real pixels are fully supervised. Synthetic pixels are controlled by the
    # ensemble reliability and later receive extra focus only where difficulty
    # remains reliable.
    pixel_weight = reliability * (1.0 + hard_gain * hardness)
    pixel_weight = pixel_weight.clamp(
        float(args.min_pixel_weight),
        float(args.max_pixel_weight),
    )

    # Segmentation: pixel-weighted BCE + Dice + Focal Tversky.
    seg_bce_map = F.binary_cross_entropy_with_logits(
        pred_m,
        gts,
        reduction="none",
    )
    seg_bce_i = weighted_spatial_mean(seg_bce_map, pixel_weight)
    seg_dice_i = weighted_dice_loss_per_sample(pred_m, gts, pixel_weight)
    seg_tversky_i = weighted_focal_tversky_per_sample(
        pred_m,
        gts,
        pixel_weight,
        alpha_fp=args.tversky_alpha_fp,
        beta_fn=args.tversky_beta_fn,
        gamma=args.tversky_gamma,
    )
    seg_loss_i = (
        args.seg_bce_w * seg_bce_i
        + args.seg_dice_w * seg_dice_i
        + args.seg_tversky_w * seg_tversky_i
    )

    # Boundary: one prediction head supervised by 3/5/7-pixel bands.
    boundary_targets = [
        bnds,
        build_boundary_band(gts, 5),
        build_boundary_band(gts, 7),
    ]
    scale_weights = [1.0, 0.5, 0.25]
    bnd_loss_i = torch.zeros_like(seg_loss_i)
    scale_sum = 0.0
    for target, scale_weight in zip(boundary_targets, scale_weights):
        pos = target.sum(dim=(1, 2, 3))
        total = float(target[0].numel())
        neg = total - pos
        pos_weight = (neg / (pos + 1e-8)).clamp(1.0, 20.0).view(-1, 1, 1, 1)
        bce_map = F.binary_cross_entropy_with_logits(
            pred_b,
            target,
            reduction="none",
        )
        class_weight = 1.0 + (pos_weight - 1.0) * target
        bce_i = weighted_spatial_mean(
            bce_map * class_weight,
            pixel_weight,
        )
        dice_i = weighted_dice_loss_per_sample(
            pred_b,
            target,
            pixel_weight,
        )
        bnd_loss_i += float(scale_weight) * (0.5 * bce_i + 0.5 * dice_i)
        scale_sum += float(scale_weight)
    bnd_loss_i = bnd_loss_i / scale_sum

    # Signed-distance task and mask-distance consistency.
    if pred_d is not None:
        pred_distance = torch.tanh(pred_d)
        distance_focus = (
            float(args.distance_far_weight)
            + (1.0 - float(args.distance_far_weight))
            * torch.exp(
                -distance_targets.abs() / max(float(args.distance_focus_tau), 1e-6)
            )
        )
        dist_pixel_weight = pixel_weight * distance_focus
        distance_map = F.smooth_l1_loss(
            pred_distance,
            distance_targets,
            reduction="none",
            beta=float(args.distance_smooth_l1_beta),
        )
        distance_loss_i = weighted_spatial_mean(distance_map, dist_pixel_weight)

        mask_prob = torch.sigmoid(pred_m)
        distance_prob = torch.sigmoid(pred_d * float(args.distance_temperature))
        consistency_map = (mask_prob - distance_prob).pow(2)
        consistency_loss_i = weighted_spatial_mean(consistency_map, pixel_weight)
    else:
        distance_loss_i = torch.zeros_like(seg_loss_i)
        consistency_loss_i = torch.zeros_like(seg_loss_i)

    loss_i = (
        float(args.alpha) * seg_loss_i
        + float(args.beta) * bnd_loss_i
        + float(args.distance_w) * distance_loss_i
        + float(args.consistency_w) * consistency_loss_i
    )

    sample_weights = sanitize_weights(
        sample_weights,
        max_weight=args.max_weight,
    ).view(-1)
    curriculum_factor = (
        real_mask
        + synthetic_mask * synthetic_ramp
        + pseudo_mask * pseudo_ramp
    )
    effective_weights = sample_weights * curriculum_factor

    valid = (
        torch.isfinite(loss_i)
        & torch.isfinite(effective_weights)
        & (effective_weights > 1e-8)
    )
    valid_n = int(valid.sum().item())
    if valid_n == 0:
        return None

    final_loss = (
        effective_weights[valid] * loss_i[valid]
    ).sum() / (effective_weights[valid].sum() + 1e-8)

    return {
        "loss": final_loss,
        "seg": seg_loss_i[valid].detach().mean(),
        "bnd": bnd_loss_i[valid].detach().mean(),
        "dist": distance_loss_i[valid].detach().mean(),
        "cons": consistency_loss_i[valid].detach().mean(),
        "raw": loss_i[valid].detach().mean(),
        "valid_n": valid_n,
        "synthetic_ramp": synthetic_ramp,
        "pseudo_ramp": pseudo_ramp,
        "hard_gain": hard_gain,
    }


@torch.no_grad()
def evaluate_dice(model, data_path, img_size, val_list=None):
    if not data_path:
        return 0.0, 0
    image_root = os.path.join(data_path, "Images")
    gt_root = os.path.join(data_path, "Masks")
    model.eval()
    loader = test_dataset(
        image_root=image_root,
        gt_root=gt_root,
        testsize=img_size,
        list_txt=val_list,
        mode="isic",
    )
    if loader.size == 0:
        return 0.0, 0

    dice_sum = torch.zeros((), device="cuda", dtype=torch.float32)
    for _ in range(loader.size):
        image, gt, _ = loader.load_data()
        gt_t = torch.from_numpy(np.asarray(gt, np.float32)).cuda()
        gt_t = gt_t / (gt_t.max() + 1e-8)
        image = image.cuda(non_blocking=True)
        pred_m, _, _ = unpack_model_outputs(model(image))
        pred_m = F.interpolate(
            pred_m,
            size=gt_t.shape,
            mode="bilinear",
            align_corners=False,
        )
        pred = (torch.sigmoid(pred_m).squeeze() >= 0.5).float()
        target = (gt_t >= 0.5).float()
        inter = (pred * target).sum()
        dice_sum += (2.0 * inter + 1.0) / (pred.sum() + target.sum() + 1.0)
    return float((dice_sum / loader.size).item()), loader.size


def set_epoch_lr(optimizer, epoch, args):
    if args.scheduler == "legacy":
        factor = args.decay_rate ** (epoch // max(args.decay_epoch, 1))
        lr = args.lr * factor
    else:
        if epoch <= args.warmup_epochs:
            lr = args.lr * epoch / max(args.warmup_epochs, 1)
        else:
            progress = (
                epoch - args.warmup_epochs
            ) / max(args.epoch - args.warmup_epochs, 1)
            progress = min(max(progress, 0.0), 1.0)
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_one_epoch(train_loader, model, ema, optimizer, epoch, args):
    model.train()
    if args.freeze_bn:
        freeze_bn_layers(model)

    meters = {name: AvgMeter() for name in ["loss", "seg", "bnd", "dist", "cons", "raw"]}
    skipped = 0
    source_counter = Counter()
    max_memory_usage = 0
    start_time = time.time()

    for step, pack in enumerate(train_loader, start=1):
        (
            images,
            gts,
            bnds,
            distance_targets,
            reliability,
            hardness,
            sample_weights,
            sources,
        ) = pack

        source_counter.update(list(sources))
        tensors = [
            images,
            gts,
            bnds,
            distance_targets,
            reliability,
            hardness,
            sample_weights,
        ]
        tensors = [x.cuda(non_blocking=True) for x in tensors]
        (
            images,
            gts,
            bnds,
            distance_targets,
            reliability,
            hardness,
            sample_weights,
        ) = tensors

        if not all(tensor_all_finite(x) for x in tensors):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        pred_m, pred_b, pred_d = unpack_model_outputs(model(images))
        if pred_b is None or pred_d is None:
            raise RuntimeError(
                "CHFS final training requires BGDNET_ENABLE_DISTANCE_HEAD=1 "
                "and a three-output BGDNet."
            )
        if not all(tensor_all_finite(x) for x in [pred_m, pred_b, pred_d]):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        result = weighted_multitask_loss(
            pred_m=pred_m,
            pred_b=pred_b,
            pred_d=pred_d,
            gts=gts,
            bnds=bnds,
            distance_targets=distance_targets,
            reliability=reliability,
            hardness=hardness,
            sample_weights=sample_weights,
            sources=sources,
            epoch=epoch,
            args=args,
        )
        if result is None or not torch.isfinite(result["loss"]):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        result["loss"].backward()
        if optimizer_has_nonfinite_grads(model):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        clip_gradient(optimizer, args.clip)
        optimizer.step()
        ema.update(model)

        if model_has_nonfinite_params(model):
            raise RuntimeError(f"Non-finite model parameters at epoch={epoch}, step={step}")

        for key in meters:
            meters[key].update(result[key].detach(), images.size(0))

        if torch.cuda.is_available():
            max_memory_usage = max(
                max_memory_usage,
                torch.cuda.max_memory_allocated(),
            )

        if step % args.print_freq == 0 or step == len(train_loader):
            msg = (
                f"{datetime.now()} Epoch [{epoch:03d}/{args.epoch:03d}] "
                f"Step [{step:04d}/{len(train_loader):04d}] "
                f"loss={safe_meter_show(meters['loss']):.4f} "
                f"seg={safe_meter_show(meters['seg']):.4f} "
                f"bnd={safe_meter_show(meters['bnd']):.4f} "
                f"dist={safe_meter_show(meters['dist']):.4f} "
                f"cons={safe_meter_show(meters['cons']):.4f} "
                f"syn_ramp={result['synthetic_ramp']:.3f} "
                f"pseudo_ramp={result['pseudo_ramp']:.3f} "
                f"hard_gain={result['hard_gain']:.3f} "
                f"valid_n={result['valid_n']} skipped={skipped} "
                f"sources={dict(source_counter)} "
                f"max_mem={max_memory_usage / (1024 ** 2):.1f}MB"
            )
            print(msg)
            logging.info(msg)

    elapsed = time.time() - start_time
    print(f"[Epoch {epoch}] time={elapsed:.2f}s skipped={skipped}")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighted_train_json", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze_bn", type=str2bool, default=False)

    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--val_list", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=352)
    parser.add_argument("--train_save", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--augmentation", type=str2bool, default=True)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--scheduler", choices=["cosine", "legacy"], default="cosine")
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--decay_epoch", type=int, default=200)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--distance_w", type=float, default=0.30)
    parser.add_argument("--consistency_w", type=float, default=0.10)
    parser.add_argument("--max_weight", type=float, default=1.5)

    parser.add_argument("--seg_bce_w", type=float, default=0.5)
    parser.add_argument("--seg_dice_w", type=float, default=1.0)
    parser.add_argument("--seg_tversky_w", type=float, default=0.5)
    parser.add_argument("--tversky_alpha_fp", type=float, default=0.3)
    parser.add_argument("--tversky_beta_fn", type=float, default=0.7)
    parser.add_argument("--tversky_gamma", type=float, default=0.75)

    parser.add_argument("--distance_max_px", type=float, default=20.0)
    parser.add_argument("--distance_focus_tau", type=float, default=0.25)
    parser.add_argument("--distance_far_weight", type=float, default=0.20)
    parser.add_argument("--distance_smooth_l1_beta", type=float, default=0.10)
    parser.add_argument("--distance_temperature", type=float, default=4.0)

    parser.add_argument("--min_pixel_weight", type=float, default=0.05)
    parser.add_argument("--max_pixel_weight", type=float, default=2.0)
    parser.add_argument("--synthetic_start_epoch", type=int, default=1)
    parser.add_argument("--synthetic_warmup_epochs", type=int, default=30)
    parser.add_argument("--pseudo_start_epoch", type=int, default=10)
    parser.add_argument("--pseudo_warmup_epochs", type=int, default=40)
    parser.add_argument("--hard_start_epoch", type=int, default=40)
    parser.add_argument("--hard_ramp_epochs", type=int, default=60)
    parser.add_argument("--hard_gain_max", type=float, default=1.0)
    args = parser.parse_args()

    if os.environ.get("BGDNET_ENABLE_DISTANCE_HEAD", "0") != "1":
        raise RuntimeError(
            "Set BGDNET_ENABLE_DISTANCE_HEAD=1 before final CHFS training."
        )

    set_seed(args.seed)
    os.makedirs(args.train_save, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.train_save, "train.log"),
        format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
        level=logging.INFO,
        filemode="a",
    )

    print("[Train BGDNet-CHFS Config]")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
        logging.info(f"{key}: {value}")

    model = BGDNet(num_classes=1).cuda(0)
    load_resume_checkpoint(model, args.resume)
    if args.freeze_bn:
        freeze_bn_layers(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

    ema = ModelEMA(model, decay=args.ema_decay)
    train_loader = get_weighted_loader(
        weighted_jsonl=args.weighted_train_json,
        batchsize=args.batchsize,
        trainsize=args.img_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        augmentation=args.augmentation,
        drop_last=True,
        distance_max_px=args.distance_max_px,
    )

    best = -1.0
    total_train_time = 0.0
    for epoch in range(1, args.epoch + 1):
        lr = set_epoch_lr(optimizer, epoch, args)
        print(f"[Epoch {epoch}] lr={lr:.8g}")
        total_train_time += train_one_epoch(
            train_loader,
            model,
            ema,
            optimizer,
            epoch,
            args,
        )

        raw_last = os.path.join(args.train_save, "BGDNet-CHFS-raw-last.pth")
        ema_last = os.path.join(args.train_save, "BGDNet-CHFS-ema-last.pth")
        torch.save(get_save_state_dict(model), raw_last)
        torch.save(ema.ema.state_dict(), ema_last)

        val_dice, val_n = evaluate_dice(
            ema.ema,
            data_path=args.val_path,
            img_size=args.img_size,
            val_list=args.val_list,
        )
        if val_n > 0:
            print(f"[Epoch {epoch}] EMA validation Dice={val_dice:.6f}")
            logging.info(f"[Epoch {epoch}] EMA validation Dice={val_dice:.6f}")
            if val_dice > best:
                best = val_dice
                torch.save(
                    ema.ema.state_dict(),
                    os.path.join(args.train_save, "BGDNet-CHFS-ema-best.pth"),
                )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # A deterministic final checkpoint exists even when no validation set was
    # supplied. It is never selected on the official test set.
    torch.save(
        ema.ema.state_dict(),
        os.path.join(args.train_save, "BGDNet-CHFS-final.pth"),
    )
    print(f"[DONE] avg train time={total_train_time / max(args.epoch, 1):.2f}s")
    print(f"[DONE] best validation Dice={best:.6f}" if best >= 0 else "[DONE] no validation set; use BGDNet-CHFS-final.pth")


if __name__ == "__main__":
    main()
