import os
import argparse
from share import *

import torch
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from tutorial_dataset import MyDataset
from ablation_cldm.logger import ImageLogger
from ablation_cldm.model import create_model, load_state_dict

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy


pl.seed_everything(42, workers=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()

    # I/O
    p.add_argument(
        "--resume",
        type=str,
        default=os.environ.get("RESUME_PATH", "./models/control_sd15_init.pth"),
    )
    p.add_argument(
        "--prompt_json",
        type=str,
        default=os.environ.get("PROMPT_JSON", "./data/prompt_train.json"),
    )
    p.add_argument(
        "--log_root",
        type=str,
        default=os.environ.get("LOG_ROOT", "/data/zjy_work/BGDiff/Ours/logs"),
    )
    p.add_argument(
        "--exp_name",
        type=str,
        default=os.environ.get("EXP_NAME", "exp_mask2img"),
    )

    # Train hyperparams
    p.add_argument(
        "--lr",
        type=float,
        default=float(os.environ.get("LR", "1e-5")),
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=int(os.environ.get("MAX_STEPS", "3000")),
    )
    p.add_argument(
        "--logger_freq",
        type=int,
        default=int(os.environ.get("LOGGER_FREQ", "400")),
    )

    # Data
    p.add_argument(
        "--img_size",
        type=int,
        default=int(os.environ.get("IMG_SIZE", "384")),
    )
    p.add_argument(
        "--empty_prompt_prob",
        type=float,
        default=float(os.environ.get("EMPTY_PROMPT_PROB", "0.05")),
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("NUM_WORKERS", "4")),
    )

    # GPU / batch
    p.add_argument(
        "--num_gpus",
        type=int,
        default=int(os.environ.get("NUM_GPUS", "1")),
    )
    p.add_argument(
        "--per_gpu_batch",
        type=int,
        default=int(os.environ.get("PER_GPU_BATCH", "1")),
    )
    p.add_argument(
        "--accum",
        type=int,
        default=int(os.environ.get("ACCUM", "4")),
    )
    p.add_argument(
        "--precision",
        type=str,
        default=os.environ.get("PRECISION", "16-mixed"),
    )

    # Model options
    p.add_argument(
        "--sd_locked",
        action="store_true",
        default=os.environ.get("SD_LOCKED", "0") == "1",
    )
    p.add_argument(
        "--only_mid_control",
        action="store_true",
        default=os.environ.get("ONLY_MID_CONTROL", "0") == "1",
    )

    # DeepSpeed option
    p.add_argument(
        "--offload_optimizer",
        action="store_true",
        default=os.environ.get("OFFLOAD_OPTIMIZER", "0") == "1",
    )

    return p.parse_args()


args = parse_args()


def print_config_block(title, keys):
    print(f"[{title}]")
    for k in keys:
        print(f"  {k} = {os.environ.get(k, '<default>')}")


# ============================================================
# Create model
# ============================================================

model = create_model(
    os.environ.get(
        "CONFIG_PATH",
        "./models/cldm_v15.yaml"
    )
).cpu()


# ============================================================
# Load checkpoint with strict=False
# ============================================================
# This is required because SBP-PG introduces optional new parameters:
#   - image_hint_block
#   - soft boundary prior related modules
# and also must tolerate old checkpoint structure.
# ============================================================

state_dict = load_state_dict(args.resume, location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("[Checkpoint Load]")
print(f"  resume: {args.resume}")
print(f"  missing keys: {len(missing)}")
print(f"  unexpected keys: {len(unexpected)}")

if len(missing) > 0:
    print("  first missing keys:")
    for k in missing[:30]:
        print(f"    {k}")

if len(unexpected) > 0:
    print("  first unexpected keys:")
    for k in unexpected[:30]:
        print(f"    {k}")


# ============================================================
# Disable gradient checkpointing
# ============================================================

def _disable_grad_checkpointing(m):
    for mod in m.modules():
        if hasattr(mod, "use_checkpoint"):
            try:
                mod.use_checkpoint = False
            except Exception:
                pass

    for attr in [
        "gradient_checkpointing",
        "checkpoint",
        "use_gradient_checkpointing",
    ]:
        if hasattr(m, attr):
            try:
                setattr(m, attr, False)
            except Exception:
                pass


_disable_grad_checkpointing(model)


# ============================================================
# Freeze VAE / CLIP
# ============================================================

if hasattr(model, "first_stage_model") and model.first_stage_model is not None:
    model.first_stage_model.eval()
    for p in model.first_stage_model.parameters():
        p.requires_grad_(False)

if hasattr(model, "cond_stage_model") and model.cond_stage_model is not None:
    model.cond_stage_model.eval()
    for p in model.cond_stage_model.parameters():
        p.requires_grad_(False)

# ============================================================
# Explicit trainable parameter policy
# ============================================================
# Important:
#   configure_optimizers controls which params are optimized,
#   but requires_grad controls whether gradients are computed.
#
# Stage1:
#   TRAIN_CONTROLNET=1
#   TRAIN_UNET_DECODER=0
#   SD_LOCKED=1
#   -> only ControlNet trainable
#
# Stage2:
#   TRAIN_CONTROLNET=0
#   TRAIN_UNET_DECODER=1
#   SD_LOCKED=0
#   -> decoder / output blocks trainable
# ============================================================

train_controlnet = os.environ.get("TRAIN_CONTROLNET", "1") == "1"
train_unet_decoder = os.environ.get("TRAIN_UNET_DECODER", "0") == "1"

# First freeze all main diffusion and control params.
if hasattr(model, "model") and model.model is not None:
    for p in model.model.parameters():
        p.requires_grad_(False)

if hasattr(model, "control_model") and model.control_model is not None:
    for p in model.control_model.parameters():
        p.requires_grad_(False)

# Unfreeze ControlNet if needed.
if train_controlnet and hasattr(model, "control_model") and model.control_model is not None:
    for p in model.control_model.parameters():
        p.requires_grad_(True)

# Unfreeze UNet decoder/output blocks if needed.
if (
    train_unet_decoder
    or not args.sd_locked
):
    diffusion_model = model.model.diffusion_model

    if hasattr(diffusion_model, "output_blocks"):
        for p in diffusion_model.output_blocks.parameters():
            p.requires_grad_(True)

    if hasattr(diffusion_model, "out"):
        for p in diffusion_model.out.parameters():
            p.requires_grad_(True)

    if hasattr(diffusion_model, "boundary_modulators"):
        for p in diffusion_model.boundary_modulators.parameters():
            p.requires_grad_(True)

    if hasattr(diffusion_model, "boundary_modulator"):
        for p in diffusion_model.boundary_modulator.parameters():
            p.requires_grad_(True)

# Print trainable parameter count.
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print("[Trainable Parameter Policy]")
print(f"  TRAIN_CONTROLNET   = {train_controlnet}")
print(f"  TRAIN_UNET_DECODER = {train_unet_decoder}")
print(f"  SD_LOCKED          = {args.sd_locked}")
print(f"  trainable params   = {trainable_params / 1e6:.2f} M")
print(f"  total params       = {total_params / 1e6:.2f} M")

# ============================================================
# Set training flags
# ============================================================

model.learning_rate = args.lr
model.sd_locked = args.sd_locked
model.only_mid_control = args.only_mid_control


# ============================================================
# Print configs
# ============================================================

print_config_block(
    "TrainConfig",
    [
        "RESUME_PATH",
        "PROMPT_JSON",
        "LOG_ROOT",
        "EXP_NAME",
        "NUM_GPUS",
        "PER_GPU_BATCH",
        "ACCUM",
        "LR",
        "MAX_STEPS",
        "IMG_SIZE",
        "EMPTY_PROMPT_PROB",
        "PRECISION",
        "NUM_WORKERS",
        "SD_LOCKED",
        "ONLY_MID_CONTROL",
        "TRAIN_CONTROLNET",
        "TRAIN_UNET_DECODER",
    ],
)

print_config_block(
    "SBP-PG Config",
    [
        "ENABLE_SOFT_BOUNDARY_PRIOR",
        "BOUNDARY_PRIOR_TAU",
        "BOUNDARY_PRIOR_RADIUS",
        "BOUNDARY_DILATE_KERNEL",
        "ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE",
        "BOUNDARY_GUIDANCE_MAX",
        "BOUNDARY_GUIDANCE_START_RATIO",
        "BOUNDARY_GUIDANCE_TEMPERATURE",
        "BOUNDARY_BRANCH_SCALE",
        "ENABLE_BOUNDARY_MODULATION",
        "BOUNDARY_MOD_SCALE",
        "BOUNDARY_MOD_START_RATIO",
        "ENABLE_TOLERANCE_BAND_LOSS",
        "LAMBDA_BAND",
        "BOUNDARY_BAND_T_GATE",
        "BOUNDARY_CONDITION_MODE",
        "BOUNDARY_ALPHA_INIT",
    ],
)

print_config_block(
    "Existing Loss / Aug Config",
    [
        "LOSS_BG_WEIGHT",
        "LOSS_FG_WEIGHT",
        "LOSS_BD_WEIGHT",
        "LAMBDA_BOUNDARY",
        "LAMBDA_MASK2IMAGE",
        "LAMBDA_MASK_REG",
        "ENABLE_ADAPTIVE_CONSISTENCY",
        "CONSISTENCY_W_MIN",
        "CONSISTENCY_W_MAX",
        "CONSISTENCY_T_GATE",
        "ENABLE_ONLINE_AUG",
        "LAMBDA_AUG",
        "AUG_PROB_MIN",
        "AUG_PROB_MAX",
        "AUG_EASY_BIAS",
    ],
)


# ============================================================
# Dataset / Loader
# ============================================================

dataset = MyDataset(
    prompt_json=args.prompt_json,
    empty_prompt_prob=args.empty_prompt_prob,
    size=args.img_size,
)

val_ratio = 0.05
n_val = max(1, int(val_ratio * len(dataset)))
n_train = len(dataset) - n_val

train_set, val_set = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_set,
    num_workers=args.num_workers,
    batch_size=args.per_gpu_batch,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_set,
    num_workers=args.num_workers,
    batch_size=args.per_gpu_batch,
    shuffle=False,
    drop_last=False,
)


# ============================================================
# Logger / callbacks
# ============================================================

tb_logger = TensorBoardLogger(
    save_dir=args.log_root,
    name=args.exp_name,
)

img_logger = ImageLogger(
    monitor="val/loss_simple_ema",
    mode="min",
    max_images=4,
)

ckpt_best = ModelCheckpoint(
    dirpath=f"{args.log_root}/{args.exp_name}/checkpoints",
    filename="best",
    monitor="val/loss_simple_ema",
    mode="min",
    save_top_k=1,
    save_last=False,
    auto_insert_metric_name=False,
)

ckpt_last = ModelCheckpoint(
    dirpath=f"{args.log_root}/{args.exp_name}/checkpoints",
    filename="last",
    save_last=True,
    save_top_k=0,
    auto_insert_metric_name=False,
)


# ============================================================
# Strategy
# ============================================================

ds_config = {
    "zero_optimization": {
        "stage": 2,
        "ignore_unused_parameters": True,
        "contiguous_gradients": True,
    },
    "fp16": {
        "enabled": True,
    },
}

if args.offload_optimizer:
    ds_config["zero_optimization"]["offload_optimizer"] = {
        "device": "cpu",
        "pin_memory": True,
    }

use_deepspeed = (
    os.environ.get("USE_DEEPSPEED", "0") == "1"
    and args.num_gpus > 1
)

ddp_strategy = DDPStrategy(
    find_unused_parameters=True,
)

trainer = pl.Trainer(
    strategy=(
        DeepSpeedStrategy(config=ds_config)
        if use_deepspeed
        else (ddp_strategy if args.num_gpus > 1 else "auto")
    ),
    accelerator="gpu",
    devices=args.num_gpus,
    precision=args.precision,
    logger=tb_logger,
    callbacks=[img_logger, ckpt_best, ckpt_last],
    deterministic=True,
    max_steps=args.max_steps,
    accumulate_grad_batches=args.accum,
    default_root_dir=f"{args.log_root}/{args.exp_name}",
)


trainer.fit(
    model,
    train_loader,
    val_loader,
)
