import os
import argparse
from share import *

import torch
import pytorch_lightning as pl
from torch.utils.data import random_split

from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, compare_weights
from pytorch_lightning.loggers import TensorBoardLogger  # Echo
from pytorch_lightning.callbacks import ModelCheckpoint  # Echo

# 新增：用 DeepSpeedStrategy 传入 ignore_unused_parameters / offload 配置
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy

pl.seed_everything(42, workers=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()

    # I/O
    p.add_argument("--resume", type=str, default=os.environ.get("RESUME_PATH", "./models/control_sd15_init.pth"))
    p.add_argument("--prompt_json", type=str, default=os.environ.get("PROMPT_JSON", "./data/prompt_train.json"))
    p.add_argument("--log_root", type=str, default=os.environ.get("LOG_ROOT", "/data/zjy_work/BGDiff/Ours/logs"))
    p.add_argument("--exp_name", type=str, default=os.environ.get("EXP_NAME", "exp_mask2img"))

    # train hyperparams
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-5")))
    p.add_argument("--max_steps", type=int, default=int(os.environ.get("MAX_STEPS", "3000")))
    p.add_argument("--logger_freq", type=int, default=int(os.environ.get("LOGGER_FREQ", "400")))

    # data
    p.add_argument("--img_size", type=int, default=int(os.environ.get("IMG_SIZE", "384")))
    p.add_argument("--empty_prompt_prob", type=float, default=float(os.environ.get("EMPTY_PROMPT_PROB", "0.05")))
    p.add_argument("--num_workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))

    # gpu / batch
    p.add_argument("--num_gpus", type=int, default=int(os.environ.get("NUM_GPUS", "1")))
    p.add_argument("--per_gpu_batch", type=int, default=int(os.environ.get("PER_GPU_BATCH", "1")))
    p.add_argument("--accum", type=int, default=int(os.environ.get("ACCUM", "4")))
    p.add_argument("--precision", type=str, default=os.environ.get("PRECISION", "16-mixed"))

    # model options
    p.add_argument("--sd_locked", action="store_true", default=os.environ.get("SD_LOCKED", "0") == "1")
    p.add_argument("--only_mid_control", action="store_true", default=os.environ.get("ONLY_MID_CONTROL", "0") == "1")

    # 新增：是否启用 ZeRO-2 的 CPU offload（显存不够时开；不影响精度但会慢）
    p.add_argument("--offload_optimizer", action="store_true",
                   default=os.environ.get("OFFLOAD_OPTIMIZER", "0") == "1")

    return p.parse_args()


args = parse_args()


# First use cpu to load models. Lightning will move it to GPUs.
model = create_model("./models/cldm_v15.yaml").cpu()
model.load_state_dict(load_state_dict(args.resume, location="cpu"), strict=True)
# ============================================================
# 强制关闭 gradient checkpointing（DeepSpeed ZeRO2 + PyTorch2.4 下非常容易触发 used-params 统计崩溃）
# ============================================================
def _disable_grad_checkpointing(m):
    # 1) 一些模块用 use_checkpoint 标志
    for mod in m.modules():
        if hasattr(mod, "use_checkpoint"):
            try:
                mod.use_checkpoint = False
            except Exception:
                pass

    # 2) 有些实现用 gradient_checkpointing / checkpoint 标志
    for attr in ["gradient_checkpointing", "checkpoint", "use_gradient_checkpointing"]:
        if hasattr(m, attr):
            try:
                setattr(m, attr, False)
            except Exception:
                pass

_disable_grad_checkpointing(model)
# ============================================================

# ===================== 关键修改：不要全解冻 =====================
# ❌ 删除你原来的：
# for p in model.parameters():
#     p.requires_grad_(True)

#  显式冻结 VAE / CLIP（它们在 LDM 中通常也是 no_grad 编码，训练它们会导致 DS 统计出错且浪费显存）
if hasattr(model, "first_stage_model") and model.first_stage_model is not None:
    model.first_stage_model.eval()
    for p in model.first_stage_model.parameters():
        p.requires_grad_(False)

if hasattr(model, "cond_stage_model") and model.cond_stage_model is not None:
    model.cond_stage_model.eval()
    for p in model.cond_stage_model.parameters():
        p.requires_grad_(False)

# 训练哪些参数：通常 ControlNet 一定要训练
# 你的 ControlLDM 一般有 control_model；如果名字不同，按实际类名改一下
if hasattr(model, "control_model") and model.control_model is not None:
    for p in model.control_model.parameters():
        p.requires_grad_(True)

# UNet（diffusion 主干）是否训练由 sd_locked 控制
# sd_locked=True -> 锁住UNet，只训ControlNet
# sd_locked=False -> UNet+ControlNet一起训（显存更吃紧）
# if hasattr(model, "model") and model.model is not None:
#     for p in model.model.parameters():
#         p.requires_grad_(not args.sd_locked)



model.learning_rate = args.lr
model.sd_locked = args.sd_locked
model.only_mid_control = args.only_mid_control
# 3.18修改
# print task-aware loss config 
print("[TaskAwareLossConfig]")
for k in [
    "LOSS_BG_WEIGHT",
    "LOSS_FG_WEIGHT",
    "LOSS_BD_WEIGHT",
    "LAMBDA_BOUNDARY",
    "LAMBDA_MASK2IMAGE",
    "LAMBDA_MASK_REG",
    "BOUNDARY_DILATE_KERNEL",
]:
    print(f"  {k} = {os.environ.get(k, '<default>')}")
# 3.18修改
# Dataset / Loader

dataset = MyDataset(
    prompt_json=args.prompt_json,
    empty_prompt_prob=args.empty_prompt_prob,
    size=args.img_size
)

# ---- split train/val ----
val_ratio = 0.05
n_val = max(1, int(val_ratio * len(dataset)))
n_train = len(dataset) - n_val

train_set, val_set = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_set,
    num_workers=args.num_workers,
    batch_size=args.per_gpu_batch,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_set,
    num_workers=args.num_workers,
    batch_size=args.per_gpu_batch,
    shuffle=False,
    drop_last=False
)


# Loggers / callbacks
tb_logger = TensorBoardLogger(save_dir=args.log_root, name=args.exp_name)

# 1) ImageLogger: only save BEST + LATEST images on validation epoch end
img_logger = ImageLogger(
    monitor="val/loss_simple_ema",
    mode="min",
    max_images=4,
)

# 2) Checkpoints: keep only BEST + LAST
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


# 关键：DeepSpeed ZeRO-2 配置
# - ignore_unused_parameters=True 解决你遇到的 NoneType grad_fn / next_functions 报错
# - 可选 offload_optimizer 降显存（单卡 16G 常用）
ds_config = {
    "zero_optimization": {
        "stage": 2,
        "ignore_unused_parameters": True,   #  必须开
        "contiguous_gradients": True,
    },
    "fp16": {"enabled": True},
    # "zero_force_ds_cpu_optimizer": False,
}



if args.offload_optimizer:
    ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

use_deepspeed = (os.environ.get("USE_DEEPSPEED", "0") == "1") and (args.num_gpus > 1)

# DDP 策略：必须打开 find_unused_parameters=True，否则你这种动态图/分支会炸
ddp_strategy = DDPStrategy(
    find_unused_parameters=True,
    # 下面这两个一般不必配；如果你用的 PyTorch2.4 + Lightning2.x 仍然抖，再考虑：
    # gradient_as_bucket_view=True,
    # static_graph=False,
)

trainer = pl.Trainer(
    strategy=DeepSpeedStrategy(config=ds_config) if use_deepspeed else (ddp_strategy if args.num_gpus > 1 else "auto"),
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


trainer.fit(model, train_loader, val_loader)
