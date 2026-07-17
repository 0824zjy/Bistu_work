#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
LOG_ROOT="/data/zjy_work/BGDiff/Ours/logs"
EXP_NAME="exp_mask2img_stage1_cn_morph_adapt_only"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"
cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES=0

export PL_GLOBAL_SEED=0
export PYTHONHASHSEED=0

export RESUME_PATH="${PROJECT_DIR}/Ours/models/control_sd15_init.pth"
export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"

export NUM_GPUS=1
export PER_GPU_BATCH=1
export ACCUM=8
export LR=1e-5
export MAX_STEPS=20000
export LOGGER_FREQ=2000
export LOG_ROOT="${LOG_ROOT}"
export EXP_NAME="${EXP_NAME}"
export IMG_SIZE=384
export EMPTY_PROMPT_PROB=0.02
export PRECISION=16-mixed
export NUM_WORKERS=0

# ===== stage1 freeze policy =====
export TRAIN_UNET_DECODER=0
export SD_LOCKED=1
export TRAIN_CONTROLNET=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ===== new method =====
# 只开自适应一致性，关闭在线增强(二阶段在尝试)，避免额外 forward 导致 OOM
export ENABLE_ADAPTIVE_CONSISTENCY=1
export CONSISTENCY_W_MIN=0.20
export CONSISTENCY_W_MAX=1.00
export CONSISTENCY_T_GATE=300

export ENABLE_ONLINE_AUG=0
export LAMBDA_AUG=0.50
export AUG_PROB_MIN=0.15
export AUG_PROB_MAX=0.60
export AUG_EASY_BIAS=1.0
export BOUNDARY_DILATE_KERNEL=3

# 建议减少显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u "${PROJECT_DIR}/tutorial_train.py" \
  > "${LOG_ROOT}/${EXP_NAME}/train_stage1_cn_morph_adapt_only_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] stage1 morph-adapt-only started. logs in ${LOG_ROOT}/${EXP_NAME}"
