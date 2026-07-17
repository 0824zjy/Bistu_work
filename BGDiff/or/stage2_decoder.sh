#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
LOG_ROOT="/data/zjy_work/BGDiff/Ours/logs"
EXP_NAME="exp_mask2img_stage2_decoder"

STAGE1_EXP="exp_mask2img_stage1_cn"
STAGE1_CKPT="${LOG_ROOT}/${STAGE1_EXP}/checkpoints/last.ckpt"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"
cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES=2

export PL_GLOBAL_SEED=0
export PYTHONHASHSEED=0

export RESUME_PATH="${STAGE1_CKPT}"
export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"

export NUM_GPUS=1
export PER_GPU_BATCH=1
export ACCUM=8

#  更稳的 decoder 学习率（你可在 3e-6~5e-6 之间试）
export LR=3e-6
export MAX_STEPS=20000

export LOGGER_FREQ=2000
export LOG_ROOT="${LOG_ROOT}"
export EXP_NAME="${EXP_NAME}"
export IMG_SIZE=384

#  stage2 更建议收紧空prompt
export EMPTY_PROMPT_PROB=0.0

export PRECISION=16-mixed
export NUM_WORKERS=0

export TRAIN_UNET_DECODER=1
export SD_LOCKED=0
export TRAIN_CONTROLNET=0 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

nohup python -u "${PROJECT_DIR}/tutorial_train.py" \
  > "${LOG_ROOT}/${EXP_NAME}/train_stage2_decoder_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] stage2 started (CN+decoder). resume=${STAGE1_CKPT}"
