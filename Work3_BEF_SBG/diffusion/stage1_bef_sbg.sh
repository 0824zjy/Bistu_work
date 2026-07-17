#!/usr/bin/env bash
set -e
export PYTHONPATH="/data/zjy_work/env_patch:${PYTHONPATH:-}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

PROJECT_DIR="/data/zjy_work/BGDiff"
WORK_ROOT="/data/zjy_work/Work3_BEF_SBG"

LOG_ROOT="${WORK_ROOT}/logs/diffusion"
PID_DIR="${WORK_ROOT}/logs/pids"

EXP_NAME="stage1_bef_sbg"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"
mkdir -p "${PID_DIR}"

cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PROJECT_DIR="${PROJECT_DIR}"
export PL_GLOBAL_SEED=0
export PYTHONHASHSEED=0

# 可外部覆盖：
# RESUME_PATH=/path/to/xxx.pth bash stage1_bef_sbg.sh
export RESUME_PATH="${RESUME_PATH:-${PROJECT_DIR}/Ours/models/control_sd15_region_boundary_init.pth}"
export PROMPT_JSON="${WORK_ROOT}/feedback/prompt_bef_train.json"

export LOG_ROOT="${LOG_ROOT}"
export EXP_NAME="${EXP_NAME}"

export NUM_GPUS=1
export PER_GPU_BATCH=1
export ACCUM=8
export LR=1e-5
export MAX_STEPS=20000
export LOGGER_FREQ=2000
export IMG_SIZE=384
export EMPTY_PROMPT_PROB=0.02
export PRECISION=16-mixed
export NUM_WORKERS=0

# Stage1: only train ControlNet.
export SD_LOCKED=1
export ONLY_MID_CONTROL=0
export TRAIN_CONTROLNET=1
export TRAIN_UNET_DECODER=0

# BEF-SBG: use external adaptive boundary prior directly.
export BOUNDARY_PRIOR_MODE=external

# Fallback compatibility only.
export ENABLE_SOFT_BOUNDARY_PRIOR=1
export BOUNDARY_PRIOR_TAU=4.0
export BOUNDARY_PRIOR_RADIUS=12
export BOUNDARY_DILATE_KERNEL=3

# Progressive boundary guidance.
export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE=1
export BOUNDARY_GUIDANCE_MAX=0.12
export BOUNDARY_GUIDANCE_START_RATIO=0.35
export BOUNDARY_GUIDANCE_TEMPERATURE=0.05
export BOUNDARY_BRANCH_SCALE=0.10

# Stage1 disables decoder modulation.
export ENABLE_BOUNDARY_MODULATION=0
export BOUNDARY_MOD_SCALE=0.00
export BOUNDARY_MOD_START_RATIO=0.75

# Weak tolerance-band loss.
export ENABLE_TOLERANCE_BAND_LOSS=1
export LAMBDA_BAND=0.01
export BOUNDARY_BAND_T_GATE=200

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

LOG_FILE="${LOG_ROOT}/${EXP_NAME}/train_stage1_bef_sbg_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/stage1_bef_sbg.pid"

nohup python -u "${PROJECT_DIR}/tutorial_train_bef.py" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] BEF-SBG Stage1 training started."
echo "     exp:    ${EXP_NAME}"
echo "     gpu:    ${CUDA_VISIBLE_DEVICES}"
echo "     resume: ${RESUME_PATH}"
echo "     prompt: ${PROMPT_JSON}"
echo "     pid:    $(cat ${PID_FILE})"
echo "     log:    ${LOG_FILE}"
echo "     kill:   kill \$(cat ${PID_FILE})"
