#!/usr/bin/env bash
set -e
export PYTHONPATH="/data/zjy_work/env_patch:${PYTHONPATH:-}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

PROJECT_DIR="/data/zjy_work/BGDiff"
WORK_ROOT="/data/zjy_work/Work3_BEF_SBG"

LOG_ROOT="${WORK_ROOT}/logs/diffusion"
PID_DIR="${WORK_ROOT}/logs/pids"

EXP_NAME="infer_bef_sbg"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"
mkdir -p "${PID_DIR}"

cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PROJECT_DIR="${PROJECT_DIR}"

# 默认使用 Stage2 last.ckpt。
# 也可外部覆盖：
# CKPT_PATH=/path/to/best.ckpt bash infer_bef_sbg.sh
export CKPT_PATH="${CKPT_PATH:-${WORK_ROOT}/logs/diffusion/stage2_bef_sbg_decoder/checkpoints/last.ckpt}"

export PROMPT_JSON="${WORK_ROOT}/feedback/prompt_bef_train.json"
export OUT_DIR="${WORK_ROOT}/results/generated_bef_sbg"

export DEVICE="cuda:0"
export BATCH_SIZE=1
export N_SAMPLES=2
export NUM_WORKERS=4
export IMG_SIZE=384

export SAMPLE_SEED_BASE="${SAMPLE_SEED_BASE:-0}"

export DDIM_STEPS=70
export CFG=9.0

export USE_IMAGE_CONTROL=0

# BEF-SBG: use external adaptive boundary prior directly.
export BOUNDARY_PRIOR_MODE=external

# Fallback compatibility only.
export ENABLE_SOFT_BOUNDARY_PRIOR=1
export BOUNDARY_PRIOR_TAU=4.0
export BOUNDARY_PRIOR_RADIUS=12
export BOUNDARY_DILATE_KERNEL=3

export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE=1
export BOUNDARY_GUIDANCE_MAX=0.15
export BOUNDARY_GUIDANCE_START_RATIO=0.35
export BOUNDARY_GUIDANCE_TEMPERATURE=0.05
export BOUNDARY_BRANCH_SCALE=0.10

export ENABLE_BOUNDARY_MODULATION=1
export BOUNDARY_MOD_SCALE=0.10
export BOUNDARY_MOD_START_RATIO=0.75

# No loss during inference.
export ENABLE_TOLERANCE_BAND_LOSS=0
export LAMBDA_BAND=0.0

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "${OUT_DIR}"

LOG_FILE="${LOG_ROOT}/${EXP_NAME}/infer_bef_sbg_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/infer_bef_sbg.pid"

nohup python -u "${PROJECT_DIR}/tutorial_inference_bef.py" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] BEF-SBG inference started."
echo "     gpu:     ${CUDA_VISIBLE_DEVICES}"
echo "     ckpt:    ${CKPT_PATH}"
echo "     prompt:  ${PROMPT_JSON}"
echo "     out_dir: ${OUT_DIR}"
echo "     pid:     $(cat ${PID_FILE})"
echo "     log:     ${LOG_FILE}"
echo "     kill:    kill \$(cat ${PID_FILE})"
