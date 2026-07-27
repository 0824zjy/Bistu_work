#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

mkdir -p "${DIFF_STAGE2_LOG_DIR}"
mkdir -p "${PID_DIR}"

STAGE1_CKPT="${DIFF_STAGE1_LOG_DIR}/checkpoints/best.ckpt"

if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "[ERROR] Stage-1 checkpoint not found:"
    echo "        ${STAGE1_CKPT}"
    exit 1
fi

# ============================================================
# Basic configuration
# ============================================================
export BOUNDARY_ALPHA_INIT="${BOUNDARY_ALPHA_INIT:-0.5}"
export PROJECT_DIR="${BGDIFF_ROOT}"

export PL_GLOBAL_SEED="${PL_GLOBAL_SEED:-${SEED:-0}}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED:-0}}"

export RESUME_PATH="${RESUME_PATH:-${STAGE1_CKPT}}"
export PROMPT_JSON="${PROMPT_JSON:-${BEF_TRAIN_PROMPT_JSON}}"

export LOG_ROOT="${DIFF_LOG_ROOT}"
export EXP_NAME="${DIFF_STAGE2_EXP}"

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
export ACCUM="${ACCUM:-8}"
export LR="${LR:-3e-6}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export LOGGER_FREQ="${LOGGER_FREQ:-2000}"
export IMG_SIZE="${IMG_SIZE:-384}"
export EMPTY_PROMPT_PROB="${EMPTY_PROMPT_PROB:-0.0}"
export PRECISION="${PRECISION:-16-mixed}"
export NUM_WORKERS="${NUM_WORKERS:-0}"

# ============================================================
# Stage-2 trainable parameter policy
# ============================================================

export SD_LOCKED="${SD_LOCKED:-0}"
export ONLY_MID_CONTROL="${ONLY_MID_CONTROL:-0}"
export TRAIN_CONTROLNET="${TRAIN_CONTROLNET:-0}"
export TRAIN_UNET_DECODER="${TRAIN_UNET_DECODER:-1}"

# ============================================================
# BEF-SBG configuration
# ============================================================

export BOUNDARY_PRIOR_MODE="${BOUNDARY_PRIOR_MODE:-external}"

export ENABLE_SOFT_BOUNDARY_PRIOR="${ENABLE_SOFT_BOUNDARY_PRIOR:-1}"
export BOUNDARY_PRIOR_TAU="${BOUNDARY_PRIOR_TAU:-4.0}"
export BOUNDARY_PRIOR_RADIUS="${BOUNDARY_PRIOR_RADIUS:-12}"
export BOUNDARY_DILATE_KERNEL="${BOUNDARY_DILATE_KERNEL:-3}"

export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE="${ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE:-1}"
export BOUNDARY_GUIDANCE_MAX="${BOUNDARY_GUIDANCE_MAX:-0.15}"
export BOUNDARY_GUIDANCE_START_RATIO="${BOUNDARY_GUIDANCE_START_RATIO:-0.35}"
export BOUNDARY_GUIDANCE_TEMPERATURE="${BOUNDARY_GUIDANCE_TEMPERATURE:-0.05}"
export BOUNDARY_BRANCH_SCALE="${BOUNDARY_BRANCH_SCALE:-0.10}"

export ENABLE_BOUNDARY_MODULATION="${ENABLE_BOUNDARY_MODULATION:-1}"
export BOUNDARY_MOD_SCALE="${BOUNDARY_MOD_SCALE:-0.10}"
export BOUNDARY_MOD_START_RATIO="${BOUNDARY_MOD_START_RATIO:-0.75}"

export ENABLE_TOLERANCE_BAND_LOSS="${ENABLE_TOLERANCE_BAND_LOSS:-1}"
export LAMBDA_BAND="${LAMBDA_BAND:-0.02}"
export BOUNDARY_BAND_T_GATE="${BOUNDARY_BAND_T_GATE:-200}"

# ============================================================
# CUDA stability configuration
# ============================================================

# 必须由 tutorial_train_bef.py 读取。
export DIFF_DISABLE_CUDNN="${DIFF_DISABLE_CUDNN:-1}"

# 禁止 Lightning 在正式训练前执行两批 validation。
export DIFF_NUM_SANITY_VAL_STEPS="${DIFF_NUM_SANITY_VAL_STEPS:-0}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# 当前环境不要设置 expandable_segments=True。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# 降低模型启动时的 CUDA 模块加载压力。
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

export TORCH_DISABLE_ADDR2LINE=1
unset TORCH_SHOW_CPP_STACKTRACES

LOG_FILE="${DIFF_STAGE2_LOG_DIR}/train_stage2_bef_sbg_${RATIO_TAG}_cudnn${DIFF_DISABLE_CUDNN}_sanity${DIFF_NUM_SANITY_VAL_STEPS}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/08_train_diffusion_stage2_${RATIO_TAG}.pid"

nohup bash -c '
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

cd "${BGDIFF_ROOT}"

echo "============================================================"
echo "[Diffusion Stage-2 Training]"
echo "  RATIO_TAG=${RATIO_TAG}"
echo "  BGDIFF_ROOT=${BGDIFF_ROOT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""
echo "  RESUME_PATH=${RESUME_PATH}"
echo "  PROMPT_JSON=${PROMPT_JSON}"
echo "  LOG_ROOT=${LOG_ROOT}"
echo "  EXP_NAME=${EXP_NAME}"
echo ""
echo "  NUM_GPUS=${NUM_GPUS}"
echo "  PER_GPU_BATCH=${PER_GPU_BATCH}"
echo "  ACCUM=${ACCUM}"
echo "  LR=${LR}"
echo "  MAX_STEPS=${MAX_STEPS}"
echo "  IMG_SIZE=${IMG_SIZE}"
echo "  PRECISION=${PRECISION}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo ""
echo "  SD_LOCKED=${SD_LOCKED}"
echo "  TRAIN_CONTROLNET=${TRAIN_CONTROLNET}"
echo "  TRAIN_UNET_DECODER=${TRAIN_UNET_DECODER}"
echo ""
echo "  DIFF_DISABLE_CUDNN=${DIFF_DISABLE_CUDNN}"
echo "  DIFF_NUM_SANITY_VAL_STEPS=${DIFF_NUM_SANITY_VAL_STEPS}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "  CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING}"
echo "============================================================"

exec python -u "${BGDIFF_ROOT}/tutorial_train_bef.py"
' > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "[OK] Step 08 started: train diffusion stage2"
echo "     ratio:        ${RATIO_TAG}"
echo "     gpu:          ${CUDA_VISIBLE_DEVICES}"
echo "     resume:       ${RESUME_PATH}"
echo "     prompt:       ${PROMPT_JSON}"
echo "     batch:        ${PER_GPU_BATCH}"
echo "     accumulation: ${ACCUM}"
echo "     precision:    ${PRECISION}"
echo "     cudnn off:    ${DIFF_DISABLE_CUDNN}"
echo "     sanity steps: ${DIFF_NUM_SANITY_VAL_STEPS}"
echo "     allocator:    ${PYTORCH_CUDA_ALLOC_CONF}"
echo "     pid:          $(cat "${PID_FILE}")"
echo "     log:          ${LOG_FILE}"
echo "     kill:         kill \$(cat ${PID_FILE})"
