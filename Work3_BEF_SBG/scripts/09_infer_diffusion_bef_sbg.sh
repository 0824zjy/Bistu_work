#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

INFER_LOG_DIR="${LOG_ROOT}/diffusion/infer_bef_sbg_${RATIO_TAG}"
mkdir -p "${INFER_LOG_DIR}"
mkdir -p "${GEN_OUT_DIR}"
mkdir -p "${PID_DIR}"

cd "${BGDIFF_ROOT}"

# ============================================================
# Basic configuration
# ============================================================

export PROJECT_DIR="${BGDIFF_ROOT}"
export BOUNDARY_ALPHA_INIT="${BOUNDARY_ALPHA_INIT:-0.5}"
export PL_GLOBAL_SEED="${PL_GLOBAL_SEED:-${SEED:-0}}"
export DIFF_SEED="${DIFF_SEED:-${SEED:-0}}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED:-0}}"

export CKPT_PATH="${CKPT_PATH:-${DIFF_STAGE2_LOG_DIR}/checkpoints/last.ckpt}"
export PROMPT_JSON="${PROMPT_JSON:-${BEF_SAMPLE_PROMPT_JSON}}"
export OUT_DIR="${OUT_DIR:-${GEN_OUT_DIR}}"

export DEVICE="${DEVICE:-cuda:0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"

# 首次排错建议先用 1，确认跑通后再改回 2
export N_SAMPLES="${N_SAMPLES:-1}"

# 推理 DataLoader 不建议先开 4，避免 fork/显存/CPU worker 干扰排错
export NUM_WORKERS="${NUM_WORKERS:-0}"

export IMG_SIZE="${IMG_SIZE:-384}"

export SAMPLE_SEED_BASE="${SAMPLE_SEED_BASE:-${SEED:-0}}"

export DDIM_STEPS="${DDIM_STEPS:-70}"
export CFG="${CFG:-9.0}"

export USE_IMAGE_CONTROL="${USE_IMAGE_CONTROL:-0}"

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

export ENABLE_TOLERANCE_BAND_LOSS="${ENABLE_TOLERANCE_BAND_LOSS:-0}"
export LAMBDA_BAND="${LAMBDA_BAND:-0.0}"

# ============================================================
# CUDA stability configuration
# ============================================================

# 推理阶段也使用 BGDiff 专用 cuDNN 开关
export DIFF_DISABLE_CUDNN="${DIFF_DISABLE_CUDNN:-1}"

# 当前环境不要用 expandable_segments=True，沿用 max_split_size_mb 更稳
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# 降低 CUDA 模块初始化压力
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

export TORCH_DISABLE_ADDR2LINE=1
unset TORCH_SHOW_CPP_STACKTRACES

LOG_FILE="${INFER_LOG_DIR}/infer_bef_sbg_${RATIO_TAG}_cudnn${DIFF_DISABLE_CUDNN}_ns${N_SAMPLES}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/09_infer_diffusion_bef_sbg_${RATIO_TAG}.pid"

nohup bash -c '
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

cd "${BGDIFF_ROOT}"

echo "============================================================"
echo "[Diffusion BEF-SBG Inference]"
echo "  RATIO_TAG=${RATIO_TAG}"
echo "  BGDIFF_ROOT=${BGDIFF_ROOT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""
echo "  CKPT_PATH=${CKPT_PATH}"
echo "  PROMPT_JSON=${PROMPT_JSON}"
echo "  OUT_DIR=${OUT_DIR}"
echo ""
echo "  DEVICE=${DEVICE}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  N_SAMPLES=${N_SAMPLES}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  IMG_SIZE=${IMG_SIZE}"
echo "  DDIM_STEPS=${DDIM_STEPS}"
echo "  CFG=${CFG}"
echo ""
echo "  DIFF_DISABLE_CUDNN=${DIFF_DISABLE_CUDNN}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "  CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING}"
echo "============================================================"

exec python -u "${BGDIFF_ROOT}/tutorial_inference_bef.py"
' > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "[OK] Step 09 started: infer BEF-SBG diffusion samples"
echo "     ratio:     ${RATIO_TAG}"
echo "     gpu:       ${CUDA_VISIBLE_DEVICES}"
echo "     ckpt:      ${CKPT_PATH}"
echo "     prompt:    ${PROMPT_JSON}"
echo "     out_dir:   ${OUT_DIR}"
echo "     samples:   ${N_SAMPLES}"
echo "     cudnn off: ${DIFF_DISABLE_CUDNN}"
echo "     allocator: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "     pid:       $(cat "${PID_FILE}")"
echo "     log:       ${LOG_FILE}"
echo "     kill:      kill \$(cat ${PID_FILE})"
