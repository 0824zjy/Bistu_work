#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
ABLATION_ROOT="${PROJECT_DIR}/Ours/ablation_sbg"
LOG_ROOT="${ABLATION_ROOT}/logs"

# ============================================================
# User-editable settings
# ============================================================
# Available VARIANT:
#   decoder_mod
#   full_sbg
#
# Available DATA_SPLIT:
#   real5
#   real100
# ============================================================
VARIANT="${VARIANT:-full_sbg}"
DATA_SPLIT="${DATA_SPLIT:-real5}"
GPU_ID="${GPU_ID:-3}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

export PL_GLOBAL_SEED=0
export PYTHONHASHSEED=0

export PYTHONPATH="${ABLATION_ROOT}/code:${PYTHONPATH}"
export CONFIG_PATH="${ABLATION_ROOT}/code/configs/cldm_v15_ablation.yaml"

if [ "${DATA_SPLIT}" = "real5" ]; then
  export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train_5p_seed1.json"
  export MAX_STEPS="${MAX_STEPS:-12000}"
elif [ "${DATA_SPLIT}" = "real100" ]; then
  export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"
  export MAX_STEPS="${MAX_STEPS:-20000}"
else
  echo "[ERROR] Unknown DATA_SPLIT=${DATA_SPLIT}"
  exit 1
fi

# ============================================================
# Variant-specific resume and exp name
# ============================================================
if [ "${VARIANT}" = "decoder_mod" ]; then
  if [ "${DATA_SPLIT}" != "real5" ]; then
    echo "[ERROR] decoder_mod ablation is intended for real5 only."
    exit 1
  fi

  STAGE1_EXP="abl_seed1_progressive_real5_stage1"
  EXP_NAME="abl_seed1_decoder_mod_real5_stage2"

  export ENABLE_TOLERANCE_BAND_LOSS=0
  export LAMBDA_BAND=0.00

elif [ "${VARIANT}" = "full_sbg" ]; then
  STAGE1_EXP="abl_PH2_seed0_full_sbg_${DATA_SPLIT}_stage1"
  EXP_NAME="abl_PH2_seed0_full_sbg_${DATA_SPLIT}_stage2"

  export ENABLE_TOLERANCE_BAND_LOSS=1
  export LAMBDA_BAND=0.02

else
  echo "[ERROR] Unknown VARIANT=${VARIANT}"
  exit 1
fi

STAGE1_CKPT="${LOG_ROOT}/${STAGE1_EXP}/checkpoints/last.ckpt"
export RESUME_PATH="${STAGE1_CKPT}"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"

# ============================================================
# Training basic config
# ============================================================
export NUM_GPUS=1
export PER_GPU_BATCH=1
export ACCUM=8
export LR=3e-6
export LOGGER_FREQ=2000
export LOG_ROOT="${LOG_ROOT}"
export EXP_NAME="${EXP_NAME}"
export IMG_SIZE=384
export EMPTY_PROMPT_PROB=0.0
export PRECISION=16-mixed
export NUM_WORKERS=0

# ============================================================
# Stage2 freeze policy
# ============================================================
export SD_LOCKED=0
export ONLY_MID_CONTROL=0
export TRAIN_CONTROLNET=0
export TRAIN_UNET_DECODER=1

# ============================================================
# SBG-Diff settings
# ============================================================
export BOUNDARY_CONDITION_MODE=soft_from_mask
export BOUNDARY_ALPHA_INIT=1.0

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

export BOUNDARY_BAND_T_GATE=200

# Existing mechanisms
export ENABLE_ADAPTIVE_CONSISTENCY=1
export CONSISTENCY_W_MIN=0.10
export CONSISTENCY_W_MAX=0.60
export CONSISTENCY_T_GATE=250

export ENABLE_ONLINE_AUG=1
export LAMBDA_AUG=0.25
export AUG_PROB_MIN=0.10
export AUG_PROB_MAX=0.40
export AUG_EASY_BIAS=1.0

export LOSS_BG_WEIGHT=1.0
export LOSS_FG_WEIGHT=1.0
export LOSS_BD_WEIGHT=1.0
export LAMBDA_BOUNDARY=0.0
export LAMBDA_MASK2IMAGE=0.0
export LAMBDA_MASK_REG=0.0

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u "${ABLATION_ROOT}/code/train_ablation.py" \
  > "${LOG_ROOT}/${EXP_NAME}/train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] Stage2 ablation training started."
echo "     variant: ${VARIANT}"
echo "     split:   ${DATA_SPLIT}"
echo "     gpu:     ${GPU_ID}"
echo "     resume:  ${STAGE1_CKPT}"
echo "     logs:    ${LOG_ROOT}/${EXP_NAME}"
