#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
ABLATION_ROOT="${PROJECT_DIR}/Ours/ablation_sbg"
LOG_ROOT="${ABLATION_ROOT}/logs/infer"
RESULT_ROOT="${ABLATION_ROOT}/results"

# ============================================================
# User-editable settings
# ============================================================
# Available VARIANT:
#   maskonly
#   hard
#   softonly
#   progressive
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
  OUT_TAG="${VARIANT}_real5_syn15"
elif [ "${DATA_SPLIT}" = "real100" ]; then
  export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"
  OUT_TAG="${VARIANT}_real100_syn2"
else
  echo "[ERROR] Unknown DATA_SPLIT=${DATA_SPLIT}"
  exit 1
fi

# ============================================================
# Pick checkpoint according to variant
# ============================================================
if [ "${VARIANT}" = "maskonly" ]; then
  CKPT_EXP="abl_seed1_maskonly_${DATA_SPLIT}_stage1"

  export BOUNDARY_CONDITION_MODE=none
  export ENABLE_SOFT_BOUNDARY_PRIOR=0
  export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE=0
  export BOUNDARY_GUIDANCE_MAX=0.0
  export BOUNDARY_BRANCH_SCALE=0.0
  export ENABLE_BOUNDARY_MODULATION=0
  export BOUNDARY_MOD_SCALE=0.00

elif [ "${VARIANT}" = "hard" ]; then
  CKPT_EXP="abl_PH2_seed0_hard_${DATA_SPLIT}_stage1"

  export BOUNDARY_CONDITION_MODE=hard_from_mask
  export ENABLE_SOFT_BOUNDARY_PRIOR=0
  export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE=0
  export BOUNDARY_GUIDANCE_MAX=1.0
  export BOUNDARY_BRANCH_SCALE=1.0
  export ENABLE_BOUNDARY_MODULATION=0
  export BOUNDARY_MOD_SCALE=0.00

elif [ "${VARIANT}" = "softonly" ]; then
  CKPT_EXP="abl_seed1_softonly_${DATA_SPLIT}_stage1"

  export BOUNDARY_CONDITION_MODE=soft_from_mask
  export ENABLE_SOFT_BOUNDARY_PRIOR=1
  export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE=0
  export BOUNDARY_GUIDANCE_MAX=0.15
  export BOUNDARY_BRANCH_SCALE=0.10
  export ENABLE_BOUNDARY_MODULATION=0
  export BOUNDARY_MOD_SCALE=0.00

elif [ "${VARIANT}" = "progressive" ]; then
  CKPT_EXP="abl_seed1_progressive_${DATA_SPLIT}_stage1"

  export BOUNDARY_CONDITION_MODE=soft_from_mask
  export ENABLE_SOFT_BOUNDARY_PRIOR=1
  export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE=1
  export BOUNDARY_GUIDANCE_MAX=0.15
  export BOUNDARY_GUIDANCE_START_RATIO=0.35
  export BOUNDARY_GUIDANCE_TEMPERATURE=0.05
  export BOUNDARY_BRANCH_SCALE=0.10
  export ENABLE_BOUNDARY_MODULATION=0
  export BOUNDARY_MOD_SCALE=0.00

elif [ "${VARIANT}" = "decoder_mod" ]; then
  CKPT_EXP="abl_seed1_decoder_mod_real5_stage2"

  export BOUNDARY_CONDITION_MODE=soft_from_mask
  export ENABLE_SOFT_BOUNDARY_PRIOR=1
  export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE=1
  export BOUNDARY_GUIDANCE_MAX=0.15
  export BOUNDARY_GUIDANCE_START_RATIO=0.35
  export BOUNDARY_GUIDANCE_TEMPERATURE=0.05
  export BOUNDARY_BRANCH_SCALE=0.10
  export ENABLE_BOUNDARY_MODULATION=1
  export BOUNDARY_MOD_SCALE=0.10
  export BOUNDARY_MOD_START_RATIO=0.75

elif [ "${VARIANT}" = "full_sbg" ]; then

  CKPT_EXP="abl_PH2_seed0_full_sbg_${DATA_SPLIT}_stage2"

  export BOUNDARY_CONDITION_MODE=soft_from_mask
  export ENABLE_SOFT_BOUNDARY_PRIOR=1
  export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE=1
  export BOUNDARY_GUIDANCE_MAX=0.15
  export BOUNDARY_GUIDANCE_START_RATIO=0.35
  export BOUNDARY_GUIDANCE_TEMPERATURE=0.05
  export BOUNDARY_BRANCH_SCALE=0.10
  export ENABLE_BOUNDARY_MODULATION=1
  export BOUNDARY_MOD_SCALE=0.10
  export BOUNDARY_MOD_START_RATIO=0.75

else
  echo "[ERROR] Unknown VARIANT=${VARIANT}"
  exit 1
fi

export CKPT_PATH="${ABLATION_ROOT}/logs/${CKPT_EXP}/checkpoints/last.ckpt"
export OUT_DIR="${RESULT_ROOT}/${OUT_TAG}"

mkdir -p "${OUT_DIR}"

# ============================================================
# Inference basic config
# ============================================================
export DEVICE=cuda:0
export BATCH_SIZE=1
export NUM_WORKERS=0
export N_SAMPLES=3    #设置1：2
export SAMPLER=ddim
export DDIM_STEPS=50
export CFG=9.0
export USE_IMAGE_CONTROL=0
export BOUNDARY_ALPHA_INIT=1.0

export BOUNDARY_PRIOR_TAU=4.0
export BOUNDARY_PRIOR_RADIUS=12
export BOUNDARY_DILATE_KERNEL=3

nohup python -u "${ABLATION_ROOT}/code/inference_ablation.py" \
  > "${LOG_ROOT}/infer_${OUT_TAG}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] Ablation inference started."
echo "     variant: ${VARIANT}"
echo "     split:   ${DATA_SPLIT}"
echo "     gpu:     ${GPU_ID}"
echo "     ckpt:    ${CKPT_PATH}"
echo "     out_dir: ${OUT_DIR}"
echo "     logs:    ${LOG_ROOT}"
