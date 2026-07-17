#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
LOG_ROOT="/data/zjy_work/BGDiff/Ours/logs"

EXP_NAME="exp_mask2img_sbp_pg_infer"
OUT_DIR="/data/zjy_work/BGDiff/Ours/results/exp_mask2img_sbp_pg"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"
mkdir -p "${OUT_DIR}"

cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES=3

# Use merged Stage1+Stage2 model.
# Make sure your merge script saves to this path.
export CKPT_PATH="/data/zjy_work/BGDiff/Ours/models/merged_stage1_2_model.pth"

# For testing generated images on train masks, use prompt_train.json.
# For test generation, use prompt_test.json.
export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"

export OUT_DIR="${OUT_DIR}"
export DEVICE="cuda:0"

# For strict one-mask-to-n-samples correspondence.
export BATCH_SIZE=1
export N_SAMPLES=1
export NUM_WORKERS=4

# Sampling config.
export SAMPLER="ddim"
export DDIM_STEPS=70
export DPM_STEPS=20
export HYBRID_SPLIT=0.5
export CFG=9.0

# ============================================================
# Inference mode
# ============================================================
# Default inference:
#   mask + soft boundary prior
#
# Do not enable image control unless explicitly doing image-assisted inference.
export USE_IMAGE_CONTROL=0

# ============================================================
# SBP-PG: Soft Boundary Prior
# ============================================================
export ENABLE_SOFT_BOUNDARY_PRIOR=1
export BOUNDARY_PRIOR_TAU=4.0
export BOUNDARY_PRIOR_RADIUS=12
export BOUNDARY_DILATE_KERNEL=3

# ============================================================
# SBP-PG: Timestep-aware Progressive Boundary Guidance
# ============================================================
export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE=1
export BOUNDARY_GUIDANCE_MAX=0.15
export BOUNDARY_GUIDANCE_START_RATIO=0.35
export BOUNDARY_GUIDANCE_TEMPERATURE=0.05

# Boundary branch remains weak.
export BOUNDARY_BRANCH_SCALE=0.10

# ============================================================
# SBP-PG: Weak Boundary-aware Decoder Modulation
# ============================================================
export ENABLE_BOUNDARY_MODULATION=1
export BOUNDARY_MOD_SCALE=0.10
export BOUNDARY_MOD_START_RATIO=0.75

# No loss during inference.
export ENABLE_TOLERANCE_BAND_LOSS=0
export LAMBDA_BAND=0.0
export BOUNDARY_BAND_T_GATE=200

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u "${PROJECT_DIR}/tutorial_inference.py" \
  > "${LOG_ROOT}/${EXP_NAME}/infer_sbp_pg_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] SBP-PG inference started."
echo "     ckpt:    ${CKPT_PATH}"
echo "     prompts: ${PROMPT_JSON}"
echo "     results: ${OUT_DIR}"
echo "     logs:    ${LOG_ROOT}/${EXP_NAME}"
