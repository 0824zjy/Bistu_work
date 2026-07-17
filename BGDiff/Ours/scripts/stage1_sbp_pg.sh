#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
LOG_ROOT="/data/zjy_work/BGDiff/Ours/logs"
EXP_NAME="exp_mask2img_stage1_sbp_pg"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES=3

export PL_GLOBAL_SEED=0
export PYTHONHASHSEED=0

# Initialization checkpoint.
# This can be your old ControlNet init checkpoint.
export RESUME_PATH="${PROJECT_DIR}/Ours/models/control_sd15_region_boundary_init.pth"
export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"

# Training basic config.
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

# ============================================================
# Stage1 freeze policy
# ============================================================
# Stage1: learn structural control.
# In the provided tutorial_train.py, SD_LOCKED=1 means train ControlNet only.
export SD_LOCKED=1
export ONLY_MID_CONTROL=0

# These are kept for readability / future compatibility.
export TRAIN_CONTROLNET=1
export TRAIN_UNET_DECODER=0

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
export BOUNDARY_GUIDANCE_MAX=0.12
export BOUNDARY_GUIDANCE_START_RATIO=0.35
export BOUNDARY_GUIDANCE_TEMPERATURE=0.05

# Boundary branch should remain weak.
export BOUNDARY_BRANCH_SCALE=0.10

# ============================================================
# SBP-PG: Weak Boundary-aware Decoder Modulation
# ============================================================
# Stage1 recommends disabling decoder modulation.
export ENABLE_BOUNDARY_MODULATION=0
export BOUNDARY_MOD_SCALE=0.00
export BOUNDARY_MOD_START_RATIO=0.75

# ============================================================
# SBP-PG: Tolerance-band Boundary Consistency Loss
# ============================================================
# Weak regularization only. Not hard boundary supervision.
export ENABLE_TOLERANCE_BAND_LOSS=1
export LAMBDA_BAND=0.01
export BOUNDARY_BAND_T_GATE=200

# ============================================================
# Existing mechanisms
# ============================================================
# Keep previous adaptive consistency / online aug / mask regularization.
# Make sure their implementation uses regenerated soft boundary prior
# after any mask augmentation.
export ENABLE_ADAPTIVE_CONSISTENCY=1
export CONSISTENCY_W_MIN=0.20
export CONSISTENCY_W_MAX=1.00
export CONSISTENCY_T_GATE=300

export ENABLE_ONLINE_AUG=0
export LAMBDA_AUG=0.50
export AUG_PROB_MIN=0.15
export AUG_PROB_MAX=0.60
export AUG_EASY_BIAS=1.0

# If your previous p_losses has these task-aware loss terms, keep small.
export LOSS_BG_WEIGHT=1.0
export LOSS_FG_WEIGHT=1.0
export LOSS_BD_WEIGHT=1.0
export LAMBDA_BOUNDARY=0.0
export LAMBDA_MASK2IMAGE=0.0
export LAMBDA_MASK_REG=0.0

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u "${PROJECT_DIR}/tutorial_train.py" \
  > "${LOG_ROOT}/${EXP_NAME}/train_stage1_sbp_pg_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] Stage1 SBP-PG training started."
echo "     logs: ${LOG_ROOT}/${EXP_NAME}"
