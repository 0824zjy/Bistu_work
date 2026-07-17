#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
LOG_ROOT="/data/zjy_work/BGDiff/Ours/logs"

STAGE1_EXP="exp_mask2img_stage1_sbp_pg"
EXP_NAME="exp_mask2img_stage2_sbp_pg_decoder"

STAGE1_CKPT="${LOG_ROOT}/${STAGE1_EXP}/checkpoints/last.ckpt"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES=3

export PL_GLOBAL_SEED=0
export PYTHONHASHSEED=0

export RESUME_PATH="${STAGE1_CKPT}"
export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"

# Training basic config.
export NUM_GPUS=1
export PER_GPU_BATCH=1
export ACCUM=8
export LR=3e-6
export MAX_STEPS=20000
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
# In the provided ControlLDM.configure_optimizers:
#   SD_LOCKED=0 trains ControlNet + UNet output blocks / decoder.
#
# If you want strictly decoder-only Stage2, configure_optimizers should
# additionally respect TRAIN_CONTROLNET=0.
export SD_LOCKED=0
export ONLY_MID_CONTROL=0

# Kept for readability / future compatibility.
export TRAIN_CONTROLNET=0
export TRAIN_UNET_DECODER=1

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

# ============================================================
# SBP-PG: Tolerance-band Boundary Consistency Loss
# ============================================================
# If GPU memory is tight, set ENABLE_TOLERANCE_BAND_LOSS=0.
export ENABLE_TOLERANCE_BAND_LOSS=1
export LAMBDA_BAND=0.02
export BOUNDARY_BAND_T_GATE=200

# ============================================================
# Existing mechanisms
# ============================================================
# Keep previous consistency / aug logic, but boundary prior must be regenerated
# from augmented mask after online augmentation.
export ENABLE_ADAPTIVE_CONSISTENCY=1
export CONSISTENCY_W_MIN=0.10
export CONSISTENCY_W_MAX=0.60
export CONSISTENCY_T_GATE=250

export ENABLE_ONLINE_AUG=1
export LAMBDA_AUG=0.25
export AUG_PROB_MIN=0.10
export AUG_PROB_MAX=0.40
export AUG_EASY_BIAS=1.0

# Avoid strong boundary supervision.
export LOSS_BG_WEIGHT=1.0
export LOSS_FG_WEIGHT=1.0
export LOSS_BD_WEIGHT=1.0
export LAMBDA_BOUNDARY=0.0
export LAMBDA_MASK2IMAGE=0.0
export LAMBDA_MASK_REG=0.0

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u "${PROJECT_DIR}/tutorial_train.py" \
  > "${LOG_ROOT}/${EXP_NAME}/train_stage2_sbp_pg_decoder_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] Stage2 SBP-PG decoder training started."
echo "     resume: ${STAGE1_CKPT}"
echo "     logs:   ${LOG_ROOT}/${EXP_NAME}"
