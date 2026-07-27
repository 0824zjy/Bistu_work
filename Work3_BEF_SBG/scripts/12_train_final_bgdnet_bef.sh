#!/usr/bin/env bash
set -euo pipefail

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

export BGDNET_SWIN_VARIANT="standard"
export BGDNET_LOAD_BACKBONE_PRETRAIN="${BGDNET_LOAD_BACKBONE_PRETRAIN:-1}"
export BGDNET_STRICT_PRETRAIN="${BGDNET_STRICT_PRETRAIN:-1}"
export BGDNET_ENABLE_DISTANCE_HEAD="1"
export BGDNET_DISTANCE_REFINE_SCALE="${BGDNET_DISTANCE_REFINE_SCALE:-0.10}"
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"
export BGDNET_FREEZE_BN="${BGDNET_FREEZE_BN:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

mkdir -p "${LOG_ROOT}/segmentation" "${PID_DIR}" "${FINAL_SEG_SAVE_DIR}"

TRAIN_JSON="${TRAIN_JSON:-${WEIGHTED_TRAIN_JSON}}"
if [ ! -f "${TRAIN_JSON}" ]; then
  echo "[ERROR] missing weighted JSON: ${TRAIN_JSON}"
  exit 1
fi

# Never default validation to the official test set. Set these only when a
# separate validation root/list is available.
FINAL_VAL_ROOT="${FINAL_VAL_ROOT:-}"
FINAL_VAL_LIST="${FINAL_VAL_LIST:-}"
VAL_ARGS=()
if [ -n "${FINAL_VAL_ROOT}" ]; then
  VAL_ARGS+=(--val_path "${FINAL_VAL_ROOT}")
fi
if [ -n "${FINAL_VAL_LIST}" ]; then
  VAL_ARGS+=(--val_list "${FINAL_VAL_LIST}")
fi

LOG_FILE="${LOG_ROOT}/segmentation/train_chfs_${RATIO_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/12_train_chfs_${RATIO_TAG}_seed${SEED}.pid"

nohup python -u "${WORK_ROOT}/segmentation/train_BGDNet_BEF.py" \
  --weighted_train_json "${TRAIN_JSON}" \
  "${VAL_ARGS[@]}" \
  --epoch "${FINAL_EPOCH:-200}" \
  --lr "${FINAL_LR:-1e-4}" \
  --min_lr "${FINAL_MIN_LR:-1e-6}" \
  --warmup_epochs "${FINAL_WARMUP_EPOCHS:-5}" \
  --batchsize "${FINAL_BATCHSIZE:-4}" \
  --img_size "${FINAL_IMG_SIZE:-352}" \
  --train_save "${FINAL_SEG_SAVE_DIR}" \
  --alpha "${FINAL_ALPHA:-1.0}" \
  --beta "${FINAL_BETA:-0.25}" \
  --distance_w "${FINAL_DISTANCE_W:-0.30}" \
  --consistency_w "${FINAL_CONSISTENCY_W:-0.10}" \
  --clip "${FINAL_CLIP:-0.5}" \
  --augmentation "${FINAL_AUG:-True}" \
  --num_workers "${FINAL_NUM_WORKERS:-8}" \
  --optimizer "${FINAL_OPTIMIZER:-AdamW}" \
  --scheduler "${FINAL_SCHEDULER:-cosine}" \
  --decay_rate "${FINAL_DECAY_RATE:-0.1}" \
  --decay_epoch "${FINAL_DECAY_EPOCH:-200}" \
  --seed "${SEED}" \
  --freeze_bn "${BGDNET_FREEZE_BN}" \
  --ema_decay "${FINAL_EMA_DECAY:-0.999}" \
  --max_weight "${FINAL_MAX_WEIGHT:-1.5}" \
  --seg_bce_w "${FINAL_SEG_BCE_W:-0.5}" \
  --seg_dice_w "${FINAL_SEG_DICE_W:-1.0}" \
  --seg_tversky_w "${FINAL_SEG_TVERSKY_W:-0.5}" \
  --tversky_alpha_fp "${FINAL_TVERSKY_ALPHA_FP:-0.3}" \
  --tversky_beta_fn "${FINAL_TVERSKY_BETA_FN:-0.7}" \
  --tversky_gamma "${FINAL_TVERSKY_GAMMA:-0.75}" \
  --distance_max_px "${FINAL_DISTANCE_MAX_PX:-20}" \
  --distance_focus_tau "${FINAL_DISTANCE_FOCUS_TAU:-0.25}" \
  --distance_far_weight "${FINAL_DISTANCE_FAR_WEIGHT:-0.20}" \
  --distance_temperature "${FINAL_DISTANCE_TEMPERATURE:-4.0}" \
  --synthetic_start_epoch "${FINAL_SYN_START_EPOCH:-1}" \
  --synthetic_warmup_epochs "${FINAL_SYN_WARMUP_EPOCHS:-30}" \
  --pseudo_start_epoch "${FINAL_PSEUDO_START_EPOCH:-10}" \
  --pseudo_warmup_epochs "${FINAL_PSEUDO_WARMUP_EPOCHS:-40}" \
  --hard_start_epoch "${FINAL_HARD_START_EPOCH:-40}" \
  --hard_ramp_epochs "${FINAL_HARD_RAMP_EPOCHS:-60}" \
  --hard_gain_max "${FINAL_HARD_GAIN_MAX:-1.0}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[OK] Step 12 CHFS training started"
echo "     distance head: ${BGDNET_ENABLE_DISTANCE_HEAD}"
echo "     validation:    ${FINAL_VAL_ROOT:-<none; no test leakage>}"
echo "     output:        ${FINAL_SEG_SAVE_DIR}/BGDNet-CHFS-final.pth"
echo "     pid:           $(cat "${PID_FILE}")"
echo "     log:           ${LOG_FILE}"
