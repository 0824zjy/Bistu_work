#!/usr/bin/env bash
set -euo pipefail

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

mkdir -p "${FEEDBACK_OUT_DIR}" "${LOG_ROOT}/feedback"
LIST_TXT="${SPLIT_DIR}/low_${RATIO_TAG}_all.txt"
USE_HETERO_OOF="${USE_HETERO_OOF:-1}"

if [ "${USE_HETERO_OOF}" = "1" ]; then
  PRED_ROOT="${OOF_HETERO_OUT_DIR}"
  if [ ! -d "${PRED_ROOT}/pred_masks" ]; then
    echo "[ERROR] Heterogeneous OOF predictions not found: ${PRED_ROOT}"
    echo "Run Step 03b, 03c and 03d first, or set USE_HETERO_OOF=0."
    exit 1
  fi
  OPTIONAL_ARGS=(
    --pred_variance_dir "${PRED_ROOT}/pred_variance"
    --pred_reliability_dir "${PRED_ROOT}/pred_reliability"
    --lambda_epi "${BEF_LAMBDA_EPI:-0.75}"
    --reliability_floor "${BEF_RELIABILITY_FLOOR:-0.05}"
  )
else
  PRED_ROOT="${OOF_OUT_DIR}"
  OPTIONAL_ARGS=()
fi

LOG_FILE="${LOG_ROOT}/feedback/build_boundary_feedback_${RATIO_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/04_build_boundary_feedback_${RATIO_TAG}_seed${SEED}.pid"

nohup python -u "${WORK_ROOT}/feedback/build_boundary_feedback.py" \
  --image_dir "${ISIC_TRAIN_IMAGE_DIR}" \
  --mask_dir "${ISIC_TRAIN_MASK_DIR}" \
  --pred_mask_dir "${PRED_ROOT}/pred_masks" \
  --pred_boundary_dir "${PRED_ROOT}/pred_boundaries" \
  --list_txt "${LIST_TXT}" \
  --out_dir "${FEEDBACK_OUT_DIR}" \
  --radius "${BEF_RADIUS:-12}" \
  --tau0 "${BEF_TAU0:-4.0}" \
  --kernel "${BEF_KERNEL:-3}" \
  --gamma "${BEF_GAMMA:-3.0}" \
  --lambda_u "${BEF_LAMBDA_U:-0.75}" \
  --w_fn "${BEF_W_FN:-1.5}" \
  --w_fp "${BEF_W_FP:-1.0}" \
  "${OPTIONAL_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[OK] Step 04 started: build boundary feedback"
echo "     prediction root: ${PRED_ROOT}"
echo "     heterogeneous: ${USE_HETERO_OOF}"
echo "     log: ${LOG_FILE}"
