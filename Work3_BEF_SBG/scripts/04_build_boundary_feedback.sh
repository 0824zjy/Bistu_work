#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

mkdir -p "${FEEDBACK_OUT_DIR}"
mkdir -p "${LOG_ROOT}/feedback"

LIST_TXT="${SPLIT_DIR}/low_${RATIO_TAG}_all.txt"

LOG_FILE="${LOG_ROOT}/feedback/build_boundary_feedback_${RATIO_TAG}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/04_build_boundary_feedback_${RATIO_TAG}.pid"

nohup python -u "${WORK_ROOT}/feedback/build_boundary_feedback.py" \
  --image_dir "${ISIC_TRAIN_IMAGE_DIR}" \
  --mask_dir "${ISIC_TRAIN_MASK_DIR}" \
  --pred_mask_dir "${OOF_OUT_DIR}/pred_masks" \
  --pred_boundary_dir "${OOF_OUT_DIR}/pred_boundaries" \
  --list_txt "${LIST_TXT}" \
  --out_dir "${FEEDBACK_OUT_DIR}" \
  --radius "${BEF_RADIUS:-12}" \
  --tau0 "${BEF_TAU0:-4.0}" \
  --kernel "${BEF_KERNEL:-3}" \
  --gamma "${BEF_GAMMA:-3.0}" \
  --lambda_u "${BEF_LAMBDA_U:-0.75}" \
  --w_fn "${BEF_W_FN:-1.5}" \
  --w_fp "${BEF_W_FP:-1.0}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 04 started: build boundary feedback"
echo "     ratio: ${RATIO_TAG}"
echo "     list:  ${LIST_TXT}"
echo "     pid:   $(cat ${PID_FILE})"
echo "     log:   ${LOG_FILE}"
echo "     kill:  kill \$(cat ${PID_FILE})"
