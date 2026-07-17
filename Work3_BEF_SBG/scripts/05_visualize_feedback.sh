#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

mkdir -p "${LOG_ROOT}/visual"

VIS_OUT_DIR="${WORK_ROOT}/results/visual_feedback/ISIC2018_${RATIO_TAG}"
LIST_TXT="${SPLIT_DIR}/low_${RATIO_TAG}_all.txt"

LOG_FILE="${LOG_ROOT}/visual/visualize_feedback_${RATIO_TAG}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/05_visualize_feedback_${RATIO_TAG}.pid"

nohup python -u "${WORK_ROOT}/feedback/visualize_feedback.py" \
  --image_dir "${ISIC_TRAIN_IMAGE_DIR}" \
  --mask_dir "${ISIC_TRAIN_MASK_DIR}" \
  --pred_mask_dir "${OOF_OUT_DIR}/pred_masks" \
  --feedback_dir "${FEEDBACK_OUT_DIR}" \
  --list_txt "${LIST_TXT}" \
  --out_dir "${VIS_OUT_DIR}" \
  --max_samples 12 \
  --tile_size 224 \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 05 started: visualize feedback"
echo "     out:  ${VIS_OUT_DIR}"
echo "     pid:  $(cat ${PID_FILE})"
echo "     log:  ${LOG_FILE}"
echo "     kill: kill \$(cat ${PID_FILE})"
