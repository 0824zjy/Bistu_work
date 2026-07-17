#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

mkdir -p "${SPLIT_DIR}"

LOG_FILE="${LOG_ROOT}/splits/make_low_label_splits_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/01_make_low_label_splits.pid"

nohup python -u "${WORK_ROOT}/splits/make_low_label_splits.py" \
  --image_dir "${ISIC_TRAIN_IMAGE_DIR}" \
  --mask_dir "${ISIC_TRAIN_MASK_DIR}" \
  --out_dir "${SPLIT_DIR}" \
  --ratios 0.05,0.10,0.20,1.0 \
  --n_folds "${N_FOLDS}" \
  --seed "${SEED}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 01 started: make low-label splits"
echo "     split_dir: ${SPLIT_DIR}"
echo "     pid:       $(cat ${PID_FILE})"
echo "     log:       ${LOG_FILE}"
echo "     kill:      kill \$(cat ${PID_FILE})"
