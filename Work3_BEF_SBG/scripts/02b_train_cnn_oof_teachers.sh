#!/usr/bin/env bash
set -euo pipefail

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

CNN_EPOCHS="${CNN_EPOCHS:-160}"
CNN_BATCHSIZE="${CNN_BATCHSIZE:-4}"
CNN_LR="${CNN_LR:-1e-4}"
CNN_NUM_WORKERS="${CNN_NUM_WORKERS:-8}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-$((N_FOLDS - 1))}"
export CNN_EPOCHS CNN_BATCHSIZE CNN_LR CNN_NUM_WORKERS START_FOLD END_FOLD
mkdir -p "${CNN_TEACHER_CKPT_ROOT}" "${LOG_ROOT}/teacher"

LOG_FILE="${LOG_ROOT}/teacher/train_cnn_oof_${RATIO_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/02b_train_cnn_oof_${RATIO_TAG}_seed${SEED}.pid"

nohup bash -c '
set -euo pipefail
source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
export PYTHONPATH="/data/zjy_work:${PYTHONPATH:-}"
for fold in $(seq "${START_FOLD}" "${END_FOLD}"); do
  train_list="${SPLIT_DIR}/low_${RATIO_TAG}_fold${fold}_train.txt"
  val_list="${SPLIT_DIR}/low_${RATIO_TAG}_fold${fold}_val.txt"
  save_dir="${CNN_TEACHER_CKPT_ROOT}/fold${fold}"
  mkdir -p "${save_dir}"
  python -u "${WORK_ROOT}/teacher/train_cnn_teacher.py" \
    --image_dir "${ISIC_TRAIN_IMAGE_DIR}" \
    --mask_dir "${ISIC_TRAIN_MASK_DIR}" \
    --train_list "${train_list}" \
    --val_list "${val_list}" \
    --save_dir "${save_dir}" \
    --epochs "${CNN_EPOCHS}" \
    --batch_size "${CNN_BATCHSIZE}" \
    --num_workers "${CNN_NUM_WORKERS}" \
    --lr "${CNN_LR}" \
    --image_size 352 \
    --seed "$((SEED + fold))" \
    --device cuda:0 \
    --amp
  echo "[OK] CNN fold ${fold} finished"
done
' > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[OK] Step 02b started: CNN OOF teachers"
echo "     pid: $(cat "${PID_FILE}")"
echo "     log: ${LOG_FILE}"
