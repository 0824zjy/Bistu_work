#!/usr/bin/env bash
set -euo pipefail
source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

mkdir -p "${OOF_CNN_OUT_DIR}" "${LOG_ROOT}/infer"
CNN_INFER_BATCHSIZE="${CNN_INFER_BATCHSIZE:-8}"
export CNN_INFER_BATCHSIZE
LOG_FILE="${LOG_ROOT}/infer/infer_cnn_oof_${RATIO_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/03b_infer_cnn_oof_${RATIO_TAG}_seed${SEED}.pid"

nohup bash -c '
set -euo pipefail
source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
export PYTHONPATH="/data/zjy_work:${PYTHONPATH:-}"
for fold in $(seq 0 $((N_FOLDS - 1))); do
  checkpoint="${CNN_TEACHER_CKPT_ROOT}/fold${fold}/cnn-best.pth"
  list_txt="${SPLIT_DIR}/low_${RATIO_TAG}_fold${fold}_val.txt"
  if [ ! -f "${checkpoint}" ]; then
    echo "[ERROR] Missing CNN checkpoint: ${checkpoint}"; exit 1
  fi
  python -u "${WORK_ROOT}/teacher/predict_directory_with_teacher.py" \
    --teacher_type cnn \
    --checkpoint "${checkpoint}" \
    --image_dir "${ISIC_TRAIN_IMAGE_DIR}" \
    --list_txt "${list_txt}" \
    --out_dir "${OOF_CNN_OUT_DIR}" \
    --device cuda:0 \
    --image_size 352 \
    --batch_size "${CNN_INFER_BATCHSIZE}" \
    --num_workers 4 \
    --amp
done
' > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[OK] Step 03b started: CNN OOF inference"
echo "     log: ${LOG_FILE}"
