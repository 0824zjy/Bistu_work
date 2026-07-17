#!/usr/bin/env bash
set -e

WORK_ROOT="/data/zjy_work/Work3_BEF_SBG"

LOG_DIR="${WORK_ROOT}/logs/segmentation"
PID_DIR="${WORK_ROOT}/logs/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TRAIN_SAVE="${WORK_ROOT}/results/final_bgdnet_bef"
mkdir -p "${TRAIN_SAVE}"

LOG_FILE="${LOG_DIR}/train_final_bef_bgdnet_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/train_final_bef_bgdnet.pid"

nohup python -u "${WORK_ROOT}/segmentation/train_BGDNet_BEF.py" \
  --weighted_train_json "${WORK_ROOT}/results/weighted_train_5p_bef.jsonl" \
  --test_path /data/zjy_work/ISIC2018/test/ \
  --test_list "" \
  --epoch 200 \
  --lr 1e-4 \
  --batchsize 4 \
  --img_size 352 \
  --train_save "${TRAIN_SAVE}" \
  --alpha 1.0 \
  --beta 0.4 \
  --clip 0.5 \
  --augmentation False \
  --num_workers 8 \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] final BGDNet-BEF training started."
echo "     gpu:       ${CUDA_VISIBLE_DEVICES}"
echo "     train_json:${WORK_ROOT}/results/weighted_train_5p_bef.jsonl"
echo "     save_dir:  ${TRAIN_SAVE}"
echo "     pid:       $(cat ${PID_FILE})"
echo "     log:       ${LOG_FILE}"
echo "     kill:      kill \$(cat ${PID_FILE})"
