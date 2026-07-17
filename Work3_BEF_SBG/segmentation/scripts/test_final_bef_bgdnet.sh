#!/usr/bin/env bash
set -e

WORK_ROOT="/data/zjy_work/Work3_BEF_SBG"

LOG_DIR="${WORK_ROOT}/logs/segmentation"
PID_DIR="${WORK_ROOT}/logs/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PTH_PATH="${PTH_PATH:-${WORK_ROOT}/results/final_bgdnet_bef/BGDNet-BEF-best.pth}"
OUT_DIR="${WORK_ROOT}/results/eval_final_bef"

mkdir -p "${OUT_DIR}"

LOG_FILE="${LOG_DIR}/test_final_bef_bgdnet_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/test_final_bef_bgdnet.pid"

nohup python -u "${WORK_ROOT}/segmentation/test_BGDNet_BEF.py" \
  --pth_path "${PTH_PATH}" \
  --test_data_path /data/zjy_work/ISIC2018/test/ \
  --test_list "" \
  --out_dir "${OUT_DIR}" \
  --testsize 352 \
  --device cuda:0 \
  --boundary_tolerance 2 \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] final BGDNet-BEF testing started."
echo "     gpu:     ${CUDA_VISIBLE_DEVICES}"
echo "     pth:     ${PTH_PATH}"
echo "     out_dir: ${OUT_DIR}"
echo "     pid:     $(cat ${PID_FILE})"
echo "     log:     ${LOG_FILE}"
echo "     kill:    kill \$(cat ${PID_FILE})"
