#!/usr/bin/env bash
set -e

WORK_ROOT="/data/zjy_work/Work3_BEF_SBG"

LOG_DIR="${WORK_ROOT}/logs/diffusion/make_prompt"
PID_DIR="${WORK_ROOT}/logs/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}" "${WORK_ROOT}/feedback"

RATIO_TAG="${RATIO_TAG:-5p}"

LOG_FILE="${LOG_DIR}/make_bef_prompt_json_${RATIO_TAG}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/make_bef_prompt_json_${RATIO_TAG}.pid"

nohup python -u "${WORK_ROOT}/feedback/make_bef_prompt_json.py" \
  --img_dir /data/zjy_work/ISIC2018/train/Images \
  --mask_dir /data/zjy_work/ISIC2018/train/Masks \
  --adaptive_prior_dir "${WORK_ROOT}/results/boundary_feedback/ISIC2018_${RATIO_TAG}/adaptive_boundary_prior" \
  --difficulty_dir "${WORK_ROOT}/results/boundary_feedback/ISIC2018_${RATIO_TAG}/difficulty" \
  --list_txt "${WORK_ROOT}/splits/low_${RATIO_TAG}_all.txt" \
  --out "${WORK_ROOT}/feedback/prompt_bef_train.json" \
  --prompt "dermoscopic image" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] BEF prompt json generation started."
echo "     ratio: ${RATIO_TAG}"
echo "     pid:   $(cat ${PID_FILE})"
echo "     log:   ${LOG_FILE}"
echo "     out:   ${WORK_ROOT}/feedback/prompt_bef_train.json"
echo "     kill:  kill \$(cat ${PID_FILE})"
