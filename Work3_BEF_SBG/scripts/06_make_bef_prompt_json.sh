#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

mkdir -p "${WORK_ROOT}/feedback"
mkdir -p "${LOG_ROOT}/diffusion/make_prompt"

LIST_TXT="${SPLIT_DIR}/low_${RATIO_TAG}_all.txt"

LOG_FILE="${LOG_ROOT}/diffusion/make_prompt/make_bef_prompt_json_${RATIO_TAG}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/06_make_bef_prompt_json_${RATIO_TAG}.pid"

nohup python -u "${WORK_ROOT}/feedback/make_bef_prompt_json.py" \
  --img_dir "${ISIC_TRAIN_IMAGE_DIR}" \
  --mask_dir "${ISIC_TRAIN_MASK_DIR}" \
  --adaptive_prior_dir "${FEEDBACK_OUT_DIR}/adaptive_boundary_prior" \
  --difficulty_dir "${FEEDBACK_OUT_DIR}/difficulty" \
  --list_txt "${LIST_TXT}" \
  --out "${BEF_PROMPT_JSON}" \
  --prompt "dermoscopic image" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 06 started: make BEF prompt json"
echo "     ratio: ${RATIO_TAG}"
echo "     out:   ${BEF_PROMPT_JSON}"
echo "     pid:   $(cat ${PID_FILE})"
echo "     log:   ${LOG_FILE}"
echo "     kill:  kill \$(cat ${PID_FILE})"
