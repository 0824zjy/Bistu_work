#!/usr/bin/env bash
set -euo pipefail

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

: "${RATIO_TAG:?Please set RATIO_TAG, for example: RATIO_TAG=5p}"
: "${SEED:=0}"

FEEDBACK_SUMMARY_CSV="${FEEDBACK_SUMMARY_CSV:-${FEEDBACK_OUT_DIR}/feedback_summary.csv}"

BEF_TRAIN_PROMPT_JSON="${BEF_TRAIN_PROMPT_JSON:-${WORK_ROOT}/feedback/bef_train_prompt_${RATIO_TAG}.json}"

BEF_SAMPLE_PROMPT_JSON="${BEF_SAMPLE_PROMPT_JSON:-${WORK_ROOT}/feedback/bef_sample_prompt_${RATIO_TAG}.json}"

LIST_TXT="${SPLIT_DIR}/low_${RATIO_TAG}_all.txt"

LOG_FILE="${LOG_ROOT}/diffusion/make_prompt/make_bef_prompt_json_${RATIO_TAG}_$(date +%Y%m%d_%H%M%S).log"

PID_FILE="${PID_DIR}/06_make_bef_prompt_json_${RATIO_TAG}.pid"

PYTHON_SCRIPT="${WORK_ROOT}/feedback/make_bef_prompt_json.py"

required_files=(
  "${PYTHON_SCRIPT}"
  "${FEEDBACK_SUMMARY_CSV}"
  "${LIST_TXT}"
)

for file_path in "${required_files[@]}"; do
  if [[ ! -f "${file_path}" ]]; then
    echo "[ERROR] Required file does not exist:" >&2
    echo "        ${file_path}" >&2
    exit 1
  fi
done

required_dirs=(
  "${ISIC_TRAIN_IMAGE_DIR}"
  "${ISIC_TRAIN_MASK_DIR}"
  "${FEEDBACK_OUT_DIR}/adaptive_boundary_prior"
  "${FEEDBACK_OUT_DIR}/difficulty"
)

for dir_path in "${required_dirs[@]}"; do
  if [[ ! -d "${dir_path}" ]]; then
    echo "[ERROR] Required directory does not exist:" >&2
    echo "        ${dir_path}" >&2
    exit 1
  fi
done

mkdir -p \
  "${WORK_ROOT}/feedback" \
  "${LOG_ROOT}/diffusion/make_prompt" \
  "${PID_DIR}" \
  "$(dirname "${BEF_TRAIN_PROMPT_JSON}")" \
  "$(dirname "${BEF_SAMPLE_PROMPT_JSON}")"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"

  if [[ -n "${OLD_PID}" ]] &&
     kill -0 "${OLD_PID}" 2>/dev/null
  then
    echo "[ERROR] Step 06 is already running." >&2
    echo "        pid: ${OLD_PID}" >&2
    exit 1
  fi

  rm -f "${PID_FILE}"
fi

nohup python -u "${PYTHON_SCRIPT}" \
  --img_dir "${ISIC_TRAIN_IMAGE_DIR}" \
  --mask_dir "${ISIC_TRAIN_MASK_DIR}" \
  --adaptive_prior_dir "${FEEDBACK_OUT_DIR}/adaptive_boundary_prior" \
  --difficulty_dir "${FEEDBACK_OUT_DIR}/difficulty" \
  --feedback_csv "${FEEDBACK_SUMMARY_CSV}" \
  --list_txt "${LIST_TXT}" \
  --train_out "${BEF_TRAIN_PROMPT_JSON}" \
  --sample_out "${BEF_SAMPLE_PROMPT_JSON}" \
  --prompt "${BEF_PROMPT_TEXT:-dermoscopic image}" \
  --sample_multiplier "${SAMPLE_MULTIPLIER:-2.0}" \
  --easy_quantile "${EASY_QUANTILE:-0.30}" \
  --hard_quantile "${HARD_QUANTILE:-0.70}" \
  --easy_ratio "${EASY_RATIO:-0.50}" \
  --medium_ratio "${MEDIUM_RATIO:-0.35}" \
  --hard_ratio "${HARD_RATIO:-0.15}" \
  --seed "${SEED}" \
  --require_difficulty_map \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

sleep 1

if kill -0 "${PID}" 2>/dev/null; then
  echo "[OK] Step 06 started."
  echo "     ratio:        ${RATIO_TAG}"
  echo "     seed:         ${SEED}"
  echo "     feedback csv: ${FEEDBACK_SUMMARY_CSV}"
  echo "     train json:   ${BEF_TRAIN_PROMPT_JSON}"
  echo "     sample json:  ${BEF_SAMPLE_PROMPT_JSON}"
  echo "     pid:          ${PID}"
  echo "     log:          ${LOG_FILE}"
  echo "     tail:         tail -f ${LOG_FILE}"
  echo "     stop:         kill -TERM \$(cat ${PID_FILE})"
  exit 0
fi

if [[ -s "${BEF_TRAIN_PROMPT_JSON}" ]] &&
   [[ -s "${BEF_SAMPLE_PROMPT_JSON}" ]] &&
   grep -qF "[DONE] BEF train/sample prompt JSON generated." "${LOG_FILE}"
then
  rm -f "${PID_FILE}"

  echo "[OK] Step 06 completed successfully."
  echo "     train json:  ${BEF_TRAIN_PROMPT_JSON}"
  echo "     sample json: ${BEF_SAMPLE_PROMPT_JSON}"
  echo "     log:         ${LOG_FILE}"
  exit 0
fi

rm -f "${PID_FILE}"

echo "[ERROR] Step 06 failed." >&2
echo "        log: ${LOG_FILE}" >&2
tail -n 100 "${LOG_FILE}" >&2 || true
exit 1
