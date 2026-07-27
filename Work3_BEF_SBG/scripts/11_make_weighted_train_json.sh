#!/usr/bin/env bash
set -euo pipefail

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

mkdir -p "${LOG_ROOT}/scoring" "${PID_DIR}"
REAL_LIST="${REAL_LIST:-${SPLIT_DIR}/low_${RATIO_TAG}_all.txt}"
PSEUDO_JSONL="${PSEUDO_JSONL:-}"
PSEUDO_ARGS=()
if [ -n "${PSEUDO_JSONL}" ]; then
  PSEUDO_ARGS+=(--pseudo_jsonl "${PSEUDO_JSONL}" --pseudo_ratio "${PSEUDO_RATIO:-2.0}")
fi
LOG_FILE="${LOG_ROOT}/scoring/make_weighted_chfs_${RATIO_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/11_make_weighted_train_json_${RATIO_TAG}_seed${SEED}.pid"

nohup python -u "${WORK_ROOT}/scoring/make_weighted_seg_dataset.py" \
  --real_image_dir "${ISIC_TRAIN_IMAGE_DIR}" \
  --real_mask_dir "${ISIC_TRAIN_MASK_DIR}" \
  --real_list_txt "${REAL_LIST}" \
  --gen_jsonl "${GEN_ACCEPTED_JSONL}" \
  "${PSEUDO_ARGS[@]}" \
  --out_jsonl "${WEIGHTED_TRAIN_JSON}" \
  --synthetic_ratio "${SYNTHETIC_RATIO:-2.0}" \
  --sort_gen_by_weight \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[OK] Step 11 CHFS weighted JSON started"
echo "     out: ${WEIGHTED_TRAIN_JSON}"
echo "     pid: $(cat "${PID_FILE}")"
echo "     log: ${LOG_FILE}"
