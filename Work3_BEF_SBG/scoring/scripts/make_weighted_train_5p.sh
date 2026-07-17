#!/usr/bin/env bash
set -e

WORK_ROOT="/data/zjy_work/Work3_BEF_SBG"

LOG_DIR="${WORK_ROOT}/logs/scoring"
PID_DIR="${WORK_ROOT}/logs/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}" "${WORK_ROOT}/results"

# 用户给出的默认路径：
REAL_LIST="${REAL_LIST:-${WORK_ROOT}/splits/ISIC2018_seed0/low_5p_all.txt}"

# 如果你的第一阶段实际输出是 /splits/low_5p_all.txt，可自动 fallback。
if [ ! -f "${REAL_LIST}" ]; then
  ALT_LIST="${WORK_ROOT}/splits/low_5p_all.txt"
  if [ -f "${ALT_LIST}" ]; then
    echo "[WARN] REAL_LIST not found: ${REAL_LIST}"
    echo "[WARN] fallback to: ${ALT_LIST}"
    REAL_LIST="${ALT_LIST}"
  fi
fi

LOG_FILE="${LOG_DIR}/make_weighted_train_5p_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/make_weighted_train_5p.pid"

nohup python -u "${WORK_ROOT}/scoring/make_weighted_seg_dataset.py" \
  --real_image_dir /data/zjy_work/ISIC2018/train/Images \
  --real_mask_dir /data/zjy_work/ISIC2018/train/Masks \
  --real_list_txt "${REAL_LIST}" \
  --gen_jsonl "${WORK_ROOT}/results/generated_bef_sbg_accepted_5p.jsonl" \
  --out_jsonl "${WORK_ROOT}/results/weighted_train_5p_bef.jsonl" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] weighted segmentation jsonl generation started."
echo "     real_list: ${REAL_LIST}"
echo "     pid:       $(cat ${PID_FILE})"
echo "     log:       ${LOG_FILE}"
echo "     out:       ${WORK_ROOT}/results/weighted_train_5p_bef.jsonl"
echo "     kill:      kill \$(cat ${PID_FILE})"
