#!/usr/bin/env bash

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

echo "============================================================"
echo " Work3 BEF-SBG Status"
echo "============================================================"
echo "WORK_ROOT=${WORK_ROOT}"
echo "RATIO_TAG=${RATIO_TAG}"
echo "SEED=${SEED}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

echo "[PID files]"
ls -lh "${PID_DIR}" || true

echo ""
echo "[Running status]"
for f in "${PID_DIR}"/*.pid; do
  [ -e "$f" ] || continue
  name=$(basename "$f")
  pid=$(cat "$f")

  if ps -p "${pid}" > /dev/null 2>&1; then
    echo "${name}: RUNNING, pid=${pid}"
  else
    echo "${name}: NOT RUNNING, pid=${pid}"
  fi
done

echo ""
echo "[Important outputs]"
echo "SPLIT_DIR=${SPLIT_DIR}"
echo "OOF_OUT_DIR=${OOF_OUT_DIR}"
echo "FEEDBACK_OUT_DIR=${FEEDBACK_OUT_DIR}"
echo "BEF_PROMPT_JSON=${BEF_PROMPT_JSON}"
echo "GEN_OUT_DIR=${GEN_OUT_DIR}"
echo "GEN_ACCEPTED_JSONL=${GEN_ACCEPTED_JSONL}"
echo "WEIGHTED_TRAIN_JSON=${WEIGHTED_TRAIN_JSON}"
echo "FINAL_SEG_SAVE_DIR=${FINAL_SEG_SAVE_DIR}"
echo "FINAL_EVAL_DIR=${FINAL_EVAL_DIR}"
