#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
LOG_ROOT="/data/zjy_work/BGDiff/Ours/logs"
EXP_NAME="exp_mask2img"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"
cd "${PROJECT_DIR}"

nohup python -u /data/zjy_work/BGDiff/tool_merge_control.py \
  > "${LOG_ROOT}/${EXP_NAME}/merge_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] merge started. logs in ${LOG_ROOT}/${EXP_NAME}"


