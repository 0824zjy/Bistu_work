#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash kill_by_pid.sh <pid_file_or_step_keyword>"
  echo ""
  echo "Examples:"
  echo "  bash kill_by_pid.sh /data/zjy_work/Work3_BEF_SBG/logs/pids/07_train_diffusion_stage1_5p.pid"
  echo "  bash kill_by_pid.sh 07_train_diffusion_stage1_5p"
  exit 1
fi

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

TARGET="$1"

if [ -f "${TARGET}" ]; then
  PID_FILE="${TARGET}"
else
  PID_FILE="${PID_DIR}/${TARGET}.pid"
fi

if [ ! -f "${PID_FILE}" ]; then
  echo "[ERROR] pid file not found: ${PID_FILE}"
  exit 1
fi

PID=$(cat "${PID_FILE}")

if ps -p "${PID}" > /dev/null 2>&1; then
  kill "${PID}"
  echo "[OK] killed pid=${PID}"
  echo "     pid_file=${PID_FILE}"
else
  echo "[WARN] pid=${PID} is not running."
  echo "       pid_file=${PID_FILE}"
fi
