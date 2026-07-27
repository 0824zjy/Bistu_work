#!/usr/bin/env bash
set -euo pipefail

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BGDNET_SWIN_VARIANT="standard"
export BGDNET_LOAD_BACKBONE_PRETRAIN="0"
export BGDNET_STRICT_PRETRAIN="0"
export BGDNET_ENABLE_DISTANCE_HEAD="1"
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

mkdir -p "${LOG_ROOT}/segmentation" "${PID_DIR}" "${FINAL_EVAL_DIR}"

PTH_PATH="${PTH_PATH:-${FINAL_SEG_SAVE_DIR}/BGDNet-CHFS-final.pth}"
if [ ! -f "${PTH_PATH}" ]; then
  echo "[ERROR] missing checkpoint: ${PTH_PATH}"
  exit 1
fi

FINAL_TEST_LIST="${FINAL_TEST_LIST:-}"
TEST_LIST_ARGS=()
if [ -n "${FINAL_TEST_LIST}" ]; then
  TEST_LIST_ARGS+=(--test_list "${FINAL_TEST_LIST}")
fi

TTA_ARGS=()
if [ "${TEST_TTA:-1}" = "1" ]; then
  TTA_ARGS+=(--tta)
fi
if [ "${TEST_LCC:-0}" = "1" ]; then
  TTA_ARGS+=(--postprocess_lcc)
fi

LOG_FILE="${LOG_ROOT}/segmentation/test_chfs_${RATIO_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/13_test_chfs_${RATIO_TAG}_seed${SEED}.pid"

nohup python -u "${WORK_ROOT}/segmentation/test_BGDNet_BEF.py" \
  --pth_path "${PTH_PATH}" \
  --test_data_path "${ISIC_TEST_ROOT}" \
  "${TEST_LIST_ARGS[@]}" \
  --out_dir "${FINAL_EVAL_DIR}" \
  --testsize "${TEST_IMG_SIZE:-352}" \
  --device "${TEST_DEVICE:-cuda:0}" \
  --threshold "${TEST_THRESHOLD:-0.5}" \
  --boundary_threshold "${BOUNDARY_THRESHOLD:-0.5}" \
  --boundary_tolerance "${BOUNDARY_TOLERANCE:-2}" \
  --distance_fusion_weight "${DISTANCE_FUSION_WEIGHT:-0.20}" \
  --distance_temperature "${DISTANCE_TEMPERATURE:-4.0}" \
  "${TTA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[OK] Step 13 CHFS test started"
echo "     checkpoint: ${PTH_PATH}"
echo "     TTA:        ${TEST_TTA:-1}"
echo "     fusion:     ${DISTANCE_FUSION_WEIGHT:-0.20}"
echo "     pid:        $(cat "${PID_FILE}")"
echo "     log:        ${LOG_FILE}"
