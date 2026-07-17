#!/usr/bin/env bash
set -e

# ============================================================
# Step 13: Test final BGDNet-BEF
# ============================================================

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

mkdir -p "${LOG_ROOT}/segmentation"
mkdir -p "${PID_DIR}"
mkdir -p "${FINAL_EVAL_DIR}"

# -----------------------------
# CUDA / stability settings
# -----------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# BGDNet testing also disables cuDNN in this environment.
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"


# -----------------------------
# Test settings
# -----------------------------
PTH_PATH="${PTH_PATH:-${FINAL_SEG_SAVE_DIR}/BGDNet-BEF-best.pth}"

TEST_IMG_SIZE="${TEST_IMG_SIZE:-352}"
TEST_DEVICE="${TEST_DEVICE:-cuda:0}"

# Segmentation threshold.
# For formal test, keep 0.5 unless you selected a threshold on validation set.
TEST_THRESHOLD="${TEST_THRESHOLD:-0.5}"

# Boundary head threshold.
BOUNDARY_THRESHOLD="${BOUNDARY_THRESHOLD:-0.5}"

# Base tolerance on 352x352 scale.
# test_BGDNet_BEF.py will scale tolerance for original-resolution metrics.
BOUNDARY_TOLERANCE="${BOUNDARY_TOLERANCE:-2}"

# Optional test list.
# Leave empty to evaluate all images under ISIC_TEST_ROOT/Images and Masks.
FINAL_TEST_LIST="${FINAL_TEST_LIST:-}"

if [ ! -f "${PTH_PATH}" ]; then
  echo "[ERROR] checkpoint not found:"
  echo "        ${PTH_PATH}"
  echo ""
  echo "Please check FINAL_SEG_SAVE_DIR or pass PTH_PATH manually, for example:"
  echo "  PTH_PATH=/path/to/BGDNet-BEF-best.pth bash /data/zjy_work/Work3_BEF_SBG/scripts/13_test_final_bgdnet_bef.sh"
  exit 1
fi

TEST_LIST_ARGS=()
if [ -n "${FINAL_TEST_LIST}" ]; then
  if [ ! -f "${FINAL_TEST_LIST}" ]; then
    echo "[ERROR] FINAL_TEST_LIST does not exist: ${FINAL_TEST_LIST}"
    exit 1
  fi
  TEST_LIST_ARGS=(--test_list "${FINAL_TEST_LIST}")
fi

# -----------------------------
# Log / PID
# -----------------------------
LOG_FILE="${LOG_ROOT}/segmentation/test_final_bgdnet_bef_${RATIO_TAG}_seed${SEED}_th${TEST_THRESHOLD}_bth${BOUNDARY_THRESHOLD}_tol${BOUNDARY_TOLERANCE}_cudnn${BGDNET_DISABLE_CUDNN}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/13_test_final_bgdnet_bef_${RATIO_TAG}.pid"

# -----------------------------
# Start testing
# -----------------------------
nohup python -u "${WORK_ROOT}/segmentation/test_BGDNet_BEF.py" \
  --pth_path "${PTH_PATH}" \
  --test_data_path "${ISIC_TEST_ROOT}" \
  "${TEST_LIST_ARGS[@]}" \
  --out_dir "${FINAL_EVAL_DIR}" \
  --testsize "${TEST_IMG_SIZE}" \
  --device "${TEST_DEVICE}" \
  --threshold "${TEST_THRESHOLD}" \
  --boundary_threshold "${BOUNDARY_THRESHOLD}" \
  --boundary_tolerance "${BOUNDARY_TOLERANCE}" \
  ${TEST_TTA:+--tta} \
  ${TEST_LCC:+--postprocess_lcc} \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 13 started: test final BGDNet-BEF"
echo "     ratio:              ${RATIO_TAG}"
echo "     seed:               ${SEED}"
echo "     gpu:                ${CUDA_VISIBLE_DEVICES}"
echo "     cudnn off:          ${BGDNET_DISABLE_CUDNN}"
echo "     pth:                ${PTH_PATH}"
echo "     test_root:          ${ISIC_TEST_ROOT}"
echo "     test_list:          ${FINAL_TEST_LIST:-<all test images>}"
echo "     out_dir:            ${FINAL_EVAL_DIR}"
echo "     testsize:           ${TEST_IMG_SIZE}"
echo "     threshold:          ${TEST_THRESHOLD}"
echo "     boundary_threshold: ${BOUNDARY_THRESHOLD}"
echo "     boundary_tolerance: ${BOUNDARY_TOLERANCE}"
echo "     pid:                $(cat ${PID_FILE})"
echo "     log:                ${LOG_FILE}"
echo "     kill:               kill \$(cat ${PID_FILE})"
echo ""
echo "Check log:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "After finished, check summary:"
echo "  cat ${FINAL_EVAL_DIR}/summary_metrics.csv"
