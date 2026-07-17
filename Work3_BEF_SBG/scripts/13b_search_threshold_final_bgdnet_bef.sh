#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

TH_LIST="${TH_LIST:-0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70}"

for th in ${TH_LIST}; do
  echo "============================================================"
  echo "[Threshold Search] threshold=${th}"
  echo "============================================================"

  RATIO_TAG="${RATIO_TAG}" SEED="${SEED}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  TEST_THRESHOLD="${th}" \
  BOUNDARY_THRESHOLD="${BOUNDARY_THRESHOLD:-0.5}" \
  BOUNDARY_TOLERANCE="${BOUNDARY_TOLERANCE:-2}" \
  TEST_TTA="${TEST_TTA:-1}" \
  TEST_LCC="${TEST_LCC:-1}" \
  BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}" \
  CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}" \
  bash /data/zjy_work/Work3_BEF_SBG/scripts/13_test_final_bgdnet_bef.sh

  echo ""
done