#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
ABLATION_ROOT="${PROJECT_DIR}/Ours/ablation_sbg"
LOG_ROOT="${ABLATION_ROOT}/logs/eval"

# ============================================================
# User-editable settings
# ============================================================
# VARIANT examples:
#   maskonly
#   hard
#   softonly
#   progressive
#   decoder_mod
#   full_sbg
#
# DATA_SPLIT:
#   real5
#   real100
# ============================================================
VARIANT="${VARIANT:-full_sbg}"
DATA_SPLIT="${DATA_SPLIT:-real5}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_DIR}"

if [ "${DATA_SPLIT}" = "real5" ]; then
  OUT_TAG="${VARIANT}_real5_syn10"
elif [ "${DATA_SPLIT}" = "real100" ]; then
  OUT_TAG="${VARIANT}_real100_syn2"
else
  echo "[ERROR] Unknown DATA_SPLIT=${DATA_SPLIT}"
  exit 1
fi

IMAGE_DIR="${ABLATION_ROOT}/results/${OUT_TAG}/images"
MASK_DIR="${ABLATION_ROOT}/results/${OUT_TAG}/masks"
OUT_CSV="${ABLATION_ROOT}/eval_outputs/boundary_metrics/${OUT_TAG}.csv"

if [ ! -d "${IMAGE_DIR}" ]; then
  echo "[ERROR] IMAGE_DIR does not exist: ${IMAGE_DIR}"
  exit 1
fi

if [ ! -d "${MASK_DIR}" ]; then
  echo "[ERROR] MASK_DIR does not exist: ${MASK_DIR}"
  exit 1
fi

nohup python -u "${ABLATION_ROOT}/eval/eval_boundary_naturalness.py" \
  --image_dir "${IMAGE_DIR}" \
  --mask_dir "${MASK_DIR}" \
  --out_csv "${OUT_CSV}" \
  --radius 12 \
  --tau 4.0 \
  --kernel_size 3 \
  > "${LOG_ROOT}/boundary_metrics_${OUT_TAG}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] Boundary naturalness evaluation started."
echo "     variant: ${VARIANT}"
echo "     split:   ${DATA_SPLIT}"
echo "     images:  ${IMAGE_DIR}"
echo "     masks:   ${MASK_DIR}"
echo "     out_csv: ${OUT_CSV}"
echo "     logs:    ${LOG_ROOT}"




# VARIANT=maskonly DATA_SPLIT=real5 \
# bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/06_boundary_metrics_variant.sh

# VARIANT=hard DATA_SPLIT=real5 \
# bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/06_boundary_metrics_variant.sh

# VARIANT=full_sbg DATA_SPLIT=real5 \
# bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/06_boundary_metrics_variant.sh

