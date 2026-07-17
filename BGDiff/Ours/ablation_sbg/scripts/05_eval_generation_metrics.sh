#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
ABLATION_ROOT="${PROJECT_DIR}/Ours/ablation_sbg"

LOG_ROOT="${ABLATION_ROOT}/logs/eval"
METRIC_ROOT="${ABLATION_ROOT}/eval_outputs/generation_metrics"
WORK_ROOT="${ABLATION_ROOT}/eval_outputs/metric_inputs"

# ============================================================
# User-editable settings
# ============================================================
# VARIANT:
#   maskonly
#   hard
#   softonly
#   progressive
#   decoder_mod
#   full_sbg
#   siamese
#
# DATA_SPLIT:
#   real5
#   real100
# ============================================================
VARIANT="${VARIANT:-full_sbg}"
DATA_SPLIT="${DATA_SPLIT:-real5}"
GPU_ID="${GPU_ID:-3}"

EVAL_SIZE="${EVAL_SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-16}"

MOS_MODEL="${MOS_MODEL:-musiq}"

PYTHON_BIN="/opt/conda/envs/BGDiff/bin/python"

mkdir -p "${LOG_ROOT}"
mkdir -p "${METRIC_ROOT}"
mkdir -p "${WORK_ROOT}"

cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ABLATION_ROOT}/code:${PYTHONPATH}"

# ============================================================
# Resolve output tag and prompt json
# ============================================================
if [ "${DATA_SPLIT}" = "real5" ]; then
  PROMPT_JSON="${PROJECT_DIR}/data/prompt_train_5p_seed0.json"
  OUT_TAG="${VARIANT}_real5_syn10"
elif [ "${DATA_SPLIT}" = "real100" ]; then
  PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"
  OUT_TAG="${VARIANT}_real100_syn2"
else
  echo "[ERROR] Unknown DATA_SPLIT=${DATA_SPLIT}"
  exit 1
fi

# Siamese-Diffusion 如果你整理成这个目录，也可以直接评估
if [ "${VARIANT}" = "siamese" ]; then
  if [ "${DATA_SPLIT}" = "real5" ]; then
    OUT_TAG="siamese_real5_syn10"
  else
    OUT_TAG="siamese_real100_syn2"
  fi
fi

REAL_DIR="${ABLATION_ROOT}/eval_outputs/real_train_images"
SYN_DIR="${ABLATION_ROOT}/results/${OUT_TAG}/images"

if [ ! -d "${REAL_DIR}" ]; then
  echo "[ERROR] REAL_DIR does not exist:"
  echo "        ${REAL_DIR}"
  echo "Please run:"
  echo "        bash ${ABLATION_ROOT}/scripts/04_collect_real_images.sh"
  exit 1
fi

if [ ! -d "${SYN_DIR}" ]; then
  echo "[ERROR] SYN_DIR does not exist:"
  echo "        ${SYN_DIR}"
  exit 1
fi

OUT_CSV="${METRIC_ROOT}/${OUT_TAG}_metrics.csv"
OUT_JSON="${METRIC_ROOT}/${OUT_TAG}_metrics.json"
WORK_DIR="${WORK_ROOT}/${OUT_TAG}"

LOG_FILE="${LOG_ROOT}/metrics_${OUT_TAG}_$(date +%Y%m%d_%H%M%S).log"

nohup "${PYTHON_BIN}" -u "${ABLATION_ROOT}/eval/eval_generation_metrics.py" \
  --real_dir "${REAL_DIR}" \
  --syn_dir "${SYN_DIR}" \
  --prompt_json "${PROMPT_JSON}" \
  --work_dir "${WORK_DIR}" \
  --out_csv "${OUT_CSV}" \
  --out_json "${OUT_JSON}" \
  --size "${EVAL_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --device cuda:0 \
  --mos_model "${MOS_MODEL}" \
  > "${LOG_FILE}" 2>&1 &

echo "[OK] Generation metric evaluation started."
echo "     variant:     ${VARIANT}"
echo "     split:       ${DATA_SPLIT}"
echo "     physical gpu:${GPU_ID}"
echo "     device:      cuda:0"
echo "     real_dir:    ${REAL_DIR}"
echo "     syn_dir:     ${SYN_DIR}"
echo "     prompt_json: ${PROMPT_JSON}"
echo "     out_csv:     ${OUT_CSV}"
echo "     out_json:    ${OUT_JSON}"
echo "     log:         ${LOG_FILE}"
