#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
ABLATION_ROOT="${PROJECT_DIR}/Ours/ablation_sbg"

PYTHON_BIN="/opt/conda/envs/BGDiff/bin/python"

GPU_ID="${GPU_ID:-3}"

# ============================================================
# Official ISIC2018 train real image root
# ============================================================
REAL_ROOT="/data/zjy_work/ISIC2018/train"

# 用于 CLIP-I / LPIPS 的配对真实图路径。
# 如果你的这些生成结果都是基于 Real-5 生成的，就使用这个。
PROMPT_JSON="${PROMPT_JSON:-${PROJECT_DIR}/data/prompt_train_5p_seed0.json}"

OUT_ROOT="${ABLATION_ROOT}/eval_outputs/isic2018_train_metrics"
WORK_DIR="${OUT_ROOT}/work"
LOG_ROOT="${OUT_ROOT}/logs"
SUMMARY_CSV="${OUT_ROOT}/summary_metrics.csv"
SUMMARY_TSV="${OUT_ROOT}/summary_metrics.tsv"

mkdir -p "${OUT_ROOT}" "${WORK_DIR}" "${LOG_ROOT}"

cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PROJECT_DIR}:${ABLATION_ROOT}/code:${PYTHONPATH}"

echo "============================================================"
echo "[CONFIG]"
echo "PROJECT_DIR = ${PROJECT_DIR}"
echo "REAL_ROOT   = ${REAL_ROOT}"
echo "PROMPT_JSON = ${PROMPT_JSON}"
echo "GPU_ID      = ${GPU_ID}"
echo "OUT_ROOT    = ${OUT_ROOT}"
echo "============================================================"

if [ ! -d "${REAL_ROOT}" ]; then
  echo "[ERROR] REAL_ROOT does not exist: ${REAL_ROOT}"
  exit 1
fi

if [ ! -f "${ABLATION_ROOT}/eval/eval_generation_metrics_isic.py" ]; then
  echo "[ERROR] evaluator not found:"
  echo "  ${ABLATION_ROOT}/eval/eval_generation_metrics_isic.py"
  exit 1
fi

# 重新生成总表，避免旧结果混入
rm -f "${SUMMARY_CSV}" "${SUMMARY_TSV}"

run_one () {
  METHOD="$1"
  RESULT_DIR="$2"

  LOG_FILE="${LOG_ROOT}/${METHOD}_$(date +%Y%m%d_%H%M%S).log"

  echo "============================================================"
  echo "[RUN] ${METHOD}"
  echo "result_dir = ${RESULT_DIR}"
  echo "log        = ${LOG_FILE}"

  if [ ! -d "${RESULT_DIR}" ]; then
    echo "[ERROR] missing result dir: ${RESULT_DIR}"
    return
  fi

  "${PYTHON_BIN}" -u "${ABLATION_ROOT}/eval/eval_generation_metrics_isic.py" \
    --method "${METHOD}" \
    --real_root "${REAL_ROOT}" \
    --result_dir "${RESULT_DIR}" \
    --prompt_json "${PROMPT_JSON}" \
    --work_dir "${WORK_DIR}" \
    --out_csv "${SUMMARY_CSV}" \
    --device "cuda:0" \
    --batch_size 32 \
    --fid_size 299 \
    --pair_size 256 \
    --max_real_images 0 \
    --max_syn_images 0 \
    --clip_model_name "ViT-B-32" \
    --clip_pretrained "openai" \
    --mos_model "musiq" \
    > "${LOG_FILE}" 2>&1

  echo "[DONE] ${METHOD}"
}

# ============================================================
# Your generated result directories
# ============================================================

run_one "decoder_mod" \
  "/data/zjy_work/BGDiff/Ours/ablation_sbg/results/decoder_mod_real5_syn10"

run_one "hard" \
  "/data/zjy_work/BGDiff/Ours/ablation_sbg/results/hard_real5_syn10"

run_one "maskonly" \
  "/data/zjy_work/BGDiff/Ours/ablation_sbg/results/maskonly_real5_syn10"

run_one "progressive" \
  "/data/zjy_work/BGDiff/Ours/ablation_sbg/results/progressive_real5_syn10"

run_one "softonly" \
  "/data/zjy_work/BGDiff/Ours/ablation_sbg/results/softonly_real5_syn10"

if [ -f "${SUMMARY_CSV}" ]; then
  tr ',' '\t' < "${SUMMARY_CSV}" > "${SUMMARY_TSV}"

  echo "============================================================"
  echo "[ALL DONE]"
  echo "Summary CSV: ${SUMMARY_CSV}"
  echo "Summary TSV: ${SUMMARY_TSV}"
  echo ""
  cat "${SUMMARY_TSV}"
else
  echo "[ERROR] summary csv not generated."
  exit 1
fi


# GPU_ID=3 PROMPT_JSON=/data/zjy_work/BGDiff/data/prompt_train_5p_seed0.json bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/07_eval_all_given_results.sh
