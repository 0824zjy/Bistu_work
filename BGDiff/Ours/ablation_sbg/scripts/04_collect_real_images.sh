#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
ABLATION_ROOT="${PROJECT_DIR}/Ours/ablation_sbg"
LOG_ROOT="${ABLATION_ROOT}/logs/eval"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_DIR}"

export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"
export OUT_DIR="${ABLATION_ROOT}/eval_outputs/real_train_images"

nohup python -u "${ABLATION_ROOT}/eval/collect_real_images.py" \
  --prompt_json "${PROMPT_JSON}" \
  --out_dir "${OUT_DIR}" \
  > "${LOG_ROOT}/collect_real_images_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] Collecting real images started."
echo "     prompt_json: ${PROMPT_JSON}"
echo "     out_dir:     ${OUT_DIR}"
echo "     logs:        ${LOG_ROOT}"
