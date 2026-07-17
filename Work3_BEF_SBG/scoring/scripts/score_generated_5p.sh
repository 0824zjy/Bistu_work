#!/usr/bin/env bash
set -e

WORK_ROOT="/data/zjy_work/Work3_BEF_SBG"

LOG_DIR="${WORK_ROOT}/logs/scoring"
PID_DIR="${WORK_ROOT}/logs/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}" "${WORK_ROOT}/results"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 建议使用第一阶段 OOF teacher 中某个表现稳定的 BGDNet 权重，
# 也可以改成工作一/二中训练好的质量评估模型。
QUALITY_MODEL_PTH="${QUALITY_MODEL_PTH:-${WORK_ROOT}/teacher/checkpoints/ISIC2018_5p/fold0/BGDNet-best.pth}"

LOG_FILE="${LOG_DIR}/score_generated_5p_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/score_generated_5p.pid"

nohup python -u "${WORK_ROOT}/scoring/score_generated_samples.py" \
  --gen_image_dir "${WORK_ROOT}/results/generated_bef_sbg/images" \
  --gen_mask_dir "${WORK_ROOT}/results/generated_bef_sbg/masks" \
  --gen_prior_dir "${WORK_ROOT}/results/generated_bef_sbg/boundary_prior" \
  --quality_model_pth "${QUALITY_MODEL_PTH}" \
  --out_csv "${WORK_ROOT}/results/generated_bef_sbg_scores_5p.csv" \
  --out_jsonl "${WORK_ROOT}/results/generated_bef_sbg_accepted_5p.jsonl" \
  --device cuda:0 \
  --img_size 352 \
  --cons_threshold 0.75 \
  --beta_hard 0.5 \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] generated sample scoring started."
echo "     gpu:     ${CUDA_VISIBLE_DEVICES}"
echo "     quality: ${QUALITY_MODEL_PTH}"
echo "     pid:     $(cat ${PID_FILE})"
echo "     log:     ${LOG_FILE}"
echo "     kill:    kill \$(cat ${PID_FILE})"
