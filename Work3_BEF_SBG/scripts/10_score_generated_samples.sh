#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

mkdir -p "${LOG_ROOT}/scoring"

# Generated sample scoring uses new standard BGDNet teachers.
export BGDNET_SWIN_VARIANT="standard"

# A complete teacher checkpoint will be loaded.
export BGDNET_LOAD_BACKBONE_PRETRAIN="0"
export BGDNET_STRICT_PRETRAIN="0"

export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# Default quality model.
# You can override:
#   QUALITY_MODEL_PTH=/path/to/model.pth bash 10_score_generated_samples.sh
export QUALITY_MODEL_PTH="${QUALITY_MODEL_PTH:-${TEACHER_CKPT_ROOT}/fold0/BGDNet-best.pth}"
if [ -z "${QUALITY_MODEL_PTHS:-}" ]; then
  QUALITY_MODEL_PTHS=""
  for fold in $(seq 0 $((N_FOLDS - 1))); do
    P="${TEACHER_CKPT_ROOT}/fold${fold}/BGDNet-best.pth"
    if [ -f "${P}" ]; then
      if [ -z "${QUALITY_MODEL_PTHS}" ]; then
        QUALITY_MODEL_PTHS="${P}"
      else
        QUALITY_MODEL_PTHS="${QUALITY_MODEL_PTHS},${P}"
      fi
    fi
  done
fi

export QUALITY_MODEL_PTHS
LOG_FILE="${LOG_ROOT}/scoring/score_generated_${RATIO_TAG}_cudnn${BGDNET_DISABLE_CUDNN}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/10_score_generated_samples_${RATIO_TAG}.pid"

nohup python -u "${WORK_ROOT}/scoring/score_generated_samples.py" \
  --gen_image_dir "${GEN_OUT_DIR}/images" \
  --gen_mask_dir "${GEN_OUT_DIR}/masks" \
  --gen_prior_dir "${GEN_OUT_DIR}/boundary_prior" \
  --quality_model_pth "${QUALITY_MODEL_PTH}" \
  --quality_model_pths "${QUALITY_MODEL_PTHS}" \
  --out_csv "${GEN_SCORE_CSV}" \
  --out_jsonl "${GEN_ACCEPTED_JSONL}" \
  --device cuda:0 \
  --img_size 352 \
  --cons_threshold "${CONS_THRESHOLD:-0.80}" \
  --beta_hard "${BETA_HARD:-0.5}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 10 started: score generated samples"
echo "     ratio:     ${RATIO_TAG}"
echo "     gpu:       ${CUDA_VISIBLE_DEVICES}"
echo "     cudnn off: ${BGDNET_DISABLE_CUDNN}"
echo "     quality:   ${QUALITY_MODEL_PTH}"
echo "     csv:       ${GEN_SCORE_CSV}"
echo "     jsonl:     ${GEN_ACCEPTED_JSONL}"
echo "     pid:       $(cat ${PID_FILE})"
echo "     log:       ${LOG_FILE}"
echo "     kill:      kill \$(cat ${PID_FILE})"
