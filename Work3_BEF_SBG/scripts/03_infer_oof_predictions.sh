#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh
# Load complete standard teacher checkpoints.
export BGDNET_SWIN_VARIANT="standard"
export BGDNET_LOAD_BACKBONE_PRETRAIN="0"
export BGDNET_STRICT_PRETRAIN="0"

mkdir -p "${OOF_OUT_DIR}"
mkdir -p "${LOG_ROOT}/infer"

# BGDNet inference also disables cuDNN for this environment.
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

LOG_FILE="${LOG_ROOT}/infer/infer_oof_predictions_${RATIO_TAG}_cudnn${BGDNET_DISABLE_CUDNN}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/03_infer_oof_predictions_${RATIO_TAG}.pid"

nohup python -u "${WORK_ROOT}/teacher/infer_oof_predictions.py" \
  --bgdnet_root "${BGDNET_ROOT}" \
  --test_data_path "${ISIC_TRAIN_ROOT}" \
  --split_dir "${SPLIT_DIR}" \
  --ratio_tag "${RATIO_TAG}" \
  --n_folds "${N_FOLDS}" \
  --ckpt_root "${TEACHER_CKPT_ROOT}" \
  --out_dir "${OOF_OUT_DIR}" \
  --testsize 352 \
  --device cuda:0 \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 03 started: infer OOF teacher predictions"
echo "     ratio:     ${RATIO_TAG}"
echo "     gpu:       ${CUDA_VISIBLE_DEVICES}"
echo "     cudnn off: ${BGDNET_DISABLE_CUDNN}"
echo "     pid:       $(cat ${PID_FILE})"
echo "     log:       ${LOG_FILE}"
echo "     kill:      kill \$(cat ${PID_FILE})"
