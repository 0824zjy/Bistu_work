#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# BEF-SBG Work3 Stage-1:
# 5-fold OOF teacher training with BGDNet
# ============================================================

WORK_ROOT="${WORK_ROOT:-/data/zjy_work/Work3_BEF_SBG}"
BGDNET_ROOT="${BGDNET_ROOT:-/data/zjy_work/BGDNet}"
ISIC_TRAIN_ROOT="${ISIC_TRAIN_ROOT:-/data/zjy_work/ISIC2018/train}"

SPLIT_DIR="${SPLIT_DIR:-${WORK_ROOT}/splits}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs/teacher}"

# 默认跑 5% 低标注 teacher。
# 可通过 RATIO_TAG=10p / 20p / 100p 修改。
RATIO_TAG="${RATIO_TAG:-5p}"
N_FOLDS="${N_FOLDS:-5}"

CKPT_ROOT="${CKPT_ROOT:-${WORK_ROOT}/teacher/checkpoints/ISIC2018_${RATIO_TAG}}"

# 支持外部指定 GPU：
#   CUDA_VISIBLE_DEVICES=0 bash train_oof_teachers.sh
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

mkdir -p "${CKPT_ROOT}"
mkdir -p "${LOG_DIR}"

echo "[BEF-SBG Teacher Training]"
echo "  WORK_ROOT           = ${WORK_ROOT}"
echo "  BGDNET_ROOT         = ${BGDNET_ROOT}"
echo "  ISIC_TRAIN_ROOT     = ${ISIC_TRAIN_ROOT}"
echo "  SPLIT_DIR           = ${SPLIT_DIR}"
echo "  RATIO_TAG           = ${RATIO_TAG}"
echo "  N_FOLDS             = ${N_FOLDS}"
echo "  CKPT_ROOT           = ${CKPT_ROOT}"
echo "  CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES}"

cd "${BGDNET_ROOT}"

for fold in $(seq 0 $((N_FOLDS - 1))); do
  TRAIN_LIST="${SPLIT_DIR}/low_${RATIO_TAG}_fold${fold}_train.txt"
  VAL_LIST="${SPLIT_DIR}/low_${RATIO_TAG}_fold${fold}_val.txt"
  SAVE_DIR="${CKPT_ROOT}/fold${fold}"
  LOG_FILE="${LOG_DIR}/train_teacher_ISIC2018_${RATIO_TAG}_fold${fold}.log"

  if [[ ! -f "${TRAIN_LIST}" ]]; then
    echo "[ERROR] Missing train list: ${TRAIN_LIST}"
    exit 1
  fi

  if [[ ! -f "${VAL_LIST}" ]]; then
    echo "[ERROR] Missing val list: ${VAL_LIST}"
    exit 1
  fi

  mkdir -p "${SAVE_DIR}"

  echo "============================================================"
  echo "[Fold ${fold}]"
  echo "  train_list = ${TRAIN_LIST}"
  echo "  val_list   = ${VAL_LIST}"
  echo "  save_dir   = ${SAVE_DIR}"
  echo "  log_file   = ${LOG_FILE}"
  echo "============================================================"

  python -u "${BGDNET_ROOT}/BGDiff_train.py" \
    --epoch 200 \
    --lr 1e-4 \
    --optimizer AdamW \
    --batchsize 4 \
    --img_size 352 \
    --clip 0.5 \
    --decay_rate 0.1 \
    --decay_epoch 200 \
    --train_path1 "${ISIC_TRAIN_ROOT}" \
    --test_path "${ISIC_TRAIN_ROOT}" \
    --train_list1 "${TRAIN_LIST}" \
    --test_list "${VAL_LIST}" \
    --train_save "${SAVE_DIR}" \
    --alpha 1.0 \
    --beta 0.4 \
    2>&1 | tee "${LOG_FILE}"

  echo "[OK] Fold ${fold} finished."
done

echo "[DONE] All OOF teacher folds finished."
