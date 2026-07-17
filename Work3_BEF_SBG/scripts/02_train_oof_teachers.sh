#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh
# Training a new standard BGDNet teacher.
export BGDNET_SWIN_VARIANT="standard"
export BGDNET_LOAD_BACKBONE_PRETRAIN="1"

# Fail if no pretrained Swin tensors are matched.
export BGDNET_STRICT_PRETRAIN="1"

mkdir -p "${TEACHER_CKPT_ROOT}"
mkdir -p "${LOG_ROOT}/teacher"

# ============================================================
# Teacher OOF training config
# ============================================================

# 默认 batchsize=4。
# 如果 batchsize=4 仍不稳定，可临时 TEACHER_BATCHSIZE=2 或 1。
export TEACHER_BATCHSIZE="${TEACHER_BATCHSIZE:-4}"
export TEACHER_EPOCH="${TEACHER_EPOCH:-200}"

# Fold range, useful for debugging or resuming.
export START_FOLD="${START_FOLD:-0}"
export END_FOLD="${END_FOLD:-$((N_FOLDS - 1))}"

# 当前环境 cuDNN 初始化不稳定，BGDNet 默认禁用 cuDNN。
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"

# batchsize=4 时不冻结 BN。
# batchsize=1 时建议手动设置 BGDNET_FREEZE_BN=1。
export BGDNET_FREEZE_BN="${BGDNET_FREEZE_BN:-0}"

# IMPORTANT:
# Do NOT use expandable_segments=True here.
# It can trigger:
#   !block->expandable_segment_ INTERNAL ASSERT FAILED
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# Lazy CUDA module loading can reduce initial CUDA pressure.
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

# Avoid very long C++ symbolization logs.
export TORCH_DISABLE_ADDR2LINE=1
unset TORCH_SHOW_CPP_STACKTRACES

LOG_FILE="${LOG_ROOT}/teacher/train_oof_teachers_${RATIO_TAG}_bs${TEACHER_BATCHSIZE}_cudnn${BGDNET_DISABLE_CUDNN}_freezebn${BGDNET_FREEZE_BN}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/02_train_oof_teachers_${RATIO_TAG}.pid"

nohup bash -c '
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

cd "${BGDNET_ROOT}"

echo "[Teacher OOF Training]"
echo "  RATIO_TAG=${RATIO_TAG}"
echo "  SPLIT_DIR=${SPLIT_DIR}"
echo "  TEACHER_CKPT_ROOT=${TEACHER_CKPT_ROOT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  TEACHER_BATCHSIZE=${TEACHER_BATCHSIZE}"
echo "  TEACHER_EPOCH=${TEACHER_EPOCH}"
echo "  START_FOLD=${START_FOLD}"
echo "  END_FOLD=${END_FOLD}"
echo "  BGDNET_DISABLE_CUDNN=${BGDNET_DISABLE_CUDNN}"
echo "  BGDNET_FREEZE_BN=${BGDNET_FREEZE_BN}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "  CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING}"

for fold in $(seq ${START_FOLD} ${END_FOLD}); do
  TRAIN_LIST="${SPLIT_DIR}/low_${RATIO_TAG}_fold${fold}_train.txt"
  VAL_LIST="${SPLIT_DIR}/low_${RATIO_TAG}_fold${fold}_val.txt"
  SAVE_DIR="${TEACHER_CKPT_ROOT}/fold${fold}"

  if [ ! -f "${TRAIN_LIST}" ]; then
    echo "[ERROR] Missing train list: ${TRAIN_LIST}"
    exit 1
  fi

  if [ ! -f "${VAL_LIST}" ]; then
    echo "[ERROR] Missing val list: ${VAL_LIST}"
    exit 1
  fi

  mkdir -p "${SAVE_DIR}"

  echo "============================================================"
  echo "[Fold ${fold}]"
  echo "  train_list=${TRAIN_LIST}"
  echo "  val_list=${VAL_LIST}"
  echo "  save_dir=${SAVE_DIR}"
  echo "============================================================"

  python -u "${BGDNET_ROOT}/BGDiff_train.py" \
    --epoch "${TEACHER_EPOCH}" \
    --lr 1e-4 \
    --optimizer AdamW \
    --batchsize "${TEACHER_BATCHSIZE}" \
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
    --beta 0.4

  echo "[OK] Fold ${fold} finished."
done

echo "[DONE] Selected OOF teacher folds finished."
' > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 02 started: train OOF teachers"
echo "     ratio:      ${RATIO_TAG}"
echo "     gpu:        ${CUDA_VISIBLE_DEVICES}"
echo "     batchsize:  ${TEACHER_BATCHSIZE}"
echo "     epoch:      ${TEACHER_EPOCH}"
echo "     fold range: ${START_FOLD}-${END_FOLD}"
echo "     cudnn off:  ${BGDNET_DISABLE_CUDNN}"
echo "     freeze bn:  ${BGDNET_FREEZE_BN}"
echo "     allocator:  ${PYTORCH_CUDA_ALLOC_CONF}"
echo "     pid:        $(cat ${PID_FILE})"
echo "     log:        ${LOG_FILE}"
echo "     kill:       kill \$(cat ${PID_FILE})"
