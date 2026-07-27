#!/usr/bin/env bash
set -euo pipefail

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

if [ ! -f "${SAM_BASE_CHECKPOINT}" ]; then
  echo "[ERROR] Missing SAM checkpoint: ${SAM_BASE_CHECKPOINT}"
  echo "Run scripts/00_install_hetero_teacher_deps.sh first."
  exit 1
fi

SAM_EPOCHS="${SAM_EPOCHS:-120}"
SAM_BATCHSIZE="${SAM_BATCHSIZE:-1}"
SAM_LR="${SAM_LR:-2e-4}"
SAM_NUM_WORKERS="${SAM_NUM_WORKERS:-4}"
SAM_UNFREEZE_LAST_BLOCKS="${SAM_UNFREEZE_LAST_BLOCKS:-0}"
SAM_ADAPTER_BOTTLENECK="${SAM_ADAPTER_BOTTLENECK:-64}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-$((N_FOLDS - 1))}"
export SAM_EPOCHS SAM_BATCHSIZE SAM_LR SAM_NUM_WORKERS SAM_UNFREEZE_LAST_BLOCKS SAM_ADAPTER_BOTTLENECK START_FOLD END_FOLD
mkdir -p "${SAM_TEACHER_CKPT_ROOT}" "${LOG_ROOT}/teacher"

LOG_FILE="${LOG_ROOT}/teacher/train_sam_oof_${RATIO_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/02c_train_sam_oof_${RATIO_TAG}_seed${SEED}.pid"

nohup bash -c '
set -euo pipefail
source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
export PYTHONPATH="/data/zjy_work:${PYTHONPATH:-}"
for fold in $(seq "${START_FOLD}" "${END_FOLD}"); do
  train_list="${SPLIT_DIR}/low_${RATIO_TAG}_fold${fold}_train.txt"
  val_list="${SPLIT_DIR}/low_${RATIO_TAG}_fold${fold}_val.txt"
  save_dir="${SAM_TEACHER_CKPT_ROOT}/fold${fold}"
  mkdir -p "${save_dir}"
  python -u "${WORK_ROOT}/teacher/train_sam_adapter_teacher.py" \
    --image_dir "${ISIC_TRAIN_IMAGE_DIR}" \
    --mask_dir "${ISIC_TRAIN_MASK_DIR}" \
    --train_list "${train_list}" \
    --val_list "${val_list}" \
    --save_dir "${save_dir}" \
    --epochs "${SAM_EPOCHS}" \
    --batch_size "${SAM_BATCHSIZE}" \
    --num_workers "${SAM_NUM_WORKERS}" \
    --lr "${SAM_LR}" \
    --sam_type vit_b \
    --sam_base_checkpoint "${SAM_BASE_CHECKPOINT}" \
    --sam_unfreeze_last_blocks "${SAM_UNFREEZE_LAST_BLOCKS}" \
    --sam_adapter_bottleneck "${SAM_ADAPTER_BOTTLENECK}" \
    --image_size 1024 \
    --seed "$((SEED + fold + 100))" \
    --device cuda:0 \
    --amp
  echo "[OK] SAM adapter fold ${fold} finished"
done
' > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[OK] Step 02c started: SAM adapter OOF teachers"
echo "     pid: $(cat "${PID_FILE}")"
echo "     log: ${LOG_FILE}"
