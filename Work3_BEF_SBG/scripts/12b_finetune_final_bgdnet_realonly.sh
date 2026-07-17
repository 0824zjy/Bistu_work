#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

mkdir -p "${LOG_ROOT}/segmentation"
mkdir -p "${PID_DIR}"

REAL_ONLY_JSON="${WORK_ROOT}/results/real_only_train_${RATIO_TAG}.jsonl"
REAL_LIST="${REAL_LIST:-${SPLIT_DIR}/low_${RATIO_TAG}_all.txt}"

BASE_CKPT="${BASE_CKPT:-${FINAL_SEG_SAVE_DIR}/BGDNet-BEF-best.pth}"
FT_SAVE_DIR="${FINAL_SEG_SAVE_DIR}_realonly_ft"

if [ ! -f "${BASE_CKPT}" ]; then
  echo "[ERROR] base checkpoint not found:"
  echo "        ${BASE_CKPT}"
  exit 1
fi

mkdir -p "${FT_SAVE_DIR}"

# Build real-only jsonl.
python -u "${WORK_ROOT}/scoring/make_weighted_seg_dataset.py" \
  --real_image_dir "${ISIC_TRAIN_IMAGE_DIR}" \
  --real_mask_dir "${ISIC_TRAIN_MASK_DIR}" \
  --real_list_txt "${REAL_LIST}" \
  --gen_jsonl "/tmp/non_existing_bef_sbg_gen.jsonl" \
  --out_jsonl "${REAL_ONLY_JSON}" \
  --synthetic_ratio 0.0 \
  --sort_gen_by_weight

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"
export BGDNET_FREEZE_BN="${BGDNET_FREEZE_BN:-0}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

FT_EPOCH="${FT_EPOCH:-40}"
FT_BATCHSIZE="${FT_BATCHSIZE:-4}"
FT_LR="${FT_LR:-1e-5}"
FT_BETA="${FT_BETA:-0.25}"
FT_AUG="${FT_AUG:-True}"
FT_IMG_SIZE="${FT_IMG_SIZE:-352}"
FT_NUM_WORKERS="${FT_NUM_WORKERS:-8}"

LOG_FILE="${LOG_ROOT}/segmentation/finetune_realonly_${RATIO_TAG}_seed${SEED}_lr${FT_LR}_epoch${FT_EPOCH}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/12b_finetune_realonly_${RATIO_TAG}.pid"

nohup python -u "${WORK_ROOT}/segmentation/train_BGDNet_BEF.py" \
  --weighted_train_json "${REAL_ONLY_JSON}" \
  --test_path "${ISIC_TEST_ROOT}" \
  --epoch "${FT_EPOCH}" \
  --lr "${FT_LR}" \
  --batchsize "${FT_BATCHSIZE}" \
  --img_size "${FT_IMG_SIZE}" \
  --train_save "${FT_SAVE_DIR}" \
  --alpha 1.0 \
  --beta "${FT_BETA}" \
  --clip 0.5 \
  --augmentation "${FT_AUG}" \
  --num_workers "${FT_NUM_WORKERS}" \
  --optimizer AdamW \
  --decay_rate 0.1 \
  --decay_epoch "${FT_EPOCH}" \
  --resume "${BASE_CKPT}" \
  --seed "${SEED}" \
  --freeze_bn "${BGDNET_FREEZE_BN}" \
  --max_weight "${FT_MAX_WEIGHT:-1.5}" \
  --use_tversky "${FT_USE_TVERSKY:-True}" \
  --seg_bce_w "${FT_SEG_BCE_W:-0.5}" \
  --seg_dice_w "${FT_SEG_DICE_W:-1.0}" \
  --seg_tversky_w "${FT_SEG_TVERSKY_W:-0.5}" \
  --tversky_alpha_fp "${FT_TVERSKY_ALPHA_FP:-0.3}" \
  --tversky_beta_fn "${FT_TVERSKY_BETA_FN:-0.7}" \
  --tversky_gamma "${FT_TVERSKY_GAMMA:-0.75}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 12b started: real-only fine-tune"
echo "     ratio:      ${RATIO_TAG}"
echo "     seed:       ${SEED}"
echo "     base_ckpt:  ${BASE_CKPT}"
echo "     real_json:  ${REAL_ONLY_JSON}"
echo "     save_dir:   ${FT_SAVE_DIR}"
echo "     epoch:      ${FT_EPOCH}"
echo "     lr:         ${FT_LR}"
echo "     beta:       ${FT_BETA}"
echo "     pid:        $(cat ${PID_FILE})"
echo "     log:        ${LOG_FILE}"
echo ""
echo "Check log:"
echo "  tail -f ${LOG_FILE}"