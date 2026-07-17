#!/usr/bin/env bash
set -e

# ============================================================
# Step 12: Train final BGDNet-BEF
# ============================================================

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh
# Train a new standard final model.
export BGDNET_SWIN_VARIANT="standard"
export BGDNET_LOAD_BACKBONE_PRETRAIN="1"
export BGDNET_STRICT_PRETRAIN="1"

mkdir -p "${LOG_ROOT}/segmentation"
mkdir -p "${PID_DIR}"
mkdir -p "${FINAL_SEG_SAVE_DIR}"

# -----------------------------
# CUDA / stability settings
# -----------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# This environment has cuDNN instability for BGDNet backbone.
# Keep disabled by default.
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"

# batchsize=4: do not freeze BN by default.
# If you later set FINAL_BATCHSIZE=1, you may set BGDNET_FREEZE_BN=1 manually.
export BGDNET_FREEZE_BN="${BGDNET_FREEZE_BN:-0}"

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# -----------------------------
# Training hyperparameters
# -----------------------------
FINAL_EPOCH="${FINAL_EPOCH:-200}"
FINAL_BATCHSIZE="${FINAL_BATCHSIZE:-4}"
FINAL_LR="${FINAL_LR:-1e-4}"
FINAL_ALPHA="${FINAL_ALPHA:-1.0}"

# Important:
# after changing loss to BCE + Dice and balanced boundary loss,
# beta=0.25 is safer than 0.4.
FINAL_BETA="${FINAL_BETA:-0.25}"

FINAL_CLIP="${FINAL_CLIP:-0.5}"
FINAL_IMG_SIZE="${FINAL_IMG_SIZE:-352}"
FINAL_NUM_WORKERS="${FINAL_NUM_WORKERS:-8}"

# Important:
# real labeled data is limited, so enable augmentation by default.
FINAL_AUG="${FINAL_AUG:-True}"

FINAL_OPTIMIZER="${FINAL_OPTIMIZER:-AdamW}"
FINAL_DECAY_RATE="${FINAL_DECAY_RATE:-0.1}"
FINAL_DECAY_EPOCH="${FINAL_DECAY_EPOCH:-200}"

TRAIN_JSON="${TRAIN_JSON:-${WEIGHTED_TRAIN_JSON}}"

# Optional validation/test list.
# Leave empty to use all images under ISIC_TEST_ROOT/Images and Masks.
FINAL_TEST_LIST="${FINAL_TEST_LIST:-}"

if [ ! -f "${TRAIN_JSON}" ]; then
  echo "[ERROR] weighted train json not found:"
  echo "        ${TRAIN_JSON}"
  echo ""
  echo "Please run Step 11 first:"
  echo "  RATIO_TAG=${RATIO_TAG} SEED=${SEED} bash /data/zjy_work/Work3_BEF_SBG/scripts/11_make_weighted_train_json.sh"
  exit 1
fi

# Optional test list argument.
TEST_LIST_ARGS=()
if [ -n "${FINAL_TEST_LIST}" ]; then
  if [ ! -f "${FINAL_TEST_LIST}" ]; then
    echo "[ERROR] FINAL_TEST_LIST does not exist: ${FINAL_TEST_LIST}"
    exit 1
  fi
  TEST_LIST_ARGS=(--test_list "${FINAL_TEST_LIST}")
fi

# -----------------------------
# Preflight: check bad weights
# -----------------------------
python - <<PY
import json, math, os, sys

path = "${TRAIN_JSON}"
bad = 0
total = 0
zero = 0
missing = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        total += 1
        item = json.loads(line)

        image = item.get("image", "")
        mask = item.get("mask", "")

        if not os.path.exists(image) or not os.path.exists(mask):
            missing += 1

        try:
            w = float(item.get("weight", 1.0))
        except Exception:
            bad += 1
            continue

        if not math.isfinite(w):
            bad += 1
        elif w <= 1e-8:
            zero += 1

print("[Preflight weighted json]")
print(f"  path    = {path}")
print(f"  total   = {total}")
print(f"  bad_w   = {bad}")
print(f"  zero_w  = {zero}")
print(f"  missing = {missing}")

if total == 0:
    raise SystemExit("[ERROR] weighted json is empty.")

if bad > 0:
    raise SystemExit("[ERROR] bad weights found in weighted json. Please fix before training.")

if missing > 0:
    raise SystemExit("[ERROR] missing image/mask paths found in weighted json. Please fix before training.")
PY

# -----------------------------
# Log / PID
# -----------------------------
LOG_FILE="${LOG_ROOT}/segmentation/train_final_bgdnet_bef_${RATIO_TAG}_seed${SEED}_bs${FINAL_BATCHSIZE}_beta${FINAL_BETA}_aug${FINAL_AUG}_cudnn${BGDNET_DISABLE_CUDNN}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/12_train_final_bgdnet_bef_${RATIO_TAG}.pid"

# -----------------------------
# Start training
# -----------------------------
nohup python -u "${WORK_ROOT}/segmentation/train_BGDNet_BEF.py" \
  --weighted_train_json "${TRAIN_JSON}" \
  --test_path "${ISIC_TEST_ROOT}" \
  "${TEST_LIST_ARGS[@]}" \
  --epoch "${FINAL_EPOCH}" \
  --lr "${FINAL_LR}" \
  --batchsize "${FINAL_BATCHSIZE}" \
  --img_size "${FINAL_IMG_SIZE}" \
  --train_save "${FINAL_SEG_SAVE_DIR}" \
  --alpha "${FINAL_ALPHA}" \
  --beta "${FINAL_BETA}" \
  --clip "${FINAL_CLIP}" \
  --augmentation "${FINAL_AUG}" \
  --num_workers "${FINAL_NUM_WORKERS}" \
  --optimizer "${FINAL_OPTIMIZER}" \
  --decay_rate "${FINAL_DECAY_RATE}" \
  --decay_epoch "${FINAL_DECAY_EPOCH}" \
  --seed "${SEED}" \
  --freeze_bn "${BGDNET_FREEZE_BN}" \
  --max_weight "${FINAL_MAX_WEIGHT:-1.5}" \
  --use_tversky "${FINAL_USE_TVERSKY:-True}" \
  --seg_bce_w "${FINAL_SEG_BCE_W:-0.5}" \
  --seg_dice_w "${FINAL_SEG_DICE_W:-1.0}" \
  --seg_tversky_w "${FINAL_SEG_TVERSKY_W:-0.5}" \
  --tversky_alpha_fp "${FINAL_TVERSKY_ALPHA_FP:-0.3}" \
  --tversky_beta_fn "${FINAL_TVERSKY_BETA_FN:-0.7}" \
  --tversky_gamma "${FINAL_TVERSKY_GAMMA:-0.75}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 12 started: train final BGDNet-BEF"
echo "     ratio:       ${RATIO_TAG}"
echo "     seed:        ${SEED}"
echo "     gpu:         ${CUDA_VISIBLE_DEVICES}"
echo "     cudnn off:   ${BGDNET_DISABLE_CUDNN}"
echo "     freeze bn:   ${BGDNET_FREEZE_BN}"
echo "     train_json:  ${TRAIN_JSON}"
echo "     test_root:   ${ISIC_TEST_ROOT}"
echo "     test_list:   ${FINAL_TEST_LIST:-<all test images>}"
echo "     save_dir:    ${FINAL_SEG_SAVE_DIR}"
echo "     epoch:       ${FINAL_EPOCH}"
echo "     batchsize:   ${FINAL_BATCHSIZE}"
echo "     lr:          ${FINAL_LR}"
echo "     beta:        ${FINAL_BETA}"
echo "     aug:         ${FINAL_AUG}"
echo "     pid:         $(cat ${PID_FILE})"
echo "     log:         ${LOG_FILE}"
echo "     kill:        kill \$(cat ${PID_FILE})"
echo ""
echo "Check log:"
echo "  tail -f ${LOG_FILE}"
