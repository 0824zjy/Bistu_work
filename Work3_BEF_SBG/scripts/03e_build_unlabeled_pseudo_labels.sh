#!/usr/bin/env bash
set -euo pipefail

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

UNLABELED_IMAGE_DIR="${UNLABELED_IMAGE_DIR:-}"
if [ -z "${UNLABELED_IMAGE_DIR}" ] || [ ! -d "${UNLABELED_IMAGE_DIR}" ]; then
  echo "[ERROR] Set UNLABELED_IMAGE_DIR to a valid image directory."
  exit 1
fi

PSEUDO_OUT_ROOT="${PSEUDO_OUT_ROOT:-${WORK_ROOT}/results/unlabeled_pseudo_${RATIO_TAG}_seed${SEED}}"
PSEUDO_JSONL="${PSEUDO_JSONL:-${PSEUDO_OUT_ROOT}/accepted_pseudo.jsonl}"
UNLABELED_LIST="${UNLABELED_HETERO_PRED_ROOT}/unlabeled_stems.txt"
SOURCE_JSON="${UNLABELED_HETERO_PRED_ROOT}/sources.json"
FUSED_DIR="${UNLABELED_HETERO_PRED_ROOT}/fused"
mkdir -p "${UNLABELED_HETERO_PRED_ROOT}" "${PSEUDO_OUT_ROOT}" "${LOG_ROOT}/infer"

export UNLABELED_IMAGE_DIR PSEUDO_OUT_ROOT PSEUDO_JSONL UNLABELED_LIST SOURCE_JSON FUSED_DIR
export BGDNET_ARCH_WEIGHT="${BGDNET_ARCH_WEIGHT:-1.0}"
export CNN_ARCH_WEIGHT="${CNN_ARCH_WEIGHT:-1.0}"
export SAM_ARCH_WEIGHT="${SAM_ARCH_WEIGHT:-1.0}"
export BGDNET_PSEUDO_BATCHSIZE="${BGDNET_PSEUDO_BATCHSIZE:-4}"
export CNN_PSEUDO_BATCHSIZE="${CNN_PSEUDO_BATCHSIZE:-8}"
export SAM_PSEUDO_BATCHSIZE="${SAM_PSEUDO_BATCHSIZE:-1}"

LOG_FILE="${LOG_ROOT}/infer/build_unlabeled_pseudo_hetero_${RATIO_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/03e_build_unlabeled_pseudo_${RATIO_TAG}_seed${SEED}.pid"

nohup bash -c '
set -euo pipefail
source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
export PYTHONPATH="/data/zjy_work:${PYTHONPATH:-}"

python - <<PY
import os
root = r"${UNLABELED_IMAGE_DIR}"
out = r"${UNLABELED_LIST}"
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
names = sorted(name for name in os.listdir(root) if os.path.splitext(name)[1].lower() in exts)
with open(out, "w", encoding="utf-8") as f:
    for name in names:
        f.write(os.path.splitext(name)[0] + "\n")
print("unlabeled images:", len(names))
PY

for fold in $(seq 0 $((N_FOLDS - 1))); do
  python -u "${WORK_ROOT}/teacher/predict_directory_with_teacher.py" \
    --teacher_type bgdnet \
    --checkpoint "${TEACHER_CKPT_ROOT}/fold${fold}/BGDNet-best.pth" \
    --image_dir "${UNLABELED_IMAGE_DIR}" \
    --list_txt "${UNLABELED_LIST}" \
    --out_dir "${UNLABELED_HETERO_PRED_ROOT}/bgdnet/fold${fold}" \
    --device cuda:0 --image_size 352 \
    --batch_size "${BGDNET_PSEUDO_BATCHSIZE}" --num_workers 4 \
    --bgdnet_root "${BGDNET_ROOT}"
done

for fold in $(seq 0 $((N_FOLDS - 1))); do
  python -u "${WORK_ROOT}/teacher/predict_directory_with_teacher.py" \
    --teacher_type cnn \
    --checkpoint "${CNN_TEACHER_CKPT_ROOT}/fold${fold}/cnn-best.pth" \
    --image_dir "${UNLABELED_IMAGE_DIR}" \
    --list_txt "${UNLABELED_LIST}" \
    --out_dir "${UNLABELED_HETERO_PRED_ROOT}/cnn/fold${fold}" \
    --device cuda:0 --image_size 352 \
    --batch_size "${CNN_PSEUDO_BATCHSIZE}" --num_workers 4 --amp
done

for fold in $(seq 0 $((N_FOLDS - 1))); do
  python -u "${WORK_ROOT}/teacher/predict_directory_with_teacher.py" \
    --teacher_type sam_adapter \
    --checkpoint "${SAM_TEACHER_CKPT_ROOT}/fold${fold}/sam_adapter-best.pth" \
    --image_dir "${UNLABELED_IMAGE_DIR}" \
    --list_txt "${UNLABELED_LIST}" \
    --out_dir "${UNLABELED_HETERO_PRED_ROOT}/sam/fold${fold}" \
    --device cuda:0 --image_size 1024 \
    --batch_size "${SAM_PSEUDO_BATCHSIZE}" --num_workers 2 --amp
done

python - <<PY
import json
n = int("${N_FOLDS}")
sources = []
for fold in range(n):
    for name, root, weight in [
        ("BGDNet", r"${UNLABELED_HETERO_PRED_ROOT}/bgdnet", float("${BGDNET_ARCH_WEIGHT}")),
        ("ConvNeXt", r"${UNLABELED_HETERO_PRED_ROOT}/cnn", float("${CNN_ARCH_WEIGHT}")),
        ("SAMAdapter", r"${UNLABELED_HETERO_PRED_ROOT}/sam", float("${SAM_ARCH_WEIGHT}")),
    ]:
        base = f"{root}/fold{fold}"
        sources.append({
            "name": f"{name}-fold{fold}",
            "pred_mask_dir": f"{base}/pred_masks",
            "pred_boundary_dir": f"{base}/pred_boundaries",
            "weight": weight / n,
        })
with open(r"${SOURCE_JSON}", "w", encoding="utf-8") as f:
    json.dump({"sources": sources}, f, ensure_ascii=False, indent=2)
PY

python -u "${WORK_ROOT}/teacher/fuse_prediction_directories.py" \
  --sources_json "${SOURCE_JSON}" \
  --list_txt "${UNLABELED_LIST}" \
  --out_dir "${FUSED_DIR}" \
  --entropy_lambda "${HETERO_ENTROPY_LAMBDA:-1.0}" \
  --variance_lambda "${HETERO_VARIANCE_LAMBDA:-2.0}" \
  --confidence_gamma "${HETERO_CONFIDENCE_GAMMA:-1.0}"

python -u "${WORK_ROOT}/teacher/build_unlabeled_pseudo_from_predictions.py" \
  --unlabeled_image_dir "${UNLABELED_IMAGE_DIR}" \
  --fused_prediction_dir "${FUSED_DIR}" \
  --out_root "${PSEUDO_OUT_ROOT}" \
  --out_jsonl "${PSEUDO_JSONL}" \
  --pseudo_threshold "${PSEUDO_THRESHOLD:-0.5}" \
  --min_mean_reliability "${PSEUDO_MIN_MEAN_REL:-0.70}" \
  --min_boundary_reliability "${PSEUDO_MIN_BOUNDARY_REL:-0.55}" \
  --min_foreground_ratio "${PSEUDO_MIN_FG_RATIO:-0.005}" \
  --max_foreground_ratio "${PSEUDO_MAX_FG_RATIO:-0.80}"
' > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[OK] Step 03e started: heterogeneous unlabeled pseudo labels"
echo "     image dir: ${UNLABELED_IMAGE_DIR}"
echo "     jsonl: ${PSEUDO_JSONL}"
echo "     log: ${LOG_FILE}"
