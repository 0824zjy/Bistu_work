#!/usr/bin/env bash
set -euo pipefail

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
source /data/zjy_work/Work3_BEF_SBG/scripts/00_standard_env.sh

mkdir -p "${LOG_ROOT}/scoring" "${GEN_HETERO_PRED_ROOT}"
export BGDNET_SWIN_VARIANT="standard"
export BGDNET_LOAD_BACKBONE_PRETRAIN="0"
export BGDNET_STRICT_PRETRAIN="0"
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

GEN_LIST="${GEN_HETERO_PRED_ROOT}/generated_stems.txt"
SOURCE_JSON="${GEN_HETERO_PRED_ROOT}/sources.json"
FUSED_DIR="${GEN_HETERO_PRED_ROOT}/fused"
RELIABILITY_DIR="${GEN_OUT_DIR}/teacher_reliability"
HARDNESS_DIR="${GEN_OUT_DIR}/teacher_hardness"

LOG_FILE="${LOG_ROOT}/scoring/score_generated_hetero_${RATIO_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/10_score_generated_hetero_${RATIO_TAG}_seed${SEED}.pid"

export GEN_LIST SOURCE_JSON FUSED_DIR RELIABILITY_DIR HARDNESS_DIR
export BGDNET_ARCH_WEIGHT="${BGDNET_ARCH_WEIGHT:-1.0}"
export CNN_ARCH_WEIGHT="${CNN_ARCH_WEIGHT:-1.0}"
export SAM_ARCH_WEIGHT="${SAM_ARCH_WEIGHT:-1.0}"
export BGDNET_SCORE_BATCHSIZE="${BGDNET_SCORE_BATCHSIZE:-4}"
export CNN_SCORE_BATCHSIZE="${CNN_SCORE_BATCHSIZE:-8}"
export SAM_SCORE_BATCHSIZE="${SAM_SCORE_BATCHSIZE:-1}"

nohup bash -c '
set -euo pipefail
source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh
export PYTHONPATH="/data/zjy_work:${PYTHONPATH:-}"

python - <<PY
import os
root = r"${GEN_OUT_DIR}/images"
out = r"${GEN_LIST}"
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
names = sorted(name for name in os.listdir(root) if os.path.splitext(name)[1].lower() in exts)
with open(out, "w", encoding="utf-8") as f:
    for name in names:
        f.write(os.path.splitext(name)[0] + "\n")
print("generated samples:", len(names))
PY

for fold in $(seq 0 $((N_FOLDS - 1))); do
  checkpoint="${TEACHER_CKPT_ROOT}/fold${fold}/BGDNet-best.pth"
  out_dir="${GEN_HETERO_PRED_ROOT}/bgdnet/fold${fold}"
  python -u "${WORK_ROOT}/teacher/predict_directory_with_teacher.py" \
    --teacher_type bgdnet \
    --checkpoint "${checkpoint}" \
    --image_dir "${GEN_OUT_DIR}/images" \
    --list_txt "${GEN_LIST}" \
    --out_dir "${out_dir}" \
    --device cuda:0 \
    --image_size 352 \
    --batch_size "${BGDNET_SCORE_BATCHSIZE}" \
    --num_workers 4 \
    --bgdnet_root "${BGDNET_ROOT}"
done

for fold in $(seq 0 $((N_FOLDS - 1))); do
  checkpoint="${CNN_TEACHER_CKPT_ROOT}/fold${fold}/cnn-best.pth"
  out_dir="${GEN_HETERO_PRED_ROOT}/cnn/fold${fold}"
  python -u "${WORK_ROOT}/teacher/predict_directory_with_teacher.py" \
    --teacher_type cnn \
    --checkpoint "${checkpoint}" \
    --image_dir "${GEN_OUT_DIR}/images" \
    --list_txt "${GEN_LIST}" \
    --out_dir "${out_dir}" \
    --device cuda:0 \
    --image_size 352 \
    --batch_size "${CNN_SCORE_BATCHSIZE}" \
    --num_workers 4 \
    --amp
done

for fold in $(seq 0 $((N_FOLDS - 1))); do
  checkpoint="${SAM_TEACHER_CKPT_ROOT}/fold${fold}/sam_adapter-best.pth"
  out_dir="${GEN_HETERO_PRED_ROOT}/sam/fold${fold}"
  python -u "${WORK_ROOT}/teacher/predict_directory_with_teacher.py" \
    --teacher_type sam_adapter \
    --checkpoint "${checkpoint}" \
    --image_dir "${GEN_OUT_DIR}/images" \
    --list_txt "${GEN_LIST}" \
    --out_dir "${out_dir}" \
    --device cuda:0 \
    --image_size 1024 \
    --batch_size "${SAM_SCORE_BATCHSIZE}" \
    --num_workers 2 \
    --amp
done

python - <<PY
import json
sources = []
n = int("${N_FOLDS}")
for fold in range(n):
    for teacher_type, root, weight in [
        ("BGDNet", r"${GEN_HETERO_PRED_ROOT}/bgdnet", float("${BGDNET_ARCH_WEIGHT}")),
        ("ConvNeXt", r"${GEN_HETERO_PRED_ROOT}/cnn", float("${CNN_ARCH_WEIGHT}")),
        ("SAMAdapter", r"${GEN_HETERO_PRED_ROOT}/sam", float("${SAM_ARCH_WEIGHT}")),
    ]:
        base = f"{root}/fold{fold}"
        sources.append({
            "name": f"{teacher_type}-fold{fold}",
            "pred_mask_dir": f"{base}/pred_masks",
            "pred_boundary_dir": f"{base}/pred_boundaries",
            "weight": weight / n,
        })
with open(r"${SOURCE_JSON}", "w", encoding="utf-8") as f:
    json.dump({"sources": sources}, f, ensure_ascii=False, indent=2)
print("sources:", len(sources))
PY

python -u "${WORK_ROOT}/teacher/fuse_prediction_directories.py" \
  --sources_json "${SOURCE_JSON}" \
  --list_txt "${GEN_LIST}" \
  --out_dir "${FUSED_DIR}" \
  --entropy_lambda "${HETERO_ENTROPY_LAMBDA:-1.0}" \
  --variance_lambda "${HETERO_VARIANCE_LAMBDA:-2.0}" \
  --confidence_gamma "${HETERO_CONFIDENCE_GAMMA:-1.0}"

python -u "${WORK_ROOT}/scoring/score_generated_samples.py" \
  --gen_image_dir "${GEN_OUT_DIR}/images" \
  --gen_mask_dir "${GEN_OUT_DIR}/masks" \
  --gen_prior_dir "${GEN_OUT_DIR}/boundary_prior" \
  --fused_prediction_dir "${FUSED_DIR}" \
  --out_csv "${GEN_SCORE_CSV}" \
  --out_jsonl "${GEN_ACCEPTED_JSONL}" \
  --out_reliability_dir "${RELIABILITY_DIR}" \
  --out_hardness_dir "${HARDNESS_DIR}" \
  --cons_threshold "${CONS_THRESHOLD:-0.80}" \
  --beta_hard "${BETA_HARD:-0.5}" \
  --agreement_gamma "${RELIABILITY_AGREEMENT_GAMMA:-1.0}" \
  --reliability_floor "${RELIABILITY_FLOOR:-0.05}"
' > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[OK] Step 10 started: heterogeneous teacher scoring"
echo "     prediction root: ${GEN_HETERO_PRED_ROOT}"
echo "     csv: ${GEN_SCORE_CSV}"
echo "     jsonl: ${GEN_ACCEPTED_JSONL}"
echo "     log: ${LOG_FILE}"
