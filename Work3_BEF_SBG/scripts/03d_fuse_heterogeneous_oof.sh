#!/usr/bin/env bash
set -euo pipefail
source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

LIST_TXT="${SPLIT_DIR}/low_${RATIO_TAG}_all.txt"
SOURCE_JSON="${OOF_HETERO_OUT_DIR}/sources.json"
mkdir -p "${OOF_HETERO_OUT_DIR}" "${LOG_ROOT}/infer"

BGDNET_ARCH_WEIGHT="${BGDNET_ARCH_WEIGHT:-1.0}"
CNN_ARCH_WEIGHT="${CNN_ARCH_WEIGHT:-1.0}"
SAM_ARCH_WEIGHT="${SAM_ARCH_WEIGHT:-1.0}"
cat > "${SOURCE_JSON}" <<JSON
{
  "sources": [
    {
      "name": "BGDNet-OOF",
      "pred_mask_dir": "${OOF_OUT_DIR}/pred_masks",
      "pred_boundary_dir": "${OOF_OUT_DIR}/pred_boundaries",
      "weight": ${BGDNET_ARCH_WEIGHT}
    },
    {
      "name": "ConvNeXt-OOF",
      "pred_mask_dir": "${OOF_CNN_OUT_DIR}/pred_masks",
      "pred_boundary_dir": "${OOF_CNN_OUT_DIR}/pred_boundaries",
      "weight": ${CNN_ARCH_WEIGHT}
    },
    {
      "name": "SAMAdapter-OOF",
      "pred_mask_dir": "${OOF_SAM_OUT_DIR}/pred_masks",
      "pred_boundary_dir": "${OOF_SAM_OUT_DIR}/pred_boundaries",
      "weight": ${SAM_ARCH_WEIGHT}
    }
  ]
}
JSON

LOG_FILE="${LOG_ROOT}/infer/fuse_hetero_oof_${RATIO_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/03d_fuse_hetero_oof_${RATIO_TAG}_seed${SEED}.pid"

nohup env PYTHONPATH="/data/zjy_work:${PYTHONPATH:-}" \
python -u "${WORK_ROOT}/teacher/fuse_prediction_directories.py" \
  --sources_json "${SOURCE_JSON}" \
  --list_txt "${LIST_TXT}" \
  --mask_dir "${ISIC_TRAIN_MASK_DIR}" \
  --out_dir "${OOF_HETERO_OUT_DIR}" \
  --entropy_lambda "${HETERO_ENTROPY_LAMBDA:-1.0}" \
  --variance_lambda "${HETERO_VARIANCE_LAMBDA:-2.0}" \
  --confidence_gamma "${HETERO_CONFIDENCE_GAMMA:-1.0}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[OK] Step 03d started: fuse heterogeneous OOF"
echo "     source json: ${SOURCE_JSON}"
echo "     log: ${LOG_FILE}"
