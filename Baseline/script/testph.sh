#!/bin/bash

cd /data/zjy_work/Baseline || exit 1

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

LOG_DIR="/data/zjy_work/Baseline/log"

MODEL_DIR="/data/zjy_work/Baseline/model_out/PH2_ours_Seed0/UNet_real_gen_full_sbg_real5_syn15_seed0_ph2fix"

OUT_DIR="/data/zjy_work/Baseline/model_out/PH2_eval_Seed0/UNet_real_gen_full_sbg_real5_syn15_seed0_ph2fix"

PTH_PATH="${MODEL_DIR}/unet-best.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_DIR}"

LOG_FILE="${LOG_DIR}/test_ph2_unet_full_sbg_real5_syn15_seed0_ph2fix_$(date +%Y%m%d_%H%M%S).log"

nohup python -u test_seg_baseline.py \
  --model unet \
  --num_classes 1 \
  --testsize 352 \
  --pth_path "${PTH_PATH}" \
  --test_data_path /data/zjy_work/PH2/test/ \
  --test_list /data/zjy_work/data_txt/test_PH2.txt \
  --test_mode ph2 \
  --threshold 0.5 \
  --out_dir "${OUT_DIR}" \
  --save_csv per_image_metrics.csv \
  --save_summary_csv summary_metrics.csv \
  --device cuda:0 \
  --print_freq 10 \
  > "${LOG_FILE}" 2>&1 &

PID=$!

echo "PH2 UNet testing started."
echo "PID: ${PID}"
echo "Checkpoint: ${PTH_PATH}"
echo "Output: ${OUT_DIR}"
echo "Log: ${LOG_FILE}"
