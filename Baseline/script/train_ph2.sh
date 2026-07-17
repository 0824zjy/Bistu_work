#!/bin/bash

cd /data/zjy_work/Baseline || exit 1

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

LOG_DIR="/data/zjy_work/Baseline/log"
SAVE_DIR="/data/zjy_work/Baseline/model_out/PH2_ours_Seed0/UNet_real_gen_full_sbg_real5_syn15_seed0_ph2fix"

mkdir -p "${LOG_DIR}"
mkdir -p "${SAVE_DIR}"

LOG_FILE="${LOG_DIR}/train_ph2_unet_full_sbg_real5_syn15_seed0_ph2fix_$(date +%Y%m%d_%H%M%S).log"

nohup python -u train_seg_baseline.py \
  --model unet \
  --epoch 200 \
  --lr 1e-4 \
  --optimizer AdamW \
  --batchsize 4 \
  --img_size 352 \
  --train_path1 /data/zjy_work/PH2/train/ \
  --train_path2 /data/zjy_work/BGDiff/Ours/ablation_sbg/results/full_sbg_real5_syn15 \
  --test_path /data/zjy_work/PH2/test/ \
  --train_list1 /data/zjy_work/data_txt/train_PH2_5p.txt \
  --train_list2 /data/zjy_work/data_txt/train_PH2_full_sbg_real5_syn15_seed0.txt \
  --test_list /data/zjy_work/data_txt/test_PH2.txt \
  --train_mode1 ph2 \
  --train_mode2 paired \
  --test_mode ph2 \
  --train_save "${SAVE_DIR}" \
  --device cuda:0 \
  --num_workers 16 \
  --print_freq 50 \
  > "${LOG_FILE}" 2>&1 &

PID=$!

echo "UNet PH2 real+generated training started."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Save directory: ${SAVE_DIR}"
