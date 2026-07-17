#!/bin/bash

cd /data/zjy_work/Baseline || exit 1

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

mkdir -p /data/zjy_work/Baseline/log
mkdir -p /data/zjy_work/Baseline/model_out/ISIC2018/UNet_all

LOG_FILE="/data/zjy_work/Baseline/log/train_unet_all_$(date +%Y%m%d_%H%M%S).log"

nohup python -u train_seg_baseline.py \
  --model unet \
  --epoch 200 \
  --lr 1e-4 \
  --optimizer AdamW \
  --batchsize 4 \
  --img_size 352 \
  --train_path1 /data/zjy_work/ISIC2018/train/ \
  --test_path /data/zjy_work/ISIC2018/test/ \
  --train_list1 /data/zjy_work/data_txt/train_all.txt \
  --test_list /data/zjy_work/data_txt/test.txt \
  --train_save /data/zjy_work/Baseline/model_out/ISIC2018/UNet_all/ \
  --device cuda:0 \
  --num_workers 16 \
  --print_freq 50 \
  > "${LOG_FILE}" 2>&1 &

echo "UNet training started."
echo "PID: $!"
echo "Log: ${LOG_FILE}"


# UNet training started.5% data
# PID: 228270
# Log: /data/zjy_work/Baseline/log/train_unet_5%_20260511_220140.log