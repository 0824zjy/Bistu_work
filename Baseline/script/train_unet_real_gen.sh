#!/bin/bash

cd /data/zjy_work/Baseline || exit 1

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

mkdir -p /data/zjy_work/Baseline/log
mkdir -p /data/zjy_work/Baseline/model_out/PH2_ours_Seed0/UNet_real_gen_full_sbg_real5_syn15_seed0

LOG_FILE="/data/zjy_work/Baseline/log/train_ph2_unet_full_sbg_real5_syn15_seed0_$(date +%Y%m%d_%H%M%S).log"

nohup python -u train_seg_baseline.py \
  --model unet \
  --epoch 200 \
  --lr 1e-4 \
  --optimizer AdamW \
  --batchsize 4 \
  --img_size 352 \
  --train_path1 /data/zjy_work/PH2/train/ \
  --train_path2 /data/zjy_work/BGDiff/Ours/ablation_sbg/results/full_sbg_real5_syn15\
  --test_path /data/zjy_work/PH2/test \
  --train_list1 /data/zjy_work/data_txt/train_PH2_5p.txt \
  --train_list2 /data/zjy_work/data_txt/train_PH2_full_sbg_real5_syn15_seed0.txt\
  --test_list /data/zjy_work/data_txt/test_PH2.txt \
  --train_save /data/zjy_work/Baseline/model_out/PH2_ours_Seed0/UNet_real_gen_full_sbg_real5_syn15_seed0 \
  --device cuda:0 \
  --num_workers 16 \
  --print_freq 50 \
  > "${LOG_FILE}" 2>&1 &

echo "UNet real+gen training started."
echo "PID: $!"
echo "Log: ${LOG_FILE}"


# (ISIC) root@de5b9aa5f049:/data/zjy_work/Baseline/script# bash train_unet_real_gen.sh 
# UNet real+gen training started.
# PID: 32706
# Log: /data/zjy_work/Baseline/log/train_unet_full_sbg_real5_syn10_seed1_20260602_100727.log
# (ISIC) root@de5b9aa5f049:/data/zjy_work/Baseline/script# bash train_unet_real_gen.sh 
# UNet real+gen training started.
# PID: 33549
# Log: /data/zjy_work/Baseline/log/train_unet_maskonly_real5_syn10_seed1_20260602_100828.log