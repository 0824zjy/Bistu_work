#!/usr/bin/env bash
set -e

cd /data/zjy_work/BGDNet

export CUDA_VISIBLE_DEVICES=1

SAVE_DIR=/data/zjy_work/BGDNet/model_out/ISIC2018_real_only/20p
mkdir -p ${SAVE_DIR}

nohup python BGDiff_train.py \
  --epoch 200 \
  --batchsize 4 \
  --img_size 352 \
  --n_gpu 1 \
  --train_path1 /data/zjy_work/ISIC2018/train/ \
  --train_list1 /data/zjy_work/data_txt/train_20p.txt \
  --test_path /data/zjy_work/ISIC2018/test/ \
  --test_list /data/zjy_work/data_txt/test.txt \
  --train_save ${SAVE_DIR} \
  > ${SAVE_DIR}/nohup_train_real_only_20p.log 2>&1 &
echo "Training started. Log: ${SAVE_DIR}/nohup_train_real_only_20p.log"
