#!/usr/bin/env bash
set -e

cd /data/zjy_work/BGDNet

export CUDA_VISIBLE_DEVICES=0,1

SAVE_DIR=./model_out/ISIC2018/5p_cn_only_r1g1_2gpu
mkdir -p ${SAVE_DIR}

nohup python BGDiff_train.py \
  --epoch 200 \
  --batchsize 8 \
  --img_size 352 \
  --n_gpu 2 \
  --train_path1 /data/zjy_work/ISIC2018/train/ \
  --train_path2 /data/zjy_work/ISIC2018/exp_mask2img_cn_only/ \
  --train_list1 /data/zjy_work/data_txt/train_5p.txt \
  --train_list2 /data/zjy_work/data_txt/mix_ratio/train_5p__cn_only__real1_gen1__nreal130_ngen130.txt \
  --test_path /data/zjy_work/ISIC2018/test/ \
  --test_list /data/zjy_work/data_txt/test.txt \
  --train_save ${SAVE_DIR} \
  > ${SAVE_DIR}/nohup_train.log 2>&1 &

echo "2-GPU training started. Log: ${SAVE_DIR}/nohup_train.log"
