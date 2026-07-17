#!/usr/bin/env bash
set -e

cd /data/zjy_work/BGDNet
export CUDA_VISIBLE_DEVICES=2

PTH=/data/zjy_work/BGDNet/model_out/ISIC2018_real_gen/10p_stage1-2_r3g1_4_30/BGDNet-best.pth

OUT_DIR=/data/zjy_work/BGDNet/model_out/ISIC2018_real_gen/10p_stage1-2_r3g1_4_30/eval_test
mkdir -p ${OUT_DIR}

nohup python /data/zjy_work/BGDNet/BGDiff_test.py \
  --pth_path ${PTH} \
  --testsize 352 \
  --test_data_path /data/zjy_work/ISIC2018/test/ \
  --test_list /data/zjy_work/data_txt/test.txt \
  --out_dir ${OUT_DIR} \
  --save_csv per_image_metrics.csv \
  --save_summary_csv summary_metrics.csv \
  > ${OUT_DIR}/nohup_test.log 2>&1 &

echo "Testing started. Log: ${OUT_DIR}/nohup_test.log"
